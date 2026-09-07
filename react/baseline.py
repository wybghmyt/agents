import argparse
import asyncio
import json
import os
import re
import time
from concurrent.futures import ThreadPoolExecutor
from functools import partial

import numpy as np
import yaml
from openai import AsyncOpenAI

os.environ.setdefault("ALFWORLD_DATA", "/root/autodl-tmp/alfworld")

from alfworld.agents.environment.alfred_tw_env import AlfredTWEnv

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)

# 任务类型识别: 顺序很重要, 先匹配更具体的类型
TASK_TYPE_PATTERNS = [
    ("pick_two_obj_and_place", "pick_two_obj"),
    ("look_at_obj_in_light", "look_at_obj"),
    ("pick_clean_then_place_in_recep", "pick_clean_then_place"),
    ("pick_heat_then_place_in_recep", "pick_heat_then_place"),
    ("pick_cool_then_place_in_recep", "pick_cool_then_place"),
    ("pick_and_place_simple", "pick_and_place"),
]

# 展示顺序 (与结果表模板一致)
TASK_TYPE_ORDER = [
    "pick_and_place",
    "pick_clean_then_place",
    "pick_heat_then_place",
    "pick_cool_then_place",
    "look_at_obj",
    "pick_two_obj",
]

# 结果表中的任务类型展示名 (alfworld 官方全名)
DISPLAY_NAME = {
    "pick_and_place": "pick_and_place",
    "pick_clean_then_place": "pick_clean_then_place",
    "pick_heat_then_place": "pick_heat_then_place",
    "pick_cool_then_place": "pick_cool_then_place",
    "look_at_obj": "look_at_obj_in_light",
    "pick_two_obj": "pick_two_obj_and_place",
}

# prompt 文件中的键名简写,此处用于动态绑定对应任务的few-shot (react_<key>_0/1/2)
PROMPT_KEY = {
    "pick_and_place": "put",
    "pick_clean_then_place": "clean",
    "pick_heat_then_place": "heat",
    "pick_cool_then_place": "cool",
    "look_at_obj": "examine",
    "pick_two_obj": "puttwo",
}

# ALFWorld 合法动作动词 (think 允许带冒号, 如 "think: I need to ...")
VALID_VERBS = (
    "think",
    "go to",
    "take",
    "put",
    "open",
    "close",
    "clean",
    "heat",
    "cool",
    "use",
    "examine",
    "look",
    "inventory",
)

# 上下文字符预算 (约 6500 token, 留有余量; max-model-len=8192)
CTX_CHAR_BUDGET = 26000


# 将文件名中的游戏类型进行简化
def parse_task_type(gamefile: str) -> str:
    for substr, short in TASK_TYPE_PATTERNS:
        if substr in gamefile:
            return short
    raise ValueError(f"无法识别任务类型: {gamefile}")


def normalize_action(raw: str):
    """动作规范化: 去掉 Action: 前缀 / 引号 / 句号 / 数字编号等, 返回 (规范化动作, 是否合法)。"""
    a = raw.strip()
    a = re.sub(r"^>+\s*", "", a)  # "> go to ..."
    a = re.sub(r"^(action|act)\s*[:：]\s*", "", a, flags=re.I)  # "Action: ..."
    a = re.sub(r"^\d+[.)、]\s*", "", a)  # "1. go to ..." / "2) ..."
    a = re.sub(r"^step\s*\d+\s*[:：]\s*", "", a, flags=re.I)  # "Step 1: ..."
    a = a.strip().strip("\"'`").strip()  # 包裹引号
    a = re.sub(r"[.。!！]+\s*$", "", a).strip()  # 结尾标点
    a = re.sub(r"\s+", " ", a).strip()  # 压缩空白
    if not a:
        return "", False
    low = a.lower()
    valid = any(
        low == v or low.startswith(v + " ") or low.startswith(v + ":")
        for v in VALID_VERBS
    )
    return a, valid


def adapt_action(action: str) -> str:
    # alfworld的版本问题0.4以上需要将put改为move
    m = re.match(r"^put\s+(.+?)\s+in/on\s+(.+)$", action, flags=re.I)
    if m:
        return f"move {m.group(1)} to {m.group(2)}"
    return action


def build_query(base_ctx: str, init_obs: str, turns):
    """组装 prompt: few-shot + 初始环境观察 + mem; 超预算时截断为最近 N 步。

    turns: list of (action, obs)。返回 (prompt, 是否发生截断)。
    """
    ctx = base_ctx + "\n" + init_obs
    tail = "".join(f"\n> {a}\n{o}" for a, o in turns)
    if len(ctx) + len(tail) <= CTX_CHAR_BUDGET:
        return ctx + tail, False
    budget = CTX_CHAR_BUDGET - len(ctx)
    keep, used = [], 0
    for a, o in reversed(turns):
        seg = f"\n> {a}\n{o}"
        if used + len(seg) > budget and keep:
            break
        keep.append(seg)
        used += len(seg)
    return ctx + "".join(reversed(keep)), True


class Worker:
    def __init__(self, wid, num_workers, args, prompts, client, env_lock, game_pool):
        self.wid = wid
        self.num_workers = num_workers
        self.args = args
        self.prompts = prompts
        self.client = client
        self.env_lock = env_lock
        self.game_pool = game_pool
        # 单线程 executor: 保证该 worker 的 env 始终在同一线程中被访问
        self.executor = ThreadPoolExecutor(
            max_workers=1, thread_name_prefix=f"alfenv-{wid}"
        )
        self.env = None

    async def env_call(self, fn, *a, **kw):
        # 把一些同步操作丢到线程池执行
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(self.executor, partial(fn, *a, **kw))

    async def setup(self):
        def _make():
            with open(self.args.config) as f:
                config = yaml.safe_load(f)
            env = AlfredTWEnv(config, train_eval="eval_out_of_distribution")
            env = env.init_env(batch_size=1)
            return env

        # 建立游戏环境
        self.env = await self.env_call(_make)

    async def llm(self, prompt):
        for attempt in range(3):
            try:
                resp = await self.client.completions.create(
                    model=self.args.model,
                    prompt=prompt,
                    temperature=0,
                    max_tokens=self.args.max_tokens,
                    stop=["\n"],
                    seed=self.args.seed,
                )
                return resp.choices[0].text.strip()
            except Exception as e:
                if attempt == 2:
                    raise
                await asyncio.sleep(2**attempt)

    async def run_episode(self, episode_id):
        env = self.env

        # 显式分配游戏: 把该 env 的内部迭代器替换为只含目标游戏的一次性迭代器,
        # 随后 reset() 会精确加载它。不依赖迭代器自动轮转:
        # shuffled_cycle 耗尽一轮后会用已漂移的 RNG 重新洗牌, 并发下会造成重复/遗漏
        def _assign_game():
            env._gamefiles_iterator = iter([self.game_pool[episode_id]])

        async with self.env_lock:
            await self.env_call(_assign_game)
            obs, info = await self.env_call(env.reset)
        gamefile = info["extra.gamefile"]
        if isinstance(gamefile, (list, tuple)):
            gamefile = gamefile[0]
        task_type = parse_task_type(gamefile)
        prompt_key = PROMPT_KEY[task_type]
        ex = (
            "Interact with a household to solve a task. Here are two examples.\n"
            + self.prompts[f"react_{prompt_key}_1"]
            + self.prompts[f"react_{prompt_key}_0"]
            + "\nHere is the task.\n"
        )
        init_obs = obs[0]
        # 去掉 TextWorld 欢迎 banner: 该 banner 是 few-shot 示例里没有的陌生前缀,
        # 会破坏「任务情境 → > think:」的模式匹配。实测仅去掉它, pick_and_place 首步
        # P(think) 就从 0.185 回到 0.681 (7B 模型 ICL 弱、对输入格式扰动敏感);
        # 去除后真实任务输入与示例开头 ("You are in the middle of a room...") 对齐。
        init_obs = re.sub(
            r"^-\s*=?\s*Welcome to TextWorld, ALFRED!\s*=?-?\s*", "", init_obs
        )
        # 再压掉初始观测内部的空行: few-shot 示例中 "You are in the middle of a room..."
        # 与 "Your task is to: ..." 是紧邻两行, 而环境返回的观测在两者之间夹了一个空行,
        # 保留会削弱「示例格式 → 真实输入」的模式匹配 (6 类任务结构一致, 仅此一处空行)。
        init_obs = re.sub(r"\n\s*\n+", "\n", init_obs).strip()

        turns = []  # (action_sent, obs)
        raw_outputs = []  # 每步完整记录
        success = False
        truncated_times = 0
        invalid_actions = 0
        t0 = time.time()

        for step in range(self.args.max_steps):
            query, truncated = build_query(ex, init_obs, turns)
            truncated_times += int(truncated)
            raw = await self.llm(query + "\n> ")
            action, valid = normalize_action(raw)
            if not valid:
                invalid_actions += 1

            # think 动作不发送给环境, 直接返回 "OK." (与 few-shot 格式一致)
            if action.lower().startswith("think"):
                new_obs = "OK."
                reward_val = 0.0
                done_val = False
            else:
                sent = adapt_action(action.lower()) if action else "look"
                async with self.env_lock:
                    obs, reward, done, info = await self.env_call(env.step, [sent])
                new_obs = obs[0]
                reward_val = float(reward[0])
                done_val = bool(done[0])

            sent = action.lower() if action else "look"
            turns.append((sent, new_obs))
            raw_outputs.append(
                {
                    "step": step + 1,
                    "raw": raw,
                    "action": sent,
                    "valid": bool(valid),
                    "obs": new_obs,
                    "reward": reward_val,
                    "done": done_val,
                }
            )
            # done为true，说明任务完成，游戏结束
            if done_val:
                success = bool(reward_val > 0) or bool(info.get("won", [0])[0])
                break

        record = {
            "episode_id": episode_id,
            "worker": self.wid,
            "gamefile": gamefile,
            "task_type": task_type,
            "success": success,
            "num_steps": len(turns),
            "invalid_actions": invalid_actions,
            "llm_calls": len(raw_outputs),
            "invalid_rate": round(invalid_actions / max(len(raw_outputs), 1), 4),
            "truncated_ctx_times": truncated_times,
            "duration_sec": round(time.time() - t0, 1),
            "init_obs": init_obs,
            "turns": raw_outputs,
        }
        return record


async def worker_loop(
    wid,
    num_workers,
    total_games,
    queue,
    args,
    prompts,
    client,
    write_lock,
    env_lock,
    game_pool,
    traj_fp,
    stats,
):
    worker = Worker(wid, num_workers, args, prompts, client, env_lock, game_pool)
    # 建立游戏
    await worker.setup()
    try:
        while True:
            try:
                episode_id = queue.get_nowait()
            except asyncio.QueueEmpty:
                break
                # 记录游戏号为episode_id的游戏记录
            record = await worker.run_episode(episode_id)
            # 保证不同协程进行写操作时不会出现数据污染
            async with write_lock:
                traj_fp.write(json.dumps(record, ensure_ascii=False) + "\n")
                traj_fp.flush()
                # 记录已完成的动作数
                stats["done"] += 1
                stats["success"] += int(record["success"])
            tag = "SUCCESS" if record["success"] else "FAIL"
            print(
                f"[{stats['done']:>3}/{total_games}] {tag:<7} "
                f"ep={episode_id:>3} type={record['task_type']:<22} "
                f"steps={record['num_steps']:>2} invalid={record['invalid_actions']}",
                flush=True,
            )
    finally:
        await worker.env_call(worker.env.close)
        worker.executor.shutdown(wait=False)


def summarize(traj_path, out_dir):
    records = [json.loads(l) for l in open(traj_path) if l.strip()]
    gamefiles = [r["gamefile"] for r in records]
    if len(set(gamefiles)) != len(gamefiles):
        print("警告: 存在重复的 gamefile, 并发分配可能有误!")

    per = {
        t: {"n": 0, "succ": 0, "steps": 0, "succ_steps": 0, "invalid": 0, "calls": 0}
        for t in TASK_TYPE_ORDER
    }
    for r in records:
        s = per[r["task_type"]]
        s["n"] += 1
        s["succ"] += int(r["success"])
        s["steps"] += r["num_steps"]
        # 平均步数只统计成功局: 失败局常烧满 max_steps, 混入会严重抬高均值
        if r["success"]:
            s["succ_steps"] += r["num_steps"]
        s["invalid"] += r["invalid_actions"]
        s["calls"] += r["llm_calls"]

    lines = [
        "| 任务类型 | 数量 | 成功率 | 平均步数(成功局) | 平均无效动作率 |",
        "| --- | --- | --- | --- | --- |",
    ]
    summary = {"per_task": {}, "overall": {}}
    for t in TASK_TYPE_ORDER:
        s = per[t]
        if s["n"] == 0:
            continue
        sr = s["succ"] / s["n"]
        avg_steps = s["succ_steps"] / s["succ"] if s["succ"] else None  # 仅成功局
        avg_steps_all = s["steps"] / s["n"]  # 全部局 (辅助)
        inv = s["invalid"] / max(s["calls"], 1)
        summary["per_task"][t] = {
            "count": s["n"],
            "success": s["succ"],
            "success_rate": round(sr, 4),
            "avg_steps": round(avg_steps, 2) if avg_steps is not None else None,
            "avg_steps_all": round(avg_steps_all, 2),
            "invalid_action_rate": round(inv, 4),
        }
        steps_txt = f"{avg_steps:.2f}" if avg_steps is not None else "-"
        lines.append(
            f"| {DISPLAY_NAME[t]} | {s['n']} | {sr:.1%} | {steps_txt} | {inv:.1%} |"
        )

    n = len(records)
    succ = sum(int(r["success"]) for r in records)
    steps = sum(r["num_steps"] for r in records)
    succ_steps = sum(r["num_steps"] for r in records if r["success"])
    invalid = sum(r["invalid_actions"] for r in records)
    calls = sum(r["llm_calls"] for r in records)
    avg_steps_ov = succ_steps / succ if succ else None  # 仅成功局
    steps_txt_ov = f"{avg_steps_ov:.2f}" if avg_steps_ov is not None else "-"
    summary["overall"] = {
        "count": n,
        "success": succ,
        "success_rate": round(succ / max(n, 1), 4),
        "avg_steps": round(avg_steps_ov, 2) if avg_steps_ov is not None else None,
        "avg_steps_all": round(steps / max(n, 1), 2),
        "invalid_action_rate": round(invalid / max(calls, 1), 4),
        "unique_games": len(set(gamefiles)),
    }
    lines.append(
        f"| 总体 | {n} | {succ / max(n, 1):.1%} | {steps_txt_ov} | "
        f"{invalid / max(calls, 1):.1%} |"
    )

    with open(os.path.join(out_dir, "summary.json"), "w") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    with open(os.path.join(out_dir, "summary.md"), "w") as f:
        f.write("\n".join(lines) + "\n")
    print("\n=== 评测结果 ===")
    print("\n".join(lines))
    return summary


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--config", default=os.path.join(PROJECT_ROOT, "play-log/base_config.yaml")
    )
    ap.add_argument(
        "--prompts", default=os.path.join(PROJECT_ROOT, "react/alfworld_3prompts.json")
    )
    ap.add_argument(
        "--output-dir", default=os.path.join(PROJECT_ROOT, "results", "baseline")
    )
    ap.add_argument("--base-url", default="http://127.0.0.1:8000/v1")
    ap.add_argument("--model", default="qwen")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--max-steps", type=int, default=50)
    ap.add_argument("--max-tokens", type=int, default=100)
    ap.add_argument(
        "--seed",
        type=int,
        default=1234,
        help="随机种子: 控制游戏洗牌/分配顺序, 并透传给 vLLM",
    )
    ap.add_argument("--limit", type=int, default=0, help="只跑前 N 局 (smoke test 用)")
    args = ap.parse_args()

    with open(args.prompts) as f:
        prompts = json.load(f)

    os.makedirs(args.output_dir, exist_ok=True)
    traj_path = os.path.join(args.output_dir, "trajectories.jsonl")

    client = AsyncOpenAI(base_url=args.base_url, api_key="EMPTY")

    # 用一个临时 env 获取游戏列表总数 (游戏数固定: unseen 全量)
    with open(args.config) as f:
        config = yaml.safe_load(f)
    tmp = AlfredTWEnv(config, train_eval="eval_out_of_distribution")
    total_games = tmp.num_games
    if args.limit > 0:
        total_games = min(total_games, args.limit)
    # 全局游戏分配: 复刻 textworld.gym 的洗牌逻辑 (seed=1234),
    # 得到确定的全局顺序后按 episode_id 分配, 保证无重复无遗漏
    print(f"评测集规模: {total_games} 局, 并发: {args.workers} 路, seed: {args.seed}")
    rng = np.random.RandomState(args.seed)
    game_pool = list(tmp.game_files)
    rng.shuffle(game_pool)

    # 存放任务队列，如果有未开始但已搭建好的游戏，就先入队
    queue = asyncio.Queue()
    for i in range(total_games):
        queue.put_nowait(i)

    stats = {"done": 0, "success": 0}
    write_lock = asyncio.Lock()
    env_lock = asyncio.Lock()
    t0 = time.time()
    with open(traj_path, "w") as traj_fp:
        tasks = [
            worker_loop(
                w,
                args.workers,
                total_games,
                queue,
                args,
                prompts,
                client,
                write_lock,
                env_lock,
                game_pool,
                traj_fp,
                stats,
            )
            for w in range(args.workers)
        ]
        await asyncio.gather(*tasks)
    print(f"\n全部完成, 耗时 {time.time() - t0:.0f}s, 轨迹写入 {traj_path}")

    summarize(traj_path, args.output_dir)


if __name__ == "__main__":
    asyncio.run(main())
