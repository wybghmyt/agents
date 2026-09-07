"""两阶段自由生成: Phase1 自由思考 → Phase2 从合法列表选择动作 (纯 prompt 约束, 不用 guided_choice)。

与现有方案的差异:
- inject_commands: 列表塞进 prompt, 一次自由生成, 无思考 → 命中率仅 27.7%
- inject_strong:   强指令让模型从列表选, 一次生成, 无思考 → 命中率 92.5% 但成功率仅 6%
- inject_constrained: guided_choice 硬约束 → 程序约束, 无思考
- **本方案 (think_choose)**: 每步两次自由生成, 约束完全通过 prompt 指令实现:
    Phase1: prompt 末尾追加合法列表 + "Let me think."  → 模型自由输出思考内容
    Phase2: prompt 末尾追加合法列表 + 强指令 "choose from list" → 模型自由输出动作
  两次生成都是自由的, 如果模型不遵循列表则视为无效动作。

用法:
    conda activate alfworld
    python react/think_choose.py                 # 全量 134 局
    python react/think_choose.py --limit 4       # smoke test
"""

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

from alfworld.agents.environment.alfred_tw_env import AlfredTWEnv  # noqa: E402

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)

VARIANT = "think_then_choose"

TASK_TYPE_PATTERNS = [
    ("pick_two_obj_and_place", "pick_two_obj"),
    ("look_at_obj_in_light", "look_at_obj"),
    ("pick_clean_then_place_in_recep", "pick_clean_then_place"),
    ("pick_heat_then_place_in_recep", "pick_heat_then_place"),
    ("pick_cool_then_place_in_recep", "pick_cool_then_place"),
    ("pick_and_place_simple", "pick_and_place"),
]

TASK_TYPE_ORDER = [
    "pick_and_place",
    "pick_clean_then_place",
    "pick_heat_then_place",
    "pick_cool_then_place",
    "look_at_obj",
    "pick_two_obj",
]

DISPLAY_NAME = {
    "pick_and_place": "pick_and_place",
    "pick_clean_then_place": "pick_clean_then_place",
    "pick_heat_then_place": "pick_heat_then_place",
    "pick_cool_then_place": "pick_cool_then_place",
    "look_at_obj": "look_at_obj_in_light",
    "pick_two_obj": "pick_two_obj_and_place",
}

PROMPT_KEY = {
    "pick_and_place": "put",
    "pick_clean_then_place": "clean",
    "pick_heat_then_place": "heat",
    "pick_cool_then_place": "cool",
    "look_at_obj": "examine",
    "pick_two_obj": "puttwo",
}

# 每步合法列表最多展示条数 (控制上下文占用)
MAX_ADMISSIBLE_SHOW = 30

# 上下文字符预算
CTX_CHAR_BUDGET = 26000


def parse_task_type(gamefile: str) -> str:
    for substr, short in TASK_TYPE_PATTERNS:
        if substr in gamefile:
            return short
    raise ValueError(f"无法识别任务类型: {gamefile}")


def normalize_action(raw: str):
    """动作规范化: 去掉 Action: 前缀 / 引号 / 句号等, 返回 (规范化动作, 是否合法)。"""
    a = raw.strip()
    a = re.sub(r"^>+\s*", "", a)
    a = re.sub(r"^(action|act)\s*[:：]\s*", "", a, flags=re.I)
    a = re.sub(r"^\d+[.)、]\s*", "", a)
    a = re.sub(r"^step\s*\d+\s*[:：]\s*", "", a, flags=re.I)
    a = a.strip().strip('"\'`').strip()
    a = re.sub(r"[.。!！]+\s*$", "", a).strip()
    a = re.sub(r"\s+", " ", a).strip()
    if not a:
        return "", False
    low = a.lower()
    valid = any(
        low == v or low.startswith(v + " ") or low.startswith(v + ":")
        for v in ("think", "go to", "take", "put", "open", "close",
                  "clean", "heat", "cool", "use", "examine", "look", "inventory")
    )
    return a, valid


def adapt_action(action: str) -> str:
    """语法适配: alfworld 0.4+ 把 `put X in/on Y` 改为 `move X to Y`。"""
    m = re.match(r"^put\s+(.+?)\s+in/on\s+(.+)$", action, flags=re.I)
    if m:
        return f"move {m.group(1)} to {m.group(2)}"
    return action


def format_admissible(admissible, for_phase="think"):
    """把合法动作列表格式化为纯英文段落, 末尾追加阶段对应的指令。

    for_phase="think":  Phase1 用, 引导模型先思考
    for_phase="choose": Phase2 用, 强指令要求从列表中原样选择
    """
    shown = admissible[:MAX_ADMISSIBLE_SHOW]
    lines = "\n".join(shown)
    if len(admissible) > MAX_ADMISSIBLE_SHOW:
        lines += f"\n... ({len(admissible)} commands total)"

    if for_phase == "think":
        # Phase1: 让模型先思考, 不需要它选择动作
        instruction = (
            "\nThink step by step about what to do next based on the current state "
            "and the admissible commands above. What is the best next action?"
        )
    else:
        # Phase2: 强指令要求从列表中逐字选择
        instruction = (
            "\nYou MUST choose exactly one command from the admissible list above. "
            "Copy the command exactly as written. "
            "Do NOT write any command that is not in the list."
        )
    return f"\nAdmissible commands:\n{lines}{instruction}"


def build_query(base_ctx, init_obs, turns, admissible, phase):
    """组装 prompt: few-shot + 初始观测 + 历史 + 合法列表 + 阶段指令。

    phase: "think" 或 "choose"
    turns: list of (action, obs)
    返回 (prompt, 是否截断)
    """
    ctx = base_ctx + "\n" + init_obs
    tail = "".join(f"\n> {a}\n{o}" for a, o in turns)
    adm = format_admissible(admissible, for_phase=phase)

    if len(ctx) + len(tail) + len(adm) <= CTX_CHAR_BUDGET:
        return ctx + tail + adm, False

    # 超预算: 先截历史, 再截列表
    budget = CTX_CHAR_BUDGET - len(ctx) - len(adm)
    keep, used = [], 0
    if budget > 0:
        for a, o in reversed(turns):
            seg = f"\n> {a}\n{o}"
            if used + len(seg) > budget and keep:
                break
            keep.append(seg)
            used += len(seg)
    history = "".join(reversed(keep))

    budget = CTX_CHAR_BUDGET - len(ctx) - len(history)
    if len(adm) > budget:
        adm = adm[:max(budget, 0)]
    return ctx + history + adm, True


class Worker:
    def __init__(self, wid, num_workers, args, prompts, client, env_lock, game_pool):
        self.wid = wid
        self.num_workers = num_workers
        self.args = args
        self.prompts = prompts
        self.client = client
        self.env_lock = env_lock
        self.game_pool = game_pool
        self.executor = ThreadPoolExecutor(
            max_workers=1, thread_name_prefix=f"alfenv-{wid}"
        )
        self.env = None

    async def env_call(self, fn, *a, **kw):
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(self.executor, partial(fn, *a, **kw))

    async def setup(self):
        def _make():
            with open(self.args.config) as f:
                config = yaml.safe_load(f)
            env = AlfredTWEnv(config, train_eval="eval_out_of_distribution")
            env = env.init_env(batch_size=1)
            return env
        self.env = await self.env_call(_make)

    async def llm(self, prompt, max_tokens=100):
        for attempt in range(3):
            try:
                resp = await self.client.completions.create(
                    model=self.args.model,
                    prompt=prompt,
                    temperature=0,
                    max_tokens=max_tokens,
                    stop=["\n"],
                )
                return resp.choices[0].text.strip()
            except Exception as e:
                if attempt == 2:
                    raise
                await asyncio.sleep(2 ** attempt)

    async def run_episode(self, episode_id):
        env = self.env

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
        ex = "\n".join(
            self.prompts[f"react_{prompt_key}_{i}"] for i in range(2)
        )
        init_obs = obs[0]
        admissible = list(info["admissible_commands"][0])

        turns = []          # (action_sent, obs)
        raw_outputs = []    # 每步完整记录
        success = False
        truncated_times = 0
        invalid_actions = 0
        admissible_hits = 0
        total_llm_calls = 0
        num_game_steps = 0  # 实际游戏步数 (每步包含 think + choose 两次调用)
        t0 = time.time()

        for step in range(self.args.max_steps):
            # ========== Phase 1: 自由思考 ==========
            num_game_steps += 1
            q_think, trunc1 = build_query(ex, init_obs, turns, admissible, phase="think")
            truncated_times += int(trunc1)
            # prompt 末尾加 "> think: " 引导模型输出思考内容
            raw_think = await self.llm(q_think + "\n> think: ", max_tokens=150)
            total_llm_calls += 1

            think_action = f"think: {raw_think}"
            turns.append((think_action, "OK."))
            raw_outputs.append({
                "step": step + 1,
                "phase": "think",
                "raw": raw_think,
                "action": think_action,
                "valid": True,
                "in_admissible": False,
                "obs": "OK.",
                "reward": 0.0,
                "done": False,
            })

            # ========== Phase 2: 从合法列表选择动作 ==========
            q_choose, trunc2 = build_query(ex, init_obs, turns, admissible, phase="choose")
            truncated_times += int(trunc2)
            # prompt 末尾加 "> " 让模型输出动作
            raw_choose = await self.llm(q_choose + "\n> ", max_tokens=100)
            total_llm_calls += 1

            action, valid = normalize_action(raw_choose)
            if not valid:
                invalid_actions += 1

            # think 动作不发送给环境
            if action.lower().startswith("think"):
                new_obs = "OK."
                reward_val = 0.0
                done_val = False
                in_adm = False
            else:
                sent = adapt_action(action.lower()) if action else "look"
                in_adm = sent in admissible
                if in_adm:
                    admissible_hits += 1
                async with self.env_lock:
                    obs, reward, done, info = await self.env_call(env.step, [sent])
                new_obs = obs[0]
                reward_val = float(reward[0])
                done_val = bool(done[0])

            sent_action = action.lower() if action else "look"
            turns.append((sent_action, new_obs))
            raw_outputs.append({
                "step": step + 1,
                "phase": "choose",
                "raw": raw_choose,
                "action": sent_action,
                "valid": bool(valid),
                "in_admissible": in_adm,
                "obs": new_obs,
                "reward": reward_val,
                "done": done_val,
            })

            # 更新合法动作列表
            if not action.lower().startswith("think"):
                admissible = list(info["admissible_commands"][0])

            if done_val:
                success = bool(reward_val > 0) or bool(info.get("won", [0])[0])
                break

        record = {
            "variant": VARIANT,
            "episode_id": episode_id,
            "worker": self.wid,
            "gamefile": gamefile,
            "task_type": task_type,
            "success": success,
            "num_steps": num_game_steps,
            "invalid_actions": invalid_actions,
            "llm_calls": total_llm_calls,
            "invalid_rate": round(invalid_actions / max(total_llm_calls, 1), 4),
            "admissible_hit_rate": round(admissible_hits / max(num_game_steps, 1), 4),
            "truncated_ctx_times": truncated_times,
            "duration_sec": round(time.time() - t0, 1),
            "init_obs": init_obs,
            "turns": raw_outputs,
        }
        return record


async def worker_loop(wid, num_workers, total_games, queue, args, prompts, client,
                      write_lock, env_lock, game_pool, traj_fp, stats):
    worker = Worker(wid, num_workers, args, prompts, client, env_lock, game_pool)
    await worker.setup()
    try:
        while True:
            try:
                episode_id = queue.get_nowait()
            except asyncio.QueueEmpty:
                break
            record = await worker.run_episode(episode_id)
            async with write_lock:
                traj_fp.write(json.dumps(record, ensure_ascii=False) + "\n")
                traj_fp.flush()
                stats["done"] += 1
                stats["success"] += int(record["success"])
            tag = "SUCCESS" if record["success"] else "FAIL"
            print(
                f"[{stats['done']:>3}/{total_games}] {tag:<7} "
                f"ep={episode_id:>3} type={record['task_type']:<22} "
                f"steps={record['num_steps']:>2} invalid={record['invalid_actions']} "
                f"adm_hit={record['admissible_hit_rate']:.1%}",
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

    per = {t: {"n": 0, "succ": 0, "steps": 0, "invalid": 0, "calls": 0, "hits": 0}
           for t in TASK_TYPE_ORDER}
    for r in records:
        s = per[r["task_type"]]
        s["n"] += 1
        s["succ"] += int(r["success"])
        s["steps"] += r["num_steps"]
        s["invalid"] += r["invalid_actions"]
        s["calls"] += r["llm_calls"]
        s["hits"] += round(r["admissible_hit_rate"] * r["num_steps"])

    lines = [
        f"# 两阶段自由生成: {VARIANT}",
        "",
        "| 任务类型 | 数量 | 成功率 | 平均步数 | 平均无效动作率 | 合法命中率 |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    summary = {"variant": VARIANT, "per_task": {}, "overall": {}}
    for t in TASK_TYPE_ORDER:
        s = per[t]
        if s["n"] == 0:
            continue
        sr = s["succ"] / s["n"]
        avg_steps = s["steps"] / s["n"]
        inv = s["invalid"] / max(s["calls"], 1)
        hit = s["hits"] / max(s["n"], 1)
        summary["per_task"][t] = {
            "count": s["n"], "success": s["succ"],
            "success_rate": round(sr, 4),
            "avg_steps": round(avg_steps, 2),
            "invalid_action_rate": round(inv, 4),
            "admissible_hit_rate": round(hit, 4),
        }
        lines.append(
            f"| {DISPLAY_NAME[t]} | {s['n']} | {sr:.1%} | {avg_steps:.2f} | "
            f"{inv:.1%} | {hit:.1%} |"
        )

    n = len(records)
    succ = sum(int(r["success"]) for r in records)
    steps = sum(r["num_steps"] for r in records)
    invalid = sum(r["invalid_actions"] for r in records)
    calls = sum(r["llm_calls"] for r in records)
    hits = sum(round(r["admissible_hit_rate"] * r["num_steps"]) for r in records)
    summary["overall"] = {
        "count": n, "success": succ,
        "success_rate": round(succ / max(n, 1), 4),
        "avg_steps": round(steps / max(n, 1), 2),
        "invalid_action_rate": round(invalid / max(calls, 1), 4),
        "admissible_hit_rate": round(hits / max(n, 1), 4),
        "unique_games": len(set(gamefiles)),
    }
    lines.append(
        f"| 总体 | {n} | {succ / max(n, 1):.1%} | {steps / max(n, 1):.2f} | "
        f"{invalid / max(calls, 1):.1%} | {hits / max(n, 1):.1%} |"
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
    ap.add_argument("--config", default=os.path.join(PROJECT_ROOT, "play-log/base_config.yaml"))
    ap.add_argument("--prompts", default=os.path.join(PROJECT_ROOT, "react/alfworld_3prompts.json"))
    ap.add_argument("--output-dir", default=os.path.join(PROJECT_ROOT, "results", "think-choose"))
    ap.add_argument("--base-url", default="http://127.0.0.1:8000/v1")
    ap.add_argument("--model", default="qwen")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--max-steps", type=int, default=50)
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    with open(args.prompts) as f:
        prompts = json.load(f)

    os.makedirs(args.output_dir, exist_ok=True)
    traj_path = os.path.join(args.output_dir, "trajectories.jsonl")

    client = AsyncOpenAI(base_url=args.base_url, api_key="EMPTY")

    with open(args.config) as f:
        config = yaml.safe_load(f)
    tmp = AlfredTWEnv(config, train_eval="eval_out_of_distribution")
    total_games = tmp.num_games
    if args.limit > 0:
        total_games = min(total_games, args.limit)

    print(f"评测集规模: {total_games} 局, 并发: {args.workers} 路, seed: {args.seed}")
    rng = np.random.RandomState(args.seed)
    game_pool = list(tmp.game_files)
    rng.shuffle(game_pool)

    queue = asyncio.Queue()
    for i in range(total_games):
        queue.put_nowait(i)

    stats = {"done": 0, "success": 0}
    write_lock = asyncio.Lock()
    env_lock = asyncio.Lock()
    t0 = time.time()
    with open(traj_path, "w") as traj_fp:
        tasks = [
            worker_loop(w, args.workers, total_games, queue, args, prompts,
                        client, write_lock, env_lock, game_pool, traj_fp, stats)
            for w in range(args.workers)
        ]
        await asyncio.gather(*tasks)
    print(f"\n全部完成, 耗时 {time.time() - t0:.0f}s, 轨迹写入 {traj_path}")

    summarize(traj_path, args.output_dir)


if __name__ == "__main__":
    asyncio.run(main())
