"""离线统计所有已完成实验的平均 prompt token (覆盖全部变体)。

原理: 各变体的 prompt 组装是确定性的, 轨迹落盘了 init_obs 与每步
(action, obs[, warned]), 因此可逐调用重建 prompt 并用 Qwen2.5 tokenizer 计数:
  prompt = few-shot 示例 + "\n" + init_obs + 历史轨迹 + 续写后缀
各变体差异:
- baseline / add-warn / add-think / admissible-commands: react_<key>_0/1 (2-shot)
- act-only: act_<key>_0/1 (清理残留 "OK." 行)
- add-warn: 无效果步观测后追加 WARN_TEXT
- add-think: 上一步为 "Nothing happens." 时, 续写后缀为 "\n> think: " 而非 "\n> "
- zero-shot: 无示例 (ctx 以 "\n" 开头, 与实现一致)
- shot-2rand-typed: 任务内 3 示例随机抽 2, 由 (seed, episode_id) 复现
- shot-2rand (完全随机): 全部 18 示例随机抽 2, 同上复现
- all-6-shot: 6 种任务各第 1 个示例
- admissible-commands: 合法动作列表未落盘, 只重建历史部分 (下界)
"""
import json
import os
from random import Random

from transformers import AutoTokenizer

ROOT = "/root/autodl-tmp/agents"
CTX_CHAR_BUDGET = 26000
WARN_TEXT = "(the previous action had no effect, please choose another action)"

PROMPT_KEY = {
    "pick_and_place": "put",
    "pick_clean_then_place": "clean",
    "pick_heat_then_place": "heat",
    "pick_cool_then_place": "cool",
    "look_at_obj": "examine",
    "pick_two_obj": "puttwo",
}
TASK_TYPE_ORDER = [
    "pick_and_place", "pick_clean_then_place", "pick_heat_then_place",
    "pick_cool_then_place", "look_at_obj", "pick_two_obj",
]
ALL6_PROMPT_KEYS = [PROMPT_KEY[t] for t in TASK_TYPE_ORDER]
ALL_EXAMPLE_KEYS = [f"react_{k}_{i}" for k in PROMPT_KEY.values() for i in range(3)]


def load_prompts(path):
    with open(path) as f:
        prompts = json.load(f)
    for key, ex in prompts.items():
        if key.startswith("act_"):
            lines = [ln for ln in ex.split("\n") if ln.strip() != "OK."]
            prompts[key] = "\n".join(lines)
    return prompts


def build_query(base_ctx, init_obs, turns, with_warn=False):
    def seg_of(t):
        seg = f"\n> {t[0]}\n{t[1]}"
        if with_warn and len(t) > 2 and t[2]:
            seg += f"\n{WARN_TEXT}"
        return seg

    ctx = base_ctx + "\n" + init_obs
    tail = "".join(seg_of(t) for t in turns)
    if len(ctx) + len(tail) <= CTX_CHAR_BUDGET:
        return ctx + tail
    budget = CTX_CHAR_BUDGET - len(ctx)
    keep, used = [], 0
    for t in reversed(turns):
        seg = seg_of(t)
        if used + len(seg) > budget and keep:
            break
        keep.append(seg)
        used += len(seg)
    return ctx + "".join(reversed(keep))


def ex_builder_factory(kind, prompts):
    """返回 ex(task_type, episode_id, seed) 构造器。"""
    if kind == "react2":
        return lambda t, ep, sd: "\n".join(
            prompts[f"react_{PROMPT_KEY[t]}_{i}"] for i in range(2))
    if kind == "act2":
        return lambda t, ep, sd: "\n".join(
            prompts[f"act_{PROMPT_KEY[t]}_{i}"] for i in range(2))
    if kind == "zero":
        return lambda t, ep, sd: ""
    if kind == "typed2":
        def f(t, ep, sd):
            ids = Random(sd * 100003 + ep).sample(range(3), 2)
            return "\n".join(prompts[f"react_{PROMPT_KEY[t]}_{i}"] for i in ids)
        return f
    if kind == "fullrand2":
        def f(t, ep, sd):
            keys = Random(sd * 100003 + ep).sample(ALL_EXAMPLE_KEYS, 2)
            return "\n".join(prompts[k] for k in keys)
        return f
    if kind == "all6":
        return lambda t, ep, sd: "\n".join(
            prompts[f"react_{k}_0"] for k in ALL6_PROMPT_KEYS)
    raise ValueError(kind)


VARIANTS = [
    # (展示名, 轨迹目录, 示例类型, with_warn, think后缀变体, 该次运行seed, 备注)
    ("baseline", "results/baseline", "react2", False, False, 1234, ""),
    ("act-only", "results/act-only", "act2", False, False, 1234, ""),
    ("add-warn", "results/add-warn", "react2", True, False, 1234, ""),
    ("add-think", "results/add-think", "react2", False, True, 1234, ""),
    ("zero-shot", "results/zero-shot", "zero", False, False, 1234, ""),
    ("shot-2rand-typed", "results/shot-2rand-typed", "typed2", False, False, 1234, ""),
    ("shot-2rand", "results/shot-2rand", "fullrand2", False, False, 1234, ""),
    ("all-6-shot", "results/all-shot", "all6", False, False, 1234, ""),
    ("admissible-commands", "results/admissible-commands", "react2", False, False, 1234,
     "下界(不含列表)"),
]


def main():
    tok = AutoTokenizer.from_pretrained(
        os.path.join(ROOT, "models", "Qwen2.5-7B-Instruct"),
        trust_remote_code=True,
    )
    prompts = load_prompts(os.path.join(ROOT, "react", "alfworld_3prompts.json"))

    print(f"{'variant':<22}{'calls':>8}{'avg_prompt_tokens':>20}  备注")
    for name, rel, kind, with_warn, think_variant, seed, note in VARIANTS:
        path = os.path.join(ROOT, rel, "trajectories.jsonl")
        if not os.path.exists(path):
            print(f"{name:<22}{'-':>8}{'(未完成)':>20}")
            continue
        recs = [json.loads(l) for l in open(path) if l.strip()]
        ex_fn = ex_builder_factory(kind, prompts)
        total_tokens, total_calls = 0, 0
        for r in recs:
            ex = ex_fn(r["task_type"], r["episode_id"], seed)
            turns = []
            for t in r["turns"]:
                query = build_query(ex, r["init_obs"], turns, with_warn)
                if think_variant and turns and turns[-1][1] == "Nothing happens.":
                    query += "\n> think: "
                else:
                    query += "\n> "
                total_tokens += len(tok.encode(query, add_special_tokens=False))
                total_calls += 1
                turns.append((t["action"], t["obs"], t.get("warned", False)))
        assert total_calls == sum(r["llm_calls"] for r in recs), name
        print(f"{name:<22}{total_calls:>8}{total_tokens / total_calls:>20.1f}  {note}")


if __name__ == "__main__":
    main()
