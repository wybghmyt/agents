"""离线估算各消融实验的平均 prompt token。

评测脚本未记录 usage.prompt_tokens, 但轨迹中保存了 init_obs 与每步
(action, obs[, warned]), 且各变体的 prompt 组装是确定性的, 因此可以
逐调用重建 prompt 并用 Qwen2.5 tokenizer 计数:
  prompt = few-shot(2 示例) + "\n" + init_obs + 历史轨迹 [+ "\n> "]
- baseline / add-warn / inject-commands: react_<key>_0/1 示例
- act-only: act_<key>_0/1 示例 (清理过残留 "OK." 行)
- add-warn: warned 步在观测后追加 WARN_TEXT
- 超预算截断逻辑与评测脚本完全一致 (add-warn 共 6 次触发)
- inject-commands 的 admissible 列表未落盘, 只能重建历史部分 (下界)
"""
import json
import os
from alfworld.agents.environment.alfred_tw_env import AlfredTWEnv
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

VARIANTS = {
    "baseline": ("results/baseline", "react_", False),
    "act-only": ("results/act-only", "act_", False),
    "add-warn": ("results/add-warn", "react_", True),
    "admissible-commands": ("results/admissible-commands", "react_", False),
}


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


def main():
    tok = AutoTokenizer.from_pretrained(
        os.path.join(ROOT, "models", "Qwen2.5-7B-Instruct"),
        trust_remote_code=True,
    )
    prompts = load_prompts(os.path.join(ROOT, "react", "alfworld_3prompts.json"))

    print(f"{'variant':<22}{'calls':>8}{'avg_prompt_tokens':>20}")
    for name, (rel, prefix, with_warn) in VARIANTS.items():
        recs = [json.loads(l) for l in
                open(os.path.join(ROOT, rel, "trajectories.jsonl")) if l.strip()]
        total_tokens, total_calls = 0, 0
        for r in recs:
            ex = "\n".join(prompts[f"{prefix}{PROMPT_KEY[r['task_type']]}_{i}"]
                           for i in range(2))
            turns = []
            for t in r["turns"]:
                query = build_query(ex, r["init_obs"], turns, with_warn) + "\n> "
                total_tokens += len(tok.encode(query, add_special_tokens=False))
                total_calls += 1
                turns.append((t["action"], t["obs"], t.get("warned", False)))
        note = "  (不含未落盘的 admissible 列表)" if name == "admissible-commands" else ""
        print(f"{name:<22}{total_calls:>8}{total_tokens / total_calls:>20.1f}{note}")
        assert total_calls == sum(r["llm_calls"] for r in recs)


if __name__ == "__main__":
    main()
