"""把四个变体在某一步实际发给模型的完整 prompt 打印出来, 便于人工对比差异。

取 baseline 轨迹中第一条 pick_and_place, 模拟跑到第 2 步后 (已有 2 个
(action, obs) 历史) 组装第 3 步的 prompt, 分别按四个变体的组装规则输出。
"""
import json
import os

ROOT = "/root/autodl-tmp/agents"
CTX_CHAR_BUDGET = 26000
WARN_TEXT = "(the previous action had no effect, please choose another action)"

PROMPT_KEY = {
    "pick_and_place": "put", "pick_clean_then_place": "clean",
    "pick_heat_then_place": "heat", "pick_cool_then_place": "cool",
    "look_at_obj": "examine", "pick_two_obj": "puttwo",
}


def load_prompts(path):
    with open(path) as f:
        prompts = json.load(f)
    for key, ex in prompts.items():
        if key.startswith("act_"):
            lines = [ln for ln in ex.split("\n") if ln.strip() != "OK."]
            prompts[key] = "\n".join(lines)
    return prompts


def build_query(base_ctx, init_obs, turns, with_warn=False, admissible=None):
    def seg_of(t):
        seg = f"\n> {t[0]}\n{t[1]}"
        if with_warn and len(t) > 2 and t[2]:
            seg += f"\n{WARN_TEXT}"
        return seg

    ctx = base_ctx + "\n" + init_obs
    tail = "".join(seg_of(t) for t in turns)
    adm = ""
    if admissible is not None:
        adm = ("\nAdmissible commands:\n" + "\n".join(admissible)
               + "\nChoose an action from the admissible commands above.")
    return ctx + tail + adm


def main():
    prompts = load_prompts(os.path.join(ROOT, "react", "alfworld_3prompts.json"))
    recs = [json.loads(l) for l in
            open(os.path.join(ROOT, "results", "baseline",
                              "trajectories.jsonl")) if l.strip()]
    # 找一条 pick_and_place 且 step>=2 的轨迹
    r = next(x for x in recs if x["task_type"] == "pick_and_place"
             and len(x["turns"]) >= 2)
    key = PROMPT_KEY[r["task_type"]]
    init_obs = r["init_obs"]
    turns = [(t["action"], t["obs"], t.get("warned", False)) for t in r["turns"][:2]]

    react_ex = "\n".join(prompts[f"react_{key}_{i}"] for i in range(2))
    act_ex = "\n".join(prompts[f"act_{key}_{i}"] for i in range(2))
    # 演示用假 admissible 列表
    fake_adm = ["go to cabinet 1", "go to cabinet 2", "take mug 1 from cabinet 1",
                "look", "inventory", "examine cabinet 1"]

    sections = [
        ("1) baseline  (react few-shot + 历史)",
         build_query(react_ex, init_obs, turns, False, None)),
        ("2) act-only  (act few-shot + 历史, 无 think 行)",
         build_query(act_ex, init_obs, turns, False, None)),
        ("3) add-warn  (react few-shot + 历史, warned 步追加警告)",
         build_query(react_ex, init_obs, turns, True, None)),
        ("4) admissible-commands  (react few-shot + 历史 + 合法动作列表)",
         build_query(react_ex, init_obs, turns, False, fake_adm)),
    ]
    for title, prompt in sections:
        print("=" * 78)
        print(title)
        print("=" * 78)
        print(prompt + "\n> ", end="")
        print("\n")


if __name__ == "__main__":
    main()
