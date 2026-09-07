"""采样统计 inject_commands 变体每步注入的 admissible_commands 列表 token 开销。

轨迹中未落盘 admissible 列表, 这里从每类任务各抽 1 局, 用随机动作驱动
环境若干步, 收集每步 info["admissible_commands"][0] 并 tokenize 计数,
得到平均每步的列表开销, 用于修正离线估算值。
"""
import json
import os
import random

import yaml
from transformers import AutoTokenizer

os.environ.setdefault("ALFWORLD_DATA", "/root/autodl-tmp/alfworld")
from alfworld.agents.environment.alfred_tw_env import AlfredTWEnv  # noqa: E402

ROOT = "/root/autodl-tmp/agents"
CONFIG = os.path.join(ROOT, "play-log", "base_config.yaml")
MAX_STEPS = 30
SEED = 0


def main():
    tok = AutoTokenizer.from_pretrained(
        os.path.join(ROOT, "models", "Qwen2.5-7B-Instruct"),
        trust_remote_code=True,
    )
    recs = [json.loads(l) for l in
            open(os.path.join(ROOT, "results", "admissible-commands",
                              "trajectories.jsonl")) if l.strip()]
    # 每类任务取一局
    seen, games = set(), []
    for r in recs:
        if r["task_type"] not in seen:
            seen.add(r["task_type"])
            games.append((r["task_type"], r["gamefile"]))

    with open(CONFIG) as f:
        config = yaml.safe_load(f)

    rng = random.Random(SEED)
    tokens_per_state, n_states = [], 0
    for task_type, gamefile in games:
        env = AlfredTWEnv(config, train_eval="eval_out_of_distribution")
        env = env.init_env(batch_size=1)
        env._gamefiles_iterator = iter([gamefile])
        obs, info = env.reset()
        admissible = list(info["admissible_commands"][0])
        for step in range(MAX_STEPS):
            # 格式化开销与评测脚本 format_admissible 一致
            text = ("\nAdmissible commands:\n" + "\n".join(admissible)
                    + "\nChoose an action from the admissible commands above.")
            tokens_per_state.append(len(tok.encode(text, add_special_tokens=False)))
            n_states += 1
            act = rng.choice(admissible) if admissible else "look"
            obs, reward, done, info = env.step([act])
            if done[0]:
                break
            admissible = list(info["admissible_commands"][0])
        env.close()
        print(f"{task_type:<26} states={n_states:>4} "
              f"avg_adm_tokens={sum(tokens_per_state) / len(tokens_per_state):.1f}")

    print(f"\n采样状态数: {len(tokens_per_state)}, "
          f"平均每步 admissible 列表开销: {sum(tokens_per_state) / len(tokens_per_state):.1f} tokens")


if __name__ == "__main__":
    main()
