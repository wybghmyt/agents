from alfworld.agents.environment.alfred_tw_env import AlfredTWEnv
import os
from datetime import datetime

os.environ.setdefault("ALFWORLD_DATA", "/root/autodl-tmp/alfworld")

import yaml
import alfworld.agents.environment as environment

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
CONFIG_PATH = os.path.join(SCRIPT_DIR, "base_config.yaml")
LOG_PATH = os.path.join(SCRIPT_DIR, "play-log")


def print_admissible(admissible):
    print(f"\n当前 admissible_commands (共 {len(admissible)} 条):")
    for cmd in admissible:
        print(f"  - {cmd}")


def append_log(gamefile, goal, turns, won):
    # 把本局交互以可读的结构化文本追加到日志文件。
    lines = [
        "=" * 60,
        f"时间      : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"游戏文件  : {gamefile}",
        f"任务目标  : {goal}",
        f"步数      : {len(turns)}",
        f"结果      : {'通关' if won else '未通关'}",
        "-" * 60,
    ]
    for t in turns:
        lines.append(f"第 {t['step']:>2} 步 | 输入: {t['command']}")
        lines.append(f"         | 反馈: {t['feedback']}")
    lines.append("=" * 60)
    lines.append("")

    with open(LOG_PATH, "a", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print(f"本局日志已追加到: {LOG_PATH}")


def main():
    with open(CONFIG_PATH) as f:
        config = yaml.safe_load(f)

    # 文本环境用 AlfredTWEnv
    env = AlfredTWEnv(config, train_eval="eval_out_of_distribution")
    env = env.init_env(batch_size=1)

    obs, info = env.reset()
    gamefile = info["extra.gamefile"]
    gamefile = gamefile[0] if isinstance(gamefile, (list, tuple)) else gamefile
    goal = obs[0]

    print("【任务目标】")
    print(goal)
    print_admissible(info["admissible_commands"][0])

    turns = []
    won = False
    while True:
        try:
            command = input(f"\n[第 {len(turns) + 1} 步] 请输入命令 > ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n输入中断, 结束本局。")
            break
        if not command:
            continue

        obs, reward, done, info = env.step([command])
        feedback = obs[0]
        turns.append({"step": len(turns) + 1, "command": command, "feedback": feedback})
        print(f"\n{feedback}")

        if done[0]:
            # 步数跳完了
            won = reward[0] > 0
            if won:
                print("\n游戏通关")
            else:
                print("\n游戏结束, 未能在限定步数内通关。")
            break

        print_admissible(info["admissible_commands"][0])

    append_log(gamefile, goal, turns, won)
    env.close()


if __name__ == "__main__":
    main()
