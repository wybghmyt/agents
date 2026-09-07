"""按用户指定格式输出评测结果表:
任务类型 | 局数 | 成功率 | 平均步数(成功局) | 无效动作率 (不含成功数列)
成功局平均步数从轨迹逐局统计; 多种子运行给出均值±标准差。
"""
import json
import os
import statistics
import sys

# 数据根目录: 默认 results/, 可用环境变量 RESULT_BASE 切到 result/ (新一轮实验落盘位置)
BASE = os.environ.get("RESULT_BASE", "/root/autodl-tmp/agents/results")
TASK_ORDER = [
    ("pick_and_place", "pick_and_place"),
    ("pick_clean_then_place", "pick_clean_then_place"),
    ("pick_heat_then_place", "pick_heat_then_place"),
    ("pick_cool_then_place", "pick_cool_then_place"),
    ("look_at_obj", "look_at_obj_in_light"),
    ("pick_two_obj", "pick_two_obj_and_place"),
]

def load_runs(dirs):
    runs = []
    for d in dirs:
        path = os.path.join(BASE, d, "trajectories.jsonl")
        runs.append([json.loads(l) for l in open(path) if l.strip()])
    return runs

def per_task_stats(records):
    """单次运行 -> {task: {n, succ, succ_rate, avg_steps_succ, invalid_rate}}"""
    stats = {}
    for t, _ in TASK_ORDER:
        eps = [r for r in records if r["task_type"] == t]
        succ_eps = [r for r in eps if r["success"]]
        calls = sum(r["llm_calls"] for r in eps)
        inv = sum(r["invalid_actions"] for r in eps)
        stats[t] = {
            "n": len(eps),
            "succ": len(succ_eps),
            "succ_rate": len(succ_eps) / max(len(eps), 1),
            "avg_steps_succ": (statistics.mean(r["num_steps"] for r in succ_eps)
                               if succ_eps else None),
            "invalid_rate": inv / max(calls, 1),
        }
    return stats

def fmt(mean, std=None, pct=False, digits=1):
    if mean is None:
        return "—"
    if pct:
        mean *= 100
        if std is not None:
            std *= 100
    if std is None:
        return f"{mean:.{digits}f}{'%' if pct else ''}"
    return f"{mean:.{digits}f} ± {std:.{digits}f}{'%' if pct else ''}"

def render(dirs, single=False):
    runs = load_runs(dirs)
    per_runs = [per_task_stats(r) for r in runs]

    lines = ["| 任务类型 | 局数 | 成功率 | 平均步数（成功局） | 无效动作率 |",
             "| --- | --- | --- | --- | --- |"]
    tot = {"n": 0, "succ": [], "steps_succ_runs": [], "inv": []}
    for t, disp in TASK_ORDER:
        ns = [pr[t]["n"] for pr in per_runs]
        succs = [pr[t]["succ"] for pr in per_runs]
        rates = [pr[t]["succ_rate"] for pr in per_runs]
        steps = [pr[t]["avg_steps_succ"] for pr in per_runs]
        invs = [pr[t]["invalid_rate"] for pr in per_runs]
        tot["n"] += ns[0]
        tot["succ"].append(succs)
        tot["steps_succ_runs"].append(steps)
        tot["inv"].append(invs)
        if single:
            row = [disp, ns[0], fmt(rates[0], pct=True),
                   fmt(steps[0]), fmt(invs[0], pct=True)]
        else:
            # 成功局平均步数: 只对有成功局的运行取均值 (无成功局的运行不参与)
            valid_steps = [s for s in steps if s is not None]
            row = [disp, ns[0],
                   fmt(statistics.mean(rates), statistics.stdev(rates) if len(rates) > 1 else None, pct=True),
                   fmt(statistics.mean(valid_steps),
                       statistics.stdev(valid_steps) if len(valid_steps) > 1 else None) if valid_steps else "—",
                   fmt(statistics.mean(invs), statistics.stdev(invs) if len(invs) > 1 else None, pct=True)]
        lines.append("| " + " | ".join(str(x) for x in row) + " |")

    # 总计行
    n = tot["n"]
    succ_per_run = [sum(s[i] for s in tot["succ"]) for i in range(len(per_runs))]
    rate_per_run = [s / n for s in succ_per_run]
    steps_per_run = []
    for i in range(len(per_runs)):
        ss = [s for s in tot["steps_succ_runs"][i] if s is not None]
        if ss:
            # 按成功局数加权回各运行的成功局平均步数
            steps_per_run.append(None)  # 占位, 总计成功局步数直接按成功局加权
    # 总计成功局平均步数: 直接汇总所有成功局
    all_succ_steps = []
    for records in runs:
        all_succ_steps += [r["num_steps"] for r in records if r["success"]]
    inv_per_run = [sum(r["invalid_actions"] for r in records) /
                   max(sum(r["llm_calls"] for r in records), 1) for records in runs]
    if single:
        lines.append(f"| 总计 | {n} | {fmt(rate_per_run[0], pct=True)} | "
                     f"{fmt(statistics.mean(all_succ_steps) if all_succ_steps else None)} | "
                     f"{fmt(inv_per_run[0], pct=True)} |")
    else:
        lines.append(
            f"| 总计 | {n} | "
            f"{fmt(statistics.mean(rate_per_run), statistics.stdev(rate_per_run) if len(rate_per_run) > 1 else None, pct=True)} | "
            f"{fmt(statistics.mean(all_succ_steps), statistics.stdev(all_succ_steps) if len(all_succ_steps) > 1 else None) if all_succ_steps else '—'} | "
            f"{fmt(statistics.mean(inv_per_run), statistics.stdev(inv_per_run) if len(inv_per_run) > 1 else None, pct=True)} |"
        )
    print("\n".join(lines))

if __name__ == "__main__":
    label = sys.argv[1]
    dirs = sys.argv[2:]
    print(f"=== {label} ===")
    render(dirs, single=len(dirs) == 1)
