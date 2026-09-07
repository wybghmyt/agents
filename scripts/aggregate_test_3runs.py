"""聚合 test 三次运行 (seed 1234/42/2024) 的结果, 计算均值/标准差。"""
import json
import os
import statistics

BASE = "/root/autodl-tmp/agents/results"
RUNS = [
    ("seed1234", "test"),
    ("seed42", "test-seed42"),
    ("seed2024", "test-seed2024"),
]
TASKS = [
    "pick_and_place", "pick_clean_then_place", "pick_heat_then_place",
    "pick_cool_then_place", "look_at_obj", "pick_two_obj",
]
METRICS = ["success_rate", "avg_steps", "invalid_action_rate"]

summaries = {name: json.load(open(os.path.join(BASE, d, "summary.json")))
             for name, d in RUNS}

agg = {"runs": {}}
for name, d in RUNS:
    s = summaries[name]
    agg["runs"][name] = {
        "dir": d,
        "per_task": s["per_task"],
        "overall": s["overall"],
    }

def stat_block(values):
    return {
        "mean": round(statistics.mean(values), 4),
        "std": round(statistics.stdev(values), 4),
        "runs": values,
    }

agg["aggregate"] = {}
for t in TASKS:
    agg["aggregate"][t] = {
        m: stat_block([round(summaries[n]["per_task"][t][m] * (100 if m != "avg_steps" else 1), 2)
                       for n, _ in RUNS])
        for m in METRICS
    }
agg["aggregate"]["总体"] = {
    m: stat_block([round(summaries[n]["overall"][m] * (100 if m != "avg_steps" else 1), 2)
                   for n, _ in RUNS])
    for m in METRICS
}

out = os.path.join(BASE, "test-3runs-aggregate.json")
with open(out, "w") as f:
    json.dump(agg, f, ensure_ascii=False, indent=2)
print("已写入", out)

# 打印可读表格
print("\n=== 三次运行聚合结果 ===")
print(f"{'任务类型':<25} {'成功率(均值±std)':<20} {'平均步数(均值±std)':<20} {'无效动作率(均值±std)':<20}")
print("-" * 90)
for t in TASKS + ["总体"]:
    sr = agg["aggregate"][t]["success_rate"]
    st = agg["aggregate"][t]["avg_steps"]
    ir = agg["aggregate"][t]["invalid_action_rate"]
    print(f"{t:<25} {sr['mean']:.1f}% ± {sr['std']:.1f}%     {st['mean']:.1f} ± {st['std']:.1f}      {ir['mean']:.1f}% ± {ir['std']:.1f}%")

print("\n=== 各次运行总体成功率 ===")
for name, _ in RUNS:
    print(f"  {name}: {summaries[name]['overall']['success_rate']*100:.1f}%")
