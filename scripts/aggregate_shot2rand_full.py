"""聚合完全随机 2-shot 三次运行 (seed 1234/42/2024) 的结果, 计算均值/标准差。"""
import json
import os
import statistics

BASE = "/root/autodl-tmp/agents/results"
RUNS = [
    ("seed1234", "shot-2rand"),
    ("seed42", "shot-2rand-seed42"),
    ("seed2024", "shot-2rand-seed2024"),
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

out = os.path.join(BASE, "shot-2rand-fullrandom-3runs-aggregate.json")
with open(out, "w") as f:
    json.dump(agg, f, ensure_ascii=False, indent=2)
print("已写入", out)
print(json.dumps(agg["aggregate"]["总体"], ensure_ascii=False, indent=2))
