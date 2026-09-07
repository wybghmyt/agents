"""聚合 result/baseline 三遍运行 (seed 1234/42/2024), 计算均值/标准差。

数据源: result/baseline/seed{1234,42,2024}/trajectories.jsonl (直接重算, 不依赖 summary.json)
指标口径:
  - 成功率     = 成功局 / 总局
  - 平均步数   = 仅统计成功对局 (成功局步数总和 / 成功局数); 某 seed 某类 0 成功则记 N/A,
                 跨 seed 聚合时只用有成功局的 seed, 并标注有效样本数 n_valid
  - 无效动作率 = 总无效动作 / 总 llm 调用
输出:
  - result/baseline/aggregate.json
  - result/baseline/aggregate.md
"""
import json
import os
import statistics

PROJECT_ROOT = "/root/autodl-tmp/agents"
BASE = os.path.join(PROJECT_ROOT, "result", "baseline")
RUNS = [("seed1234", "seed1234"), ("seed42", "seed42"), ("seed2024", "seed2024")]
TASKS = [
    "pick_and_place", "pick_clean_then_place", "pick_heat_then_place",
    "pick_cool_then_place", "look_at_obj", "pick_two_obj",
]
DISPLAY_NAME = {
    "pick_and_place": "pick_and_place",
    "pick_clean_then_place": "pick_clean_then_place",
    "pick_heat_then_place": "pick_heat_then_place",
    "pick_cool_then_place": "pick_cool_then_place",
    "look_at_obj": "look_at_obj_in_light",
    "pick_two_obj": "pick_two_obj_and_place",
}


def load_traj(d):
    path = os.path.join(BASE, d, "trajectories.jsonl")
    if not os.path.exists(path):
        raise FileNotFoundError(f"缺少 {path}, 请先跑完三遍 baseline")
    return [json.loads(l) for l in open(path) if l.strip()]


def stats_of(recs, key):
    """key=None 表示总体; 否则为某任务类型。平均步数仅统计成功局。"""
    sub = recs if key is None else [r for r in recs if r["task_type"] == key]
    n = len(sub)
    succ = sum(int(r["success"]) for r in sub)
    succ_steps = sum(r["num_steps"] for r in sub if r["success"])
    invalid = sum(r["invalid_actions"] for r in sub)
    calls = sum(r["llm_calls"] for r in sub)
    return {
        "count": n,
        "success": succ,
        "success_rate": round(succ / n, 4) if n else 0.0,
        # 仅成功对局的平均步数; 无成功局则 None (N/A)
        "avg_steps_success": round(succ_steps / succ, 2) if succ > 0 else None,
        "invalid_action_rate": round(invalid / calls, 4) if calls else 0.0,
    }


def agg_block(vals, pct=False):
    """跨 seed 聚合; vals 可含 None (该类某 seed 无成功局)。pct=True 时转百分数。"""
    def conv(v):
        if v is None:
            return None
        return round(v * 100, 2) if pct else round(v, 2)
    conv_vals = [conv(v) for v in vals]
    clean = [v for v in conv_vals if v is not None]
    if not clean:
        return {"mean": None, "std": None, "runs": conv_vals, "n_valid": 0}
    return {
        "mean": round(statistics.mean(clean), 4),
        "std": round(statistics.stdev(clean), 4) if len(clean) > 1 else 0.0,
        "runs": conv_vals,
        "n_valid": len(clean),
    }


def get_stat(per_run, name, key, field):
    """key=None 取总体, 否则取某任务类型的指定字段。"""
    if key is None:
        return per_run[name]["overall"][field]
    return per_run[name]["per_task"][key][field]


def fmt_sr(block, unit="%"):
    if block["mean"] is None:
        return "N/A"
    return f"{block['mean']:.2f}{unit} ± {block['std']:.2f}{unit}"


def main():
    trajs = {name: load_traj(d) for name, d in RUNS}

    # 每 seed 每类 + 总体的原始统计
    per_run = {}
    for name, _ in RUNS:
        per_run[name] = {
            "per_task": {t: stats_of(trajs[name], t) for t in TASKS},
            "overall": stats_of(trajs[name], None),
        }

    agg = {"runs": {name: {"dir": d, **per_run[name]} for name, d in RUNS},
           "aggregate": {}}

    keys = TASKS + [None]   # None = 总体
    for key in keys:
        label = "总体" if key is None else key
        sr = agg_block([get_stat(per_run, n, key, "success_rate") for n, _ in RUNS], pct=True)
        st = agg_block([get_stat(per_run, n, key, "avg_steps_success") for n, _ in RUNS], pct=False)
        ir = agg_block([get_stat(per_run, n, key, "invalid_action_rate") for n, _ in RUNS], pct=True)
        cnt = get_stat(per_run, RUNS[0][0], key, "count")
        agg["aggregate"][label] = {
            "count": cnt,
            "success_rate": sr,
            "avg_steps_success": st,
            "invalid_action_rate": ir,
        }

    out_json = os.path.join(BASE, "aggregate.json")
    with open(out_json, "w") as f:
        json.dump(agg, f, ensure_ascii=False, indent=2)
    print("已写入", out_json)

    # Markdown 表格
    lines = [
        "# Baseline 三遍聚合 (首步归一化版, seed 1234 / 42 / 2024)",
        "",
        "注: 平均步数仅统计成功对局; 某类在某 seed 无成功局则该 seed 记 N/A, 均值±标准差只用有成功局的 seed (见 n)。",
        "",
        "| 任务类型 | 数量 | 成功率 (mean ± std) | 平均步数(仅成功局, mean ± std) | 无效动作率 (mean ± std) |",
        "| --- | --- | --- | --- | --- |",
    ]
    for key in keys:
        label = "总体" if key is None else key
        a = agg["aggregate"][label]
        name = "总体" if key is None else DISPLAY_NAME[key]
        st = a["avg_steps_success"]
        st_txt = "N/A" if st["mean"] is None else f"{st['mean']:.2f} ± {st['std']:.2f}"
        if st["n_valid"] not in (0, len(RUNS)):
            st_txt += f" (n={st['n_valid']})"
        lines.append(
            f"| {name} | {a['count']} | {fmt_sr(a['success_rate'])} "
            f"| {st_txt} | {fmt_sr(a['invalid_action_rate'])} |"
        )

    # 各 seed 总体明细
    lines += ["", "## 各次运行总体指标", "",
              "| 运行 | 成功率 | 平均步数(仅成功局) | 无效动作率 |",
              "| --- | --- | --- | --- |"]
    for name, _ in RUNS:
        o = per_run[name]["overall"]
        st = o["avg_steps_success"]
        st_txt = "N/A" if st is None else f"{st:.2f}"
        lines.append(
            f"| {name} | {o['success_rate']*100:.2f}% | {st_txt} "
            f"| {o['invalid_action_rate']*100:.2f}% |"
        )

    out_md = os.path.join(BASE, "aggregate.md")
    with open(out_md, "w") as f:
        f.write("\n".join(lines) + "\n")
    print("已写入", out_md)
    print()
    print("\n".join(lines))


if __name__ == "__main__":
    main()
