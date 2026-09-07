"""add-think 变体机制验证与对比分析"""
import json

recs = [json.loads(l) for l in open('results/add-think/trajectories.jsonl')]

think_after_nothing, nothing_count = 0, 0
ouch_eps = 0
for r in recs:
    turns = r['turns']
    has_ouch_run = any(
        all(t['raw'].strip().lower().startswith('ouch') for t in turns[i:i+3])
        for i in range(len(turns) - 2)
    )
    ouch_eps += has_ouch_run
    for i in range(1, len(turns)):
        if turns[i-1]['obs'] == 'Nothing happens.':
            nothing_count += 1
            if turns[i]['action'].startswith('think'):
                think_after_nothing += 1

n_think_avg = sum(
    sum(1 for t in r['turns'] if t['action'].startswith('think')) for r in recs
) / len(recs)
print(f"平均每局 think 数: {n_think_avg:.1f} (baseline 约 0.9)")
print(f"'Nothing happens.' 后紧跟 think: {think_after_nothing}/{nothing_count} "
      f"({think_after_nothing/max(nothing_count,1):.0%}) (baseline 为 0%)")
print(f"仍出现 ouch 退化循环的局: {ouch_eps}/134 (baseline 失败局中为 66/76)")

# 抽一条看强制 think 的实际内容
found = False
for r in recs:
    if found:
        break
    for i in range(1, len(r['turns']) - 1):
        if r['turns'][i-1]['obs'] == 'Nothing happens.' and r['turns'][i]['action'].startswith('think'):
            print(f"\n示例 (ep={r['episode_id']} {r['task_type']}):")
            print(f"  #{i} {r['turns'][i-1]['action']!r} -> Nothing happens.")
            print(f"  #{i+1} {r['turns'][i]['raw'][:110]!r}")
            print(f"  #{i+2} {r['turns'][i+1]['action'][:80]!r} -> "
                  f"{r['turns'][i+1]['obs'][:60]!r}")
            found = True
            break
