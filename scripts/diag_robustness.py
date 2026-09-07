"""稳健性验证: 区分「补 OK. 系统性恶化首步崩溃」vs「贪心解码混沌扰动」。

背景: 2×2 实验发现给 put 示例 think#3 补 OK. 后, pick_and_place 首步崩溃率从 37.5%→66.7%。
但被补的 think#3 在示例中后段, 而首步崩溃模仿的是示例首行 think#1 (两版都带 OK.)。
本脚本用新拼接固定, 只改 prompt, 测多组变体的首步崩溃率:

  V0  put_0/put_1 都用 checkpoint(缺 think#3 的 OK.)          —— 基线 ≈ C
  V1  只补 put_0 的 OK.
  V2  只补 put_1 的 OK.
  V3  put_0/put_1 都补(当前版)                                —— ≈ D
  --- 以下为「无关扰动」对照组: 完全不碰 think/OK. 结构, 只改措辞 ---
  P1  V0 + wrapper 加一个词 "short"
  P2  V0 + wrapper 改成 "Here are two examples for you."
  P3  V0 + react_put_1 首行 think#1 句式微调(同义, 仍带 OK.)

若 P1/P2/P3 这类无关扰动也让崩溃率大幅偏离 V0, 则证明首步崩溃受贪心解码混沌支配,
「补 OK. 恶化」并非该改动的特定语义后果, 而是 token 扰动的偶然传播。
"""
import asyncio
import json
import re

from openai import AsyncOpenAI

NEW = json.load(open('/root/autodl-tmp/agents/react/alfworld_3prompts.json'))
OLD = json.load(open('/root/autodl-tmp/agents/react/.ipynb_checkpoints/alfworld_3prompts-checkpoint.json'))
TRAJ = '/root/autodl-tmp/agents/result/baseline/seed1234/trajectories.jsonl'
client = AsyncOpenAI(base_url='http://127.0.0.1:8000/v1', api_key='EMPTY')

WRAP_HEAD = 'Interact with a household to solve a task. Here are two examples.\n'


def asm(put0, put1, head=WRAP_HEAD):
    return head + put1 + put0 + '\nHere is the task.\n'


def build(ex, init_obs):
    return ex + "\n" + init_obs + "\n> "


def is_crash(raw):
    a = re.sub(r'^>+\s*', '', raw.strip().lower())
    vv = ("think", "go to", "take", "put", "open", "close",
          "clean", "heat", "cool", "use", "examine", "look", "inventory")
    return not any(a == v or a.startswith(v + " ") or a.startswith(v + ":") for v in vv)


# put_1 首行 think#1 同义微调 (仍带 OK., 不碰 think#3)
PUT1_THINK_TWEAK = (
    OLD['react_put_1']
    .replace('> think: To solve the task, I need to find and take an apple, then put it in sidetable.',
             '> think: To solve this task, I need to find and take an apple, then put it in sidetable.')
)

VARIANTS = {
    'V0 都缺OK.(checkpoint)':      asm(OLD['react_put_0'], OLD['react_put_1']),
    'V1 只补put_0 OK.':            asm(NEW['react_put_0'], OLD['react_put_1']),
    'V2 只补put_1 OK.':            asm(OLD['react_put_0'], NEW['react_put_1']),
    'V3 都补OK.(当前版)':          asm(NEW['react_put_0'], NEW['react_put_1']),
    'P1 V0+wrapper加"short"':      asm(OLD['react_put_0'], OLD['react_put_1'],
                                       'Interact with a household to solve a task. Here are two short examples.\n'),
    'P2 V0+wrapper改措辞':         asm(OLD['react_put_0'], OLD['react_put_1'],
                                       'Interact with a household to solve a task. Here are two examples for you.\n'),
    'P3 V0+think#1同义微调':       asm(OLD['react_put_0'], PUT1_THINK_TWEAK),
}


async def query(prompt, sem):
    async with sem:
        r = await client.completions.create(model='qwen', prompt=prompt,
                                            temperature=0, max_tokens=100, stop=["\n"])
        return r.choices[0].text.strip()


async def main():
    recs = [json.loads(l) for l in open(TRAJ) if l.strip()]
    pa = [r for r in recs if r['task_type'] == 'pick_and_place']
    sem = asyncio.Semaphore(16)
    print(f"pick_and_place n={len(pa)}\n")
    print(f"{'变体':<28} {'崩溃率':<14}")
    print('-' * 46)
    out = {}
    for name, ex in VARIANTS.items():
        raws = await asyncio.gather(*[query(build(ex, r['init_obs']), sem) for r in pa])
        c = sum(is_crash(x) for x in raws)
        out[name] = {'crash': c, 'n': len(pa), 'rate': round(c / len(pa), 4), 'raws': raws}
        print(f"{name:<28} {c}/{len(pa)} = {c/len(pa):.1%}")

    json.dump(out, open('/root/autodl-tmp/agents/result/baseline/diag_robustness.json', 'w'),
              ensure_ascii=False, indent=2)

    base = out['V0 都缺OK.(checkpoint)']['rate']
    print(f"\n以 V0({base:.1%}) 为基线, 各无关扰动组的偏移:")
    for p in ['P1 V0+wrapper加"short"', 'P2 V0+wrapper改措辞', 'P3 V0+think#1同义微调']:
        d = out[p]['rate'] - base
        print(f"  {p}: {out[p]['rate']:.1%}  (Δ={d:+.1%})")
    print(f"\n补OK.组: V1={out['V1 只补put_0 OK.']['rate']:.1%}, "
          f"V2={out['V2 只补put_1 OK.']['rate']:.1%}, V3={out['V3 都补OK.(当前版)']['rate']:.1%}")
    print("\n已写入 result/baseline/diag_robustness.json")


if __name__ == '__main__':
    asyncio.run(main())
