"""2×2 受控诊断: 隔离「prompt 完好率」与「baseline.py 拼接方式」对 pick_and_place 首步格式崩溃的影响。

仅生成首步 (不跑完整 episode), 复用 result/baseline/seed1234 中已落盘的 24 局 pick_and_place init_obs。
对每局用 4 种组合构建 prompt, 以 temperature=0 调用 vLLM, 统计首步是否丢失 think: 前缀。

组合:
  A: 旧拼接 + 旧prompt(checkpoint, put 75%完好)   —— 复现 8-30 历史运行
  B: 旧拼接 + 新prompt(当前, 100%完好)
  C: 新拼接 + 旧prompt(checkpoint)
  D: 新拼接 + 新prompt(当前)                      —— 复现今天运行 (预期 ~62.5% 崩溃)
"""
import asyncio
import json
import re

from openai import AsyncOpenAI

PROMPT_NEW = json.load(open('/root/autodl-tmp/agents/react/alfworld_3prompts.json'))
PROMPT_OLD = json.load(open('/root/autodl-tmp/agents/react/.ipynb_checkpoints/alfworld_3prompts-checkpoint.json'))

TRAJ = '/root/autodl-tmp/agents/result/baseline/seed1234/trajectories.jsonl'
BASE_URL = 'http://127.0.0.1:8000/v1'
MODEL = 'qwen'


def assemble_old(prompts, key='put'):
    """8-30 历史运行的拼接: '\\n'.join(react_put_0, react_put_1), 无 wrapper。"""
    return "\n".join(prompts[f'react_{key}_{i}'] for i in range(2))


def assemble_new(prompts, key='put'):
    """当前 baseline.py 的拼接: wrapper + react_put_1 + react_put_0 + 'Here is the task.'。"""
    return ('Interact with a household to solve a task. Here are two examples.\n'
            + prompts[f'react_{key}_1']
            + prompts[f'react_{key}_0']
            + '\nHere is the task.\n')


def build_first_prompt(ex, init_obs):
    """复刻 baseline.py: ctx = ex + '\\n' + init_obs, 再追加 '\\n> '。"""
    return ex + "\n" + init_obs + "\n> "


def is_crash(raw):
    """首步崩溃判定: 不以 think:/合法动词开头 (即丢失 think: 前缀的裸自然语言)。"""
    a = raw.strip().lower()
    a = re.sub(r'^>+\s*', '', a)
    valid_verbs = ("think", "go to", "take", "put", "open", "close",
                   "clean", "heat", "cool", "use", "examine", "look", "inventory")
    return not any(a == v or a.startswith(v + " ") or a.startswith(v + ":") for v in valid_verbs)


async def query(client, prompt, sem):
    async with sem:
        resp = await client.completions.create(
            model=MODEL, prompt=prompt, temperature=0, max_tokens=100, stop=["\n"],
        )
        return resp.choices[0].text.strip()


async def main():
    recs = [json.loads(l) for l in open(TRAJ) if l.strip()]
    pa = [r for r in recs if r['task_type'] == 'pick_and_place']
    print(f"pick_and_place 局数: {len(pa)}")

    combos = {
        'A 旧拼接+旧prompt(75%)': (assemble_old, PROMPT_OLD),
        'B 旧拼接+新prompt(100%)': (assemble_old, PROMPT_NEW),
        'C 新拼接+旧prompt(75%)': (assemble_new, PROMPT_OLD),
        'D 新拼接+新prompt(100%)': (assemble_new, PROMPT_NEW),
    }

    client = AsyncOpenAI(base_url=BASE_URL, api_key='EMPTY')
    sem = asyncio.Semaphore(16)

    results = {}
    for name, (asm, prompts) in combos.items():
        ex = asm(prompts)
        tasks = [query(client, build_first_prompt(ex, r['init_obs']), sem) for r in pa]
        raws = await asyncio.gather(*tasks)
        crashes = sum(is_crash(x) for x in raws)
        results[name] = (crashes, len(pa), raws)
        print(f"\n=== {name}: 首步崩溃 {crashes}/{len(pa)} = {crashes/len(pa):.1%} ===")
        for r, raw in zip(pa, raws):
            mark = 'CRASH' if is_crash(raw) else 'ok   '
            print(f"  ep={r['episode_id']:>3} [{mark}] {raw[:75]!r}")

    print("\n=== 汇总: 首步崩溃率 ===")
    for name, (c, n, _) in results.items():
        print(f"  {name}: {c}/{n} = {c/n:.1%}")

    json.dump({name: {'crash': c, 'n': n, 'rate': round(c/n, 4),
                      'raws': raws}
               for name, (c, n, raws) in results.items()},
              open('/root/autodl-tmp/agents/result/baseline/diag_2x2_firststep.json', 'w'),
              ensure_ascii=False, indent=2)
    print("\n已写入 result/baseline/diag_2x2_firststep.json")


if __name__ == '__main__':
    asyncio.run(main())
