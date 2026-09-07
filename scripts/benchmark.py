import time
from concurrent.futures import ThreadPoolExecutor
from openai import OpenAI

client = OpenAI(
    base_url="http://127.0.0.1:8000/v1",
    api_key="EMPTY",
)

def request():
    response = client.chat.completions.create(
        model="qwen",
        messages=[
            {"role": "system", "content": "你是Qwen-2.5-7B-Instruct，一个由Qwen团队训练的大语言模型，具有强大的自然语言处理能力。"},
            {
                "role": "user",
                "content": "请详细介绍一下Transformer模型。",
            }
        ],
        max_tokens=256,
        stream=False
    )
    print(response.choices[0].message.content)

    return response.usage.completion_tokens


concurrency = 4

start = time.perf_counter()

with ThreadPoolExecutor(max_workers=concurrency) as executor:
    results = list(executor.map(lambda _: request(), range(concurrency)))

elapsed = time.perf_counter() - start

total_tokens = sum(results)

print(f"并发数: {concurrency}")
print(f"总生成 tokens: {total_tokens}")
print(f"耗时: {elapsed:.2f} s")
print(f"吞吐: {total_tokens / elapsed:.2f} tokens/s")