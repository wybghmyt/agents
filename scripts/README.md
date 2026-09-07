## vLLM 推理服务测试

### 硬件

- GPU：NVIDIA RTX 4090 24GB
- CUDA：12.8

### 模型

- Qwen3-8B

### 启动参数

```bash
vllm serve Qwen/Qwen2.5-7B-Instruct \
    --served-model-name qwen \
    --host 127.0.0.1 \
    --port 8000 \
    --max-model-len 8192 \
    --gpu-memory-utilization 0.75