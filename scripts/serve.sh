#!/bin/bash

set -e

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd -- "$SCRIPT_DIR/.." && pwd)"
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
MODEL="./models/Qwen2.5-7B-Instruct"

vllm serve "/root/autodl-tmp/agents/models/Qwen2.5-7B-Instruct" \
    --served-model-name qwen \
    --host 127.0.0.1 \
    --port 8000 \
    --max-model-len 8192 \
    --gpu-memory-utilization 0.75