#!/usr/bin/env bash
# 用 3 个 GRPO 种子各自的最好检查点（results/grpo/seed*/best）跑闭环评测（agent/grounded_qa.py）。
# 结果写入 results/grounded/grpo-seed<N>-best.json。
set -u
cd "$(dirname "$0")/.."

for seed in 0 1 2; do
  .venv/bin/python -m nanovllm.entrypoints.openai_server --model results/grpo/seed$seed/best \
      --served-model-name grpo-seed$seed --port 8000 --gpu-memory-utilization 0.6 --max-model-len 8192 \
      > results/server_grounded_seed$seed.log 2>&1 &
  SERVER_PID=$!
  until curl -s --noproxy "*" localhost:8000/health > /dev/null; do
    kill -0 $SERVER_PID 2> /dev/null || { echo "server failed to start, see results/"; exit 1; }
    sleep 5
  done
  .venv/bin/python -m agent.grounded_qa --model grpo-seed$seed --label grpo-seed$seed-best | grep -v -i warn
  kill $SERVER_PID
  wait $SERVER_PID 2> /dev/null
  until [ "$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits)" -lt 1000 ]; do sleep 3; done
done
echo ALL_DONE
