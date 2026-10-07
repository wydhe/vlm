#!/usr/bin/env bash
# PubMed 语料下的两组补充实验：
#   1. 闭环评测（agent/grounded_qa.py --corpus pubmed）：原始 Qwen3-1.7B 和 3 个 GRPO 种子的最好检查点
#      → results/grounded/<label>-pubmed.json
#   2. Agentic RAG（agent/medical_agent.py --corpus pubmed）：Qwen3-8B，和教材语料用同一批 1000 题
#      → results/agent/agent-qwen3-8b-pubmed.json
set -u
cd "$(dirname "$0")/.."

SERVER_PID=""

serve() {   # serve <模型目录> <服务名> <显存比例>
  .venv/bin/python -m nanovllm.entrypoints.openai_server --model "$1" --served-model-name "$2" --port 8000 \
      --gpu-memory-utilization "$3" --max-model-len 8192 --max-num-batched-tokens 2048 \
      > "results/server_$2.log" 2>&1 &
  SERVER_PID=$!
  until curl -s --noproxy "*" localhost:8000/health > /dev/null; do
    kill -0 $SERVER_PID 2> /dev/null || { echo "server $2 failed to start, see results/server_$2.log"; exit 1; }
    sleep 5
  done
}

stop() {
  kill $SERVER_PID
  wait $SERVER_PID 2> /dev/null
  until [ "$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits)" -lt 1000 ]; do sleep 3; done
}

serve weights/Qwen3-1.7B Qwen3-1.7B 0.6
.venv/bin/python -u -m agent.grounded_qa --model Qwen3-1.7B --corpus pubmed --label qwen3-1.7b-base-pubmed
stop

for seed in 0 1 2; do
  serve results/grpo/seed$seed/best grpo-seed$seed 0.6
  .venv/bin/python -u -m agent.grounded_qa --model grpo-seed$seed --corpus pubmed --label grpo-seed$seed-best-pubmed
  stop
done

serve weights/Qwen3-8B Qwen3-8B 0.75
.venv/bin/python -u -m agent.medical_agent --model Qwen3-8B --corpus pubmed --per-dataset 200 \
    --compare results/rag/qwen3-8b-textbooks-k5.json --label agent-qwen3-8b-pubmed
stop
echo ALL_DONE
