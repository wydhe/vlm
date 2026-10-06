#!/usr/bin/env bash
# 启动 Qwen3-8B 服务，依次跑 PubMed 检索评测和中文医学考试题评测。
# 结果：results/rag/qwen3-8b-pubmed.json、results/rag/qwen3-8b-zh.json
set -u
cd "$(dirname "$0")/.."

.venv/bin/python -m nanovllm.entrypoints.openai_server --model weights/Qwen3-8B --port 8000 \
    --max-num-batched-tokens 2048 --max-model-len 8192 --gpu-memory-utilization 0.75 \
    > results/server_rag_evals.log 2>&1 &
SERVER_PID=$!
until curl -s --noproxy "*" localhost:8000/health > /dev/null; do
  kill -0 $SERVER_PID 2> /dev/null || { echo "server failed to start, see results/"; exit 1; }
  sleep 5
done

.venv/bin/python -u -m rag.eval_pubmed --model Qwen3-8B --label qwen3-8b-pubmed
.venv/bin/python -u -m rag.eval_chinese --model Qwen3-8B --label qwen3-8b-zh

kill $SERVER_PID
wait $SERVER_PID 2> /dev/null
echo ALL_DONE
