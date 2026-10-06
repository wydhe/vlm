#!/usr/bin/env bash
# 重复压测：上游调度 / 本项目最终配置 / 官方 vLLM，各测 rr = 8, 16, inf，每档 3 次。
# 结果写入 results/serving/rep-<配置>-rr<速率>-rep<次序>_<时间>.json，
# 用 `uv run python -m bench.aggregate --prefix rep-` 汇总。
set -u
cd "$(dirname "$0")/.."
MODEL=weights/Qwen3-8B
COMMON="--max-num-seqs 256 --max-model-len 4096 --gpu-memory-utilization 0.9"

start_server() {   # start_server <配置名>，后台启动服务，把 PID 写到 SERVER_PID
  case $1 in
    upstream) .venv/bin/python -m nanovllm.entrypoints.openai_server --model $MODEL --port 8000 $COMMON \
                --max-num-batched-tokens 16384 --scheduling-policy prefill_first --no-piecewise-cudagraph \
                > results/server_rep_$1.log 2>&1 & ;;
    ours)     .venv/bin/python -m nanovllm.entrypoints.openai_server --model $MODEL --port 8000 $COMMON \
                --max-num-batched-tokens 1024 > results/server_rep_$1.log 2>&1 & ;;
    vllm)     .venv-vllm/bin/vllm serve $MODEL --served-model-name Qwen3-8B --port 8000 $COMMON \
                --max-num-batched-tokens 16384 --disable-log-requests > results/server_rep_$1.log 2>&1 & ;;
  esac
  SERVER_PID=$!
  until curl -s --noproxy "*" localhost:8000/health > /dev/null; do sleep 5; done
}

stop_server() {    # 结束服务，并等显存释放
  kill $SERVER_PID
  wait $SERVER_PID 2> /dev/null
  until [ "$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits)" -lt 1000 ]; do sleep 3; done
}

for config in upstream ours vllm; do
  start_server $config
  for rate in 8 16 inf; do
    for rep in 1 2 3; do
      .venv/bin/python -m bench.serving --tokenizer $MODEL --model Qwen3-8B --num-prompts 500 \
        --request-rate $rate --label rep-$config-rr$rate-rep$rep | grep -E "req/s|itl"
    done
  done
  stop_server
done
echo ALL_DONE
