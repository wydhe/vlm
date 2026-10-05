# VLM Project

面向医疗多模态文档问答的全栈系统：自研轻量推理引擎（基于 nano-vllm 魔改）+ GRPO 对齐 + Agentic RAG。

| 目录 | 内容 |
|---|---|
| `engine/` | 推理引擎（nano-vllm，git subtree，MIT） |
| `align/` | GRPO 对齐 |
| `rag/` | 文档解析、混合检索、重排 |
| `agent/` | 工具调用 Agent |
| `bench/` | 压测与评测 |
| `docs/` | **[项目总报告](docs/PROJECT_REPORT.md)**、[路线图](docs/roadmap.md)、各阶段报告 |

## 结果速览（详见 [项目总报告](docs/PROJECT_REPORT.md)）

- 推理引擎（Qwen3-8B，16 req/s）：混合批 + 分段 CUDA Graph 使 ITL P99 235 → 39 ms、TPOT 45.7 → 19.0 ms，吞吐与 vLLM 持平
  （[P1](docs/p1_online_serving.md) · [P2](docs/p2_mixed_batching.md) · [P2.5](docs/p25_piecewise_cudagraph.md)）
- Qwen3-VL：与 HF 61/64 逐 token 一致，多模态前缀缓存（图片内容哈希）（[P3](docs/p3_qwen3_vl.md)）
- 医疗 RAG / Agent：MIRAGE 系统评估，含负面结果（[P5](docs/p5_rag_agent.md)）
- GRPO：自研引擎同卡 rollout，Qwen3-1.7B 引用准确率 0.71 → 0.90（[P6](docs/p6_grpo.md)）

## 环境

```bash
uv sync                      # 创建/同步 .venv（torch 2.8 + cu126, flash-attn 2.8.3）
hf download Qwen/Qwen3-0.6B --local-dir weights/Qwen3-0.6B
MODEL_DIR=weights/Qwen3-0.6B uv run python -m bench.offline_throughput
```
