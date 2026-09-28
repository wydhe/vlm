# VLM Project

面向医疗多模态文档问答的全栈系统：自研轻量推理引擎（基于 nano-vllm 魔改）+ GRPO 对齐 + Agentic RAG。

| 目录 | 内容 |
|---|---|
| `engine/` | 推理引擎（nano-vllm，git subtree，MIT） |
| `align/` | GRPO 对齐 |
| `rag/` | 文档解析、混合检索、重排 |
| `agent/` | 工具调用 Agent |
| `bench/` | 压测与评测 |
| `docs/` | [路线图](docs/roadmap.md)、压测报告 |

## 环境

```bash
uv sync                      # 创建/同步 .venv（torch 2.8 + cu126, flash-attn 2.8.3）
hf download Qwen/Qwen3-0.6B --local-dir weights/Qwen3-0.6B
MODEL_DIR=weights/Qwen3-0.6B uv run python -m bench.offline_throughput
```
