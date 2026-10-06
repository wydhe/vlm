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

- 推理引擎（Qwen3-8B，16 req/s，3 次重复）：混合批 + 分段 CUDA Graph + 混合步 attention 拆分，ITL P99 248 → 35 ms、
  TPOT 43.6 → 18.1 ms，吞吐与 vLLM 持平（[P1](docs/p1_online_serving.md) · [P2](docs/p2_mixed_batching.md) ·
  [P2.5](docs/p25_piecewise_cudagraph.md) · [P8](docs/p8_robustness.md) · [P9](docs/p9_step_profiling.md) ·
  [博客](docs/blog_engine_profiling.md)）
- Qwen3-VL：与 HF 60/64 逐 token 一致；多模态前缀缓存；批量视觉编码 2.1×；服务端图片输入与演示界面（[P3](docs/p3_qwen3_vl.md)）
- 医疗 RAG：PubMed 2390 万条语料两阶段检索，BioASQ 0.794 → 0.871；教材 / 中文百科语料下检索有害的负面结果
  （[P5](docs/p5_rag_agent.md) · [P10](docs/p10_pubmed_rag.md) · [P11](docs/p11_chinese_rag.md)）
- GRPO（3 个种子）：自研引擎同卡 rollout，引用准确率 0.71 → 0.92；部署回检索流程后有据作答 0.43 → 0.63
  （[P6](docs/p6_grpo.md) · [P7](docs/p7_closed_loop.md)）

## 环境

```bash
uv sync                      # 创建/同步 .venv（torch 2.8 + cu126, flash-attn 2.8.3）
hf download Qwen/Qwen3-0.6B --local-dir weights/Qwen3-0.6B
MODEL_DIR=weights/Qwen3-0.6B uv run python -m bench.offline_throughput
```
