# Roadmap

目标：面向医疗多模态文档问答的轻量推理引擎 + GRPO 对齐 + Agentic RAG 系统。
硬件：单卡 A800-SXM4-80GB（Ampere，无 FP8；单卡无 TP 实验）。

## 上游 nano-vllm 现状（subtree 引入于 bb823b3，2026-04-26）

已具备（**不能**作为自己的贡献写进简历）：
- PagedAttention 块管理、CUDA Graph、torch.compile、Tensor Parallel。
- 基于 xxhash 的整块前缀缓存（block_size 固定为 256 的整数倍，只复用完整块）。
- Chunked prefill：**仅允许 batch 中第一条序列被分块**。

已知空白（本项目的切入点）：
1. **Prefill 与 Decode 不混批**：只要调度到 prefill，本轮就跳过所有 decode（`scheduler.schedule`），
   长 prompt 到达时正在生成的请求全部停顿 → ITL 尖刺。→ 实现 stall-free 混合批（Sarathi-Serve 思路）。
2. **只有离线 `generate` 接口**：无在线服务、无流式输出，无法测 TTFT/ITL。
   → 异步引擎循环 + OpenAI 兼容流式 API（也是 Agent 的服务底座）。
3. **只支持纯文本 Qwen3**。→ Qwen3-VL 支持（视觉编码器、图像 token 占位合并、M-RoPE），
   以及以图像内容哈希为键的多模态前缀缓存。
4. 前缀缓存粒度粗（256 token 一块）、淘汰策略为空闲队列 FIFO。
   → Radix Tree 索引 + 显式 LRU，对比 RAG / Agent 负载下的命中率。
5. 进阶：Qwen3.5 混合注意力（Gated DeltaNet 3 : Full Attention 1）——分页 KV 与线性注意力状态缓存共存。

## 阶段计划

| 阶段 | 内容 | 验收标准 |
|---|---|---|
| P0 | 环境、subtree、原版 baseline（离线吞吐） | `docs/baseline.md` 有可复现数字 |
| P1 | 在线服务：异步引擎 + 流式 API + 压测客户端（TTFT/ITL/P99） | 与官方 vLLM 同负载对比 |
| P2 | Prefill/Decode 混合批 | 长短混合负载下 P99 ITL 对比 + 消融 |
| P3 | Qwen3-VL-2B/4B 支持 | 与 HF transformers 逐 token 对齐 |
| P4 | Radix Tree 前缀缓存 + 多模态前缀缓存 | RAG/多轮负载命中率、TTFT |
| P5 | RAG（MedRAG 语料 + MIRAGE 评测）+ Agent 循环 | Recall@K、MRR、端到端准确率 |
| P6 | GRPO（PubMedQA 构造引用/拒答任务 → 多轮工具调用 RL） | SFT vs GRPO 对比 + 失败案例 |
| P7（进阶） | Qwen3.5 混合注意力支持 | 输出对齐 + 长上下文显存对比 |

## 数据

- RAG：MedRAG 语料（PubMed / StatPearls / Textbooks / Wikipedia）+ MIRAGE 评测；中文补充 MedQA(MCMLE)、CMB。
- GRPO：PubMedQA（依据摘要 + 干扰文档，自动判定答案/引用/拒答）；MedQA / MedMCQA 做实现验证。
- VLM：SLAKE（中英双语）、VQA-RAD、PathVQA、PMC-VQA。
- 压测：ShareGPT（通用）、自有 RAG/Agent 请求轨迹（前缀复用）。
