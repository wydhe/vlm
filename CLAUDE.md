# Project: Vision-Language Model (VLM) System

## 1. 架构目标
- 专注于视觉多模态大模型的训练/微调/推理框架。
- 采用按系统层级的模块化设计：
  - `engine/`：推理引擎，基于 nano-vllm（git subtree 引入，上游 MIT），在此魔改。
  - `align/`：GRPO 等后训练对齐。
  - `rag/`：文档解析、混合检索、重排。
  - `agent/`：工具调用 Agent 循环。
  - `bench/`：压测与评测脚本；`tests/`：单元测试；`docs/`：路线图、压测报告。
- 数据、权重、实验输出分别放 `data/`、`weights/`、`results/`（均被 .gitignore 忽略，不要放代码）。
- 环境：仓库根目录 `.venv`，由 `uv sync` 管理（`pyproject.toml` + `uv.lock`）；模型路径通过 `MODEL_DIR` 等环境变量或 CLI 参数传入。
- 同步上游 nano-vllm：`git subtree pull --prefix=engine https://github.com/GeeeekExplorer/nano-vllm.git main`。

## 2. 代码与开发规范
- 语言环境：Python 3.10+, PyTorch 2.x。
- 遵循 PEP8 规范，核心函数与类必须包含 Type Hints 与 Docstrings。
- 严禁在代码中硬编码任何 API 密钥或服务器绝对路径，均通过配置或环境变量传参。

## 3. Git 提交规范
- 每次完成一个独立功能模块后，请提示用户提交或生成标准 Conventional Commits 信息：
  - feat: 新功能
  - fix: 修复 bug
  - refactor: 重构代码
  - chore: 依赖或环境配置变更
