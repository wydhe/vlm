# Project: Vision-Language Model (VLM) System

## 1. 架构目标
- 专注于视觉多模态大模型的训练/微调/推理框架。
- 采用模块化设计：`models/`, `data/`, `training/`, `eval/`, `utils/`。

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
