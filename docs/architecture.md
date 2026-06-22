# M2 项目骨架设计

本文件记录 M2 阶段的代码骨架决策，对应 `docs/PROJECT_OUTLINE.md` 的 M2「项目骨架」。

## 语言与依赖

- 主要语言：Python 3.11+。
- 包布局：`src/llm_inference/`。
- 初始运行时依赖：无第三方依赖。
- 初始测试框架：标准库 `unittest`，避免项目早期依赖安装阻塞。
- 依赖管理入口：`pyproject.toml`。

## 后端选择

本项目采用 backend-selective 架构。一次请求默认只在一个后端内执行：

- `mock`：确定性测试后端。
- `cpu`：独立 CPU 执行后端，用于兼容、小模型和正确性参考。
- `cuda`：GPU 最快路径边界，当前 M2 仅保留接口，后续接入 CUDA、Triton、FlashInfer 或 CUDA Graph。

CPU 与 GPU 共享 backend contract、请求/响应类型、采样语义和 audit event，但不共享性能实现。

## 当前模块边界

- `api`：请求与响应类型。
- `engine`：请求生命周期编排。
- `scheduler`：初始 FIFO scheduler，以及 prefill/decode 状态类型。
- `memory`：最小 KV cache block accounting。
- `runtime`：backend contract、backend registry 和 `cuda`、`cpu`、`mock` 后端目录。
- `decoding`：停止条件和采样接口。
- `kernels`：`cuda` 与 `cpu` kernel 边界。
- `observability`：基础日志配置。
- `audit`：结构化事件记录。
- `distributed`：未来分布式入口占位。

## M3 最小推理链路更新

M3-1 到 M3-3 已将早期占位式字符串生成改为 token/logits 驱动的最小推理链路：

- `tokenization` 定义 tokenizer protocol，并提供 deterministic `MockTokenizer`。
- `mock` backend 使用 deterministic mock model 产生 logits。
- `cpu` backend 保持独立可选路径，并使用 CPU reference model 产生 logits。
- `engine.generate()` 负责 prompt encode、prefill、decode step、greedy sampling、token decode、stop condition 和 audit event。
- `cuda` backend 仍保持独立边界，当前显式报 `BackendUnavailable`，不接续 CPU 路径。

## 前人经验映射

- vLLM / LMDeploy / DeepSpeed-MII：`memory` 预留 paged/blocked KV cache 方向。
- SGLang：后续 prefix/radix cache 应进入 `memory` 与 `scheduler`。
- TensorRT-LLM / FlashInfer：CUDA 后端和 GPU kernel 边界独立保留。
- TGI / Ray Serve LLM：API、streaming、metrics 和 observability 后续进入 `api` 与 `observability`。
- llama.cpp / ONNX Runtime GenAI / OpenVINO GenAI：CPU 后端作为独立执行路径，而不是 GPU 热路径的接续阶段。
