# 其他 LLM 推理系统第一轮调研

本目录对应 `docs/PROJECT_OUTLINE.md` 的 M1「调研与搜索机制」。本轮目标是先回答一个基础问题：现有 LLM inference 系统一般包含哪些内容，以及这些内容如何映射到本项目后续模块。

检索日期：2026-06-22  
资料口径：优先使用官方文档、官方 GitHub 仓库、官方博客和论文。搜索中出现的百科、新闻和二手文章仅用于发现线索，不作为技术结论依据。

## 文件结构

- `source_registry.md`：本轮保留的来源登记，包含可信度、分类和用途。
- `search_log.md`：本轮搜索查询、打开结果和排除原因。
- `01-serving-engines.md`：高吞吐 LLM 服务引擎横向整理。
- `02-runtime-kernel-backends.md`：本地/跨平台运行时、底层内核和硬件后端整理。
- `03-production-orchestration.md`：生产编排、路由、可观测性和部署层整理。
- `04-papers-and-techniques.md`：论文和关键优化技术整理。
- `feature_taxonomy.md`：从外部系统抽取出的功能分类表，供后续架构拆分使用。

## 初步结论

主流 LLM inference 系统通常不是单一“模型调用器”，而是一组围绕请求生命周期组织的组件：

1. API 与服务层：OpenAI-compatible API、streaming、batch/offline API、工具调用、结构化输出。
2. 请求调度：continuous batching / in-flight batching、chunked prefill、prefill/decode disaggregation、优先级、取消和超时。
3. KV Cache 与内存：paged/blocked/dynamic KV cache、prefix/radix cache、KV offload、KV 量化、eviction、cache sharing。
4. 模型运行时：模型加载、tokenizer/chat template、权重格式、LoRA/multi-LoRA、量化格式、多模态输入。
5. 执行后端与内核：FlashAttention/FlashInfer/TensorRT/Triton/CUDA graph/GEMM/MoE kernel/torch.compile/TVM/OpenVINO/ONNX Runtime。
6. 解码与采样：greedy、beam search、temperature、top-k/top-p、logprobs、stop sequence、speculative decoding、guided/structured decoding。
7. 并行与分布式：tensor/pipeline/data/expert/context parallelism、multi-node、replica、prefix-aware routing。
8. 可观测性与验收：Prometheus/OpenTelemetry/Grafana、benchmark、profiling、event log、错误恢复、性能回归对比。
9. 部署与运维：Docker、Kubernetes、Helm、autoscaling、load balancing、硬件支持矩阵、配置热更新。

## 对本项目的直接启发

- M2 项目骨架应预留 `api`、`engine`、`scheduler`、`memory`、`runtime`、`decoding`、`kernels`、`distributed`、`observability`、`audit` 等边界。
- M3 最小推理链路可以先实现 mock runtime、基本 tokenizer 协议、单请求 decode loop 和基础采样。
- M4 服务与调度应优先补 streaming、请求队列、continuous batching 的接口边界，而不急于实现所有高性能策略。
- M5 内存与性能优化应围绕 KV cache 分块管理、prefix cache、chunked prefill 和 benchmark 体系逐步推进。
- M6 审核系统应吸收外部系统的 metrics/profiling 思路，但用结构化 audit event 补足“可解释请求生命周期”。

