# LLM 推理引擎项目大纲

本文档是 `llm_Inference` 项目的主执行大纲。后续所有调研、架构设计、代码开发、测试补充和验收修正，都应先对齐本大纲；如果后续发现更优路线，应先更新本文档，再调整实现。

## 0. 项目目标与执行原则

### 0.1 总目标

构建一个可验证、可扩展、可审计、GPU-first 且 CPU-compatible 的 LLM 推理引擎，并以现有高性能推理引擎作为长期对标对象。引擎应通过 backend 选择在 GPU 或 CPU 上独立执行请求，而不是把 CPU 与 GPU 设计成默认接续执行链路。初始对标重点包含 README 中提到的 vLLM 与 SGLang；完整对标清单应在调研阶段基于最新公开资料确认。

### 0.2 核心原则

- 证据优先：优化方案必须能追溯到论文、官方文档、源码、基准测试或可复现实验。
- 前人经验优先：核心架构应显式继承或改造已有推理系统中已验证的设计原语，例如 vLLM 的 paged KV cache、SGLang 的 prefix reuse、TensorRT-LLM 的 GPU 后端优化、TGI 的生产观测和 llama.cpp / ONNX Runtime / OpenVINO 的 CPU 兼容路线。
- Backend 选择优先：CUDA/GPU 后端、CPU 后端和 mock 后端共享接口与语义，但各自拥有独立实现；一次请求默认只在一个已选择后端内完成。
- GPU 最快路径优先：CUDA 后端允许采用专用 kernel、CUDA Graph、Triton、FlashInfer 或其他高性能实现，不因 CPU 兼容路径牺牲 GPU 热路径。
- 测试驱动：核心能力先定义测试用例，再开发实现，最后以测试和基准报告确认结果。
- 模块清晰：调度、内存、执行、采样、服务、审核、测试等边界应明确，避免早期耦合。
- 可审计：请求链路、配置变更、性能结果、错误和异常行为都应能被记录、复盘和验证。
- 可迭代：先完成最小可运行引擎，再逐步引入高性能特性、复杂审核和生产化能力。

## 1. 搜索机制与现有推理引擎调研

### 1.1 目标

建立一个持续更新的调研与知识抽取机制，用来分析现有 LLM 推理引擎的架构、性能优化点和工程权衡，并沉淀可被本项目采用的设计。

### 1.2 调研对象

调研对象在执行阶段通过联网搜索、官方文档、源码和论文确认。初始必须覆盖：

- README 已声明的 vLLM。
- README 已声明的 SGLang。
- 其他当期主流或有代表性的开源、商业或研究型 LLM 推理系统。

### 1.3 搜索与筛选机制

每个调研对象应记录：

- 项目定位：服务框架、底层运行时、编译优化器、分布式推理框架或特定硬件方案。
- 核心特性：连续批处理、Paged KV Cache、Prefix Cache、Speculative Decoding、Chunked Prefill、张量并行、流水并行、专家并行、量化、LoRA 服务、多模态支持、OpenAI API 兼容等。
- 性能指标：吞吐、首 token 延迟、总延迟、显存占用、并发能力、长上下文能力、硬件利用率。
- 工程指标：部署复杂度、可观测性、错误恢复、扩展接口、社区活跃度、许可证和维护状态。
- 可采用内容：算法、数据结构、调度策略、内存策略、API 设计、测试方法或 benchmark 方法。

### 1.4 调研产物

后续应逐步创建并维护：

- `others/`：第一轮外部推理系统搜索结果和分类沉淀，作为 M1 原始调研交付。
- `docs/research/engine_survey.md`：推理引擎横向对比。
- `docs/research/source_registry.md`：资料来源、可信度和更新时间。
- `docs/research/optimization_notes.md`：可吸收的优化点和采用理由。
- `benchmarks/`：本项目与对标引擎的可复现 benchmark 脚本与结果。

### 1.5 采用规则

任何来自外部引擎的优化思路，在进入本项目实现前必须说明：

- 来源和证据。
- 解决的问题。
- 适用场景和不适用场景。
- 本项目采用方式。
- 对应测试和验收标准。

### 1.6 第一轮调研固化的设计原语

第一轮调研结论先沉淀在 `others/`，后续可迁移或归档到 `docs/research/`。进入代码实现前，以下设计原语应被视为本项目的默认起点：

- Backend 选择：借鉴 llama.cpp、ONNX Runtime GenAI、OpenVINO GenAI 的 CPU/跨平台经验，以及 TensorRT-LLM、FlashInfer、vLLM 的 GPU 路线，形成 `cuda`、`cpu`、`mock` 可选后端。
- KV Cache 中心化：借鉴 vLLM PagedAttention、LMDeploy/DeepSpeed blocked KV、SGLang RadixAttention，优先设计 paged/blocked KV cache、block table、prefix cache 和 cache event。
- Continuous batching：借鉴 vLLM、TGI、SGLang、TensorRT-LLM，让调度器支持 token/iteration 级动态 batching。
- Prefill/decode 分离：借鉴 SARATHI、TensorRT-LLM、SGLang、Ray Serve LLM，明确 prefill task 与 decode task 的调度差异。
- GPU kernel 插拔：借鉴 FlashInfer、TensorRT-LLM、FasterTransformer，保留 attention、GEMM、MoE、sampling、RoPE、norm 等 kernel 边界。
- 生产观测与审核：借鉴 TGI、vLLM、TensorRT-LLM、Ray Serve LLM 的 metrics、profiling 和 benchmark，再扩展为本项目的结构化 audit event。

## 2. LLM 推理引擎主要内容

### 2.1 总体架构

引擎主体应围绕以下模块组织：

- API 与服务层：提供本地调用接口、HTTP 服务接口，以及后续 OpenAI-compatible API。
- 请求生命周期：接收请求、解析参数、排队、调度、执行、流式返回、结束清理。
- 调度器：支持请求队列、continuous batching、prefill/decode 分离、chunked prefill、优先级、取消和超时。
- 后端选择层：通过配置或部署选择 `cuda`、`cpu`、`mock` 等后端；一次请求默认只在一个后端中完成，不将 CPU/GPU 设计为默认接续执行。
- 模型运行时：负责模型加载、权重管理、tokenizer 适配、执行图或 kernel 调用，并保持 backend contract 清晰。
- KV Cache 管理：负责块分配、回收、复用、内存上限控制和长上下文策略。
- Prefix Cache 管理：负责多轮对话、RAG、共享系统提示词等场景中的 prefix 识别、复用和失效。
- 解码与采样：支持 greedy、temperature、top-k、top-p、repetition penalty、停止条件等。
- Kernel 与后端抽象：为 CUDA、Triton、FlashInfer、自定义 GPU kernel、CPU kernel 或其他硬件后端预留边界。
- 并发与分布式：后续支持多 worker、多 GPU、张量并行、流水并行和远程 worker。
- 配置系统：统一模型、服务、内存、调度、审核、日志和 benchmark 配置。
- 可观测性：记录延迟、吞吐、显存、队列长度、错误、超时和请求状态。

### 2.2 推荐代码边界

后续代码结构可按实际语言与框架微调，但应保持下列逻辑边界：

```text
src/
  llm_inference/
    api/
    engine/
    scheduler/
    memory/
    runtime/
      backends/
        cuda/
        cpu/
        mock/
    decoding/
    kernels/
      cuda/
      cpu/
    distributed/
    observability/
    audit/
tests/
benchmarks/
docs/
others/
```

### 2.3 最小可运行目标

第一阶段最小可运行引擎应至少支持：

- 通过配置选择 `mock` 或 `cpu` 后端，并为 `cuda` 后端预留相同 backend contract。
- 加载一个小型测试模型或 mock model。
- 接收单条生成请求。
- 执行基本 token 生成循环。
- 返回完整生成结果。
- 有基础日志和错误处理。
- 有单元测试和端到端 smoke test。

### 2.4 进阶优化目标

在最小引擎稳定后，逐步加入：

- 多请求批处理。
- Continuous batching。
- Prefill/decode 调度和 chunked prefill。
- KV Cache 块管理、block table、prefix cache 和 cache event。
- Streaming 输出。
- 并发取消和超时。
- CUDA 后端最小可运行路径，并逐步接入专用 GPU kernel。
- CPU 后端作为独立可选执行路径和正确性参考路径。
- 性能 benchmark。
- 与对标引擎在相同硬件、模型、请求分布下比较。

### 2.5 Backend 执行模型

Backend 是运行时的核心边界。推荐接口至少表达：

- `load_model()`：加载模型、tokenizer、权重和 backend 私有资源。
- `prefill()`：处理 prompt/context，并建立或更新 KV cache。
- `decode_step()`：执行单步或批量 decode。
- `allocate_kv()` / `free_kv()`：由 backend 或 memory manager 管理 KV 生命周期。
- `sample()` 或 logits 输出：允许 decoding 模块统一处理采样，也允许高性能 backend 接入融合采样 kernel。

执行规则：

- `CudaBackend`：GPU 最快路径，优先优化吞吐、TTFT、ITL、显存和 GPU 利用率。
- `CpuBackend`：独立 CPU 执行路径，优先保证兼容、可验证、小模型可运行和无 GPU 环境下的独立 fallback。
- `MockBackend`：测试路径，用于 deterministic 输出、调度测试、审核事件测试。
- CPU/GPU 之间不得默认接续执行。若未来引入 KV offload、异构执行或 CPU 参与采样，必须作为显式策略记录来源、适用场景、风险、benchmark 和审核事件。

### 2.6 核心设计原语到模块映射

| 设计原语 | 来源经验 | 本项目模块 |
| --- | --- | --- |
| Paged/blocked KV cache | vLLM、LMDeploy、DeepSpeed-MII | `memory`、`runtime/backends/cuda` |
| Prefix/Radix cache | SGLang、vLLM | `memory`、`scheduler`、`audit` |
| Continuous batching | vLLM、TGI、SGLang、TensorRT-LLM | `scheduler` |
| Prefill/decode 分离与 chunked prefill | SARATHI、TensorRT-LLM、SGLang、Ray Serve LLM | `scheduler`、`engine` |
| GPU kernel 插拔 | FlashInfer、TensorRT-LLM、FasterTransformer | `kernels/cuda`、`runtime/backends/cuda` |
| CPU/跨平台执行 | llama.cpp、ONNX Runtime GenAI、OpenVINO GenAI | `runtime/backends/cpu`、`kernels/cpu` |
| Streaming 与生产观测 | TGI、vLLM、Ray Serve LLM | `api`、`observability` |
| 结构化审核 | 外部 metrics/benchmark 经验 + 本项目 M6 目标 | `audit`、`observability`、`benchmarks` |

## 3. 复杂审核系统

### 3.1 目标

构建一个覆盖开发期、运行期和验收期的审核系统，使引擎行为、性能变化、异常、配置和测试结果都可追踪。

### 3.2 审核层级

审核系统应至少包含：

- 代码审核：类型检查、lint、复杂度检查、依赖安全、许可证检查。
- 配置审核：模型路径、显存上限、并发上限、采样参数、服务暴露端口等配置校验。
- 请求审核：请求 ID、输入参数、选择的 backend、调度状态、生成状态、异常和资源消耗记录。
- KV Cache 审核：block 分配、复用、释放、prefix cache 命中、eviction、offload 和碎片情况。
- 性能审核：吞吐、延迟、显存、batch size、cache 命中率、backend 差异和错误率趋势。
- 变更审核：每次核心改动都应关联测试、benchmark 或设计说明。
- 输出审核：对生成结果的长度、停止原因、错误码和流式输出完整性进行一致性检查。

### 3.3 审核产物

后续应实现或维护：

- 审核事件 schema。
- 本地 audit log。
- 测试期审核报告。
- benchmark 审核报告。
- 失败用例与回归问题记录。

### 3.4 审核规则

核心路径变更在合入前必须满足：

- 有对应测试。
- 有可复现验证命令。
- 对性能或资源有影响时，必须提供 benchmark 说明。
- 对请求行为有影响时，必须更新接口或使用文档。
- 如果偏离本大纲，必须先更新本大纲并解释原因。

## 4. 测试用例与基于测试的开发方法

### 4.1 测试分层

项目测试应分层建设：

- 单元测试：调度器、KV Cache、采样器、配置解析、审核事件等。
- Backend contract tests：验证 `cuda`、`cpu`、`mock` 后端在请求语义、错误语义、停止条件和输出结构上遵循同一契约。
- CPU/GPU 对照测试：在可比模型、固定参数和固定随机种子下验证输出结构、停止原因和关键数值容差；不得用 CPU 性能代表 GPU 性能。
- 属性测试：缓存分配/释放、调度公平性、停止条件等不变量。
- Golden tests：固定输入下的 mock model 输出、API 响应结构、错误码。
- 集成测试：从请求进入到生成返回的完整链路。
- 并发测试：多请求、取消、超时、背压和资源上限。
- 兼容测试：OpenAI-compatible API 的请求和响应字段。
- 性能测试：吞吐、延迟、显存、水位线和 cache 命中率。
- 回归测试：每个修复过的问题都补充可重复触发的测试。

### 4.2 开发流程

后续开发遵循以下顺序：

1. 根据本大纲定位要实现的模块。
2. 写出目标行为和测试用例。
3. 运行测试，确认测试能暴露缺失能力或问题。
4. 编写最小实现。
5. 运行聚焦测试与相关集成测试。
6. 必要时运行 benchmark。
7. 更新文档、验收说明或审核规则。

### 4.3 测试方法文档

后续应维护：

- `docs/testing.md`：本地测试、GPU 测试、benchmark 和回归测试方法。
- `tests/`：自动化测试用例。
- `benchmarks/README.md`：benchmark 场景、模型、硬件和指标说明。

## 5. 验收方法

### 5.1 阶段验收

每个阶段结束时应检查：

- 产物是否对应本大纲中的明确模块。
- 测试是否覆盖新增能力。
- 审核日志或报告是否能解释核心行为。
- 文档是否足以让下一个开发步骤继续推进。

### 5.2 功能验收

功能验收至少包含：

- 模型或 mock model 能被加载。
- 请求能完整执行并返回。
- 错误输入能得到稳定错误响应。
- 并发请求不会破坏状态。
- 取消、超时、停止条件能按预期执行。

### 5.3 性能验收

性能验收应在调研后定义具体硬件、模型和请求分布。初始指标包括：

- Tokens/s。
- Time to First Token。
- Inter-token latency。
- 总请求延迟。
- 最大稳定并发。
- 显存峰值。
- KV Cache 命中率和碎片情况。
- Backend 选择带来的性能差异，尤其是 CUDA backend 不应被 CPU 兼容路径拖慢。
- Prefill/decode latency、queue latency、batch composition 和 GPU utilization。

### 5.4 审核验收

审核系统验收至少包含：

- 请求级 audit 事件完整。
- 配置错误能被拒绝或明确警告。
- 性能测试结果可追溯。
- 核心失败能关联到请求、配置和模块。
- 回归问题有对应测试。

### 5.5 发布验收

发布前必须满足：

- 全量测试通过。
- 关键 benchmark 完成并记录。
- README 与测试文档可指导使用。
- 大纲、调研、实现和验收标准保持一致。

## 6. 建议执行路线

### M0：项目大纲与基础文档

- 建立本大纲。
- 更新 README 入口。
- 明确后续开发必须对齐大纲。

### M1：调研与搜索机制

- 建立调研文档目录。
- 完成第一轮现有推理引擎横向分析。
- 提取可采用优化点。
- 将 `others/` 中的第一轮搜索结果归纳为可采用设计原语，并在进入实现前迁移或引用到 `docs/research/`。

### M2：项目骨架

- 建立源码、测试、benchmark 和文档目录。
- 确定主要语言、依赖管理和测试框架。
- 完成最小配置系统和日志系统。
- 建立 backend contract 和 `runtime/backends/{cuda,cpu,mock}` 目录边界。
- 建立 `memory`、`scheduler`、`decoding`、`kernels/{cuda,cpu}` 的接口骨架。
- 明确 `--backend cuda|cpu|mock` 或等价配置，不把 CPU/GPU 设计成默认接续执行。

### M3：最小推理链路

- 实现 mock model 或小模型推理链路。
- 支持单请求生成。
- 完成基础单元测试和端到端 smoke test。
- 实现 CPU 或 mock 后端的最小 generate loop，并验证 backend contract。
- 为 CUDA 后端保留最小接口和测试占位，后续接入 GPU 实现时不改变上层请求语义。

### M4：服务与调度

- 实现 API 层。
- 实现请求队列、基础 continuous batching 和 streaming。
- 增加取消、超时和错误处理。
- 将调度状态拆分为 prefill/decode，并记录结构化调度事件。

### M5：内存与性能优化

- 实现 KV Cache 管理。
- 实现 paged/blocked KV cache、block table、prefix cache 和 cache event。
- 引入 prefill/decode 调度优化、chunked prefill 和 GPU backend 优化。
- 接入或实现 CUDA/Triton/FlashInfer 风格的关键 kernel。
- 建立 benchmark 与对标报告。

### M6：复杂审核系统

- 实现 audit schema、audit log 和报告。
- 将审核系统接入请求链路、配置系统和 benchmark。
- 覆盖 backend 选择、KV cache 生命周期、调度决策、性能变化和 CPU/GPU 对照结果。

### M7：验收与迭代

- 完成功能、性能、审核、文档验收。
- 根据调研和 benchmark 结果继续迭代优化。

## 7. 后续变更约束

后续任何开发或修正都应遵循：

- 先说明对应本大纲的哪一阶段和模块。
- 涉及后端执行时必须说明是 `cuda`、`cpu`、`mock` 还是通用 backend contract 改动。
- 涉及性能路径时必须说明继承或偏离了哪个外部设计原语，以及验证方式。
- 新增能力必须有测试计划。
- 核心行为变化必须更新文档。
- 性能相关变化必须补充指标或说明无需 benchmark 的原因。
- 架构偏离必须先更新本大纲。
