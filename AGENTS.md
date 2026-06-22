# Agent 开发规约

本文件是 `llm_Inference` 仓库的 Agent 级开发指令。所有后续代码开发、调研、测试、审核和验收都必须遵循本文件，并以 `docs/PROJECT_OUTLINE.md` 作为项目主大纲。

## 1. 优先级

1. 用户当前明确要求。
2. 本文件中的 Agent 级开发规约。
3. `docs/PROJECT_OUTLINE.md` 中的项目大纲。
4. 已有代码结构、测试和文档约定。

如果本文件与项目大纲冲突，应先更新本文件或大纲并说明原因，再继续实现。

## 2. 每次开发前必须确认

在进行代码修改前，先确认当前任务对应 `docs/PROJECT_OUTLINE.md` 中的阶段和模块：

- M1：调研与搜索机制。
- M2：项目骨架。
- M3：最小推理链路。
- M4：服务与调度。
- M5：内存与性能优化。
- M6：复杂审核系统。
- M7：验收与迭代。

如果任务无法映射到现有大纲，先补充或修正大纲，再实现代码。

## 3. 代码开发规则

- 不把完整大纲复制进每个代码文件；代码应通过模块边界、类型、测试和少量必要注释体现大纲要求。
- 新增模块必须对应清晰职责，例如 `engine`、`scheduler`、`memory`、`runtime`、`decoding`、`kernels`、`audit`、`observability`。
- 核心逻辑应先定义可测试行为，再写实现。
- 影响请求行为、性能、配置或审核链路的改动，必须同步更新测试或文档。
- 性能优化必须说明来源、假设、适用场景和验证方式。
- 外部推理引擎的方案只能在调研记录和采用理由清楚后进入实现。

### 3.1 架构设计硬约束

- 本项目采用 GPU-first、backend-selective 的架构。GPU 后端和 CPU 后端是可选择的独立执行后端，不是默认接续执行链路。
- 一次请求默认只进入一个已选择的后端，例如 `cuda`、`cpu` 或 `mock`。除非明确实现并记录为内存策略（例如 KV offload），不得让 GPU 热路径依赖 CPU 后端完成常规推理步骤。
- CPU/GPU 可以共享请求协议、调度概念、采样规则、审核事件和 backend 接口，但不得为了复用代码牺牲 CUDA/GPU 最快路径。
- `CudaBackend` 应面向最快推理路径设计，允许使用 CUDA、Triton、FlashInfer、CUDA Graph、专用 attention/GEMM/MoE/sampling kernel 等实现。
- `CpuBackend` 应面向兼容、可验证、小模型执行和无 GPU 环境下的独立 fallback 设计，可参考 llama.cpp、ONNX Runtime GenAI、OpenVINO GenAI 等路线。
- `MockBackend` 应服务于单元测试、调度测试和审核事件测试，不能把真实硬件依赖带入基础测试。

### 3.2 前人经验采用规则

核心架构不得凭空设计。进入实现前，应优先从 `others/` 或后续 `docs/research/` 中提取已验证设计原语，并说明采用方式：

- vLLM：PagedAttention、paged/block KV cache、continuous batching、OpenAI-compatible API、benchmark 方法。
- SGLang：RadixAttention、prefix reuse、structured output、frontend/runtime 分离。
- TensorRT-LLM：in-flight batching、overlap scheduler、KV cache 配置、CUDA Graph、量化和高性能 GPU 后端。
- Hugging Face TGI：生产服务入口、SSE streaming、Prometheus、OpenTelemetry。
- LMDeploy、DeepSpeed-MII、SARATHI：blocked KV cache、Dynamic SplitFuse、chunked prefill、decode-maximal batching。
- FlashInfer：attention、GEMM、MoE、sampling 等 GPU kernel 边界。
- llama.cpp、ONNX Runtime GenAI、OpenVINO GenAI：CPU/跨平台运行时、generate loop、KV 管理和结构化输出。

如果实现方案偏离这些已有设计原语，应在文档或代码评审说明中解释原因、收益、风险和验证方式。

## 4. 测试规则

所有非纯文档改动都应考虑测试。测试优先级如下：

- 单元测试覆盖核心算法和边界条件。
- Backend contract tests 覆盖 `cuda`、`cpu`、`mock` 后端应共同满足的请求、采样、停止条件和错误语义。
- CPU/GPU 一致性测试覆盖可比模型、固定随机种子、采样参数和输出结构；性能测试不得用 CPU 路径代表 GPU 路径。
- 集成测试覆盖请求到生成结果的完整链路。
- 并发测试覆盖请求队列、取消、超时和资源上限。
- 性能测试覆盖吞吐、延迟、显存和 KV Cache 行为。
- 回归测试覆盖已修复的问题。

如果某次改动没有运行测试，最终说明必须写明原因。

## 5. 审核规则

实现复杂审核系统时，应覆盖：

- 配置审核。
- 请求审核。
- 输出审核。
- 性能审核。
- 变更审核。
- 测试和 benchmark 结果审核。

核心请求链路中产生的重要状态变化，应尽量通过结构化事件记录，而不是只依赖临时日志文本。

## 6. 验收规则

每次完成一个阶段或重要功能后，必须给出：

- 完成内容。
- 对应大纲位置。
- 测试或验证命令。
- 未完成事项或后续风险。

发布或阶段验收前，必须确认 README、测试文档、调研文档、代码实现和验收标准保持一致。

## 7. 维护规则

- `docs/PROJECT_OUTLINE.md` 是详细项目大纲。
- `AGENTS.md` 是执行约束和开发规约。
- 两者都应保持简洁、可执行、可维护。
- 当架构方向变化时，先改文档，再改代码。
- 每次完成代码、文档、配置或调研产物修正后，都必须进行一次 git 提交；提交信息必须使用中文。
- 每次完成 git 提交后，都必须将当前分支 push 到 GitHub 远端。
- 提交时只纳入本次任务相关文件，不得把无关工作区变更混入提交。
