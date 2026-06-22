# 项目当前状态与任务完成度

更新时间：2026-06-22  
对应大纲：`docs/PROJECT_OUTLINE.md`

本文档用于汇总当前已经完成的内容、亟待完成的内容，以及按阶段和模块估算的完成度。百分比表示“相对项目大纲当前目标的完成程度”，不是性能达标比例，也不是最终产品成熟度。

## 总体判断

当前项目已经完成从 M0 到 M2 的基础闭环：有明确大纲、Agent 规约、第一轮调研、Python 项目骨架、backend contract、`mock`/`cpu`/`cuda` 后端边界、基础测试和 `uv` 初始化方式。

项目尚未进入真实高性能推理阶段。当前最紧急的方向是把 M3 的最小推理链路做扎实：从“占位式 mock/cpu token 生成”推进到“有 tokenizer、mock model、可测试生成循环、稳定错误语义和端到端 smoke test”的真正最小引擎。

总体完成度估算：**约 28%**。

## 已完成内容

### 文档与规约

- 已建立 `AGENTS.md`，明确开发规约、测试规约、审核规约、中文提交和提交后 push 规则。
- 已建立 `docs/PROJECT_OUTLINE.md`，明确 M0-M7 阶段路线。
- 已建立 `docs/architecture.md`，记录 M2 项目骨架设计。
- 已建立 `docs/testing.md`，记录 `uv` 初始化和测试方法。
- 已建立 `benchmarks/README.md`，记录后续 benchmark 指标入口。
- 已在 `README.md` 中补充项目大纲、Agent 规约和本地开发入口。

### 调研与设计原语

- 已完成第一轮外部 LLM 推理系统搜索和分类，存放于 `others/`。
- 已覆盖 vLLM、SGLang、TensorRT-LLM、TGI、LMDeploy、LightLLM、DeepSpeed-MII、llama.cpp、MLC LLM、OpenVINO GenAI、ONNX Runtime GenAI、FlashInfer 等来源。
- 已抽取核心设计原语：backend 选择、paged/blocked KV cache、prefix/radix cache、continuous batching、prefill/decode 分离、GPU kernel 插拔、生产观测与审核。

### 项目骨架

- 已建立 Python `src/` 布局和 `pyproject.toml`。
- 已通过 `uv venv .venv --python 3.12` 初始化虚拟环境。
- 已通过 `uv pip install -e .` 安装本项目。
- 已生成并提交 `uv.lock`。
- 已新增 `.gitignore`，忽略 `.venv/`、缓存和构建产物。

### 核心代码边界

- `api`：已有 `GenerationRequest`、`GenerationResult`、`TokenChunk`。
- `engine`：已有同步 `InferenceEngine.generate()` 最小请求生命周期。
- `runtime`：已有 backend contract、backend registry，以及 `cuda`、`cpu`、`mock` 三类后端目录。
- `mock` backend：已有确定性 token 生成路径。
- `cpu` backend：已有独立可选执行路径的最小占位生成。
- `cuda` backend：已有明确边界，当前显式抛出 `BackendUnavailable`。
- `scheduler`：已有 FIFO scheduler 和 prefill/decode 状态类型。
- `memory`：已有最小 KV cache block allocation/free。
- `decoding`：已有 stop condition 和 greedy sampler 接口。
- `audit`：已有结构化 audit event 和事件记录器，并接入 engine。
- `observability`：已有基础 logging 配置。
- `kernels`：已有 `cuda`/`cpu` kernel 边界。

### 测试与验证

- 已有 11 个 `unittest` 测试。
- 已覆盖配置校验、backend contract、mock/cpu/cuda 边界、engine audit、scheduler 和 KV cache。
- 已验证命令：

```powershell
uv run python -m unittest discover -s tests
```

当前结果：`Ran 11 tests ... OK`

- 已验证 CLI smoke test：

```powershell
uv run python -m llm_inference --backend mock --max-new-tokens 2 hello
```

当前输出：`mock_0mock_1`

## 按阶段完成度

| 阶段 | 完成度 | 已完成 | 亟待完成 |
| --- | ---: | --- | --- |
| M0：项目大纲与基础文档 | 100% | README、AGENTS、PROJECT_OUTLINE 已建立，提交/push 规约已固化 | 后续只需随架构变化维护 |
| M1：调研与搜索机制 | 70% | `others/` 已完成第一轮横向调研和设计原语抽取 | 将核心调研迁移/整理到 `docs/research/`；补充源码级二次调研；建立可更新 source registry |
| M2：项目骨架 | 85% | Python `src/` 布局、`uv`、backend contract、基础模块、测试、文档均已建立 | 增加类型检查/lint 配置；完善配置文件加载；补 CI；进一步收紧接口文档 |
| M3：最小推理链路 | 30% | `mock`/`cpu` 后端已有最小同步生成；CLI 可跑；audit 已接入 | 实现 tokenizer 协议、mock model、真实生成循环语义、golden tests、错误响应和端到端 smoke test |
| M4：服务与调度 | 10% | 有请求类型、engine 生命周期、FIFO scheduler 状态边界 | 实现 HTTP/API 层、streaming、continuous batching、取消、超时、并发队列和 prefill/decode 调度 |
| M5：内存与性能优化 | 8% | 有 KV cache block accounting、benchmark 指标文档和 kernel 边界 | 实现 paged/blocked KV、block table、prefix cache、chunked prefill、CUDA backend、GPU benchmark |
| M6：复杂审核系统 | 15% | 有 audit event、EventRecorder，并记录请求、后端选择、KV 分配、prefill/decode、完成/失败事件 | 定义 audit schema、持久化 audit log、配置/性能/变更/benchmark 审核报告 |
| M7：验收与迭代 | 10% | 有阶段状态文档、测试文档和提交推送流程 | 建立阶段验收清单、发布验收、回归记录和性能对标报告 |

## 按模块完成度

| 模块 | 完成度 | 当前状态 | 下一步 |
| --- | ---: | --- | --- |
| `api` | 20% | 有本地请求/响应 dataclass | 定义错误结构、stream chunk、OpenAI-compatible 字段和 HTTP API |
| `engine` | 35% | 有同步 generate 生命周期和 audit 接入 | 增加状态机、异常分类、streaming、并发安全和请求清理语义 |
| `scheduler` | 20% | 有 FIFO scheduler、请求状态和 prefill/decode task 类型 | 实现 continuous batching、prefill/decode 拆分、取消、超时、优先级 |
| `runtime` contract | 45% | 有 backend 接口、registry、`cuda`/`cpu`/`mock` 目录 | 收紧 contract tests；明确 logits/KV/state 形状；加入模型加载配置 |
| `runtime/backends/mock` | 55% | 可确定性生成，用于测试 | 扩展为 mock model + tokenizer + golden output |
| `runtime/backends/cpu` | 25% | 有独立 CPU 占位生成路径 | 接入真实 CPU reference model 或最小张量模型 |
| `runtime/backends/cuda` | 5% | 只有边界和未实现异常 | 设计 CUDA 最小路径，选择 PyTorch/Triton/FlashInfer 的切入策略 |
| `memory` | 20% | 有最小 KV block 分配/释放 | 实现 block table、free list、引用计数、prefix cache、碎片统计 |
| `decoding` | 25% | 有 stop condition 和 greedy sampler | 接入 logits processor、temperature、top-k/top-p、repetition penalty |
| `kernels` | 5% | 只有 CPU reference 示例和 CUDA 占位 | 明确 attention/GEMM/sampling kernel API；后续接入 CUDA/Triton |
| `observability` | 10% | 有基础 logging | 增加 metrics、latency 记录、queue length、KV usage、backend 统计 |
| `audit` | 30% | 有结构化事件和内存 recorder | 定义 schema、序列化、持久化、本地 audit log 和测试报告 |
| `tests` | 35% | 有 11 个基础单元测试 | 增加 golden、集成、并发、错误、属性和 CPU/GPU 对照测试 |
| `benchmarks` | 10% | 有指标文档 | 增加可运行 benchmark 脚本、样例结果和对标格式 |
| `distributed` | 0% | 只有目录占位 | M7 前暂不展开；后续再设计多 worker/multi-GPU |
| `docs/research` | 0% | 尚未建立，当前调研在 `others/` | 将成熟调研沉淀到 `docs/research/engine_survey.md` 等文件 |

## 当前最紧急任务

### P0：完成真正的 M3 最小推理链路

目标：把当前占位式生成升级为可验证的最小引擎。

建议任务：

1. 定义 tokenizer protocol，包括 encode/decode、特殊 token 和 stop token。
2. 实现 deterministic mock model，输入 token 后返回可预测 logits。
3. 将 `engine.generate()` 改为真正的 prefill + decode loop，而不是后端直接拼字符串。
4. 增加 golden tests，固定 prompt、采样参数和输出。
5. 明确错误语义：空 prompt、非法采样参数、后端不可用、KV 分配失败。

### P0：收紧 backend contract

目标：保证未来 CUDA backend 接入时不推翻上层架构。

建议任务：

1. 明确 `PrefillState` 应包含哪些状态，例如 token ids、KV allocation、position、backend metadata。
2. 明确 `DecodeStep` 是返回 logits、token id，还是允许后端融合 sampling。
3. 增加 contract tests，要求 `mock` 和 `cpu` 后端遵循相同请求/错误/停止语义。
4. 保持 `cuda` 不走 CPU 接续路径，只保留独立 GPU 后端入口。

### P1：建立 docs/research 正式调研目录

目标：把 `others/` 的第一轮调研转成长期维护资料。

建议任务：

1. 创建 `docs/research/engine_survey.md`。
2. 创建 `docs/research/source_registry.md`。
3. 创建 `docs/research/optimization_notes.md`。
4. 从 `others/` 迁移成熟结论，保留 `others/` 作为原始搜索交付。

### P1：M4 调度与服务前置设计

目标：为 streaming、continuous batching 和取消/超时打基础。

建议任务：

1. 设计请求状态机。
2. 定义 scheduler 输入输出类型。
3. 明确 prefill task 与 decode task 的生命周期。
4. 增加队列、取消、超时的单元测试。

### P1：M5 KV Cache 设计文档

目标：先把性能核心设计清楚，再写实现。

建议任务：

1. 写 `docs/kv_cache_design.md`。
2. 定义 block size、block table、free list、引用计数和碎片统计。
3. 明确 prefix cache 的 key、命中条件、失效策略和 audit event。
4. 给出 vLLM PagedAttention / SGLang RadixAttention 的采用说明。

## 风险与注意事项

- 当前 `cpu` backend 只是占位生成，并不是实际 CPU 模型推理。
- 当前 `cuda` backend 尚不可运行，仅用于固定独立 GPU 后端边界。
- 当前 `memory` 只是 block accounting，还不是 paged KV cache。
- 当前没有 HTTP 服务、OpenAI-compatible API、streaming、并发队列或 benchmark 脚本。
- 当前没有类型检查、lint、CI 和正式 release 验收流程。
- Git 仍会提示无法访问用户全局 ignore 的 warning，但不影响当前提交、测试和推送。

