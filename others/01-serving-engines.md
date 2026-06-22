# 01. 高吞吐 LLM 服务引擎

本类系统的共同目标是把模型变成可并发、可流式、可观测、可横向扩展的服务。它们通常覆盖 API、调度、KV cache、runtime、采样、并行、量化、LoRA、benchmark 和部署。

## 横向对比

| 系统 | 定位 | 关键内容 | 本项目可吸收点 |
| --- | --- | --- | --- |
| vLLM | 高吞吐、内存高效的 LLM inference and serving engine | PagedAttention、continuous batching、chunked prefill、prefix cache、speculative decoding、structured outputs、OpenAI-compatible API、multi-LoRA、parallelism、metrics/benchmark、多硬件后端 | KV cache 分块模型、request scheduler、OpenAI API 边界、benchmark CLI、metrics 体系 |
| SGLang | 面向 LLM/VLM 的高性能 serving framework 和结构化生成系统 | RadixAttention、zero-overhead CPU scheduler、prefill/decode disaggregation、continuous batching、paged attention、chunked prefill、structured outputs、parallelism、FP4/FP8/INT4/AWQ/GPTQ、multi-LoRA | prefix/radix cache、结构化输出、前端语言与 runtime 分离、PD disaggregation 设计 |
| TensorRT-LLM | NVIDIA GPU 上的 LLM 优化、构建和 serving 运行时 | In-flight batching、paged attention、KV cache system、overlap scheduler、parallelism、quantization、LoRA、multimodal、speculative/guided decoding、trtllm-bench、trtllm-serve | 硬件后端抽象、性能配置矩阵、KV cache 配置、CUDA graph/compile/benchmark 思路 |
| Hugging Face TGI | 生产化文本生成服务 | launcher、OpenTelemetry、Prometheus、tensor parallelism、SSE streaming、continuous batching、Messages API、Flash/Paged Attention、quantization、speculation、JSON guidance、logprobs、stop sequences | 服务入口、SSE streaming、生产观测、请求参数覆盖、容器化启动方式 |
| LMDeploy | 压缩、部署和服务 LLM 的工具链 | TurboMind、persistent batch、blocked KV cache、dynamic split&fuse、tensor parallelism、CUDA kernels、AWQ/GPTQ、INT4/INT8 KV cache、OpenAI/Anthropic endpoints、LoRA、request distributor | blocked KV、dynamic split&fuse、请求分发、多模型服务、KV 量化 |
| LightLLM | Python-based 轻量 LLM inference and serving framework | 多进程协作、共享内存请求对象、预测式峰值内存调度、dynamic KV cache、tensor/data/expert parallelism、int8/fp8/int4、structured output、multi-result prediction | 多进程职责拆分、请求对象共享、峰值内存调度、动态 KV |
| DeepSpeed-MII / FastGen | DeepSpeed 驱动的高吞吐、低延迟 inference library | blocked KV caching、continuous batching、Dynamic SplitFuse、高性能 CUDA kernels、tensor parallelism、persistent/non-persistent deployment、gRPC server、replica/load balancing | SplitFuse、persistent deployment、简单 pipeline API、replica/load balancing |

## 共同模块抽取

### API 与请求入口

这些系统普遍提供至少两种入口：离线调用接口和在线服务接口。在线服务通常向 OpenAI-compatible API 靠拢，并支持 streaming、chat/completions、tools/function calling、reasoning parser、logprobs、stop sequence 和自定义 logits processor。

对本项目的影响：M3 可以先定义内部 `GenerationRequest` / `GenerationResult`，M4 再映射到 HTTP/OpenAI-compatible API。

### 调度与 batch

continuous batching / in-flight batching 是服务框架的核心共性。vLLM、TGI、SGLang、TensorRT-LLM、LMDeploy、DeepSpeed-MII 都把请求按 token 级或 iteration 级调度，而不是等一个静态 batch 全部完成。SGLang、TensorRT-LLM、Ray Serve LLM 又进一步强调 prefill/decode disaggregation；SARATHI、LMDeploy、DeepSpeed-MII 则强调 chunked prefill 或 split&fuse。

对本项目的影响：M4 的 scheduler 不应只设计成简单 FIFO 队列，需要预留 prefill/decode 两类任务、请求状态机、动态 batch builder、取消和超时。

### KV Cache 与内存

vLLM 的 PagedAttention、LMDeploy/DeepSpeed 的 blocked KV、LightLLM 的 dynamic KV、OpenVINO 的 KV eviction/prefix cache 都指向同一个问题：高并发和长上下文时 KV cache 是主要内存瓶颈。成熟系统会同时处理分块、复用、回收、offload、量化和 prefix sharing。

对本项目的影响：M5 应先实现可测试的 block allocator 和 block table，再逐步加入 prefix cache、offload、KV quant、eviction。

### 后端与 kernel

服务引擎往往不把 kernel 写死，而是接入多后端：FlashAttention、FlashInfer、TensorRT-LLM、Triton、CUTLASS、torch.compile、CUDA graph、OpenVINO、ONNX Runtime 等。TensorRT-LLM 和 FlashInfer 的文档尤其强调 attention/GEMM/MoE kernel、CUDA graph 和低精度计算。

对本项目的影响：M2 骨架中 `runtime` 与 `kernels` 应分离；M3 可以 mock，M5 再实现具体后端。

### 观测与 benchmark

TGI、vLLM、TensorRT-LLM、Ray Serve LLM 都把 benchmark、metrics、profiling、Prometheus/Grafana 或 OpenTelemetry 放在生产能力里。仅靠日志无法验收性能变化。

对本项目的影响：M6 的 audit event 应覆盖请求状态、queue latency、prefill/decode latency、tokens/s、KV cache 使用量、错误码和配置快照。

## 后续二次调研问题

1. vLLM V1 scheduler、hybrid KV cache manager 和 CUDA graph 细节需要单独读设计文档。
2. SGLang 的 RadixAttention 与本项目 prefix cache 结构如何对应，需要做算法级笔记。
3. TensorRT-LLM 的 KV cache config、overlap scheduler、feature combination matrix 适合作为 M5 benchmark 对照。
4. TGI 的 Rust router/server 与 Python model backend 边界值得复盘。
5. LMDeploy、DeepSpeed-MII 的 SplitFuse 与 SARATHI chunked prefill 是否本质相同，需要进一步拆分。

