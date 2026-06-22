# LLM Inference 功能分类表

本表从本轮外部调研中抽取功能分类，并映射到本项目阶段。它不是最终架构设计，但可以作为 M2 目录和 M3-M6 测试计划的输入。

| 分类 | 常见能力 | 代表系统 | 本项目阶段 |
| --- | --- | --- | --- |
| API 与服务 | offline generate、HTTP server、OpenAI-compatible chat/completions/messages/responses、streaming、SSE、gRPC、工具调用、reasoning parser、Anthropic-compatible endpoints | vLLM、SGLang、TGI、TensorRT-LLM、LMDeploy、Ray Serve LLM、llama.cpp | M3-M4 |
| 请求生命周期 | request id、参数解析、入队、prefill、decode、stream chunk、finish reason、错误码、取消、超时、清理 | vLLM、TGI、MII、ONNX Runtime GenAI | M3-M4 |
| 调度 | continuous batching、in-flight batching、iteration-level scheduling、chunked prefill、prefill/decode disaggregation、overlap scheduler、Dynamic SplitFuse、decode-maximal batching、capacity scheduling | vLLM、SGLang、TensorRT-LLM、TGI、LMDeploy、DeepSpeed-MII、SARATHI、Ray Serve LLM | M4-M5 |
| KV Cache 与内存 | paged KV、blocked KV、dynamic KV、block table、prefix/radix cache、cache sharing、copy-on-write、KV offload、KV eviction、KV quantization、long-context memory policy | vLLM、SGLang、TensorRT-LLM、LMDeploy、LightLLM、OpenVINO GenAI | M5 |
| Runtime | model loading、tokenizer、chat template、Hugging Face integration、ONNX/OpenVINO/TensorRT/TVM backend、stateful generation loop、weight format、checkpoint loader | vLLM、SGLang、TensorRT-LLM、llama.cpp、MLC LLM、OpenVINO GenAI、ONNX Runtime GenAI | M2-M3 |
| Kernel | attention kernels、FlashAttention、FlashInfer、paged/ragged attention、GEMM、MoE kernels、RoPE、RMSNorm/LayerNorm、CUDA graph、torch.compile、Triton/CUTLASS/cuDNN | FlashInfer、TensorRT-LLM、vLLM、FasterTransformer | M5 |
| Decoding 与采样 | greedy、beam search、temperature、top-k、top-p、min-p、repetition penalty、logprobs、stop sequence、custom logits processor、speculative decoding、structured/guided decoding | ONNX Runtime GenAI、TGI、TensorRT-LLM、vLLM、SGLang、FlashInfer | M3-M5 |
| 量化 | weight-only quant、AWQ、GPTQ、bitsandbytes、EETQ、FP8、FP4、INT4/INT8 KV cache、GGUF low-bit formats | TGI、LMDeploy、TensorRT-LLM、SGLang、llama.cpp、OpenVINO GenAI | M5 |
| LoRA 与模型扩展 | LoRA、multi-LoRA batching、shared base model、adapter loading、model support matrix、新模型注册 | vLLM、SGLang、TensorRT-LLM、LMDeploy、Ray Serve LLM、ONNX Runtime GenAI | M5-M7 |
| 多模态与非生成任务 | VLM、image/audio/video input、embedding、rerank、reward、classification、speech-to-text、text-to-speech | vLLM、SGLang、TensorRT-LLM、OpenVINO Model Server、ONNX Runtime GenAI | M7 |
| 并行与分布式 | tensor parallelism、pipeline parallelism、data parallelism、expert parallelism、context parallelism、data parallel attention、multi-node、replica、load balancing | vLLM、SGLang、TensorRT-LLM、TGI、LMDeploy、Ray Serve LLM、DeepSpeed-MII | M5-M7 |
| 生产部署 | Docker、Kubernetes、Helm、autoscaling、model repository、request distributor、multi-model serving、hardware support matrix、security considerations | TGI、vLLM、Ray Serve LLM、OpenVINO Model Server、TensorRT-LLM、LMDeploy | M4-M7 |
| 观测与审核 | Prometheus、OpenTelemetry、Grafana、profiling、benchmark CLI、performance dashboard、structured events、request audit、config audit、KV events | TGI、vLLM、TensorRT-LLM、Ray Serve LLM、OpenVINO Model Server、FlashInfer | M4-M6 |
| Benchmark 与验收 | latency、TTFT、ITL、tokens/s、throughput、concurrency、memory peak、KV cache hit、hardware utilization、model-feature matrix | vLLM、TensorRT-LLM、LMDeploy、LightLLM、FasterTransformer、SARATHI | M5-M7 |

## 推荐优先级

### M2 必须预留

- `api`：内部请求/响应模型，不急于完整 OpenAI 兼容。
- `engine`：请求生命周期和 runtime 调用协调。
- `scheduler`：即使早期只 FIFO，也要能表达 prefill/decode 状态。
- `memory`：先定义 KV cache 抽象，早期可 mock。
- `runtime`：模型加载、tokenizer、forward step 抽象。
- `decoding`：sampler、logits processor、stop condition。
- `observability` / `audit`：结构化事件 schema。

### M3 最小闭环

- 单请求 generate。
- mock model 或小模型 runtime。
- greedy/temperature 采样。
- stop condition 和 finish reason。
- 基础 request event。

### M4-M5 性能核心

- continuous batching。
- streaming。
- cancellation/timeout。
- paged/blocked KV cache。
- prefix cache。
- chunked prefill。
- benchmark harness。

### M6-M7 生产化

- 完整 audit log。
- OpenAI-compatible API。
- Prometheus/Grafana 或等价 metrics。
- 多 worker、多 GPU、多模型、多 LoRA。
- 性能回归和发布验收。

