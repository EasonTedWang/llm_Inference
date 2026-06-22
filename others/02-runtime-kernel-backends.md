# 02. 运行时、编译器与内核后端

本类系统更偏向“如何在不同硬件和执行后端上高效跑模型”。它们不一定都是完整服务框架，但会提供本项目后续 `runtime`、`kernels`、`memory` 模块的重要参考。

## llama.cpp

来源：<https://github.com/ggml-org/llama.cpp>

定位：C/C++ 本地 LLM inference，目标是 minimal setup 和跨硬件高性能。

关键内容：

- GGUF 模型格式与本地加载流程。
- `llama-cli` 离线推理和 `llama-server` OpenAI-compatible API server。
- Apple Silicon、x86 AVX/AVX2/AVX512/AMX、RISC-V、CUDA、HIP、Vulkan、SYCL 等硬件后端。
- 1.5-bit 到 8-bit 多种整数量化，强调低内存和本地可运行。
- CPU+GPU hybrid inference，用于模型大于 VRAM 的场景。

可吸收点：

- M3 可以借鉴其“最小 CLI + server”双入口思路。
- M5 可参考 GGUF/量化/CPU fallback 方向，但本项目早期不应被格式适配拖慢。

## MLC LLM

来源：<https://llm.mlc.ai/docs/>

定位：machine learning compiler + high-performance deployment engine。

关键内容：

- 用编译器路线把模型优化并部署到多平台。
- 文档包含模型编译、部署、Python package、TVM compiler、Wasm 构建和 microserving API。
- 目标是 native deployment 到不同平台，而不仅是 NVIDIA GPU server。

可吸收点：

- `runtime` 应避免强绑定某个后端，保留模型编译/图优化接入点。
- 后续如果走跨平台路线，需要单独调研 TVM/MLC 的模型构建流程。

## OpenVINO GenAI

来源：

- <https://docs.openvino.ai/2025/openvino-workflow-generative/inference-with-genai.html>
- <https://github.com/openvinotoolkit/openvino.genai>

定位：基于 OpenVINO Runtime 的 generative AI C++/Python API 与模型服务能力。

关键内容：

- OpenVINO GenAI 支持 speculative decoding、KV cache token eviction、sparse attention。
- continuous batching library 可用于 LLM serving，并支持 prefix caching。
- OpenVINO Model Server 提供 GenAI endpoints、OpenAI API、continuous batching、speculative decoding pipeline、NPU/GPU/CPU 部署线索。
- 文档覆盖模型准备、运行设备、性能调优、模型服务、metrics 和多 API 入口。

可吸收点：

- M5 可以把 KV eviction、sparse attention、prefix caching 作为中后期优化候选。
- M6 可以参考模型服务的 metrics、配置和性能调优文档结构。

## ONNX Runtime GenAI

来源：

- <https://onnxruntime.ai/docs/genai/>
- <https://github.com/microsoft/onnxruntime-genai>

定位：ONNX Runtime 的 generative AI extension，提供高层 `generate()` API 和低层逐 token loop。

关键内容：

- 明确包含 tokenization / preprocessing、ONNX Runtime inference、logits processing、search/sampling、KV cache management。
- 支持 greedy、beam search、TopP、TopK、repetition penalties、自定义 scoring。
- 支持 chat template、structured output/tool calling。
- 仓库支持 Python、C#、C/C++、Java 等 API，以及 CPU、CUDA、DirectML、TRT-RTX、OpenVINO、QNN、WebGPU 等硬件后端。
- 支持 Multi-LoRA、continuous decoding、constrained decoding、speculative decoding。

可吸收点：

- M3 的最小生成 loop 可以按 ONNX Runtime GenAI 的职责拆分：tokenizer、model forward、logits processor、sampler、KV state。
- M2 的接口设计要允许“高层 generate”和“低层 step loop”两种调用方式。

## FlashInfer

来源：

- <https://flashinfer.ai/>
- <https://github.com/flashinfer-ai/flashinfer>

定位：面向 LLM serving 的高性能 GPU kernel library / kernel generator。

关键内容：

- 提供 attention、GEMM、MoE 统一 API，后端包括 FlashAttention、cuDNN、CUTLASS、TensorRT-LLM 等。
- attention 覆盖 paged/ragged KV cache、decode、prefill、append、MLA、cascade attention、sparse attention、POD-attention。
- GEMM/MoE 覆盖 BF16、FP8、FP4、grouped GEMM、fused MoE、quantized MoE。
- sampling/decoding 包含 sorting-free Top-K/Top-P/Min-P 和 speculative decoding 支持。
- 支持 CUDAGraph、torch.compile、API logging、multi-node NVLink/NVSHMEM 等。

可吸收点：

- 本项目不应早期手写所有 CUDA kernel，但 `kernels` 模块应能表达 attention/GEMM/MoE/sampling 的能力边界。
- FlashInfer 的 API logging 思路可给 M6 audit/observability 提供参考。

## FasterTransformer

来源：<https://github.com/NVIDIA/FasterTransformer>

定位：NVIDIA 旧式 Transformer inference kernel 和 framework integration 项目，官方说明开发已转向 TensorRT-LLM。

关键内容：

- 源码结构包含 `kernels`、`layers`、`models`、`tensorrt_plugin`、`tf_op`、`th_op`、`triton_backend`、`utils`。
- 支持 TensorFlow、PyTorch、Triton backend、C++ 示例和 benchmark。
- 提供模型支持矩阵、性能结果、debug/profiling 环境变量。

可吸收点：

- 作为“历史结构参考”有价值，尤其是 kernel/layer/model/backend/test/benchmark 的目录拆法。
- 不建议作为新实现目标，TensorRT-LLM 才是 NVIDIA 当前主线。

