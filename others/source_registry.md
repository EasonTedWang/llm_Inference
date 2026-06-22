# 来源登记

本文件记录本轮检索中保留的主要来源。可信度等级：

- A：官方文档、官方仓库、论文。
- B：官方博客或 release 文章。
- C：二手资料，仅用于发现线索，不进入技术结论。

| ID | 分类 | 来源 | URL | 等级 | 本轮用途 |
| --- | --- | --- | --- | --- | --- |
| S001 | 服务引擎 | vLLM 官方文档 | https://docs.vllm.ai/en/latest/ | A | 确认 vLLM 功能目录、在线服务、观测、量化、speculative decoding、prefix cache 等能力。 |
| S002 | 服务引擎 | vLLM 官方仓库 | https://github.com/vllm-project/vllm | A | 确认 vLLM 定位、支持的 attention/kernel、parallelism、OpenAI API、LoRA、多硬件支持。 |
| S003 | 论文 | PagedAttention / vLLM 论文 | https://arxiv.org/abs/2309.06180 | A | 确认 PagedAttention、KV cache 分块、cache sharing、吞吐提升证据。 |
| S004 | 服务引擎 | SGLang 官方文档 | https://docs.sglang.io/ | A | 确认 SGLang 文档入口与服务框架定位。 |
| S005 | 服务引擎 | SGLang 官方仓库 | https://github.com/sgl-project/sglang | A | 确认 RadixAttention、CPU scheduler、PD disaggregation、paged attention、parallelism、structured output、multi-LoRA。 |
| S006 | 论文 | SGLang 论文 | https://arxiv.org/abs/2312.07104 | A | 确认 frontend language + runtime、RadixAttention、compressed FSM、复杂程序执行场景。 |
| S007 | 服务/运行时 | NVIDIA TensorRT-LLM 文档 | https://nvidia.github.io/TensorRT-LLM/ | A | 确认 IFB、paged attention、KV cache system、overlap scheduler、parallelism、quantization、LoRA、benchmark。 |
| S008 | 服务/运行时 | TensorRT-LLM 官方仓库 | https://github.com/NVIDIA/TensorRT-LLM | A | 确认 TensorRT-LLM 是 NVIDIA GPU 上的 LLM 优化与运行时组件。 |
| S009 | 服务引擎 | Hugging Face TGI 文档 | https://huggingface.co/docs/text-generation-inference/index | A | 确认 TGI 文档入口和 serving 定位。 |
| S010 | 服务引擎 | Hugging Face TGI 仓库 | https://github.com/huggingface/text-generation-inference | A | 确认生产特性：OpenTelemetry、Prometheus、TP、SSE streaming、continuous batching、Flash/Paged Attention、quantization、speculation、JSON guidance。 |
| S011 | 本地运行时 | llama.cpp 官方仓库 | https://github.com/ggml-org/llama.cpp | A | 确认 C/C++ 本地推理、GGUF、OpenAI-compatible server、多硬件后端、低比特量化、CPU+GPU hybrid。 |
| S012 | 编译/运行时 | MLC LLM 文档 | https://llm.mlc.ai/docs/ | A | 确认机器学习编译器 + 高性能部署引擎定位、跨平台部署方向。 |
| S013 | 服务引擎 | LMDeploy 文档 | https://lmdeploy.readthedocs.io/en/latest/ | A | 确认 TurboMind、persistent batch、blocked KV cache、dynamic split&fuse、TP、CUDA kernel、KV quant、OpenAI server、LoRA。 |
| S014 | 服务引擎 | LightLLM 英文文档 | https://lightllm-en.readthedocs.io/en/latest/ | A | 确认多进程协作、共享内存请求对象、预测式调度、dynamic KV、quantization、structured output、parallelism。 |
| S015 | 服务引擎 | LightLLM 官方仓库 | https://github.com/ModelTC/lightllm | A | 确认 LightLLM 项目定位、论文/调度相关入口、许可证信息。 |
| S016 | 服务引擎 | DeepSpeed-MII 官方仓库 | https://github.com/deepspeedai/DeepSpeed-MII | A | 确认 blocked KV caching、continuous batching、Dynamic SplitFuse、CUDA kernels、tensor parallelism、persistent gRPC deployment。 |
| S017 | 官方博客 | DeepSpeed FastGen 博客 | https://www.deepspeed.ai/2023/11/05/deepspeed-fastgen.html | B | 作为 MII/FastGen 性能和技术说明补充来源。 |
| S018 | 生产编排 | Ray Serve LLM 文档 | https://docs.ray.io/en/latest/serve/llm/index.html | A | 确认 engine-agnostic LLM serving、OpenAI API、autoscaling、PD disaggregation、prefix-aware routing、Multi-LoRA、metrics/Grafana。 |
| S019 | 运行时/模型服务 | OpenVINO GenAI 文档 | https://docs.openvino.ai/2025/openvino-workflow-generative/inference-with-genai.html | A | 确认 OpenVINO GenAI、OVMS、continuous batching、OpenAI API、speculative decoding pipeline、NPU/GPU/CPU 支持线索。 |
| S020 | 运行时 | OpenVINO GenAI 仓库 | https://github.com/openvinotoolkit/openvino.genai | A | 确认 speculative decoding、KV eviction、sparse attention、continuous batching、prefix caching。 |
| S021 | 运行时 | ONNX Runtime GenAI 文档 | https://onnxruntime.ai/docs/genai/ | A | 确认 generate API、tokenization、logits processing、search/sampling、KV cache management、structured output。 |
| S022 | 运行时 | ONNX Runtime GenAI 仓库 | https://github.com/microsoft/onnxruntime-genai | A | 确认多语言 API、多硬件后端、Multi-LoRA、continuous/constrained/speculative decoding。 |
| S023 | 内核库 | FlashInfer 官网 | https://flashinfer.ai/ | A | 确认 FlashInfer 是 LLM deployment 加速技术入口。 |
| S024 | 内核库 | FlashInfer 仓库 | https://github.com/flashinfer-ai/flashinfer | A | 确认 attention/GEMM/MoE kernel、paged/ragged KV、prefill/decode/append、MLA、cascade/sparse/POD attention、FP8/FP4、sampling kernel、通信。 |
| S025 | 传统内核 | NVIDIA FasterTransformer 仓库 | https://github.com/NVIDIA/FasterTransformer | A | 确认 FasterTransformer 已转向 TensorRT-LLM，以及旧式 kernel、layer/model、benchmark、tests 结构。 |
| S026 | 论文 | SARATHI 论文 | https://arxiv.org/abs/2308.16369 | A | 确认 chunked prefill、decode-maximal batching、prefill/decode 资源利用问题。 |
| S027 | 论文 | Speculative Decoding 论文 | https://arxiv.org/abs/2211.17192 | A | 确认 draft model + target verification、分布不变、2x-3x 加速线索。 |
| S028 | 论文 | SpecInfer 论文 | https://arxiv.org/abs/2305.09781 | A | 确认 tree-based speculative inference/verification。 |
| S029 | 论文 | TurboTransformers 论文 | https://arxiv.org/abs/2010.05680 | A | 确认早期 Transformer serving 的 kernel、memory allocation、batch scheduler 设计。 |

## 未作为技术依据的搜索结果

搜索中出现了 Wikipedia、新闻报道和硬件媒体文章等二手来源。它们只帮助发现项目名或关键词，不进入上面的横向结论。KServe 的一个页面在本轮打开时未返回正文，因此没有纳入技术整理，后续可重新补搜 Kubernetes/serving 层资料。

