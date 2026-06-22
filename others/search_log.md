# 搜索记录

检索日期：2026-06-22  
检索目标：搜索其他 LLM inference 系统包含的内容，并按适合本项目的大纲分类沉淀。

## 筛选规则

1. 技术结论只采用官方文档、官方仓库、官方博客和论文。
2. 二手来源只用于发现项目名和关键词，不作为结论来源。
3. 结果按项目可用性分类，而不是按搜索引擎排序。

## 查询与保留结果

| 查询/动作 | 保留结果 | 分类 | 是否用于结论 |
| --- | --- | --- | --- |
| `vLLM official docs PagedAttention continuous batching prefix caching speculative decoding` | https://docs.vllm.ai/en/latest/ | 服务引擎 | 是 |
| 直接打开 | https://github.com/vllm-project/vllm | 服务引擎 | 是 |
| 搜索结果 | https://arxiv.org/abs/2309.06180 | 论文 | 是 |
| `SGLang official docs RadixAttention speculative decoding structured outputs` | https://docs.sglang.io/ | 服务引擎 | 是 |
| 直接打开 | https://github.com/sgl-project/sglang | 服务引擎 | 是 |
| `SGLang Efficient Execution of Structured Language Model Programs arxiv` | https://arxiv.org/abs/2312.07104 | 论文 | 是 |
| `NVIDIA TensorRT-LLM official docs in-flight batching paged KV cache quantization` | https://nvidia.github.io/TensorRT-LLM/ | 服务/运行时 | 是 |
| 直接打开 | https://github.com/NVIDIA/TensorRT-LLM | 服务/运行时 | 是 |
| `Hugging Face Text Generation Inference official docs continuous batching tensor parallelism quantization` | https://huggingface.co/docs/text-generation-inference/index | 服务引擎 | 是 |
| 直接打开 | https://github.com/huggingface/text-generation-inference | 服务引擎 | 是 |
| 直接打开 | https://github.com/ggml-org/llama.cpp | 本地运行时 | 是 |
| 直接打开 | https://llm.mlc.ai/docs/ | 编译/运行时 | 是 |
| 直接打开 | https://lmdeploy.readthedocs.io/en/latest/ | 服务引擎 | 是 |
| 直接打开 | https://lightllm-en.readthedocs.io/en/latest/ | 服务引擎 | 是 |
| 直接打开 | https://github.com/ModelTC/lightllm | 服务引擎 | 是 |
| 直接打开 | https://github.com/deepspeedai/DeepSpeed-MII | 服务引擎 | 是 |
| 直接打开 | https://www.deepspeed.ai/2023/11/05/deepspeed-fastgen.html | 官方博客 | 是，作为补充 |
| 直接打开 | https://docs.ray.io/en/latest/serve/llm/index.html | 生产编排 | 是 |
| 直接打开 | https://docs.openvino.ai/2025/openvino-workflow-generative/inference-with-genai.html | 运行时/模型服务 | 是 |
| 直接打开 | https://github.com/openvinotoolkit/openvino.genai | 运行时 | 是 |
| 直接打开 | https://onnxruntime.ai/docs/genai/ | 运行时 | 是 |
| 直接打开 | https://github.com/microsoft/onnxruntime-genai | 运行时 | 是 |
| 直接打开 | https://flashinfer.ai/ | 内核库 | 是 |
| 直接打开 | https://github.com/flashinfer-ai/flashinfer | 内核库 | 是 |
| 直接打开 | https://github.com/NVIDIA/FasterTransformer | 传统内核 | 是，作为历史参考 |
| `Sarathi-Serve Efficient LLM Inference by Piggybacking Decodes with Chunked Prefills arxiv` | https://arxiv.org/abs/2308.16369 | 论文 | 是 |
| `Fast Inference from Transformers via Speculative Decoding arxiv` | https://arxiv.org/abs/2211.17192 | 论文 | 是 |
| `SpecInfer Accelerating Generative Large Language Model Serving tree based speculative inference arxiv` | https://arxiv.org/abs/2305.09781 | 论文 | 是 |
| `TurboTransformers efficient GPU serving system transformer models arxiv` | https://arxiv.org/abs/2010.05680 | 论文 | 是，作为历史参考 |
| 直接打开 | https://github.com/triton-inference-server/tensorrtllm_backend | 生产编排 | 登记，未深入 |

## 搜索中出现但未采用的结果

| 结果 | 未采用原因 |
| --- | --- |
| vLLM、SGLang、TensorRT、PagedAttention、Speculative decoding 等 Wikipedia 页面 | 二手来源，不作为技术结论。 |
| TechCrunch、Tom's Hardware 等新闻或媒体文章 | 二手来源，且不直接回答架构/实现内容。 |
| KServe Hugging Face text generation 页面 | 本轮打开时没有返回可用正文，后续如调研 Kubernetes serving 可重新检索。 |
| 非官方性能对比博客 | 可能有参考价值，但需要官方 benchmark 或可复现实验支撑后再使用。 |

## 下一轮建议查询

1. vLLM design docs: scheduler、hybrid KV cache manager、CUDA graph、V1 architecture。
2. SGLang RadixAttention implementation and prefix cache design。
3. TensorRT-LLM KV cache system、overlap scheduler、feature combination matrix。
4. TGI router/server source architecture。
5. Triton TensorRT-LLM backend streaming/batching/model repository examples。
6. KServe / Kubernetes LLM serving stack。

