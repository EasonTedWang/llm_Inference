# 03. 生产编排与部署层

本类资料回答的是：当单个 inference engine 可用之后，如何在生产环境中做多模型、多节点、路由、扩缩容、观测和故障处理。

## Ray Serve LLM

来源：<https://docs.ray.io/en/latest/serve/llm/index.html>

定位：engine-agnostic 的生产 LLM serving framework，可挂接 vLLM、SGLang 等底层引擎。

关键内容：

- OpenAI-compatible API。
- 自动扩缩容和负载均衡。
- 多节点、多模型统一部署。
- tensor/pipeline/expert parallelism 和 data parallel attention。
- prefill/decode disaggregation。
- prefix-aware、session-aware 或自定义请求路由。
- Multi-LoRA with shared base models。
- built-in metrics、Grafana dashboard、monitoring、fault tolerance、observability。

可吸收点：

- M4 先实现单机 API 和请求生命周期；M7 前再考虑 Ray 类编排。
- M6 audit 事件可预留 `route_decision`、`replica_id`、`engine_id`、`cache_affinity` 字段。
- prefix-aware routing 是 prefix cache 进入多副本部署后的自然延伸。

## OpenVINO Model Server / GenAI Serving

来源：<https://docs.openvino.ai/2025/openvino-workflow-generative/inference-with-genai.html>

定位：OpenVINO 生态里的模型服务层，覆盖常规模型和 GenAI endpoints。

关键内容：

- GenAI endpoints 覆盖 chat completions、completions、embeddings、reranking、image/audio 等接口线索。
- 文档包含 continuous batching serving、OpenAI API、speculative decoding pipeline、long context optimization。
- Model Server 功能覆盖 gRPC、REST、KServe API、dynamic batch、model version policy、metrics、online configuration updates、model cache、security considerations。
- 支持 CPU/GPU/NPU 等部署路线。

可吸收点：

- 对本项目 M4/M6 有价值：API surface、metrics、model version、online config、security/audit。
- 如果后续需要 CPU/NPU 路线，可把 OpenVINO 作为 runtime backend 候选，而不是主服务框架。

## Triton TensorRT-LLM Backend

来源：<https://github.com/triton-inference-server/tensorrtllm_backend>

定位：将 TensorRT-LLM 接到 NVIDIA Triton Inference Server 的 backend。

本轮只做来源登记，没有展开细读。后续如果项目进入生产部署或 NVIDIA stack 对标，应补充：

- model repository 结构。
- Triton ensemble / backend 配置。
- streaming、batching、metrics 和实例组配置。
- 与 TensorRT-LLM 的 KV cache / scheduler 参数映射。

## 生产层共同能力

| 能力 | 说明 | 建议阶段 |
| --- | --- | --- |
| API gateway | 对外暴露 OpenAI-compatible、gRPC 或自定义接口 | M4 |
| Request routing | 根据模型、session、prefix cache、负载和 SLA 选择 engine/replica | M4-M7 |
| Autoscaling | 按 queue length、tokens/s、GPU utilization 等扩缩容 | M7 |
| Multi-model serving | 多模型、多 LoRA、多 endpoint 统一部署 | M5-M7 |
| Observability | metrics、tracing、dashboard、profiling、structured event | M4-M6 |
| Fault tolerance | worker crash、GPU OOM、timeout、retry、降级 | M4-M7 |
| Config/version management | 模型版本、运行参数、热更新、兼容性检查 | M2-M6 |
| Security/audit | 外部暴露端口、鉴权、请求审计、配置审计 | M6-M7 |

## 对本项目的边界建议

生产编排层不应在 M2-M3 提前复杂化。当前更合理的路线是：

1. 先在 engine 内部实现清晰的 request lifecycle。
2. 再通过服务层暴露稳定 API。
3. 然后把 metrics/audit 接入每个状态变化。
4. 最后再考虑多 worker、多副本和外部编排。

