# Benchmarks

本目录对应 `docs/PROJECT_OUTLINE.md` 中 M5 之后的性能验证入口。M2 阶段先建立目录和指标约定，避免后续优化缺少可复现验收方式。

## 初始指标

- Tokens/s。
- Time to First Token。
- Inter-token latency。
- 总请求延迟。
- Prefill latency。
- Decode latency。
- Queue latency。
- Batch composition。
- GPU utilization。
- 显存峰值。
- KV cache block 使用量、命中率和碎片情况。

## 后续约束

- CPU backend 的性能不得代表 CUDA backend。
- CUDA backend benchmark 必须记录硬件、驱动、模型、量化方式、请求分布和并发设置。
- 涉及外部设计原语的优化，例如 PagedAttention、chunked prefill、prefix cache 或 speculative decoding，必须记录来源、假设、适用场景和对照结果。

