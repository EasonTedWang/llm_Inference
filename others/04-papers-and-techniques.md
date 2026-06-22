# 04. 论文与关键优化技术

本文件把论文和官方技术说明抽象成“要解决的问题、核心方法、可采用方式”。进入实现前仍需补充更细的算法笔记和测试标准。

## PagedAttention / vLLM

来源：<https://arxiv.org/abs/2309.06180>

解决的问题：LLM serving 中 KV cache 随序列长度动态增长，传统连续预留内存会产生碎片和冗余复制，限制 batch size。

核心方法：

- 将每个 sequence 的 KV cache 切成固定大小 block。
- 用 block table 把 logical block 映射到 physical GPU memory block。
- 按需分配，减少 internal/external fragmentation。
- 通过引用计数和 copy-on-write 支持跨请求或多采样分支共享 KV cache。

本项目采用候选：

- M5 先实现 block allocator、block table、free list 和 request-to-block 映射。
- Prefix cache、parallel sampling、beam search 的共享逻辑后置。
- 验收指标：峰值 KV 使用量、碎片率、cache hit、吞吐和 TTFT/ITL。

## SGLang / RadixAttention

来源：<https://arxiv.org/abs/2312.07104>

解决的问题：复杂 LLM 应用包含多轮生成、分支、并行和结构化输出，重复 prefix 计算会浪费大量 KV cache 和算力。

核心方法：

- SGLang 将系统拆成 Python-embedded frontend language 和 backend runtime。
- Runtime 用 RadixAttention 复用 KV cache。
- 用 compressed finite state machine 加速 structured output decoding。
- 面向 agent、RAG、多轮对话、JSON decoding 等复杂程序。

本项目采用候选：

- M4/M5 引入 prefix cache 时，可先实现 trie/radix-like key 结构和 cache reuse 事件。
- M6 audit 应记录 prefix cache 命中、复用来源和失效原因。
- Structured output 不放入 M3，等基础 sampler 稳定后再实现。

## SARATHI / Chunked Prefill

来源：<https://arxiv.org/abs/2308.16369>

解决的问题：prefill 阶段容易吃满 GPU compute，decode 阶段每次只生成一个 token，GPU 利用率低；两者混在 pipeline parallelism 里会产生不均衡和 bubble。

核心方法：

- 将长 prompt prefill 切成等长 chunk。
- 构造 decode-maximal batch：一个 prefill chunk 搭配尽可能多的 decode 请求。
- 让 decode piggyback 在 prefill 的 compute 饱和阶段上。

本项目采用候选：

- M4 scheduler 状态机应区分 prefill task 和 decode task。
- M5 再实现 chunked prefill 与 decode-maximal batch builder。
- 验收指标：TTFT、ITL、decode throughput、prefill queue delay、batch composition。

## Speculative Decoding

来源：<https://arxiv.org/abs/2211.17192>

解决的问题：自回归模型逐 token decode 延迟高，每个 token 都要跑一次大模型。

核心方法：

- 用小 draft model 先生成多个候选 token。
- 大 target model 一次 forward 验证多个候选。
- 通过采样校正保持与原始 target model 相同的输出分布。

本项目采用候选：

- M3 只保留 sampler 接口，不实现 speculative decoding。
- M5 之后可设计 `DraftModelRuntime`、`VerifierRuntime`、accepted-token metrics。
- 验收必须比较输出分布一致性、acceptance rate、latency、额外显存。

## SpecInfer

来源：<https://arxiv.org/abs/2305.09781>

解决的问题：单链 speculative decoding 对 draft 质量敏感，候选路径有限。

核心方法：

- 多个小模型或 speculative model 生成 token tree。
- 大模型并行验证整棵树，保留有效路径。
- 目标是降低分布式或 offloading 场景的端到端延迟。

本项目采用候选：

- 暂不进入早期路线。
- 作为 speculative decoding 的后续扩展，需要 tree attention / verification 的测试框架。

## Dynamic SplitFuse / DeepSpeed FastGen

来源：

- <https://github.com/deepspeedai/DeepSpeed-MII>
- <https://www.deepspeed.ai/2023/11/05/deepspeed-fastgen.html>

解决的问题：prefill/decode 工作负载混合导致吞吐和延迟难以同时优化。

核心方法：

- blocked KV caching。
- continuous batching。
- Dynamic SplitFuse。
- 高性能 CUDA kernels。

本项目采用候选：

- 与 SARATHI 一起二次调研，确认 split/fuse 与 chunked prefill 的差异。
- M4 先保留 batch composition 抽象，M5 再实现具体策略。

## TurboTransformers

来源：<https://arxiv.org/abs/2010.05680>

解决的问题：早期 Transformer 在线服务中 variable-length input、kernel hotspot 和 batch scheduling 的效率问题。

核心方法：

- 优化 Softmax、LayerNorm 等 GPU kernel。
- 针对变长输入设计 memory allocation。
- 用 batch scheduler 优化吞吐。

本项目采用候选：

- 作为历史参考，不作为主要对标对象。
- 对测试有启发：变长请求、batch scheduler、memory allocation 都应有边界用例。

