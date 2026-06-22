# 测试方法

本文件对应 `docs/PROJECT_OUTLINE.md` 的测试规则和 M2 骨架验收。

## 本地测试

当前项目使用标准库 `unittest`，不需要安装第三方测试依赖。

首选使用 `uv` 创建本地虚拟环境：

```powershell
uv venv .venv --python 3.12
.\.venv\Scripts\Activate.ps1
uv pip install -e .
$env:PYTHONPATH = "src"
python -m unittest discover -s tests
```

如果本机 PATH 中没有 `uv` 或 `python`，可使用 Codex 工作区提供的 Python 运行器执行同等测试命令；此时仍应设置 `PYTHONPATH=src`。

## 当前覆盖范围

- 配置校验。
- Backend contract 的 `mock`、`cpu`、`cuda` 边界。
- 请求生成最小闭环。
- 结构化 audit event。
- FIFO scheduler。
- KV cache block allocation/free。

## 后续扩展

- M3：增加 mock model golden tests 和端到端 smoke test。
- M4：增加 streaming、取消、超时和 continuous batching 测试。
- M5：增加 KV cache 不变量、GPU benchmark 和 CPU/GPU 对照测试。
- M6：增加 audit schema、benchmark 报告和回归测试。
