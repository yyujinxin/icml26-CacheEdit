# 使用教程

本教程演示如何用 CacheEdit 进行图像编辑、利用多轮缓存加速，并自定义配置。

## 目录

1. [安装与环境](#1-安装与环境)
2. [命令行：单图编辑](#2-命令行单图编辑)
3. [命令行：测速对比](#3-命令行测速对比)
4. [Python API：Qwen](#4-python-apiqwen)
5. [Python API：Flux Kontext](#5-python-apiflux-kontext)
6. [理解多轮缓存](#6-理解多轮缓存)
7. [自定义配置](#7-自定义配置)
8. [环境变量覆盖](#8-环境变量覆盖)

---

## 1. 安装与环境

```bash
pip install -e .
```

确认 CLI 可用：

```bash
cache-edit --version      # cache-edit 0.1.0
cache-edit --help
```

---

## 2. 命令行：单图编辑

```bash
cache-edit edit \
  --model flux \
  --image input.png \
  --prompt "turn the sky into a starry night" \
  --output out.png \
  --seed 42
```

常用参数：

| 参数 | 说明 |
| --- | --- |
| `--model {qwen,flux}` | 必填，选择后端 |
| `--image` | 输入图像路径 |
| `--prompt` | 编辑指令 |
| `--negative-prompt` | 负面提示（仅 Flux） |
| `--config` | 自定义配置；默认 `configs/{model}_default.yaml` |
| `--output` | 输出路径（默认 `output.png`） |
| `--seed` | 随机种子 |
| `--no-cache` | 关闭缓存，跑原始 pipeline |
| `--no-env` | 不应用 `CACHEEDIT_*` 环境变量 |

---

## 3. 命令行：测速对比

`benchmark` 会先跑一次无缓存基线，再跑多轮缓存，输出平均延迟与加速比：

```bash
cache-edit benchmark \
  --model qwen \
  --image input.png \
  --prompt "make it autumn" \
  --rounds 3 \
  --warmup 1 \
  --output-dir ./bench_outputs
```

输出示例：

```
================================================
  no-cache (baseline)  : 12.480s
  cache avg (3 runs) : 6.215s
  speedup              : 2.01x
================================================
```

每轮结果保存到 `--output-dir`（`no_cache.png`、`cache_round0.png` …）。

---

## 4. Python API：Qwen

```python
import torch
from PIL import Image
from cache_edit.models.qwen import init_qwen_pipeline, create_default_cache_manager

cache_manager = create_default_cache_manager(
    num_inference_steps=50,
    threshold=0.1,
    cache_interval=5,
)

pipeline = init_qwen_pipeline(
    model_path="Qwen/Qwen2-VL-7B-Instruct",
    device="cuda",
    dtype=torch.bfloat16,
    cache_manager=cache_manager,
)

image = Image.open("input.png").convert("RGB")
result = pipeline(image=image, prompt="make it autumn", num_inference_steps=50)
result.images[0].save("out.png")
```

不需要缓存时传 `cache_manager=None`。

---

## 5. Python API：Flux Kontext

```python
import torch
from PIL import Image
from cache_edit.models.flux import init_flux_pipeline, create_default_cache_manager

cache_manager = create_default_cache_manager(
    num_inference_steps=28,
    threshold=0.97,
    cache_interval=5,
)
cache_manager.num_gpus = 1

pipeline = init_flux_pipeline(
    model_path="black-forest-labs/FLUX.1-dev",
    device="cuda",
    dtype=torch.bfloat16,
    cache_manager=cache_manager,
)

image = Image.open("input.png").convert("RGB")
result = pipeline(image=image, prompt="add a red hat", num_inference_steps=28)
result.images[0].save("out.png")
```

---

## 6. 理解多轮缓存

缓存的收益来自**多轮编辑同一张图**：第 0 轮（round 0）完整计算并写入缓存，
后续轮次在非缓存步复用激活，跳过冗余计算。

- `on_step_start(step)`：每步开始时调用，`step==0` 触发轮次自增。
- 第 0 轮：`should_reuse` 恒为 `False`，全程计算并 `store_activation`。
- 第 1 轮起：非缓存步 `should_reuse` 为 `True`，直接读缓存。
- 跨轮提交：每轮结束 `flush_new_to_prev()`（Flux 为 `flush_new_cache_after_step`）。

Flux 额外引入**关键 token**：只对相对参考帧变化大的 token 重算，
其余复用，由 `compute_key_indices_fn` / `update_key_token_indices` 驱动。

手动驱动一轮的示意：

```python
cache_manager.reset()
for step in range(num_steps):
    cache_manager.on_step_start(step)
    if cache_manager.should_reuse(step):
        act = cache_manager.get_activation(...)   # 复用
    else:
        ...                                        # 计算
        cache_manager.store_activation(...)
cache_manager.flush_new_to_prev()
```

> CLI 的 `benchmark` 已封装多轮流程，日常无需手写。

---

## 7. 自定义配置

复制默认配置并修改：

```bash
cp configs/flux_default.yaml my_flux.yaml
```

```yaml
# my_flux.yaml（节选）
cache:
  threshold: 0.95
  cache_interval: 4
pipeline:
  num_inference_steps: 30
  guidance_scale: 3.0
```

```bash
cache-edit edit --model flux --config my_flux.yaml \
  --image in.png --prompt "..." --output out.png
```

Python 中加载并校验：

```python
from cache_edit.config import FluxConfig

cfg = FluxConfig.from_yaml("my_flux.yaml")
cfg.validate()
print(cfg.cache.threshold)
```

---

## 8. 环境变量覆盖

无需改文件即可临时覆盖任意字段（前缀 `CACHEEDIT_`，嵌套用双下划线）：

```bash
export CACHEEDIT_CACHE__THRESHOLD=0.9
export CACHEEDIT_MODEL__DEVICE=cuda:1
export CACHEEDIT_PIPELINE__NUM_INFERENCE_STEPS=20

cache-edit edit --model flux --image in.png --prompt "..."
```

加 `--no-env` 可禁用覆盖。优先级：**环境变量 > 配置文件 > dataclass 默认值**。

更多 API 细节见 [API 文档](API.md)。
