# CacheEdit

高效的扩散模型图像编辑缓存优化框架。

CacheEdit 通过在多轮图像编辑过程中智能缓存中间激活（activation），减少冗余计算，
从而加速 Qwen 与 Flux Kontext 的推理。

## 特性

- **多后端支持**：Qwen 图像编辑 与 Flux Kontext 两套 pipeline
- **智能激活缓存**：基于相似度阈值与缓存间隔，跨步/跨轮复用注意力激活
- **关键 token 机制（Flux）**：仅对变化显著的 token 重算，其余复用缓存
- **多 GPU 缓存放置**：缓存张量按可用显存自动选择设备
- **统一配置系统**：dataclass + YAML/JSON，支持环境变量覆盖
- **命令行工具**：`cache-edit edit` 单图编辑，`cache-edit benchmark` 测速对比

## 安装

```bash
# 从源码安装（开发模式）
git clone <repo-url>
cd icml26-CacheEdit
pip install -e .
```

依赖 `torch`、`diffusers`、`transformers`、`Pillow`、`pyyaml`。
模型相关依赖见 `Qwen-image-edit-plus/requirements.txt` 与 `Flux-kontext/requirements.txt`。

## 快速开始

### 命令行

```bash
# 单图编辑（Flux Kontext）
cache-edit edit --model flux \
  --image input.png \
  --prompt "add a red hat" \
  --output out.png

# 使用自定义配置 + 测速对比缓存收益
cache-edit benchmark --model qwen \
  --image input.png \
  --prompt "make it sunset" \
  --config configs/qwen_default.yaml \
  --rounds 3
```

### Python API

```python
import torch
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

from PIL import Image
result = pipeline(
    image=Image.open("input.png").convert("RGB"),
    prompt="make it sunset",
    num_inference_steps=50,
)
result.images[0].save("out.png")
```

## 配置

默认配置位于 `configs/`：

- `configs/qwen_default.yaml`
- `configs/flux_default.yaml`

可用环境变量覆盖任意字段（前缀 `CACHEEDIT_`，嵌套字段用双下划线）：

```bash
export CACHEEDIT_CACHE__THRESHOLD=0.95
export CACHEEDIT_MODEL__DEVICE=cuda:1
```

详见 [配置文档](docs/API.md#config-配置)。

## CLI 参考

| 命令 | 说明 |
| --- | --- |
| `cache-edit edit` | 用文本指令编辑单张图像 |
| `cache-edit benchmark` | 对比开/关缓存的推理延迟与加速比 |
| `cache-edit --version` | 打印版本 |

运行 `cache-edit edit --help` / `cache-edit benchmark --help` 查看完整参数。

## 文档

- [API 文档](docs/API.md)
- [使用教程](docs/TUTORIAL.md)

## 开发

```bash
# 运行测试
pytest tests/ -v
```

CI 通过 GitHub Actions 在 Python 3.8–3.11 上运行测试（见 `.github/workflows/test.yml`）。

## 许可证

见仓库 LICENSE。
