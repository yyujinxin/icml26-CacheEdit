# CacheEdit with LLM.265 NVENC Compression

本文档说明如何使用集成了 LLM.265 硬件视频编解码压缩的 CacheEdit 项目进行 Flux 图像编辑。

---

## 📋 目录

1. [环境准备](#环境准备)
2. [快速开始](#快速开始)
3. [完整使用指南](#完整使用指南)
4. [压缩功能说明](#压缩功能说明)
5. [性能优化建议](#性能优化建议)
6. [故障排除](#故障排除)

---

## 环境准备

### 系统要求

- **操作系统**：Linux (Ubuntu 20.04+)
- **GPU**：NVIDIA GPU with NVENC/NVDEC support (RTX 系列推荐)
  - 至少 1 张 GPU (24GB VRAM 推荐)
  - 支持多 GPU 并行 (2-4 张)
- **CUDA**：11.8+
- **Python**：3.10+

### 安装依赖

```bash
# 1. 克隆项目
cd /home/yujinxin/icml26-CacheEdit

# 2. 创建虚拟环境
python3.10 -m venv .venv
source .venv/bin/activate

# 3. 安装依赖
pip install -r requirements.txt

# 4. 安装 matplotlib (压缩功能需要)
pip install matplotlib

# 5. 验证 NVENC 支持
python -c "import torch; from cache_edit.compression.activation_compressor import NVENC_AVAILABLE; print(f'NVENC Available: {NVENC_AVAILABLE}')"
```

**预期输出**：
```
NVENC Available: True
```

---

## 快速开始

### 最简单的测试

```bash
# 激活环境
source .venv/bin/activate

# 运行基础测试（无压缩）
python scripts/run_flux_multi_gpu_optimized.py \
    --model-path /mnt/data/models/black-forest-labs/FLUX___1-Kontext-dev \
    --data-root /mnt/data/datasets/test \
    --output-dir ./outputs/flux_test \
    --use-cache \
    --num-inference-steps 28
```

### 启用压缩的测试

```bash
# 运行带压缩的测试
python scripts/run_flux_multi_gpu_optimized.py \
    --model-path /mnt/data/models/black-forest-labs/FLUX___1-Kontext-dev \
    --data-root /mnt/data/datasets/test \
    --output-dir ./outputs/flux_compressed \
    --use-cache \
    --use-cache-compression \
    --compression-bitrate 5.0 \
    --compression-codec hevc \
    --num-inference-steps 28
```

---

## 完整使用指南

### 数据准备

#### 1. 模型下载

```bash
# 使用 modelscope 下载 Flux 模型
python download_model.py
```

模型会保存到：`/mnt/data/models/black-forest-labs/FLUX___1-Kontext-dev`

#### 2. 数据集结构

```
/mnt/data/datasets/test/
├── images/
│   ├── 0000.jpg
│   ├── 0001.jpg
│   └── ...
└── metadata_multi_round.jsonl
```

#### 3. metadata 格式

```json
{
  "image_idx": "0000",
  "image_path": "images/0000.jpg",
  "rounds": [
    {
      "round": 0,
      "prompt": "A photo of a cat",
      "edit_prompt": "Change the cat to a dog"
    },
    {
      "round": 1,
      "prompt": "A photo of a dog",
      "edit_prompt": "Add a hat to the dog"
    }
  ]
}
```

---

## 运行流程

### 方式 1：标准单 GPU 运行

```bash
source .venv/bin/activate

python scripts/run_flux_multi_gpu_optimized.py \
    --model-path /mnt/data/models/black-forest-labs/FLUX___1-Kontext-dev \
    --data-root /mnt/data/datasets/test \
    --output-dir ./outputs/flux_output \
    --device cuda:0 \
    --use-cache \
    --use-cache-compression \
    --compression-bitrate 5.0 \
    --num-inference-steps 28 \
    --cache-interval 5 \
    --guidance-scale 3.5 \
    --threshold 0.97 \
    --seed 42
```

### 方式 2：多 GPU 运行

```bash
source .venv/bin/activate

PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
CUDA_VISIBLE_DEVICES=0,1,2,3 \
python scripts/run_flux_multi_gpu_optimized.py \
    --model-path /mnt/data/models/black-forest-labs/FLUX___1-Kontext-dev \
    --data-root /mnt/data/datasets/test \
    --output-dir ./outputs/flux_multi_gpu \
    --num-gpus 4 \
    --gpu-memory-limit-gb 20.0 \
    --gpu-memory-buffer-gb 3.0 \
    --use-cache \
    --use-cache-compression \
    --compression-bitrate 5.0 \
    --num-inference-steps 28
```

### 方式 3：批量测试脚本

```bash
# 对比测试：无压缩 vs 压缩
./scripts/test_compression.sh
```

这会运行三个测试：
1. 无压缩（baseline）
2. HEVC 5 Mbps 压缩
3. HEVC 3 Mbps 压缩（高压缩率）

---

## 压缩功能说明

### 压缩原理

CacheEdit 使用 NVIDIA NVENC 硬件视频编解码器来压缩 Transformer 激活缓存：

1. **第一轮编辑**：
   - 正常推理，生成激活
   - 将激活压缩为 HEVC/H.264 视频格式
   - 存储压缩后的数据到 CPU/GPU 内存

2. **后续轮次**：
   - 从缓存读取压缩数据
   - 使用 NVDEC 硬件解码
   - 复用解压后的激活继续推理

### 压缩参数说明

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--use-cache-compression` | False | 启用激活压缩 |
| `--compression-bitrate` | 5.0 | 压缩码率 (Mbps)，1-10 典型值 |
| `--compression-codec` | hevc | 编解码器：hevc 或 h264 |

**压缩码率选择**：
- **3-4 Mbps**：高压缩率（~4-5x），适合内存极度受限场景
- **5-6 Mbps**：推荐值（~3.5x），质量损失极小
- **7-10 Mbps**：低压缩率（~2-3x），几乎无损

### 性能指标

基于我们的测试（Flux，激活形状 1×10000×3072）：

| 指标 | 数值 |
|------|------|
| 压缩比 | ~3.54x |
| 原始大小 | 58.6 MB |
| 压缩后大小 | 16.5 MB |
| 压缩时间 | ~600ms |
| 解压时间 | ~400ms |
| 质量损失 (MSE) | < 0.026 |
| 最大误差 | < 1.5 |

---

## 完整参数列表

### 基础参数

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `--model-path` | str | (必需) | Flux 模型路径 |
| `--data-root` | str | (必需) | 数据集根目录 |
| `--metadata` | str | None | metadata 文件路径，默认 `{data-root}/metadata_multi_round.jsonl` |
| `--output-dir` | str | `./outputs/flux_multi_round` | 输出目录 |
| `--image-idx` | str | None | 指定处理哪张图片，`all` 处理所有图片 |
| `--device` | str | `cuda:0` | 主设备 |
| `--seed` | int | 42 | 随机种子 |

### 推理参数

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `--num-inference-steps` | int | 110 | 推理步数 |
| `--guidance-scale` | float | 2.5 | CFG guidance scale |
| `--true-cfg-scale` | float | 1.0 | True CFG scale |

### 缓存参数

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `--use-cache` | flag | False | 启用激活缓存 |
| `--threshold` | float | 0.97 | Key-token 相似度阈值 |
| `--cache-interval` | int | 5 | 缓存间隔（步数） |

### 压缩参数

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `--use-cache-compression` | flag | False | 启用缓存压缩 |
| `--compression-bitrate` | float | 5.0 | 压缩码率 (Mbps) |
| `--compression-codec` | str | hevc | 编解码器：hevc 或 h264 |

### 多 GPU 参数

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `--num-gpus` | int | 1 | 可用 GPU 数量 |
| `--gpu-memory-limit-gb` | float | 22.0 | 每张 GPU 显存上限 (GB) |
| `--gpu-memory-buffer-gb` | float | 2.0 | 显存预留 buffer (GB) |

### 优化参数

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `--offload-encoders` | flag | False | 将 encoder 卸载到 CPU |

---

## 性能优化建议

### 1. 内存优化

**问题**：GPU 显存不足 (OOM)

**解决方案**：

```bash
# 方案 A：增加显存 buffer
--gpu-memory-buffer-gb 4.0

# 方案 B：减少缓存密度
--cache-interval 10  # 增加间隔，减少缓存层数

# 方案 C：降低推理步数（测试时）
--num-inference-steps 20

# 方案 D：启用 encoder offload
--offload-encoders

# 方案 E：设置环境变量
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
```

### 2. 速度优化

**提升推理速度**：

```bash
# 方案 A：使用更快的编解码器
--compression-codec h264  # H.264 比 HEVC 快，但压缩率稍低

# 方案 B：提高码率（减少压缩时间）
--compression-bitrate 8.0  # 更高码率 = 更快编码

# 方案 C：增加缓存间隔（减少缓存操作）
--cache-interval 10

# 方案 D：使用多 GPU
--num-gpus 4
```

### 3. 质量优化

**提升输出质量**：

```bash
# 方案 A：增加推理步数
--num-inference-steps 50

# 方案 B：调整 guidance scale
--guidance-scale 3.5

# 方案 C：提高压缩码率（减少质量损失）
--compression-bitrate 8.0

# 方案 D：使用 HEVC（质量更好）
--compression-codec hevc
```

---

## 故障排除

### 问题 1：NVENC not available

**错误信息**：
```
RuntimeError: NVENC not available - cannot create ActivationCompressor
```

**解决方案**：
1. 检查 GPU 是否支持 NVENC：
   ```bash
   nvidia-smi
   ```
2. 检查 CUDA 版本：
   ```bash
   nvcc --version
   ```
3. 重新编译 NVENC 扩展：
   ```bash
   cd cache_edit/compression/pytorch_nvenc/cuda_extensions
   python setup.py install
   ```

### 问题 2：Compression failed (error 10)

**错误信息**：
```
[Compress] Failed for stepX_layerY: RegisterResource returned error 10
```

**原因**：NVENC 资源耗尽（已在最新代码中修复）

**解决方案**：
1. 确保使用最新代码（已修复编码器复用问题）
2. 降低 `max_cached_pipelines`：
   ```python
   # 在 cache_manager.py 中
   use_compression=True,
   compression_bitrate=5.0,
   max_cached_pipelines=1  # 减少 pipeline 缓存
   ```

### 问题 3：CUDA Out of Memory

**错误信息**：
```
torch.OutOfMemoryError: CUDA out of memory
```

**解决方案**：
1. 增加显存 buffer：
   ```bash
   --gpu-memory-buffer-gb 4.0
   ```
2. 减少缓存密度：
   ```bash
   --cache-interval 10
   ```
3. 使用 CPU 缓存：
   ```bash
   # 已默认启用，缓存存储在 CPU 内存
   ```
4. 启用内存碎片整理：
   ```bash
   PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
   ```

### 问题 4：ImportError for matplotlib

**错误信息**：
```
ImportError: No module named 'matplotlib'
```

**解决方案**：
```bash
source .venv/bin/activate
pip install matplotlib
```

---

## 测试验证

### 1. 单元测试

```bash
# 测试压缩器
python scripts/test_lru_cache.py

# 测试编码器复用
python scripts/test_encoder_reuse.py
```

### 2. 集成测试

```bash
# 完整 pipeline 测试
python scripts/run_flux_multi_gpu_optimized.py \
    --use-cache \
    --use-cache-compression \
    --num-inference-steps 28
```

### 3. 对比测试

```bash
# 运行对比测试脚本
./scripts/test_compression.sh
```

输出目录：
- `./outputs/flux_compression_test_no_compression/` - 无压缩 baseline
- `./outputs/flux_compression_test_hevc_5mbps/` - HEVC 5Mbps
- `./outputs/flux_compression_test_hevc_3mbps/` - HEVC 3Mbps

---

## 示例：完整工作流

```bash
#!/bin/bash
# 完整的图像编辑工作流示例

# 1. 激活环境
source .venv/bin/activate

# 2. 设置环境变量
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export CUDA_VISIBLE_DEVICES=0,1,2,3

# 3. 运行推理
python scripts/run_flux_multi_gpu_optimized.py \
    --model-path /mnt/data/models/black-forest-labs/FLUX___1-Kontext-dev \
    --data-root /mnt/data/datasets/test \
    --output-dir ./outputs/my_edits \
    --image-idx "0000" \
    --num-gpus 4 \
    --gpu-memory-limit-gb 20.0 \
    --gpu-memory-buffer-gb 3.0 \
    --use-cache \
    --use-cache-compression \
    --compression-bitrate 5.0 \
    --compression-codec hevc \
    --num-inference-steps 28 \
    --cache-interval 5 \
    --guidance-scale 3.5 \
    --threshold 0.97 \
    --seed 42

# 4. 查看结果
ls -lh ./outputs/my_edits/
```

---

## 附录

### A. 目录结构

```
cache_edit/
├── compression/              # 压缩模块
│   ├── activation_compressor.py   # 压缩器实现
│   ├── pipeline/                  # 压缩 pipeline
│   │   ├── nvenc.py              # NVENC 编解码（已修复）
│   │   ├── quantization.py       # 量化步骤
│   │   └── definitions.py        # Pipeline 定义
│   └── pytorch_nvenc/            # LLM.265 NVENC 模块
├── models/
│   └── flux/
│       ├── cache_manager.py      # 缓存管理器（支持压缩）
│       ├── pipeline.py           # Pipeline 接口
│       ├── processor.py          # 注意力处理器
│       └── blocks.py             # Transformer blocks
└── scripts/
    ├── run_flux_multi_gpu_optimized.py  # 主运行脚本
    ├── test_compression.sh              # 对比测试
    ├── test_lru_cache.py                # LRU 测试
    └── test_encoder_reuse.py            # 编码器复用测试
```

### B. 关键修复说明

**编码器资源耗尽问题修复** (已完成)：

**问题**：`MonoNVEncode` 和 `NVEncode` 类在每个 tile 循环中创建新的编码器/解码器，导致资源耗尽。

**修复**：
- `nvenc.py:228-255` - `MonoNVEncode.forward()` 复用编码器
- `nvenc.py:256-272` - `MonoNVEncode.backward()` 复用解码器
- `nvenc.py:150-170` - `NVEncode.forward()` 复用编码器
- `nvenc.py:172-187` - `NVEncode.backward()` 复用解码器

**验证**：
- ✅ 38 层独立压缩测试通过
- ✅ Flux 完整推理测试无压缩错误

### C. 性能对比

| 配置 | 内存使用 | 推理速度 | 质量 |
|------|----------|---------|------|
| 无缓存 | 基准 | 最慢 | 最好 |
| 缓存（无压缩） | 高 | 快 | 最好 |
| 缓存 + 压缩 (5 Mbps) | 中 | 快 | 极好 |
| 缓存 + 压缩 (3 Mbps) | 低 | 快 | 很好 |

---

## 联系与贡献

- **项目位置**：`/home/yujinxin/icml26-CacheEdit`
- **问题反馈**：创建 GitHub Issue
- **功能请求**：提交 Pull Request

---

**最后更新**：2025-01-XX
**版本**：1.0
**状态**：✅ 生产就绪
