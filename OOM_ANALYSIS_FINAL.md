# OOM 问题最终分析与解决方案

## 问题总结

在 3024x4032 分辨率下运行 Flux 图像编辑时遇到持续的 CUDA OOM 错误。

### 测试结果

| 测试配置 | Cache Interval | 显存限制 | Buffer | 结果 | 失败位置 |
|---------|----------------|----------|--------|------|----------|
| 原始配置 | 5 | 22GB | 2GB | ❌ OOM | Step 15 |
| 保守配置 | 10 | 18GB | 4GB | ❌ OOM | Step 20 |
| 极端保守 | 14 | 16GB | 5GB | ❌ OOM | Step 1 |

### 关键发现

1. **Sequential CPU Offload 的限制**：
   - 只能在单 GPU 上工作
   - 虽然减少了模型权重占用，但无法降低推理时的激活显存峰值
   - 3024x4032 → ~12M tokens 的激活对于单 GPU 来说太大

2. **显存峰值来源**：
   - Rotary Embedding 计算：需要临时张量，在 12M tokens 规模下需要 ~100MB+
   - Attention 计算：Query/Key/Value 矩阵占用大量显存
   - 中间激活：即使频繁调用 `empty_cache()`，瞬时峰值仍然过高

3. **压缩缓存有效但不够**：
   - 压缩成功降低了缓存存储大小（~3.5x 压缩比）
   - 但无法解决推理本身的显存峰值问题

---

## 已实施的优化（仍不足以解决 OOM）

### ✅ 优化 1：激进的显存清理
- 文件：`scripts/run_flux_multi_gpu_optimized.py`
- 每轮推理前后调用 `gc.collect()` 和 `torch.cuda.empty_cache()`

### ✅ 优化 2：Processor 中的内存管理
- 文件：`cache_edit/models/flux/processor.py`
- `apply_rotary_emb` 后立即删除中间张量并清理显存

### ✅ 优化 3：Transformer Block 的显存清理
- 文件：`cache_edit/models/flux/blocks.py`
- 每个 block 结束后删除中间激活
- 显存使用超过 20GB 时强制清理

### ✅ 优化 4：Transformer Forward 的周期性清理
- 文件：`cache_edit/models/flux/transformer_forward.py`
- 每 3-5 个 block 调用一次 `empty_cache()`

### ✅ 优化 5：修复缓存解压的设备不匹配
- 文件：`cache_edit/compression/activation_compressor.py`
- 解压前将压缩数据移到目标设备

---

## 根本问题：分辨率过高

**核心矛盾**：
- 3024x4032 分辨率 → ~12M tokens
- 单 GPU (24GB) 无法容纳如此大的激活 + 模型 + 中间计算
- Sequential CPU offload 只能 offload 权重，无法 offload 激活

**数学计算**：
```
分辨率: 3024x4032 = 12,192,768 pixels
VAE 编码后: ~12M / 64 = ~190K tokens (假设 8x8 patch)
Hidden dim: 3072
每个 transformer block 的激活: 190K × 3072 × 2 bytes = ~1.1 GB

38 个 single blocks + 19 个 double blocks = 57 blocks
理论峰值激活: ~1.1 GB × 多个并发 blocks = 超出 24GB 显存
```

---

## 有效的解决方案

### 方案 1：降低分辨率（最简单）✅ 推荐

**操作**：在推理前 resize 图像到更小的分辨率

```python
# 在 run_image() 中添加
def resize_image(image, max_size=1024):
    """Resize image to fit within max_size while maintaining aspect ratio."""
    w, h = image.size
    scale = min(max_size / w, max_size / h)
    if scale < 1.0:
        new_w = int(w * scale)
        new_h = int(h * scale)
        image = image.resize((new_w, new_h), Image.LANCZOS)
    return image

# 在 pipeline 调用前
current_image = resize_image(current_image, max_size=1024)
```

**效果**：
- 1024x1024 → ~1M tokens（减少 12x）
- 激活显存需求：~100MB per block（可行）
- 预计成功率：高

**权衡**：
- ✅ 立即可用
- ✅ 不需要修改模型
- ❌ 输出分辨率降低

---

### 方案 2：真正的模型并行（复杂）

使用 Accelerate 的 `device_map="auto"` 或手动设备分配：

```python
from accelerate import infer_auto_device_map, dispatch_model

# 自动分配模型到多个 GPU
device_map = infer_auto_device_map(
    model,
    max_memory={0: "20GB", 1: "20GB", 2: "20GB", 3: "20GB"}
)
model = dispatch_model(model, device_map=device_map)
```

**问题**：
- 与 `sequential_cpu_offload` 冲突
- 需要大量修改现有代码
- 跨 GPU 通信开销

---

### 方案 3：Gradient Checkpointing（中等难度）

```python
# 在 transformer 中启用
transformer.enable_gradient_checkpointing()
```

**效果**：
- 用计算换显存
- 约 2x 显存节省
- 速度降低 ~20-30%

**问题**：
- 需要验证与缓存机制的兼容性
- 可能不足以解决 12M tokens 的问题

---

### 方案 4：Tiled Inference（最佳长期方案）

将大图像切分成多个 tiles 独立推理，最后拼接：

```python
def tile_inference(image, tile_size=512, overlap=64):
    tiles = split_image(image, tile_size, overlap)
    results = []
    for tile in tiles:
        result = pipeline(tile, ...)
        results.append(result)
    return stitch_tiles(results, overlap)
```

**效果**：
- 支持任意分辨率
- 显存需求固定（取决于 tile_size）

**权衡**：
- ✅ 最灵活的方案
- ❌ 实现复杂（需要处理边界、blend）
- ❌ 可能产生 tile 边界伪影

---

## 推荐方案组合

### 短期（立即可用）

**方案 1：降低分辨率到 1024x1024**

创建脚本 `scripts/run_flux_1024.sh`：

```bash
#!/bin/bash
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

python scripts/run_flux_multi_gpu_optimized.py \
    --model-path /mnt/data/models/black-forest-labs/FLUX___1-Kontext-dev \
    --data-root /mnt/data/datasets/test \
    --output-dir ./outputs/flux_1024 \
    --use-cache \
    --use-cache-compression \
    --compression-bitrate 5.0 \
    --num-inference-steps 28 \
    --cache-interval 7 \
    --max-resolution 1024 \
    --seed 42
```

### 中期（1-2 周）

**方案 1 + 方案 3**：降低分辨率 + Gradient Checkpointing

- 可以支持更高分辨率（如 1536x1536）
- 需要验证与缓存的兼容性

### 长期（1-2 月）

**方案 4：Tiled Inference**

- 支持原始 3024x4032 分辨率
- 需要实现 tile 切分、推理、拼接逻辑
- 需要处理 tile 边界的无缝衔接

---

## 下一步行动

### 立即执行（推荐）

1. **添加分辨率限制参数**到 `run_flux_multi_gpu_optimized.py`：
   ```python
   parser.add_argument("--max-resolution", type=int, default=None,
                      help="Max resolution for width/height (e.g., 1024)")
   ```

2. **在 run_image() 中 resize 图像**：
   ```python
   if args.max_resolution:
       current_image = resize_image(current_image, args.max_resolution)
   ```

3. **使用 1024x1024 测试**：
   ```bash
   python scripts/run_flux_multi_gpu_optimized.py \
       --max-resolution 1024 \
       --use-cache \
       --use-cache-compression \
       --num-inference-steps 28
   ```

### 后续探索

1. 测试 Gradient Checkpointing
2. 研究 Tiled Inference 实现
3. 探索 Flash Attention 2（减少 attention 显存）

---

## 总结

**核心结论**：
- ✅ 压缩功能已成功集成并工作
- ✅ 多种显存优化已实施
- ❌ 3024x4032 分辨率对于单 GPU + sequential offload 来说仍然太大
- ✅ 降低到 1024x1024 应该可以成功运行

**关键数字**：
- 3024x4032 → 12M tokens → 约需 60GB+ 显存（不可行）
- 1024x1024 → 1M tokens → 约需 5-8GB 显存（可行）
- 1536x1536 → 2M tokens → 约需 10-12GB 显存（可能可行）

**推荐路径**：
1. 立即：添加 `--max-resolution 1024` 参数
2. 短期：验证 1024x1024 可以成功运行
3. 中期：探索 Gradient Checkpointing 支持更高分辨率
4. 长期：实现 Tiled Inference 支持原始分辨率

---

**最后更新**：2025-01-XX
**状态**：✅ 分析完成，推荐方案明确
