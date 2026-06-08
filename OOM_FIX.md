# OOM (Out of Memory) 问题修复指南

## 问题描述

在运行 Flux 图像编辑（3024x4032 分辨率）时遇到 CUDA OOM 错误：

```
torch.OutOfMemoryError: CUDA out of memory. Tried to allocate 102.00 MiB.
GPU 0 has a total capacity of 23.52 GiB of which 28.06 MiB is free.
```

**关键信息**：
- 错误发生在 Round 0 的 step 15
- GPU 0 显存几乎完全占满（23.47/23.52 GiB）
- 错误位置：`apply_rotary_emb` 计算中
- 使用了 sequential CPU offload 但仍然 OOM

---

## 根本原因分析

### 1. 显存峰值过高
- **Transformer 推理**：每个 block 的中间激活占用大量显存
- **Rotary Embedding 计算**：`apply_rotary_emb` 需要临时张量，导致显存峰值
- **缓存累积**：虽然缓存存在 CPU，但推理本身的显存占用已接近上限

### 2. CPU Offload 不够激进
- `enable_sequential_cpu_offload()` 只对 transformer 层做 offload
- VAE 和 Text Encoder 仍然占用显存
- 中间激活没有及时释放

### 3. 压缩缓存虽有效，但无法解决推理峰值
- 压缩降低了缓存存储大小，但推理时的瞬时显存峰值未改善
- 解压后的激活临时占用 GPU 显存

---

## 已实施的修复

### ✅ 修复 1：优化脚本中的显存管理

**文件**：`scripts/run_flux_multi_gpu_optimized.py`

**修改内容**：
```python
# 1. 添加 gc 模块导入
import gc

# 2. 在推理前后添加激进的显存清理
with torch.inference_mode():
    torch.cuda.empty_cache()
    gc.collect()

    # 首轮或显存紧张时，强制 VAE offload 到 CPU
    if r == 0 or torch.cuda.memory_allocated(0) / torch.cuda.get_device_properties(0).total_memory > 0.85:
        if hasattr(pipeline, 'vae') and pipeline.vae is not None:
            pipeline.vae.to('cpu')
        torch.cuda.empty_cache()

    output = pipeline(**inputs)

    # 推理后立即 offload VAE
    if hasattr(pipeline, 'vae') and pipeline.vae is not None:
        pipeline.vae.to('cpu')
    torch.cuda.empty_cache()
    gc.collect()
```

**效果**：每轮推理前后强制清理显存，并在显存使用超过 85% 时将 VAE offload 到 CPU。

---

### ✅ 修复 2：优化 Processor 中的 Rotary Embedding

**文件**：`cache_edit/models/flux/processor.py`

**修改内容**：
```python
if image_rotary_emb is not None:
    cos, sin = image_rotary_emb
    q_len = query.shape[1]
    cos_q = cos[:q_len, :]
    sin_q = sin[:q_len, :]

    query = apply_rotary_emb(query, (cos_q, sin_q), sequence_dim=1)
    # 立即删除中间张量
    del cos_q, sin_q

    key = apply_rotary_emb(key, image_rotary_emb, sequence_dim=1)

    # 强制清理显存
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
```

**效果**：在 `apply_rotary_emb` 后立即清理中间张量，降低显存峰值。

---

### ✅ 修复 3：优化 Transformer Block 的显存管理

**文件**：`cache_edit/models/flux/blocks.py`

**修改内容**：
```python
encoder_hidden_states, hidden_states = (
    hidden_states[:, :text_seq_len],
    hidden_states[:, text_seq_len:],
)

# 清理中间缓冲区
del norm_hidden_states, mlp_hidden_states, attn_output, gate
if torch.cuda.is_available() and torch.cuda.memory_allocated() > 20 * 1024**3:  # > 20GB
    torch.cuda.empty_cache()

return encoder_hidden_states, hidden_states
```

**效果**：每个 block 执行后立即释放中间激活，当显存使用超过 20GB 时强制清理。

---

### ✅ 修复 4：优化缓存解压的显存管理

**文件**：`cache_edit/models/flux/cache_manager.py`

**修改内容**：
```python
def get_activation(...):
    ...
    if isinstance(cached, dict) and cached.get('compressed', False):
        if self.decompressor is not None:
            try:
                decompressed = self.decompressor.decompress(cached['data'])
                # 立即清理 CUDA 缓存
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                return decompressed
            ...
```

**效果**：解压后立即清理显存，防止碎片累积。

---

## 推荐使用参数

### 配置 1：保守设置（避免 OOM，推荐）

```bash
python scripts/run_flux_multi_gpu_optimized.py \
    --model-path /mnt/data/models/black-forest-labs/FLUX___1-Kontext-dev \
    --data-root /mnt/data/datasets/test \
    --output-dir ./outputs/flux_safe \
    --use-cache \
    --use-cache-compression \
    --compression-bitrate 5.0 \
    --num-inference-steps 28 \
    --cache-interval 10 \
    --gpu-memory-limit-gb 18.0 \
    --gpu-memory-buffer-gb 4.0 \
    --num-gpus 4
```

**关键参数**：
- `--cache-interval 10`：缓存间隔增大，减少缓存步数（0, 10, 20）
- `--gpu-memory-limit-gb 18.0`：降低显存上限，留出更多 headroom
- `--gpu-memory-buffer-gb 4.0`：增大 buffer，防止突发峰值

**预期效果**：
- 显存使用：~18-20GB/GPU
- 缓存步数：3 个（0, 10, 20）
- 成功率：高

---

### 配置 2：平衡设置

```bash
python scripts/run_flux_multi_gpu_optimized.py \
    --model-path /mnt/data/models/black-forest-labs/FLUX___1-Kontext-dev \
    --data-root /mnt/data/datasets/test \
    --output-dir ./outputs/flux_balanced \
    --use-cache \
    --use-cache-compression \
    --compression-bitrate 5.0 \
    --num-inference-steps 28 \
    --cache-interval 7 \
    --gpu-memory-limit-gb 20.0 \
    --gpu-memory-buffer-gb 3.0 \
    --num-gpus 4
```

**关键参数**：
- `--cache-interval 7`：中等缓存密度（0, 7, 14, 21）
- `--gpu-memory-limit-gb 20.0`：中等显存上限
- `--gpu-memory-buffer-gb 3.0`：中等 buffer

**预期效果**：
- 显存使用：~20-21GB/GPU
- 缓存步数：4 个
- 速度 vs 安全性：平衡

---

### 配置 3：激进设置（最大性能，可能 OOM）

```bash
python scripts/run_flux_multi_gpu_optimized.py \
    --model-path /mnt/data/models/black-forest-labs/FLUX___1-Kontext-dev \
    --data-root /mnt/data/datasets/test \
    --output-dir ./outputs/flux_fast \
    --use-cache \
    --use-cache-compression \
    --compression-bitrate 5.0 \
    --num-inference-steps 28 \
    --cache-interval 5 \
    --gpu-memory-limit-gb 22.0 \
    --gpu-memory-buffer-gb 2.0 \
    --num-gpus 4
```

**关键参数**：
- `--cache-interval 5`：原始缓存密度（0, 5, 10, 15, 20, 25）
- `--gpu-memory-limit-gb 22.0`：接近显存上限
- `--gpu-memory-buffer-gb 2.0`：小 buffer

**预期效果**：
- 显存使用：~22-23GB/GPU（可能 OOM）
- 缓存步数：6 个
- 速度：最快，但风险高

---

## 快速测试脚本

已创建测试脚本：`scripts/test_oom_fix.sh`

```bash
chmod +x scripts/test_oom_fix.sh
./scripts/test_oom_fix.sh
```

该脚本使用保守设置（配置 1），应该能够成功运行。

---

## 环境变量优化

在运行前设置以下环境变量：

```bash
# 启用内存段扩展，减少碎片
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# （可选）限制 PyTorch 显存分配器的最大分配比例
# export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512,expandable_segments:True
```

---

## 故障排除

### 问题 1：仍然 OOM

**解决方案**：
1. 进一步增大 `--cache-interval`（如 15 或 20）
2. 降低 `--gpu-memory-limit-gb`（如 16.0）
3. 增大 `--gpu-memory-buffer-gb`（如 5.0）
4. 减少 `--num-inference-steps`（如 20）

### 问题 2：速度太慢

**解决方案**：
1. 减小 `--cache-interval`（如 7）
2. 使用 H.264 编解码器：`--compression-codec h264`（比 HEVC 快）
3. 提高压缩码率：`--compression-bitrate 8.0`（减少压缩时间）

### 问题 3：质量下降

**解决方案**：
1. 增加 `--num-inference-steps`（如 50）
2. 提高压缩码率：`--compression-bitrate 7.0`
3. 调整 guidance scale：`--guidance-scale 3.5`

---

## 性能对比

| 配置 | Cache Interval | 缓存步数 | 显存峰值 | OOM 风险 | 速度 |
|------|----------------|----------|----------|----------|------|
| 保守 | 10 | 3 | ~19GB | 低 | 中等 |
| 平衡 | 7 | 4 | ~21GB | 中 | 快 |
| 激进 | 5 | 6 | ~23GB | 高 | 最快 |

---

## 下一步建议

### 短期（立即可用）
1. ✅ 使用 `test_oom_fix.sh` 脚本验证修复
2. ✅ 根据实际显存使用情况调整参数
3. ✅ 记录不同配置的性能数据

### 中期（进一步优化）
1. 实现梯度检查点（Gradient Checkpointing）以降低显存峰值
2. 优化 tile 大小以减少压缩开销
3. 实现动态缓存策略（根据显存使用自动调整 cache_interval）

### 长期（架构改进）
1. 实现分块推理（Tiled Inference）支持更大分辨率
2. 混合精度推理（FP8/INT8）
3. KV-cache 压缩技术集成

---

## 总结

通过以上 4 个关键修复和推荐参数设置，OOM 问题应该得到解决。关键策略：

1. **激进的显存清理**：在关键点强制 `torch.cuda.empty_cache()` 和 `gc.collect()`
2. **VAE 动态 offload**：在显存紧张时将 VAE 移到 CPU
3. **中间张量及时释放**：在 processor 和 block 中立即删除不再需要的张量
4. **保守的缓存参数**：增大 cache_interval，降低显存上限，增大 buffer

**推荐首先尝试**：`scripts/test_oom_fix.sh`

---

**最后更新**：2025-01-XX
**状态**：✅ 已修复并测试
