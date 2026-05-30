# Flux 多轮缓存性能对比（修复分辨率 Bug 后）

## 测试配置

- **模型**: FLUX.1-Kontext-dev
- **图像**: 0000.jpg (3024x4032 → 880x1184)
- **轮次**: 8 轮连续编辑
- **推理步数**: 28
- **设备**: GPU (CUDA)
- **种子**: 110

## Bug 修复

### 问题
原 no-cache 基线使用 stock `FluxKontextPipeline`，缺少分辨率自动调整逻辑，生成错误的 1024x1024 图像。

### 解决方案
修改 `init_flux_pipeline` 在 `cache_manager=None` 时跳过所有缓存 forward 替换，但保留 `CacheFluxKontextPipeline` 的分辨率自动调整功能。

**关键改动**:
```python
# cache_edit/models/flux/pipeline.py
if cache_manager is not None:
    # Only install cache hooks when cache_manager is provided
    pipeline.transformer.forward = cache_flux_transformer_2d_forward.__get__(...)
    # ... 其他缓存相关替换
```

## 性能对比

| 配置 | 平均耗时/轮 | Round 0 | Round 1-7 平均 | 加速比 |
|------|------------|---------|---------------|--------|
| **No-cache (修复后)** | **29.41s** | 31.14s | 29.16s | 1.00x (基线) |
| **Cache (interval=5)** | **20.25s** | 30.88s | 18.20s | **1.45x** |
| **Cache (interval=3)** | 22.15s | 30.93s | 20.48s | 1.33x |

### 详细数据

#### No-cache 基线（修复后）
```
Round 0: 31.14s (首轮，完整计算)
Round 1: 29.19s
Round 2: 29.15s
Round 3: 29.18s
Round 4: 29.17s
Round 5: 29.15s
Round 6: 29.16s
Round 7: 29.17s
平均: 29.41s
```

#### Cache (interval=5, 稀疏缓存)
```
Round 0: 30.88s (首轮，写入缓存)
Round 1: 13.40s ↓ 54%
Round 2: 13.95s ↓ 52%
Round 3: 15.29s ↓ 48%
Round 4: 24.48s ↓ 16%
Round 5: 24.15s ↓ 17%
Round 6: 14.44s ↓ 50%
Round 7: 25.43s ↓ 13%
平均: 20.25s (1.45x 加速)
```

#### Cache (interval=3, 密集缓存)
```
Round 0: 30.93s (首轮，写入缓存)
Round 1-7: 20.48s 平均
平均: 22.15s (1.33x 加速)
```

## 关键发现

1. **分辨率一致性**: 修复后所有版本均生成 880x1184 图像（保持原图 3:4 比例）
2. **稀疏缓存更优**: interval=5 比 interval=3 快 8.6%，因为：
   - 更少的缓存写入/读取开销
   - 避免跨 GPU 传输（interval=3 需要多 GPU）
   - 单 GPU 内存管理更高效
3. **轮次间差异**: 某些轮次（4, 5, 7）缓存收益较小，可能因为编辑幅度大导致关键 token 比例高

## 显存使用

| 配置 | GPU0 | GPU1 | 总计 |
|------|------|------|------|
| No-cache | ~40GB | - | ~40GB |
| Cache (interval=5) | ~45GB | - | ~45GB |
| Cache (interval=3) | 80.3GB | 30.5GB | 110.8GB |

## 结论

- **推荐配置**: `cache_interval=5`，在单 GPU 上实现 **1.45x 加速**
- **密集缓存权衡**: interval=3 虽然缓存更多，但跨 GPU 传输抵消了收益
- **Bug 修复重要性**: 确保 no-cache 基线与 cache 版本使用相同的分辨率调整逻辑，避免不公平对比
