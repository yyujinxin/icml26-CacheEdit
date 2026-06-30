# Compression Parameter Search Notes

本文档记录 CacheEdit FLUX activation cache 压缩参数的搜索过程、淘汰原因和当前推荐配置。

## 目标

在保持图像质量的前提下提高 activation cache 的压缩率，并兼顾显存占用和运行稳定性。

当前质量门槛：

- `compressed_vs_cache PSNR >= 30`
- `compressed_vs_cache SSIM >= 0.66`
- `compressed_vs_cache LPIPS <= 0.18`
- `compression.failure_count == 0`

这些指标以 cache-only 结果作为参考，因此评估的是“压缩/解压恢复 cache activation”相对“不压缩 cache activation”的额外误差。

## 固定实验设置

除非特别说明，以下搜索使用：

- 模型：`/mnt/data/models/black-forest-labs/FLUX___1-Kontext-dev`
- 数据：`/mnt/data/datasets/test`
- 图像：`image_idx=0000`
- 推理步数：`28`
- 轮数：`2` 或 `3`
- cache interval：`5`
- threshold：`0.97`
- guidance scale：`3.5`
- seed：`42`
- 多卡：`num_gpus=4`
- 图像尺寸：不手动指定，使用 pipeline 内部 `_auto_resize`

结果目录：

- `outputs/compression_quant_probe_28step_3round`
- `outputs/bitrate_sweep_qg256_hevc_gop32p1_28step_2round`
- `outputs/best_ratio_relaxed_28step_3round`
- `outputs/codec_strength_qg3072_hevc_28step_2round`

## 搜索流程

### 1. 建立 baseline 和 cache-only 参考

每个 sweep 都先生成：

- `baseline_no_cache`：不 cache、不压缩。
- `cache_only`：只 cache，不做 activation 压缩。
- `compressed`：cache + codec 压缩/恢复。

随后用 `scripts/evaluate_image_metrics.py` 计算：

- `cache_vs_baseline`
- `compressed_vs_baseline`
- `compressed_vs_cache`

最终筛选只看 `compressed_vs_cache`，因为目标是控制压缩恢复引入的额外误差。

### 2. 量化 group size 和 GOP/P-frame 粗搜

最初搜索的是 codec 前 FP16->uint8 的量化粒度，以及 GOP 帧间压缩结构。

关键参数：

- `COMPRESSION_QUANT_GROUP_SIZE`
- `COMPRESSION_GOP_LENGTH`
- `COMPRESSION_FRAME_INTERVAL_P`
- `COMPRESSION_QUANT_OUTLIER_RATIO`

结论：

- 较小 qg 精度更稳，但 scale/offset 元数据更多，总压缩率较低。
- 较大 qg 元数据更少，总压缩率更高，但量化误差增加。
- `gop32,p1` 能利用连续 layer 作为连续帧做 P-frame 压缩，压缩率优于短 GOP。
- `qg3072,gop32,p1,outlier=0` 在放宽到 `30/0.66/0.18` 门槛后，成为后续 codec strength 搜索的固定量化/GOP 起点。

3-round lossless 参考结果中，`qg3072,gop32,p1` 已能通过质量门槛，并提供约 `5.8x` 量级总压缩率。

### 3. qg256 码率 sweep

固定：

```bash
--compression-codec hevc
--compression-gop-length 32
--compression-frame-interval-p 1
--compression-quant-group-size 256
```

搜索 VBR bitrate：`0.5/1/2/5/10 Mbps`，并保留 lossless anchor。

结果：

| 配置 | total ratio | PSNR | SSIM | LPIPS | 结论 |
|---|---:|---:|---:|---:|---|
| lossless qg256 | 3.24x | 42.995 | 0.99685 | 0.00216 | 通过 |
| HEVC 10Mbps | 10.45x | 20.21 | 0.82282 | 0.23559 | 失败 |
| HEVC 2Mbps | 29.05x | 15.63 | 0.69329 | 0.33858 | 失败 |
| HEVC 1Mbps/0.5Mbps | 更高 | 更低 | 更低 | 更高 | 失败 |
| HEVC 5Mbps | 异常 | 异常 | 异常 | 异常 | 出现 invalid cast |

结论：仅靠 VBR bitrate 调节不能保证质量。低码率会严重破坏 activation，甚至导致生成图像异常。

### 4. 放宽质量门槛后固定 qg/GOP

按用户要求使用质量门槛：

```text
PSNR >= 30, SSIM >= 0.66, LPIPS <= 0.18
```

在 lossless codec 下测试更大的 qg：

- `qg0`：channel-wise。
- `qg512`
- `qg1024`
- `qg3072`
- `qg256 + outlier=0.001`

结论：

- `qg3072,gop32,p1` 在压缩率和质量之间表现最好。
- `qg256 + outlier` 能降低最大量化误差，但会增加辅助元数据；对当前压缩率目标不是最优。
- 因此后续 codec strength 搜索固定为：

```bash
--compression-gop-length 32
--compression-frame-interval-p 1
--compression-quant-group-size 3072
--compression-quant-outlier-ratio 0
```

### 5. Codec compression strength 搜索

这一步不再使用 lossless 作为最终目标，而是固定量化/GOP，主要探索 codec 有损强度参数。

新增并使用的参数：

```bash
--compression-rc-mode {vbr,cbr,constqp}
--compression-const-qp <int>
--compression-bitrate-max-multiplier <float>
```

同时修复了 native encoder 中的一个 bitrate 配置错误：

- 原问题：`averageBitRate` 误写成 `maxBitRate`。
- 修复后：VBR/CBR 模式会正确使用传入的平均码率。

本轮脚本：

```bash
bash scripts/sweep_codec_strength_qg3072.sh
```

脚本默认候选：

```bash
hevc constqp=0
hevc constqp=4
hevc constqp=8
hevc vbr 20Mbps
hevc vbr 50Mbps
lossless anchor
```

`100Mbps VBR` 也被单独测试过，但触发 NVENC init error / OOM fallback，默认已从脚本中移除。

## Codec Strength 搜索结果

输出目录：

```text
outputs/codec_strength_qg3072_hevc_28step_2round
```

汇总文件：

- `sweep_summary.csv`
- `sweep_summary.json`
- `recommended_config.json`
- `sweep.log`

结果表：

| 配置 | total ratio | PSNR | SSIM | LPIPS | failure | 结论 |
|---|---:|---:|---:|---:|---:|---|
| lossless qg3072 | 5.88x | 38.03 | 0.991 | 0.0066 | 0 | 高质量 anchor |
| HEVC ConstQP 0 | 8.94x | 32.27 | 0.958 | 0.0476 | 0 | 通过 |
| HEVC ConstQP 4 | 11.25x | 32.47 | 0.963 | 0.0302 | 0 | 当前最佳 |
| HEVC ConstQP 8 | 16.46x | 29.17 | 0.927 | 0.0604 | 0 | PSNR 失败 |
| HEVC VBR 50Mbps | 30.61x | 21.43 | 0.851 | 0.1682 | 0 | PSNR 失败 |
| HEVC VBR 20Mbps | 64.61x | 4.11 | 0.500 | 0.5314 | 0 | 激活严重破坏 |
| HEVC VBR 100Mbps | N/A | Infinity | 1.0 | 0.0 | 36 | 全部 fallback，不可用 |

说明：

- `VBR 100Mbps` 的指标看起来完美，是因为压缩全部失败并回退到未压缩 cache，`failure_count=36`，因此被质量门槛排除。
- `VBR 20Mbps` 出现过 `invalid value encountered in cast`，说明恢复后的 activation 已破坏到影响最终图像。
- `ConstQP 8` 压缩率更高，但 PSNR 已低于 30，因此不作为推荐。
- `ConstQP 4` 在当前门槛下压缩率最高，并且没有压缩失败。

## 当前推荐配置

质量优先：

```bash
--compression-codec lossless
--compression-rc-mode constqp
--compression-const-qp 0
--compression-gop-length 32
--compression-frame-interval-p 1
--compression-quant-group-size 3072
--compression-quant-outlier-ratio 0
```

压缩率优先且满足当前质量门槛：

```bash
--compression-codec hevc
--compression-rc-mode constqp
--compression-const-qp 4
--compression-bitrate 5.0
--compression-bitrate-max-multiplier 10
--compression-gop-length 32
--compression-frame-interval-p 1
--compression-quant-group-size 3072
--compression-quant-outlier-ratio 0
```

推荐使用第二组作为当前默认探索点。

## 运行命令

重新跑当前 codec strength sweep：

```bash
bash scripts/sweep_codec_strength_qg3072.sh
```

手动跑推荐配置：

```bash
python scripts/run_flux_multi_gpu_optimized.py \
  --model-path /mnt/data/models/black-forest-labs/FLUX___1-Kontext-dev \
  --data-root /mnt/data/datasets/test \
  --image-idx 0000 \
  --num-gpus 4 \
  --gpu-memory-limit-gb 16.0 \
  --gpu-memory-buffer-gb 5.0 \
  --num-inference-steps 28 \
  --guidance-scale 3.5 \
  --seed 42 \
  --max-rounds 2 \
  --output-dir outputs/manual_hevc_constqp4_qg3072 \
  --use-cache \
  --use-cache-compression \
  --cache-interval 5 \
  --threshold 0.97 \
  --compression-codec hevc \
  --compression-rc-mode constqp \
  --compression-const-qp 4 \
  --compression-bitrate 5.0 \
  --compression-bitrate-max-multiplier 10 \
  --compression-gop-length 32 \
  --compression-frame-interval-p 1 \
  --compression-quant-group-size 3072 \
  --compression-quant-outlier-ratio 0
```

## 后续建议

当前结果仍是 `image_idx=0000`、2-round 或 3-round 的局部搜索，不应直接等同于完整数据集最终结论。

建议后续按这个顺序验证：

1. 使用 `HEVC ConstQP=4, qg3072, gop32,p1` 跑 5-round，确认误差不会跨 round 快速累积。
2. 在完整数据集上跑 baseline/cache-only/compressed 三组指标，确认平均指标和坏例。
3. 如果 5-round 或全数据集质量不稳，在 `ConstQP=2/3/4/5/6` 之间做细扫。
4. 暂不优先继续 VBR bitrate 路线；当前 VBR 的质量下降过快，并且 100Mbps 会触发 fallback/OOM。
5. 如果需要更高压缩率，优先探索更细的 ConstQP，而不是进一步增大 qg 或降低 bitrate。
