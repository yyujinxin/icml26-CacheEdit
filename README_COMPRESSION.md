# CacheEdit Activation Compression

本文档是当前项目里 cache 压缩相关的统一入口。历史优化过程记录在
`docs/OPTIMIZATION_GUIDE.md`，实际运行请优先看本文和 `scripts/` 下的脚本。

## 当前稳定配置

当前默认目标是：不手动指定输入图像尺寸，使用 pipeline 内部 `_auto_resize`，
在 28-step、`cache_interval=5` 下稳定完成多轮 FLUX 编辑。

推荐参数：

```bash
--use-cache
--cache-interval 5
--threshold 0.97
--use-cache-compression
--compression-codec lossless
--compression-bitrate 5.0
--compression-gop-length 16
--compression-frame-interval-p 16
--num-inference-steps 28
```

说明：

- `lossless` 不是保存原始 tensor，也不是绕过 codec；它使用 HEVC/NVENC lossless
  模式编码量化后的 uint8 帧。
- `hevc` / `h264` 是有损视频编码路径，`--compression-bitrate` 对这两种模式生效。
- `lossless` 下 codec 对量化帧是无损的，但 FP16 到 uint8 的量化仍可能带来误差；
  当前实现优先使用 group-wise quantization 以降低误差。
- GOP 会把同一个 diffusion step 内连续 layer 当作连续视频帧，使用 P 帧帧间预测提高压缩率。
  decoder 已处理 P-frame flush 一次返回多帧的情况，避免后续帧错位恢复。

## 常用脚本

完整 28-step、cache + lossless codec + GOP 测试：

```bash
bash scripts/test_gop28_full.sh
```

不 cache、不压缩的 28-step baseline：

```bash
bash scripts/test_no_cache_28_full.sh
```

三组质量对比并计算 PSNR / SSIM / LPIPS：

```bash
bash scripts/test_cache_quality_metrics.sh
```

完整数据集 cache+compression 续跑：

```bash
bash scripts/run_cache_compressed_full_dataset_resume.sh
```

这些脚本都会激活 `.venv`。参数直接在脚本顶部修改，不依赖外部环境变量传参。

## 压缩数据如何流动

round 0 或 cache step 正常计算激活后，cache manager 会把激活收集成 GOP：

1. FP16 activation reshape 成二维 `[tokens, hidden]`。
2. 量化成 uint8 帧；`lossless` 模式优先用 `GWQuantization(groupsize=64)`。
3. `FixedTiling` 把帧切成 NVENC 可编码 tile。
4. `MonoNVEncodeSequence` 用 NVENC 编成 HEVC/H.264 bitstream；GOP 模式下后续层可以作为 P 帧。
5. 压缩结果作为 cache entry 存在内存里，不写独立视频文件。

压缩后的 cache entry 包含：

- `bitstream` / packet sizes 等 codec payload；
- `scale` / `offset` 等量化恢复元数据；
- 原始 shape、dtype、device；
- GOP 的 `frame_index`、`group_layers`、`gop_length`、`frame_interval_p`；
- 报告字段里的 `codec`、`quantization`、payload size、auxiliary size。

后续 round 复用 cache 时：

1. 从 `prev_cache[(stream, cache_step, layer_idx)]` 找到 compressed entry。
2. 如果是 GOP entry，按 `frame_index` 取对应 layer。
3. NVDEC 一次解码整个 GOP。
4. 解 tile、裁掉 padding、按量化元数据恢复 FP16 activation。
5. 激活搬到当前 transformer layer 所在 device 后继续计算。

当前实现会把压缩 payload 和已解码 GOP cache 放在 CPU 侧，避免长期占用 GPU 显存。

## 同步与异步状态

代码中保留了异步压缩和 GOP prefetch 的结构及统计字段，但当前稳定配置将它们关闭：

- `_async_compression_max_pending = 0`
- `_gop_prefetch_window = 0`

原因是 Python worker 线程中并发调用 NVENC/NVDEC，和 transformer CUDA kernel overlap
时曾触发 CUDA context/NVENC 资源稳定性问题。当前版本以正确性和完整数据集稳定完成为优先：
GOP 压缩、GOP 解码同步执行，已解码结果仍会在 CPU decoded GOP cache 中复用，避免同一 GOP
重复解码。

报告里仍保留 `async_*` 和 `gop_prefetch_*` 字段，方便以后重新打开实验性 overlap。
当前稳定运行时，`async_compression_enabled` 应为 `false`。

## 输出报告

主运行脚本会写：

- `timings.partial.json`：运行中断时也会尽量保留已完成图片的统计。
- `timings.json`：完整运行结束后的最终报告。

压缩统计位于 `compression.summary`，重点看：

- `success_count` / `failure_count`
- `success_count_by_mode`
- `success_count_by_quantization`
- `payload_compression_ratio`
- `total_compression_ratio`
- `compressed_payload_mib`
- `compressed_auxiliary_mib`
- `compressed_total_mib`
- `decompression_failure_count`
- `gop_decode_cache_hit_count`
- `gop_decode_cache_miss_count`

`payload_compression_ratio` 只统计 codec bitstream；`total_compression_ratio` 会把量化元数据、
packet sizes 等辅助数据也算进去，更接近真实 cache 占用。

## 关键文件

- `scripts/run_flux_multi_gpu_optimized.py`：主入口。
- `cache_edit/models/flux/cache_manager.py`：cache 存储、GOP 组织、压缩报告。
- `cache_edit/compression/activation_compressor.py`：activation 到 codec frame 的压缩/解压。
- `cache_edit/compression/pipeline/nvenc.py`：NVENC/NVDEC pipeline step。
- `docs/OPTIMIZATION_GUIDE.md`：历史优化记录和 Nsight 分析结论。

## 常见问题

`RegisterResource returned error 10`：

通常是 NVENC/CUDA resource register 失败。当前代码通过限制 pipeline cache、及时 close
native encoder/decoder、失败后清理 pipeline cache，并避免后台线程并发 NVENC/NVDEC 来降低风险。

生成图变黑或出现 `invalid value encountered in cast`：

优先检查 `timings.json` 中是否有压缩或解压失败、是否回退到 baseline 流程、以及
`success_count_by_quantization` 是否符合预期。当前推荐先使用 `lossless` codec + GOP/P-frame 路径，
而不是低码率 HEVC。

压缩组和 cache-only 输出不一致：

这是预期会有一定差异的路径，因为 codec 前存在 FP16 到 uint8 的量化。`lossless` 只保证 codec
对量化帧无损，不保证 activation 完全逐 bit 等于 cache-only。
