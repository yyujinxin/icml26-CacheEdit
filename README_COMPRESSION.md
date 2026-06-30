# CacheEdit Activation Compression

本文档是当前项目里 cache 压缩相关的统一入口。历史优化过程记录在
`docs/OPTIMIZATION_GUIDE.md`，参数搜索过程和结果记录在
`docs/COMPRESSION_PARAM_SEARCH.md`，实际运行请优先看本文和 `scripts/` 下的脚本。

## 当前稳定配置

当前默认目标是：不手动指定输入图像尺寸，使用 pipeline 内部 `_auto_resize`，
在 28-step、`cache_interval=5` 下稳定完成多轮 FLUX 编辑。

当前 qg256 GOP/P-frame 探索起点：

```bash
--use-cache
--cache-interval 5
--threshold 0.97
--use-cache-compression
--compression-codec lossless
--compression-bitrate 5.0
--compression-gop-length 32
--compression-frame-interval-p 1
--compression-quant-group-size 256
--compression-quant-outlier-ratio 0.0
--num-inference-steps 28
```

当前在 `PSNR>=30, SSIM>=0.66, LPIPS<=0.18` 门槛下的 codec-strength
推荐点：

```bash
--use-cache
--cache-interval 5
--threshold 0.97
--use-cache-compression
--compression-codec hevc
--compression-rc-mode constqp
--compression-const-qp 4
--compression-bitrate 5.0
--compression-bitrate-max-multiplier 10
--compression-gop-length 32
--compression-frame-interval-p 1
--compression-quant-group-size 3072
--compression-quant-outlier-ratio 0
--num-inference-steps 28
```

该配置来自 `outputs/codec_strength_qg3072_hevc_28step_2round`：
2-round、`image_idx=0000` 下 total ratio `11.25x`，cache-vs-compressed
`PSNR=32.47`、`SSIM=0.963`、`LPIPS=0.030`，`failure_count=0`。

说明：

- `lossless` 不是保存原始 tensor，也不是绕过 codec；它使用 HEVC/NVENC lossless
  模式编码量化后的 uint8 帧。
- `hevc` / `h264` 是有损视频编码路径，`--compression-bitrate` 对这两种模式生效。
- `lossless` 下 codec 对量化帧是无损的，但 FP16 到 uint8 的量化仍可能带来误差；
  当前实现优先使用 group-wise quantization 以降低误差。
- `--compression-quant-group-size` 控制 codec 前的 FP16->uint8 group-wise
  量化粒度。较小 group 通常精度更好但 `scale/offset` 元数据更多；较大 group
  通常总压缩率更高但量化误差更大。设为 `0` 会强制使用 channel-wise quantization。
- `--compression-quant-outlier-ratio` 控制是否额外保存少量最大量化 residual。
  主 activation 仍被编码成 uint8 视频帧；该参数只增加辅助元数据。`0` 表示关闭。
- GOP 会把同一个 diffusion step 内连续 layer 当作连续视频帧，使用 P 帧帧间预测提高压缩率。
  decoder 已处理 P-frame flush 一次返回多帧的情况，避免后续帧错位恢复。
- GOP/P-frame 参数仍应按目标 round 数重新 sweep；当前 28-step、2-round
  probe 中 `gop32,p1,qg256` 是通过严格质量门槛时压缩率最高的候选。

当前 28-step、`image_idx=0000`、3-round probe 中，测试过
`64/128/256/512/0`：

- `128` 是质量最稳的质量优先配置。
- `256` 是当前优化默认值，也是 `scripts/summarize_compression_sweep.py` 默认质量门槛
  (`PSNR>=41`, `SSIM>=0.994`, `LPIPS<=0.004`, 无压缩失败) 下压缩率最高的
  候选，总压缩率从 `128` 的 `2.75x` 提升到 `3.19x`。
- `512` 可作为更偏压缩率的候选，`0` 会强制 channel-wise quantization，
  压缩率最高但质量损失明显增大。

group-wise 量化当前使用 rounded zero-point 恢复；min-offset group-wise 变体
虽然在合成 activation 上误差略低，但真实图像 probe 指标更差，因此没有作为默认实现。

5-round probe 显示质量误差会继续累积，之前 3-round 的严格质量门槛不能直接外推到更长编辑链。
因此当前建议先固定 `--compression-quant-group-size 256`，再用
`scripts/sweep_gop_params_qg256.sh` 搜索 `COMPRESSION_GOP_LENGTH` 和
`COMPRESSION_FRAME_INTERVAL_P`。

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

完整数据集论文实验：依次运行 no-cache baseline、cache-only、
cache+compression，统计全数据集延迟、cache 空间占用、PSNR / SSIM / LPIPS，
并生成可直接用于画图的 Excel/CSV/JSON 报告：

```bash
bash scripts/run_full_dataset_paper_experiment.sh
```

默认输出：

- `outputs/full_dataset_paper_28step/no_cache`
- `outputs/full_dataset_paper_28step/cache_only`
- `outputs/full_dataset_paper_28step/cache_compressed`
- `outputs/full_dataset_paper_28step/quality_metrics.json`
- `outputs/full_dataset_paper_28step/paper_report/full_dataset_report.xlsx`
- `outputs/full_dataset_paper_28step/paper_report/*.csv`

如果三组结果已经存在，也可以只重新生成论文表格：

```bash
python scripts/summarize_full_dataset_experiment.py \
  --output-root outputs/full_dataset_paper_28step \
  --baseline-name no_cache \
  --cache-only-name cache_only \
  --compressed-name cache_compressed \
  --metrics-json outputs/full_dataset_paper_28step/quality_metrics.json
```

完整数据集 cache+compression 续跑：

```bash
bash scripts/run_cache_compressed_full_dataset_resume.sh
```

压缩参数 sweep：

```bash
bash scripts/sweep_compression_quant_params.sh
```

该脚本会先跑 baseline/cache-only，然后遍历脚本顶部配置的
`QUANT_GROUP_SIZES` 和 `GOP_CONFIGS`，为每组 cache+compression 输出
`timings.json` 和 PSNR / SSIM / LPIPS 指标，并生成：

- `sweep_summary.csv`
- `sweep_summary.json`
- `recommended_config.json`

`recommended_config.json` 会同时记录质量优先配置，以及满足质量门槛时压缩率最高的配置。

固定 qg256，只搜索 GOP/P-frame：

```bash
bash scripts/sweep_gop_params_qg256.sh
```

该脚本默认测试 `gop16/gop32` 的长 GOP 候选，以及不同
`frame_interval_p`；`gop1/gop4/gop8` 因 codec 调用过碎、复用轮明显过慢而默认排除。
每组结果都会写入 `timings.json` 和 metrics JSON；`sweep_summary.csv` 会包含质量、
压缩率和 torch CUDA 峰值显存列。
当前默认是 2-round 粗扫：round 0 负责建 cache/压缩，round 1 负责解压复用。
选出候选后再把 `MAX_ROUNDS` 改成目标轮数做完整验证。

固定 qg256 和当前 GOP/P 候选，只搜索 HEVC 有损码率：

```bash
bash scripts/sweep_bitrate_qg256.sh
```

该脚本默认使用 `qg256,gop32,p1`，并测试 `0.5/1/2/5/10 Mbps`，
同时保留一个 `lossless` anchor。最新 28-step、2-round probe 的结果是：

- `lossless + qg256 + gop32,p1`：cache-vs-compressed
  `PSNR=42.995`、`SSIM=0.99685`、`LPIPS=0.00216`，总压缩率 `3.24x`，
  通过质量门槛。
- HEVC 10 Mbps：总压缩率 `10.45x`，但 cache-vs-compressed
  `PSNR=20.21`、`SSIM=0.82282`、`LPIPS=0.23559`，未通过质量门槛。
- HEVC 0.5/1/2 Mbps：总压缩率更高，但质量进一步下降。
- HEVC 5 Mbps 这次出现 `invalid value encountered in cast` 警告，图像质量
  指标明显异常，不作为可用配置。

因此当前要求“保证精度”的配置仍推荐 `lossless` codec；如果后续要继续探索
有损 codec，应先降低 codec 前量化误差或改进 activation 数值范围处理，而不是仅提高码率。

固定 `qg3072,gop32,p1`，搜索 HEVC codec compression strength：

```bash
bash scripts/sweep_codec_strength_qg3072.sh
```

该脚本默认搜索 `HEVC ConstQP=0/4/8` 和 VBR `20/50Mbps`，并保留 lossless
anchor。当前结果显示：

- `HEVC ConstQP=4` 是满足 `PSNR>=30, SSIM>=0.66, LPIPS<=0.18` 时压缩率最高的点。
- `HEVC ConstQP=8` 压缩率更高，但 PSNR 降到 `29.17`，未通过门槛。
- VBR 20/50Mbps 的 PSNR 明显低于门槛；100Mbps 曾触发 NVENC/OOM fallback，
  不再作为默认候选。

详细过程见 `docs/COMPRESSION_PARAM_SEARCH.md`。

真实 activation 量化误差探针：

```bash
bash scripts/probe_quant_error_qg256.sh
```

该脚本实际压缩仍使用 qg256，但会在同一批真实 activation 上额外估计
`COMPRESSION_QUANT_ERROR_PROBE_GROUPS` 中多个候选的 FP16->uint8 量化误差。
输出：

- `timings.json`：包含 `compression.summary.quant_error_probe_by_quantization`
- `quant_error_summary.csv`
- `quant_error_summary.json`

重点看 `rmse`、`relative_rmse`、`max_abs` 和
`metadata_over_original_ratio`。较小 qg 通常误差更低，但 scale/offset 元数据更多。

固定 `qg256,gop32,p1`，做 residual-outlier 图像级 sweep：

```bash
bash scripts/sweep_quant_outlier_qg256.sh
```

最新 28-step、2-round activation probe 结果显示：

- `qg256`：`RMSE=2.294`、`max_abs=110.0`、metadata/original `0.098%`。
- `qg256_o0p0005`：`RMSE=2.271`、`max_abs=25.14`、metadata/original `0.108%`。
- `qg256_o0p001`：`RMSE=2.256`、`max_abs=24.66`、metadata/original `0.117%`。

也就是说，保存 `0.05%~0.1%` 最坏 residual 对 RMSE 只有小幅提升，但能显著压低
最大量化误差，元数据开销仍低于 qg128 baseline。下一步应跑图像级 PSNR/SSIM/LPIPS
验证它是否能减少多轮编辑的质量漂移。

这些脚本都会激活 `.venv`。参数直接在脚本顶部修改，不依赖外部环境变量传参。

## 压缩数据如何流动

round 0 或 cache step 正常计算激活后，cache manager 会把激活收集成 GOP：

1. FP16 activation reshape 成二维 `[tokens, hidden]`。
2. 量化成 uint8 帧；`lossless` 模式优先用
   `GWQuantization(groupsize=<compression_quant_group_size>)`，默认 256。
   group-wise 路径使用 rounded zero-point；channel-wise fallback 使用 min-offset。
   如果 `compression_quant_outlier_ratio > 0`，会改用
   `GWOutlierQuantization`，在量化后把最大 residual 的 flat index 和 FP16
   residual 作为辅助元数据保存。
3. `FixedTiling` 把帧切成 NVENC 可编码 tile。
4. `MonoNVEncodeSequence` 用 NVENC 编成 HEVC/H.264 bitstream；GOP 模式下后续层可以作为 P 帧。
5. 压缩结果作为 cache entry 存在内存里，不写独立视频文件。

压缩后的 cache entry 包含：

- `bitstream` / packet sizes 等 codec payload；
- `scale` / `offset` 等量化恢复元数据；
- 可选 `outlier_indices` / `outlier_residuals`；
- 原始 shape、dtype、device；
- GOP 的 `frame_index`、`group_layers`、`gop_length`、`frame_interval_p`；
- 报告字段里的 `codec`、`quantization`、`quant_group_size`、payload size、
  auxiliary size。

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

显存统计位于 `cuda_memory`，来自 `torch.cuda`，包括每张 GPU 的
`peak_allocated_gib` 和 `peak_reserved_gib`。峰值在 pipeline 初始化后重置，
因此用于比较推理和 cache/compression 过程中的峰值占用；它不包含 NVENC 外部资源的全部细节。

压缩统计位于 `compression.summary`，重点看：

- `success_count` / `failure_count`
- `success_count_by_mode`
- `success_count_by_quantization`
- `quant_outlier_ratio`
- `quant_group_size`
- `payload_compression_ratio`
- `total_compression_ratio`
- `compressed_payload_mib`
- `compressed_auxiliary_mib`
- `compressed_total_mib`
- `decompression_failure_count`
- `gop_decode_cache_hit_count`
- `gop_decode_cache_miss_count`
- `quant_error_probe_by_quantization`，仅在启用
  `--compression-quant-error-probe-groups` 时出现

`payload_compression_ratio` 只统计 codec bitstream；`total_compression_ratio` 会把量化元数据、
packet sizes 等辅助数据也算进去，更接近真实 cache 占用。

`scripts/summarize_compression_sweep.py` 会把 `max_cuda_peak_allocated_gib` 和
`max_cuda_peak_reserved_gib` 写入汇总表。可用
`--max-peak-reserved-gib` 给推荐配置增加显存上限门槛。

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
