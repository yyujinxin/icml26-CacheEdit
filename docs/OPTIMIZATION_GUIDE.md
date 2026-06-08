# CacheEdit FLUX Cache Compression Optimization Guide

本文档记录当前项目中已经完成的一系列优化方法，重点面向
`scripts/run_flux_multi_gpu_optimized.py` 和 FLUX 多轮图像编辑缓存路径。

目标是：在不手动限制输入图像尺寸的前提下，降低多轮编辑中的显存占用，
提高缓存激活值压缩率，并尽量隐藏压缩、解压和数据加载带来的延迟。

## 约束和基线

### 固定约束

- 不在脚本中显式指定 `width` / `height`。
- 输入图像尺寸由 pipeline 内部 `_auto_resize=True` 路径处理。
- 当前 cache/GOP 完整测试使用 `cache_interval=5`。
- 28-step 测试的 cache anchor step 为 `0, 5, 10, 15, 20, 25`。

### 主要测试脚本

带 cache、压缩、GOP 的完整 28-step 测试：

```bash
cd /home/yujinxin/icml26-CacheEdit
bash scripts/test_gop28_full.sh
```

不 cache、不压缩的完整 28-step baseline：

```bash
cd /home/yujinxin/icml26-CacheEdit
bash scripts/test_no_cache_28_full.sh
```

两个脚本都会自动激活 `.venv`。脚本中的参数直接在文件内修改，不依赖外部环境变量传参。

## 优化 1：避免 OOM

### 问题

FLUX 多轮编辑在 28-step、较大输入图像、多 GPU offload 和 cache 同时启用时，
显存压力主要来自：

- transformer 激活值缓存体积大；
- cache 跨 round 保留，峰值显存随轮次增加；
- 压缩前后存在原始激活、量化 buffer、视频 buffer 和解码结果的短暂共存；
- PyTorch CUDA allocator 碎片化会放大峰值显存问题。

### 方法

1. cache 存储优先放到 CPU，避免把所有历史激活长期留在 GPU。
2. 多 GPU runner 使用软显存限制和 buffer：
   - `--gpu-memory-limit-gb`
   - `--gpu-memory-buffer-gb`
3. 启用 PyTorch expandable segments：
   - `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`
4. 压缩后尽快释放未压缩激活引用。
5. cache 复用时只把当前需要的激活搬到目标 device。

### 当前脚本参数

`scripts/test_gop28_full.sh` 当前使用：

```bash
NUM_GPUS="4"
GPU_MEMORY_LIMIT_GB="16.0"
GPU_MEMORY_BUFFER_GB="5.0"
CACHE_INTERVAL="5"
NUM_INFERENCE_STEPS="28"
```

这些参数偏保守，目的是让完整 28-step 多轮测试稳定完成。

## 优化 2：压缩精度修复，解决黑图和 NaN

### 问题

之前出现过 round 0 之后生成纯黑图，并伴随：

```text
RuntimeWarning: invalid value encountered in cast
images = (images * 255).round().astype("uint8")
```

这说明 VAE decode 后的图像张量里出现了 NaN 或 Inf。根因在 cache 激活经过
视频压缩/解压后数值重建不稳定，导致后续 transformer 或 VAE 输出异常。

### 方法

压缩激活时使用更稳健的量化和元数据记录：

- 对激活按 FP32 统计 `min` / `scale`；
- 视频编码前将浮点激活映射到可编码的整数图像表示；
- 解码后使用压缩元数据恢复到原始浮点范围；
- 对解压结果做有限值检查，避免 NaN/Inf 继续传播；
- 保留原始 shape、dtype、device 等信息，解压后按目标 device 恢复。

### 效果

修复后，后续 round 生成图像不再是纯黑图。曾完成过 28-step GOP 测试，8 个 round
输出均为 finite、非黑图。

## 优化 3：压缩统计报告

### 目的

只看运行时间无法判断压缩是否真正有效。因此在输出报告中加入压缩统计，用于观察：

- 原始激活大小；
- 压缩 payload 大小；
- 加上元数据后的总大小；
- 压缩率；
- 压缩/解压成功失败次数；
- 每次压缩/解压的耗时；
- GOP、prefetch、decoded-cache 的命中情况。

### 输出位置

运行脚本后，报告写入：

```text
<OUTPUT_DIR>/timings.json
```

例如：

```text
outputs/flux_28step_gop16_P16/timings.json
outputs/flux_28step_no_cache/timings.json
```

### 关键字段

`timings.json` 中的 `compression` 字段包含压缩摘要。重点关注：

- `enabled`
- `configured_gop_length`
- `compression_count`
- `compression_success_count`
- `compression_failure_count`
- `decompression_count`
- `decompression_success_count`
- `decompression_failure_count`
- `payload_compression_ratio`
- `total_compression_ratio`
- `total_compression_time_s`
- `total_decompression_time_s`
- `gop_prefetch_count`
- `gop_prefetch_success_count`
- `gop_prefetch_failure_count`
- `total_gop_prefetch_time_s`
- `total_gop_prefetch_wait_s`

这些字段由 `cache_edit/models/flux/cache_manager.py` 的
`get_compression_report()` 汇总，并由 `scripts/run_flux_multi_gpu_optimized.py`
写入最终 `timings.json`。

## 优化 4：帧间 GOP 压缩

### 背景

原始逐层压缩相当于每个 layer 单独作为一帧编码，层间相关性没有被利用。
但 transformer 连续 layer 的激活通常存在结构相似性，可以把连续 layer 看成视频中的连续帧，
用视频编码器的帧间预测提高压缩率。

### 策略

当前策略是 inter-layer GOP：

- 同一个 diffusion step 内，按连续 layer 组织 GOP；
- 多个连续 layer 的 activation 作为同一个视频序列的帧；
- key frame 保存 GOP 内第一层；
- 后续层使用视频编码器的帧间预测；
- GOP 解码时一次解码整个序列，再按 frame index 取对应 layer。

### 参数

`scripts/test_gop28_full.sh` 中：

```bash
COMPRESSION_GOP_LENGTH="16"
COMPRESSION_FRAME_INTERVAL_P="16"
COMPRESSION_CODEC="hevc"
COMPRESSION_BITRATE="5.0"
```

含义：

- `COMPRESSION_GOP_LENGTH`：一个 GOP 中最多包含多少个连续 layer。
- `COMPRESSION_FRAME_INTERVAL_P`：P frame 间隔参数，控制编码器如何放置预测帧。
- `COMPRESSION_CODEC`：当前推荐 `hevc`。
- `COMPRESSION_BITRATE`：码率越高，误差越小，但压缩率越低。

### 为什么当前使用 GOP 16

之前对 GOP 参数做过探索：

- GOP 太短：稳定，但层间压缩收益不足；
- GOP 太长：压缩率可能更高，但重建误差和解码代价上升；
- GOP 16 在当前 28-step、cache_interval=5、HEVC 5 Mbps 设置下，压缩率和稳定性更均衡。

该选择不是理论最优，只是当前硬件和样例任务上的经验配置。若换模型、数据或码率，
仍建议重新用同一 baseline 对比。

### 压缩数据存放方式

压缩后的激活仍保存在 cache manager 的 cache entry 中，不写成独立视频文件。
每个 compressed entry 包含：

- 编码后的二进制 payload；
- 原始 tensor shape；
- 原始 dtype；
- 原始 device 信息；
- 量化恢复需要的 min/scale 等元数据；
- GOP 元数据，例如 group id、frame index、layer 范围。

后续 round 复用 cache 时，cache manager 从上一轮保存的 cache entry 中取出 compressed data，
调用 decompressor 解码，并恢复成当前 layer 需要的 activation tensor。

## 优化 5：NVENC 注册失败和日志清理

### RegisterResource error 10

曾出现过如下错误：

```text
RegisterResource : m_nvenc.nvEncRegisterResource(...) returned error 10
```

这类错误通常发生在 NVENC 注册资源失败时，常见诱因包括：

- 同时创建或销毁过多 NVENC session；
- 压缩请求过密，资源未及时释放；
- GPU memory/resource pressure 较高；
- 编码器输入 buffer 生命周期管理不稳定。

当前优化通过减少重复解码、减少临时资源峰值、GOP 批量编码和更保守的内存设置来降低触发概率。

### Session 日志清理

NVENC native 层会打印：

```text
Session Initialization Time: ...
Session Deinitialization Time: ...
```

这些打印不影响结果，但会污染测试日志。当前在 `cache_edit/compression/pipeline/nvenc.py`
中通过 stdout suppress 包住 native encode/decode 调用，避免这些信息刷屏。

## 优化 6：Nsight Systems 延迟瓶颈分析

### 观察结果

Nsight Systems 分析显示，GOP 压缩路径的主要瓶颈不是 GPU kernel 本身，而是：

- `cuMemcpyDtoH_v2`
- `cudaMemcpyAsync`
- stream synchronize
- 同步解码带来的等待

这说明瓶颈集中在设备到主机拷贝、视频解码和同步等待上。

### 对比数据

短测 2 round / 6 step 的结果曾观察到：

| 配置 | round times | 平均 round time |
| --- | --- | --- |
| GOP，无解码缓存 | `[21.05, 67.31]` | `44.18s` |
| 不压缩 cache | `[13.45, 10.38]` | `11.92s` |
| GOP，decoded cache | `[21.20, 29.38]` | `25.29s` |
| GOP，prefetch + decoded cache | `[21.18, 26.47]` | `23.83s` |

从数据看，decoded cache 和 prefetch 能显著降低后续 round 的解压等待，但压缩路径仍比
不压缩 cache 慢。它的价值主要在显存/内存压力受限时换取可运行性和更低缓存体积。

## 优化 7：GOP 解码缓存

### 问题

GOP 解码的天然代价是：即使只需要某个 layer，也通常需要解码整个 GOP 序列。
如果每个 layer 都单独触发一次 `decompress_sequence_frame()`，会反复解码同一个 GOP。

### 方法

增加 decoded GOP cache：

- 第一次需要某个 GOP 中的任意 frame 时，解码整个 GOP；
- 将整个 GOP 的 decoded frames 暂存在 cache manager 内部；
- 后续连续 layer 直接从 decoded cache 取 frame；
- 使用小型 LRU，避免 decoded frames 长期占用大量内存。

### 实现位置

核心逻辑在 `cache_edit/models/flux/cache_manager.py`：

- `_decoded_gop_cache`
- `_decoded_gop_access_order`
- `_decoded_gop_max_entries`
- `_get_decoded_gop_frame()`
- `_install_decoded_gop_frames()`
- `_take_decoded_gop_cache()`

### 效果

短测中，GOP 解码总耗时从约 `52.21s` 降到约 `10.17s`，平均解压耗时从约
`0.227s` 降到约 `0.044s`。

## 优化 8：提前解压和 overlap

### 问题

仅有 decoded cache 仍然是在真正需要 cache 复用时才解码。这样 transformer 执行到该 layer 时，
如果 decoded cache 没有准备好，就会同步等待。

### 方法

增加 GOP prefetch：

- 在 step 开始时，根据即将复用的 cache entry 构建 prefetch plan；
- 后台线程提前解码 GOP；
- 解码结果放入 decoded GOP cache；
- 前台真正需要某个 layer 时，优先从 decoded cache 或 prefetch future 取结果；
- 如果 prefetch 还没完成，只等待对应 future；
- 每消费一个 prefetch 结果后，继续调度后面的 GOP，形成滑动窗口。

### 调度策略

当前策略：

- prefetch window 为 2；
- 使用单 worker 后台线程，避免并发 NVDEC/NVENC 资源压力过大；
- step 内按 stream 和 layer 顺序构建计划；
- 对普通 cache 复用 plan 按 compressed data id 去重；
- 对 key-token ref 需要的 late layer 单独插入 plan；
- decoded cache 默认最多保留 2 个 GOP，控制内存占用。

### 读取优先级

`_get_decoded_gop_frame()` 的读取顺序：

1. 目标 device 上的 decoded cache；
2. canonical original-device decoded cache，再 `.to(target_device)`；
3. 已提交的 prefetch future；
4. 同步 fallback 解码。

报告中的 `decompression_source_counts` 可用于确认是否仍有大量同步解码。
理想情况下，`sync_decode` 应接近 0。

### 效果

短测中，prefetch + decoded cache 后：

- `gop_prefetch_count=34`
- `gop_prefetch_success_count=34`
- `gop_prefetch_failure_count=0`
- `total_gop_prefetch_wait_s` 约为 `0.0002s`
- `sync_decode` 基本被消除
- round 平均时间进一步从约 `25.29s` 降到约 `23.83s`

这说明解码已经大部分被提前准备，前台等待被明显隐藏。

## 当前推荐参数

用于完整 28-step cache/GOP 测试：

```bash
CACHE_INTERVAL="5"
NUM_INFERENCE_STEPS="28"
GUIDANCE_SCALE="3.5"
THRESHOLD="0.97"
COMPRESSION_CODEC="hevc"
COMPRESSION_BITRATE="5.0"
COMPRESSION_GOP_LENGTH="16"
COMPRESSION_FRAME_INTERVAL_P="16"
NUM_GPUS="4"
GPU_MEMORY_LIMIT_GB="16.0"
GPU_MEMORY_BUFFER_GB="5.0"
```

不要在测试命令中添加 `--width` 或 `--height`。输入图像尺寸应由 pipeline 的
`_auto_resize` 路径处理。

## 如何评估一次新优化

建议每次只改一个变量，然后对比以下指标：

1. 是否完成完整 28-step 多轮测试。
2. 生成图是否 finite、非纯黑。
3. `timings.json` 中 round time 是否下降。
4. `compression.payload_compression_ratio` 和 `compression.total_compression_ratio` 是否提高。
5. `compression.decompression_failure_count` 是否为 0。
6. `compression.gop_prefetch_failure_count` 是否为 0。
7. `compression.total_gop_prefetch_wait_s` 是否接近 0。
8. `decompression_source_counts` 中是否还有大量 `sync_decode`。

如果压缩率提升但图像质量明显下降，优先提高 `COMPRESSION_BITRATE` 或缩短
`COMPRESSION_GOP_LENGTH`。

如果速度下降但压缩率很好，优先检查：

- GOP 解码是否重复发生；
- prefetch 是否命中；
- 是否有大量 host/device copy；
- decoded GOP cache 是否太小；
- NVENC/NVDEC session 是否过于频繁创建。

## 已知取舍

- 不压缩 cache 通常最快，但占用更高，容易在大图和多轮任务中触发 OOM。
- GOP 压缩显著降低 cache 体积，但带来编码/解码和数据搬运开销。
- decoded cache 会增加短期内存占用，但能避免重复解码。
- prefetch 可以隐藏解码等待，但无法消除后台解码本身的硬件开销。
- 更大的 GOP 通常提高压缩率，但可能增加误差和单次解码成本。

当前实现偏向“完整 28-step、多轮、大图可稳定跑完”，而不是追求单轮极限速度。
