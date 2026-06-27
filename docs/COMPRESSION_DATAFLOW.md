# 压缩模式数据流分析

本文档详细说明 CacheEdit activation 压缩的完整数据流、NVENC/NVDEC API 限制、
以及为何 bitstream 必须经过 CPU 的技术原因。

## 完整数据流

### Encode 路径（压缩）

```
1. FP16 activation tensor (GPU, torch-managed)
   ↓ quantize (GWQuantization for lossless, groupsize=64)
   
2. uint8 quantized frames (GPU, torch-managed)
   ↓ NvEncoderCuda::CopyToDeviceFrame (csrc/encode.cpp)
   
3. NVENC input buffer (GPU video memory, encoder-managed)
   ↓ nv_encoder->EncodeFrame()
   
4. Compressed bitstream (GPU video memory, encoder output buffer)
   ↓ cuMemcpyDtoD (encode.cpp:290-295)
   
5. output_bitstream tensor (GPU, torch-managed)
   ↓ .copy_() with pinned_memory=True (encode.cpp:321-328)
   
6. bitstream_cpu + packet_sizes_cpu (CPU pinned memory)
   → stored in FluxCacheManager cache dict
```

**关键点**：
- 步骤 5→6 的 D→H 传输使用 pinned memory，比 pageable memory 快 2-3x
- packet_sizes 直接在 GPU 上构造（cuMemcpyHtoD），避免了 CPU→GPU 往返
- 输出直接存 CPU，为解码做准备

### Decode 路径（解压）

```
1. bitstream + packet_sizes (CPU pinned memory, from cache)
   ↓ nv_decoder->Decode(host_ptr, size) [decode.cpp:47]
   
2. cuvidParseVideoData (NVDEC parser, CPU侧)
   ↓ NVDEC 内部把 bitstream 从 host DMA 到 GPU
   
3. NVDEC internal processing (GPU)
   ↓ nv_decoder->GetFrame()
   
4. Decoded frame (GPU video memory, decoder-managed)
   ↓ cuMemcpyDtoD (decode.cpp:95-99)
   
5. output_tensor (GPU, torch-managed)
   ↓ dequantize (GWQuantization backward)
   
6. FP16 activation tensor (GPU, torch-managed)
```

**关键点**：
- 步骤 1→2：bitstream 已经在 CPU，无需我们显式拷贝
- 步骤 2→3：NVDEC 内部隐式做 H→D（我们无法干预）
- 解码输出直接在 GPU，无需额外传输

## NVDEC API 限制：为何 bitstream 必须经过 CPU

### API 设计

NVIDIA Video Codec SDK 的解码接口要求 bitstream 在 **host memory**：

```cpp
// NvDecoder.h
int Decode(const uint8_t *pData, int nSize, ...);

// nvcuvid.h
typedef struct _CUVIDSOURCEDATAPACKET {
    const unsigned char *payload;  // host pointer
    unsigned long payload_size;
    ...
} CUVIDSOURCEDATAPACKET;

CUresult cuvidParseVideoData(CUvideoparser obj, CUVIDSOURCEDATAPACKET *pPacket);
```

**没有任何标志位或替代接口支持 device memory 输入。**

### 为什么这样设计？

1. **Parser 在 CPU 侧**
   - Bitstream parsing（NAL unit 分割、header 解析、SPS/PPS 提取）是串行、
     分支密集的操作
   - CPU 更适合这类逻辑密集型工作
   - GPU 只负责并行的解码计算（IDCT、motion compensation、deblocking）

2. **历史架构**
   - SDK 诞生于 GPU 专用 video memory 时代（PCIe 隔离）
   - 典型工作流：Host 准备 bitstream → GPU 解码 → Host 消费结果
   - 虽然现在有 unified memory 和 NVLink，但 API 未更新

3. **DMA 效率**
   - NVDEC 内部用专用 DMA engine 把 host bitstream 搬到 GPU
   - 这个路径已高度优化，bypassing CPU cache
   - 用户层面干预反而可能引入开销

## 探索过的方案

### 方案 A：绕过 parser，直接用 decoder

**结论：不可行。**

`cuvidCreateDecoder` 创建的 decoder 只能通过 parser 喂数据，没有 "raw NAL 
unit" 接口。

### 方案 B：CUDA IPC 跨进程共享 GPU memory

**结论：无意义。**

即使用 `cudaIpcGetMemHandle` / `cudaIpcOpenMemHandle` 让 bitstream 在 GPU 间
共享，最终仍需 D→H 拷贝给 parser。

### 方案 C：NVENC output 直接连 NVDEC input

**结论：理论最优，实际不支持。**

NVENC 和 NVDEC 之间没有 "zero-copy" 路径。即使在同一 GPU 上，bitstream 仍需：
- 从 NVENC output buffer 拷到某个可寻址位置
- 喂给 parser（需要 host pointer）

## 性能分析与优化

### Profile 数据（6 cache steps，每步 19 层）

| 优化前（bitstream 在 GPU 存储） | 优化后（bitstream 在 CPU 存储） |
|-------------------------------|-------------------------------|
| Compress: 6.87s               | Compress: 5.10s (**1.35x**)   |
| Decompress: 8.99s             | Decompress: 7.13s (**1.26x**) |
| **Total: 15.86s**             | **Total: 12.23s (1.30x)**     |

### 瓶颈分解

优化前：
- `cuMemcpyDtoH`（D→H）：72% 时间（5.0s）
  - 456 次调用 = 6 steps × 76 次/step
  - 每次 decode 前都 `.to(CPU)`，重复传输
- `cudaMemcpyAsync`：24.8% 时间（1.7s）
  - packet_sizes 从 CPU→GPU 的冗余拷贝

优化后：
- `cuMemcpyDtoH`（D→H）：90.4% 时间（5.0s）
  - 占比升高，但**绝对值不变**（encode 输出的必要传输）
  - decode 输入无需显式拷贝（bitstream 已在 CPU）
- `cudaMemcpyAsync`：消失（0%）
  - packet_sizes 直接在 GPU 构造
- `cudaHostAlloc`：4.8% 时间（0.27s）
  - pinned memory 分配开销（12 次调用，分摊）

### 关键洞察

即使我们把 bitstream 存在 GPU，解码时 NVDEC 内部仍会做隐式的 H→D 传输。
所以 "bitstream 驻留 GPU" 省掉的只是：
- 我们 cache 字典里的 GPU→CPU 拷贝
- **但 NVDEC 仍需 CPU→GPU（它的内部实现，用户无法绕过）**

实际场景中，一个 cache entry 压缩 1 次、复用 10+ 次：
- 优化前：1 次 encode D→H + 10 次 decode D→H = **11 次传输**
- 优化后：1 次 encode D→H + 0 次显式拷贝 = **1 次传输**（NVDEC 内部的 H→D 
  无法避免，但它已高度优化）

## 进一步优化方向

### 1. ✅ 已实现：CPU 存储 + pinned memory
- 压缩后立即 D→H，存在 CPU pinned memory
- 解码时 bitstream 已在 host，无需我们显式拷贝
- **加速 1.30x**

### 2. 异步传输 overlap（待实现）

```cpp
// encode.cpp 改成异步
cudaStream_t stream;
cudaStreamCreate(&stream);
cudaMemcpyAsync(bitstream_cpu.data_ptr(), output_bitstream.data_ptr(),
                size, cudaMemcpyDeviceToHost, stream);
// 立即返回 TensorEncodeOutput，让 D→H 与下一层 transformer 计算 overlap
```

**收益**：隐藏 D→H 延迟，理论可再快 20-30%（需实测）

### 3. 自适应压缩策略

不是每个 cache step 都压缩：
- 监控 GPU 显存使用率
- 只有接近 `gpu_memory_limit_gb` 时才启用压缩
- 显存充裕时用未压缩缓存（更快）

**Trade-off**：复杂度 vs 性能

### 4. 替代压缩算法探索

NVENC/NVDEC 优势：
- 硬件加速，吞吐高
- 对视频内容优化（空间+时间相关性）

劣势：
- 对随机激活不是最优
- 必须经过 CPU（API 限制）

可能的替代：
- 专用 tensor 压缩 CUDA kernel（如 SZ3、cuSZ）
- 但吞吐可能不如 NVENC
- 需权衡：压缩比 vs 速度 vs 显存节省

## 相关文件

- `cache_edit/compression/csrc/encode.cpp`：NVENC 封装，包含 pinned memory 
  优化
- `cache_edit/compression/csrc/decode.cpp`：NVDEC 封装，bitstream 必须在 CPU 
  检查
- `cache_edit/compression/activation_compressor.py`：Python API，CPU 存储架构
- `cache_edit/compression/pipeline/nvenc.py`：编解码 pipeline step
- `cache_edit/models/flux/cache_manager.py`：缓存管理，GOP 分组
- `docs/BUILD_NVENC.md`：构建指南
- `README_COMPRESSION.md`：使用说明
