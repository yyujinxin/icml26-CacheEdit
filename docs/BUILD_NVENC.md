# 构建 NVENC 压缩扩展

本文档说明如何在单卡 RTX PRO 6000（`a6000pro` 分支）环境下，从零构建
activation 压缩所需的原生扩展 `cache_edit.compression.ops`（NVENC/NVDEC 绑定）。

不构建此扩展时，no-cache 与 cache-only 路径仍可正常运行；只有
`--use-cache-compression` 压缩路径需要它。若扩展缺失，`FluxCacheManager`
会打印 "NVENC ops not available" 并自动回退到非压缩缓存。

## 背景

`cache_edit/compression/` 只包含薄封装：

- `csrc/`：`bind.cpp` / `encode.cpp` / `decode.cpp` / `nvcodec.h`，PyTorch 绑定。
- `pipeline/`：量化、切片、NVENC 序列编码的 Python 包装。

真正的 NVENC/NVDEC 实现来自外部的 **NVIDIA Video Codec SDK**，构建时需要把 SDK
里的 `NvEncoder.cpp` / `NvEncoderCuda.cpp` / `NvEncoderOutputInVidMemCuda.cpp` /
`NvDecoder.cpp` 一起编译进来。

## 前置条件

1. **Video Codec SDK** 放在构建脚本的默认路径：

   ```text
   /home/yujinxin/llm.265/vendor/Video_Codec_SDK_12.1.14
   ```

   该目录需含 `Interface/`（nvEncodeAPI.h、cuviddec.h、nvcuvid.h）和
   `Samples/NvCodec/{NvEncoder,NvDecoder}` 的头文件与 `.cpp` 源码。如果 SDK 在
   别处，构建时用 `--llm265-root <包含 vendor/ 的目录>` 指定。

2. **驱动运行库**（通常随 NVIDIA 驱动安装，已在系统）：

   ```text
   /usr/lib/x86_64-linux-gnu/libnvcuvid.so
   /usr/lib/x86_64-linux-gnu/libnvidia-encode.so
   /usr/lib/x86_64-linux-gnu/libcuda.so
   ```

   用 `ldconfig -p | grep -iE 'nvcuvid|nvidia-encode|libcuda'` 确认。

3. 本机**没有系统 CUDA toolkit / nvcc**（无 `/usr/local/cuda`）。下面的步骤用
   conda env `cacheedit` 里 pip 安装的 cu13 组件来提供 nvcc 和 CUDA_HOME。

## 构建步骤

所有命令在仓库根目录、conda env `cacheedit` 下执行。

### 1. 安装匹配的 nvcc

PyTorch 是 cu130 构建，需要 CUDA 13.x 的 nvcc。镜像源只有占位 stub，必须用
NVIDIA 官方 index：

```bash
pip install nvidia-cuda-nvcc==13.0.88 --extra-index-url https://pypi.nvidia.com
```

注意包名是无后缀的 `nvidia-cuda-nvcc`（不是 `-cu13`）。安装后 nvcc 位于：

```text
<env>/lib/python3.10/site-packages/nvidia/cu13/bin/nvcc
```

### 2. 安装 matplotlib

`cache_edit/compression/pipeline/__init__.py` 会 import `debug.py`，后者依赖
matplotlib。缺它会导致 `activation_compressor` 的导入失败、`NVENC_AVAILABLE`
变成 `False`（报错信息是 "No module named 'matplotlib'"）：

```bash
pip install matplotlib
```

### 3. 创建 libcudart 软链

torch 的 `CUDAExtension` 链接 `-lcudart`，但 pip 只装了带版本号的
`libcudart.so.13`，需要补一个无版本号软链：

```bash
CU13=<env>/lib/python3.10/site-packages/nvidia/cu13
ln -s libcudart.so.13 "$CU13/lib/libcudart.so"
```

### 4. 编译扩展

```bash
CU13=<env>/lib/python3.10/site-packages/nvidia/cu13
CUDA_HOME=$CU13 \
LIBRARY_PATH=$CU13/lib:/usr/lib/x86_64-linux-gnu \
python scripts/build_cacheedit_ops.py
```

`csrc/` 和 SDK 源文件全是 `.cpp`（host 代码，用 g++ 编译），所以即使 pip 的
`nvidia-nvvm` 版本与 ptxas 不完全匹配也不影响构建——不会触发设备代码编译。

成功后会在 `cache_edit/compression/` 下生成
`ops.cpython-310-x86_64-linux-gnu.so`。

## 验证

运行时无需额外环境变量：先 `import torch` 会把 cu13 运行库加载进进程，驱动库在
标准系统路径，扩展即可导入。

```bash
python -c "
from cache_edit.compression.activation_compressor import NVENC_AVAILABLE
print('NVENC_AVAILABLE:', NVENC_AVAILABLE)
"
```

应输出 `NVENC_AVAILABLE: True`。端到端验证可直接跑压缩脚本：

```bash
bash scripts/test_gop28_full.sh
```

完成后查看 `<output-dir>/timings.json` 的 `compression.summary`，确认
`failure_count` 和 `decompression_failure_count` 为 0。

## 注意事项

- 压缩是用**时间换显存**：NVENC 编码 / NVDEC 解码本身有开销，复用轮会明显变慢。
  在 96GB 单卡上若只追求速度，用 `--cache-device cuda:0`（默认 auto 即如此）不开
  压缩更快；压缩适合显存吃紧的场景。
- lossless 路径要求 hidden 维能被 64 整除（走 `GWQuantization(groupsize=64)`）。
  FLUX 的 hidden=3072 满足此条件。
- 编译产物 `ops*.so` 已被 `.gitignore` 忽略，不进代码库；换机器需按本文重建。
