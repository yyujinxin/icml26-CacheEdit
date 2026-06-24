#!/usr/bin/env python3
"""Build the local cache_edit.compression.ops CUDA/NVENC extension in place."""

from __future__ import annotations

import argparse
from pathlib import Path

import setuptools
from torch.utils.cpp_extension import BuildExtension, CUDAExtension


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--llm265-root",
        default="/home/yujinxin/llm.265",
        help="Path containing vendor/Video_Codec_SDK_12.1.14.",
    )
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[1]
    llm265_root = Path(args.llm265_root).resolve()
    sdk_root = llm265_root / "vendor" / "Video_Codec_SDK_12.1.14"

    sources = [
        repo_root / "cache_edit" / "compression" / "csrc" / "bind.cpp",
        repo_root / "cache_edit" / "compression" / "csrc" / "encode.cpp",
        repo_root / "cache_edit" / "compression" / "csrc" / "decode.cpp",
        sdk_root / "Samples" / "NvCodec" / "NvEncoder" / "NvEncoder.cpp",
        sdk_root / "Samples" / "NvCodec" / "NvEncoder" / "NvEncoderCuda.cpp",
        sdk_root
        / "Samples"
        / "NvCodec"
        / "NvEncoder"
        / "NvEncoderOutputInVidMemCuda.cpp",
        sdk_root / "Samples" / "NvCodec" / "NvDecoder" / "NvDecoder.cpp",
    ]
    include_dirs = [
        repo_root / "cache_edit" / "compression" / "csrc",
        sdk_root / "Samples" / "Utils",
        sdk_root / "Samples" / "NvCodec",
        sdk_root / "Interface",
        Path("/usr/local/cuda/include"),
    ]
    library_dirs = [
        sdk_root / "Lib" / "linux" / "stubs" / "x86_64",
        Path("/usr/local/cuda/lib64/stubs"),
        Path("/usr/lib/x86_64-linux-gnu"),
    ]

    extension = CUDAExtension(
        "cache_edit.compression.ops",
        [str(path) for path in sources],
        include_dirs=[str(path) for path in include_dirs],
        library_dirs=[str(path) for path in library_dirs],
        libraries=["nvcuvid", "nvidia-encode", "cuda"],
        extra_compile_args=["-std=c++17", "-w"],
    )

    setuptools.setup(
        name="cacheedit_ops_build",
        ext_modules=[extension],
        cmdclass={"build_ext": BuildExtension},
        script_args=["build_ext", "--inplace"],
    )


if __name__ == "__main__":
    main()
