#!/usr/bin/env python3
"""Minimal compression round-trip profile script for bottleneck analysis.

Simulates the realistic Flux activation compression/decompression path:
- 19 transformer layers per diffusion step (Flux double-stream architecture)
- 6 cache steps (0, 5, 10, 15, 20, 25) in a 28-step run
- GOP=16 inter-layer compression
- Realistic activation shape: [1, 4096, 3072] (batch, tokens, hidden)
"""

import torch
import torch.cuda.profiler as profiler
import torch.cuda.nvtx as nvtx

from cache_edit.compression.activation_compressor import (
    ActivationCompressor,
    ActivationDecompressor,
)


def main():
    device = torch.device("cuda:0")

    # Mimic Flux Kontext: 19 double-stream transformer layers, 6 cache steps
    num_layers = 19
    num_cache_steps = 6

    # Realistic activation shape from Flux: [batch, seq_len, hidden]
    # hidden=3072 is divisible by 64, so lossless uses GWQuantization
    activation_shape = (1, 4096, 3072)

    print(f"Simulating compression bottleneck profile:")
    print(f"  layers per step: {num_layers}")
    print(f"  cache steps: {num_cache_steps}")
    print(f"  activation shape: {activation_shape}")
    print(f"  dtype: float16, device: {device}")
    print()

    compressor = ActivationCompressor(codec="lossless")
    decompressor = ActivationDecompressor()

    # Warm-up: build the pipeline cache once
    nvtx.range_push("warmup")
    x = torch.randn(*activation_shape, dtype=torch.float16, device=device)
    activations = [x.clone() for _ in range(num_layers)]
    comp = compressor.compress_sequence(
        activations,
        gop_length=16,
        frame_interval_p=16,
        name="warmup",
    )
    rec = decompressor.decompress_sequence(comp)
    nvtx.range_pop()
    del x, activations, comp, rec
    torch.cuda.synchronize()

    print("Warmup complete, starting profiled run...")
    print()

    # Start CUDA profiling (for --capture-range=cudaProfilerApi)
    profiler.start()

    # Profile: simulate 6 cache steps, each caching 19 layers
    for step_idx in range(num_cache_steps):
        nvtx.range_push(f"cache_step_{step_idx}")

        # Generate activations (mimic transformer forward pass output)
        nvtx.range_push("generate_activations")
        activations = [
            torch.randn(*activation_shape, dtype=torch.float16, device=device)
            for _ in range(num_layers)
        ]
        nvtx.range_pop()

        # Compress the layer sequence (GOP encoding)
        nvtx.range_push("compress_sequence")
        compressed = compressor.compress_sequence(
            activations,
            gop_length=16,
            frame_interval_p=16,
            name=f"step{step_idx}",
        )
        nvtx.range_pop()

        # Decompress (happens on cache reuse in the next diffusion step)
        nvtx.range_push("decompress_sequence")
        recovered = decompressor.decompress_sequence(compressed)
        nvtx.range_pop()

        # Validation
        nvtx.range_push("validate")
        for orig, rec in zip(activations, recovered):
            assert orig.shape == rec.shape
            assert orig.dtype == rec.dtype
            assert orig.device == rec.device
        nvtx.range_pop()

        # Print stats AFTER decompress, not immediately after compress, to avoid
        # triggering implicit synchronization on the async D->H transfer.
        print(f"step {step_idx}: compressed {len(activations)} layers, code_size={compressed['code_size']//1024}KB")

        nvtx.range_pop()  # cache_step_{step_idx}

    profiler.stop()

    print()
    print("Profile complete. Check the .nsys-rep file with nsys-ui or nsys stats.")


if __name__ == "__main__":
    main()
