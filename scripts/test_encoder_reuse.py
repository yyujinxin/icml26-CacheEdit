#!/usr/bin/env python3
"""Test encoder/decoder reuse in MonoNVEncode."""

import torch
from cache_edit.compression.activation_compressor import ActivationCompressor, ActivationDecompressor

print("Testing Encoder Reuse Fix")
print("=" * 60)

# Create compressor
compressor = ActivationCompressor(
    bitrate=5.0,
    codec="hevc",
    max_cached_pipelines=1,  # Only 1 pipeline to stress test
)

decompressor = ActivationDecompressor(max_cached_pipelines=1)

print(f"\nCompressor created with max_cached_pipelines=1")

# Simulate multiple layers with same shape (like Flux transformer blocks)
num_layers = 38  # Same as Flux
shape = (1, 10000, 3072)  # Typical Flux activation size

print(f"\nCompressing {num_layers} layers with shape {shape}")
print("This simulates all 38 Flux transformer layers...\n")

compressed_list = []
for layer_idx in range(num_layers):
    test_tensor = torch.randn(*shape, dtype=torch.float16, device='cuda')

    try:
        compressed = compressor.compress(test_tensor, name=f"layer{layer_idx}")
        compressed_list.append(compressed)

        if layer_idx % 5 == 0:
            print(f"✓ Layer {layer_idx:2d}: Compressed successfully ({compressed['code_size'] / 1024 / 1024:.2f} MB)")
    except Exception as e:
        print(f"✗ Layer {layer_idx:2d}: FAILED - {e}")
        break

if len(compressed_list) == num_layers:
    print(f"\n{'='*60}")
    print(f"✅ SUCCESS! All {num_layers} layers compressed!")
    print(f"{'='*60}")

    # Test decompression
    print(f"\nTesting decompression of first layer...")
    try:
        recovered = decompressor.decompress(compressed_list[0])
        print(f"✓ Decompression successful: shape {recovered.shape}")
    except Exception as e:
        print(f"✗ Decompression failed: {e}")
else:
    print(f"\n{'='*60}")
    print(f"⚠️  Only {len(compressed_list)}/{num_layers} layers succeeded")
    print(f"{'='*60}")
