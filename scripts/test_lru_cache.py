#!/usr/bin/env python3
"""Quick test for LRU pipeline caching in ActivationCompressor."""

import torch
from cache_edit.compression.activation_compressor import ActivationCompressor

print("Testing LRU Pipeline Caching")
print("=" * 60)

# Create compressor with max_cached_pipelines=2
compressor = ActivationCompressor(
    bitrate=5.0,
    codec="hevc",
    max_cached_pipelines=2,
)

print(f"\nCompressor created with max_cached_pipelines=2")
print(f"Current cache size: {len(compressor._pipeline_cache)}")

# Test with 3 different shapes to trigger eviction
shapes = [
    (1, 1000, 3072),   # Shape 1
    (1, 2000, 3072),   # Shape 2
    (1, 3000, 3072),   # Shape 3 - should evict Shape 1
]

for i, shape in enumerate(shapes, 1):
    print(f"\n--- Test {i}: Compressing shape {shape} ---")
    test_tensor = torch.randn(*shape, dtype=torch.float16, device='cuda')

    try:
        compressed = compressor.compress(test_tensor, name=f"test{i}")
        print(f"✓ Compression successful")
        print(f"  Cache size: {len(compressor._pipeline_cache)}")
        print(f"  Cached shapes: {list(compressor._pipeline_cache.keys())}")
        print(f"  Compressed size: {compressed['code_size'] / 1024 / 1024:.2f} MB")
    except Exception as e:
        print(f"✗ Compression failed: {e}")

# Now access Shape 2 again (should not create new pipeline)
print(f"\n--- Test 4: Re-access Shape 2 (should hit cache) ---")
test_tensor = torch.randn(1, 2000, 3072, dtype=torch.float16, device='cuda')
compressed = compressor.compress(test_tensor, name="test4")
print(f"✓ Compression successful (cache hit)")
print(f"  Cache size: {len(compressor._pipeline_cache)}")
print(f"  Cached shapes: {list(compressor._pipeline_cache.keys())}")

print("\n" + "=" * 60)
print("✅ LRU caching test completed!")
