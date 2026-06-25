#!/bin/bash
# Test script for LLM.265 compression integration with CacheEdit

set -e

echo "=========================================="
echo "Testing CacheEdit with Compression"
echo "=========================================="

# Activate the conda environment (single RTX PRO 6000 setup).
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate cacheedit

# Test parameters
MODEL_PATH="/home/yujinxin/model/black-forest-labs/FLUX___1-Kontext-dev"
DATA_ROOT="/home/yujinxin/dataset/test"
OUTPUT_DIR="./outputs/flux_compression_test"

echo ""
echo "Test 1: Run without compression (baseline)"
echo "=========================================="
python scripts/run_flux_multi_gpu_optimized.py \
    --model-path "$MODEL_PATH" \
    --data-root "$DATA_ROOT" \
    --output-dir "${OUTPUT_DIR}_no_compression" \
    --use-cache \
    --num-inference-steps 28 \
    --cache-interval 5 \
    --guidance-scale 3.5 \
    --threshold 0.97 \
    --seed 42 \
    --gpu-memory-limit-gb 90 \
    --gpu-memory-buffer-gb 6

echo ""
echo "Test 2: Run with lossless codec compression"
echo "=========================================="
python scripts/run_flux_multi_gpu_optimized.py \
    --model-path "$MODEL_PATH" \
    --data-root "$DATA_ROOT" \
    --output-dir "${OUTPUT_DIR}_lossless_gop16" \
    --use-cache \
    --use-cache-compression \
    --compression-bitrate 5.0 \
    --compression-codec lossless \
    --compression-gop-length 16 \
    --compression-frame-interval-p 16 \
    --num-inference-steps 28 \
    --cache-interval 5 \
    --guidance-scale 3.5 \
    --threshold 0.97 \
    --seed 42 \
    --gpu-memory-limit-gb 90 \
    --gpu-memory-buffer-gb 6

echo ""
echo "Test 3: Run with lossy HEVC compression @ 5 Mbps"
echo "=========================================="
python scripts/run_flux_multi_gpu_optimized.py \
    --model-path "$MODEL_PATH" \
    --data-root "$DATA_ROOT" \
    --output-dir "${OUTPUT_DIR}_hevc_5mbps" \
    --use-cache \
    --use-cache-compression \
    --compression-bitrate 5.0 \
    --compression-codec hevc \
    --compression-gop-length 16 \
    --compression-frame-interval-p 16 \
    --num-inference-steps 28 \
    --cache-interval 5 \
    --guidance-scale 3.5 \
    --threshold 0.97 \
    --seed 42 \
    --gpu-memory-limit-gb 90 \
    --gpu-memory-buffer-gb 6

echo ""
echo "=========================================="
echo "All tests completed!"
echo "=========================================="
echo ""
echo "Compare results in:"
echo "  - ${OUTPUT_DIR}_no_compression"
echo "  - ${OUTPUT_DIR}_lossless_gop16"
echo "  - ${OUTPUT_DIR}_hevc_5mbps"
