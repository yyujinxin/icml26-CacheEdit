#!/bin/bash
# Test script with aggressive memory management to avoid OOM

set -e

# Activate the conda environment (single RTX PRO 6000 setup).
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate cacheedit

# Set memory optimization flags
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Use conservative parameters to avoid OOM:
# 1. Increased cache interval (10 instead of 5) - fewer cache steps
# 2. Increased memory buffer (4GB instead of 2GB)
# 3. Lower memory limit per GPU (18GB instead of 22GB) to leave more headroom
# 4. Reduced inference steps (28 instead of 110) for testing

echo "=========================================="
echo "Testing OOM Fix with Conservative Settings"
echo "=========================================="
echo ""
echo "Configuration:"
echo "  - Cache interval: 10 (cache steps: 0, 10, 20)"
echo "  - GPU memory limit: 18GB"
echo "  - GPU memory buffer: 4GB"
echo "  - Inference steps: 28"
echo "  - Compression: lossless codec (bitrate ignored)"
echo ""

python scripts/run_flux_multi_gpu_optimized.py \
    --model-path /home/yujinxin/model/black-forest-labs/FLUX___1-Kontext-dev \
    --data-root /home/yujinxin/dataset/test \
    --output-dir ./outputs/flux_oom_test \
    --image-idx "0000" \
    --num-gpus 1 \
    --gpu-memory-limit-gb 90.0 \
    --gpu-memory-buffer-gb 6.0 \
    --use-cache \
    --use-cache-compression \
    --compression-bitrate 5.0 \
    --compression-codec lossless \
    --compression-gop-length 16 \
    --compression-frame-interval-p 16 \
    --num-inference-steps 28 \
    --cache-interval 10 \
    --guidance-scale 3.5 \
    --threshold 0.97 \
    --seed 42

echo ""
echo "=========================================="
echo "Test completed successfully!"
echo "=========================================="
