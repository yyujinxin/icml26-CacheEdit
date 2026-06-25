#!/bin/bash
# Extreme conservative test to avoid OOM completely

set -e

# Activate the conda environment (single RTX PRO 6000 setup).
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate cacheedit

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

echo "=========================================="
echo "Testing OOM Fix - EXTREME Conservative"
echo "=========================================="
echo ""
echo "Configuration:"
echo "  - Cache interval: 5 (cache steps: 0, 5, 10, 15, 20, 25)"
echo "  - GPU memory limit: 16GB"
echo "  - GPU memory buffer: 5GB"
echo "  - Inference steps: 28"
echo "  - Compression: lossless codec (bitrate ignored)"
echo ""

python scripts/run_flux_multi_gpu_optimized.py \
    --model-path /home/yujinxin/model/black-forest-labs/FLUX___1-Kontext-dev \
    --data-root /home/yujinxin/dataset/test \
    --output-dir ./outputs/flux_full_round_nan_fix_report_ratio \
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
    --num-inference-steps 2 \
    --cache-interval 5 \
    --guidance-scale 3.5 \
    --threshold 0.97 \
    --seed 42

echo ""
echo "=========================================="
echo "Test completed!"
echo "=========================================="
