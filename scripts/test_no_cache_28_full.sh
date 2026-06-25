#!/bin/bash
# Full 28-step Flux baseline test without activation cache or compression.

set -euo pipefail

# Activate the conda environment (single RTX PRO 6000 setup).
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate cacheedit

# Edit parameters in this block directly.

# PyTorch CUDA allocator setting. expandable_segments:True reduces
# fragmentation; the no-cache baseline still needs activation headroom.
PYTORCH_CUDA_ALLOC_CONF_VALUE="expandable_segments:True"

# Local FLUX-Kontext model directory.
MODEL_PATH="/home/yujinxin/model/black-forest-labs/FLUX___1-Kontext-dev"

# Dataset root. The runner reads metadata and input images from this directory.
DATA_ROOT="/home/yujinxin/dataset/test"

# Output directory for generated images and timings.json.
OUTPUT_DIR="./outputs/flux_28step_no_cache"

# Image id from the dataset metadata to run. 0000 runs the first configured case.
IMAGE_IDX="0000"

# Number of GPUs. Single RTX PRO 6000 (96GB): keep this at 1.
NUM_GPUS="1"

# Soft GPU memory limit in GiB used by the offload/cache manager on the 96GB card.
GPU_MEMORY_LIMIT_GB="90.0"

# Extra GiB kept free as a safety buffer before placing tensors on the GPU.
GPU_MEMORY_BUFFER_GB="6.0"

# Number of denoising steps. 28 is the full baseline setting.
NUM_INFERENCE_STEPS="28"

# FLUX guidance scale. Keep aligned with the GOP test for fair comparison.
GUIDANCE_SCALE="3.5"

# Cache threshold is not used when cache is disabled, so this script does not
# pass --threshold.

# Random seed for deterministic generation when the rest of the environment is stable.
SEED="42"

export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF_VALUE}"

echo "=========================================="
echo "Full 28-step baseline test"
echo "=========================================="
echo ""
echo "Configuration:"
echo "  - Cache: disabled"
echo "  - Compression: disabled"
echo "  - Image idx: ${IMAGE_IDX}"
echo "  - Output dir: ${OUTPUT_DIR}"
echo "  - Inference steps: ${NUM_INFERENCE_STEPS}"
echo "  - Guidance scale: ${GUIDANCE_SCALE}"
echo "  - GPUs: ${NUM_GPUS}, memory limit=${GPU_MEMORY_LIMIT_GB}GB, buffer=${GPU_MEMORY_BUFFER_GB}GB"
echo "  - CUDA allocator: ${PYTORCH_CUDA_ALLOC_CONF}"
echo ""

python scripts/run_flux_multi_gpu_optimized.py \
    --model-path "${MODEL_PATH}" \
    --data-root "${DATA_ROOT}" \
    --output-dir "${OUTPUT_DIR}" \
    --image-idx "${IMAGE_IDX}" \
    --num-gpus "${NUM_GPUS}" \
    --gpu-memory-limit-gb "${GPU_MEMORY_LIMIT_GB}" \
    --gpu-memory-buffer-gb "${GPU_MEMORY_BUFFER_GB}" \
    --num-inference-steps "${NUM_INFERENCE_STEPS}" \
    --guidance-scale "${GUIDANCE_SCALE}" \
    --seed "${SEED}"

echo ""
echo "=========================================="
echo "Full baseline test completed"
echo "Report: ${OUTPUT_DIR}/timings.json"
echo "Images: ${OUTPUT_DIR}/generation"
echo "=========================================="
