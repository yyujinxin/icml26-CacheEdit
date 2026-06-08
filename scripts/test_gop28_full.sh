#!/bin/bash
# Full 28-step Flux multi-round test with inter-layer GOP compression.

set -euo pipefail

source .venv/bin/activate

# Edit parameters in this block directly.

# PyTorch CUDA allocator setting. expandable_segments:True reduces fragmentation
# and helps avoid OOM during long multi-round runs.
PYTORCH_CUDA_ALLOC_CONF_VALUE="expandable_segments:True"

# Local FLUX-Kontext model directory.
MODEL_PATH="/mnt/data/models/black-forest-labs/FLUX___1-Kontext-dev"

# Dataset root. The runner reads metadata and input images from this directory.
DATA_ROOT="/mnt/data/datasets/test"

# Output directory for generated images, timings.json, and compression report.
OUTPUT_DIR="./outputs/flux_28step_gop16_P16"

# Image id from the dataset metadata to run. 0000 runs the first configured case.
IMAGE_IDX="0000"

# Number of GPUs used by the optimized multi-GPU runner.
NUM_GPUS="4"

# Soft per-GPU memory limit in GiB used by the offload/cache manager.
GPU_MEMORY_LIMIT_GB="16.0"

# Extra GiB kept free as a safety buffer before placing tensors on a GPU.
GPU_MEMORY_BUFFER_GB="5.0"

# Cache reuse interval between diffusion steps. Keep this at 5 for the current
# 28-step test requirement; cache anchor steps are 0, 5, 10, 15, 20, and 25.
CACHE_INTERVAL="5"

# Number of denoising steps. 28 is the full test setting.
NUM_INFERENCE_STEPS="28"

# FLUX guidance scale. Higher values push edits more strongly but may reduce stability.
GUIDANCE_SCALE="3.5"

# Cache similarity/reuse threshold. Higher values are more conservative.
THRESHOLD="0.97"

# Random seed for deterministic generation when the rest of the environment is stable.
SEED="42"

# NVENC codec used for activation compression. hevc usually gives better
# compression than h264 at the same bitrate.
COMPRESSION_CODEC="hevc"

# NVENC bitrate in Mbps. Higher bitrate improves reconstructed activation
# accuracy but lowers compression ratio and increases payload size.
COMPRESSION_BITRATE="5.0"

# Inter-layer GOP length. Consecutive transformer layers are encoded as frames
# in one GOP; larger values can improve compression ratio but may increase error.
COMPRESSION_GOP_LENGTH="16"

# Distance between P frames in the GOP. 1 means IPPP... and enables dense
# inter-frame prediction between adjacent layers.
COMPRESSION_FRAME_INTERVAL_P="16"

export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF_VALUE}"

echo "=========================================="
echo "Full 28-step GOP compression test"
echo "=========================================="
echo ""
echo "Configuration:"
echo "  - Image idx: ${IMAGE_IDX}"
echo "  - Output dir: ${OUTPUT_DIR}"
echo "  - Inference steps: ${NUM_INFERENCE_STEPS}"
echo "  - Cache interval: ${CACHE_INTERVAL} (28-step cache steps: 0, 5, 10, 15, 20, 25)"
echo "  - Compression: ${COMPRESSION_CODEC} ${COMPRESSION_BITRATE}Mbps"
echo "  - Inter-layer GOP: length=${COMPRESSION_GOP_LENGTH}, frame_interval_p=${COMPRESSION_FRAME_INTERVAL_P}"
echo "  - GPUs: ${NUM_GPUS}, memory limit=${GPU_MEMORY_LIMIT_GB}GB, buffer=${GPU_MEMORY_BUFFER_GB}GB"
echo ""

python scripts/run_flux_multi_gpu_optimized.py \
    --model-path "${MODEL_PATH}" \
    --data-root "${DATA_ROOT}" \
    --output-dir "${OUTPUT_DIR}" \
    --image-idx "${IMAGE_IDX}" \
    --num-gpus "${NUM_GPUS}" \
    --gpu-memory-limit-gb "${GPU_MEMORY_LIMIT_GB}" \
    --gpu-memory-buffer-gb "${GPU_MEMORY_BUFFER_GB}" \
    --use-cache \
    --use-cache-compression \
    --compression-bitrate "${COMPRESSION_BITRATE}" \
    --compression-codec "${COMPRESSION_CODEC}" \
    --compression-gop-length "${COMPRESSION_GOP_LENGTH}" \
    --compression-frame-interval-p "${COMPRESSION_FRAME_INTERVAL_P}" \
    --num-inference-steps "${NUM_INFERENCE_STEPS}" \
    --cache-interval "${CACHE_INTERVAL}" \
    --guidance-scale "${GUIDANCE_SCALE}" \
    --threshold "${THRESHOLD}" \
    --seed "${SEED}"

echo ""
echo "=========================================="
echo "Full GOP test completed"
echo "Report: ${OUTPUT_DIR}/timings.json"
echo "Images: ${OUTPUT_DIR}/generation"
echo "=========================================="
