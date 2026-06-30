#!/bin/bash
# Run the full 28-step cache+compression dataset job with resume.
# Terminal output is also saved to the output directory.

set -euo pipefail

# Activate the conda environment (single RTX PRO 6000 setup).
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate cacheedit

PYTORCH_CUDA_ALLOC_CONF_VALUE="expandable_segments:True"
MODEL_PATH="/home/yujinxin/model/black-forest-labs/FLUX___1-Kontext-dev"
DATA_ROOT="/home/yujinxin/dataset/test"
OUTPUT_DIR="./outputs/cache_quality_metrics_28step_full_dataset/cache_compressed_onepass"

NUM_GPUS="1"
GPU_MEMORY_LIMIT_GB="90.0"
GPU_MEMORY_BUFFER_GB="6.0"
NUM_INFERENCE_STEPS="28"
GUIDANCE_SCALE="3.5"
SEED="42"
MAX_ROUNDS="8"
CACHE_INTERVAL="5"
THRESHOLD="0.97"
# Activation compression codec. lossless uses HEVC/NVENC lossless mode after
# FP16 activations are quantized to uint8 frames; hevc/h264 use lossy NVENC.
COMPRESSION_CODEC="lossless"

# NVENC bitrate in Mbps. Ignored by COMPRESSION_CODEC=lossless; only used by
# hevc/h264 lossy video compression.
COMPRESSION_BITRATE="5.0"

# Consecutive transformer layers are encoded as one inter-layer GOP.
COMPRESSION_GOP_LENGTH="16"

# P-frame interval inside the NVENC GOP.
COMPRESSION_FRAME_INTERVAL_P="16"

# Group size for lossless FP16->uint8 activation quantization before codec.
# Smaller values usually improve precision but increase scale/offset metadata;
# larger values may improve total compression ratio but can lose more precision.
# Use 0 to force channel-wise quantization.
COMPRESSION_QUANT_GROUP_SIZE="256"

mkdir -p "${OUTPUT_DIR}"
LOG_FILE="${OUTPUT_DIR}/run_$(date +%Y%m%d_%H%M%S).log"

echo "Output dir: ${OUTPUT_DIR}"
echo "Log file: ${LOG_FILE}"
echo ""

export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF_VALUE}"
export MPLCONFIGDIR="${PWD}/outputs/matplotlib_cache"
mkdir -p "${MPLCONFIGDIR}"

python -u scripts/run_flux_multi_gpu_optimized.py \
    --model-path "${MODEL_PATH}" \
    --data-root "${DATA_ROOT}" \
    --output-dir "${OUTPUT_DIR}" \
    --image-idx all \
    --num-gpus "${NUM_GPUS}" \
    --gpu-memory-limit-gb "${GPU_MEMORY_LIMIT_GB}" \
    --gpu-memory-buffer-gb "${GPU_MEMORY_BUFFER_GB}" \
    --num-inference-steps "${NUM_INFERENCE_STEPS}" \
    --guidance-scale "${GUIDANCE_SCALE}" \
    --seed "${SEED}" \
    --max-rounds "${MAX_ROUNDS}" \
    --use-cache \
    --cache-interval "${CACHE_INTERVAL}" \
    --threshold "${THRESHOLD}" \
    --use-cache-compression \
    --compression-bitrate "${COMPRESSION_BITRATE}" \
    --compression-codec "${COMPRESSION_CODEC}" \
    --compression-gop-length "${COMPRESSION_GOP_LENGTH}" \
    --compression-frame-interval-p "${COMPRESSION_FRAME_INTERVAL_P}" \
    --compression-quant-group-size "${COMPRESSION_QUANT_GROUP_SIZE}" \
    --resume-skip-complete \
    > "${LOG_FILE}" 2>&1
