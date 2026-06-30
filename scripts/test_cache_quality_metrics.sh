#!/bin/bash
# Run baseline, cache-only, and cache+activation-compression tests, then
# evaluate generated images with PSNR, SSIM, and optional LPIPS.

set -euo pipefail

# Activate the conda environment (single RTX PRO 6000 setup).
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate cacheedit

# Edit parameters in this block directly.

# PyTorch CUDA allocator setting. expandable_segments:True reduces fragmentation
# during long multi-round runs with large input images.
PYTORCH_CUDA_ALLOC_CONF_VALUE="expandable_segments:True"

# Local FLUX-Kontext model directory.
MODEL_PATH="/home/yujinxin/model/black-forest-labs/FLUX___1-Kontext-dev"

# Dataset root. The runner reads metadata and input images from this directory.
DATA_ROOT="/home/yujinxin/dataset/test"

# Image id from the dataset metadata. 0000 is the cat example used in prior tests.
IMAGE_IDX="0000"

# Number of edit rounds to run. 8 runs all rounds for image 0000 in the test set.
MAX_ROUNDS="8"

# Shared output root. Each mode writes into a subdirectory here.
OUTPUT_ROOT="./outputs/cache_quality_metrics_28step_8round"

# Number of GPUs. Single RTX PRO 6000 (96GB): keep this at 1.
NUM_GPUS="1"

# Soft GPU memory limit in GiB used by the offload/cache manager on the 96GB card.
GPU_MEMORY_LIMIT_GB="90.0"

# Extra GiB kept free as a safety buffer before placing tensors on the GPU.
GPU_MEMORY_BUFFER_GB="6.0"

# Number of denoising steps.
NUM_INFERENCE_STEPS="28"

# Cache reuse interval. For 28 steps, cache anchor steps are 0, 5, 10, 15, 20, 25.
CACHE_INTERVAL="5"

# FLUX guidance scale. Keep aligned across all three modes for fair comparison.
GUIDANCE_SCALE="3.5"

# Cache similarity/reuse threshold. Used by cache-only and cache+compression modes.
THRESHOLD="0.97"

# Random seed for deterministic generation when the runtime is otherwise stable.
SEED="42"

# Activation compression codec. lossless uses HEVC/NVENC lossless mode after
# FP16 activations are quantized to uint8 frames; hevc/h264 use lossy NVENC.
COMPRESSION_CODEC="lossless"

# NVENC bitrate in Mbps. Ignored by COMPRESSION_CODEC=lossless; only used by
# hevc/h264 lossy video compression.
COMPRESSION_BITRATE="5.0"

# Inter-layer GOP length. Consecutive layers are encoded as codec frames. In
# lossless mode the codec itself is lossless for the quantized frames.
COMPRESSION_GOP_LENGTH="16"

# P-frame interval inside the NVENC GOP.
COMPRESSION_FRAME_INTERVAL_P="16"

# Group size for lossless FP16->uint8 activation quantization before codec.
# Smaller values usually improve precision but increase scale/offset metadata;
# larger values may improve total compression ratio but can lose more precision.
# Use 0 to force channel-wise quantization.
COMPRESSION_QUANT_GROUP_SIZE="256"

# Set to 1 to skip LPIPS even if the optional lpips package is installed.
SKIP_LPIPS="0"

export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF_VALUE}"
export TORCH_HOME="${PWD}/outputs/lpips_torch_home"

BASELINE_DIR="${OUTPUT_ROOT}/baseline_no_cache"
CACHE_ONLY_DIR="${OUTPUT_ROOT}/cache_only"
COMPRESSED_DIR="${OUTPUT_ROOT}/cache_compressed"
METRICS_JSON="${OUTPUT_ROOT}/quality_metrics.json"

COMMON_ARGS=(
    --model-path "${MODEL_PATH}"
    --data-root "${DATA_ROOT}"
    --image-idx "${IMAGE_IDX}"
    --num-gpus "${NUM_GPUS}"
    --gpu-memory-limit-gb "${GPU_MEMORY_LIMIT_GB}"
    --gpu-memory-buffer-gb "${GPU_MEMORY_BUFFER_GB}"
    --num-inference-steps "${NUM_INFERENCE_STEPS}"
    --guidance-scale "${GUIDANCE_SCALE}"
    --seed "${SEED}"
    --max-rounds "${MAX_ROUNDS}"
)

echo "=========================================="
echo "Cache quality metrics test"
echo "=========================================="
echo "Output root: ${OUTPUT_ROOT}"
echo "Image idx: ${IMAGE_IDX}, rounds: ${MAX_ROUNDS}, steps: ${NUM_INFERENCE_STEPS}"
echo "Cache interval: ${CACHE_INTERVAL}, threshold: ${THRESHOLD}"
if [[ "${COMPRESSION_CODEC}" == "lossless" ]]; then
    echo "Compression: ${COMPRESSION_CODEC} codec (bitrate ignored), GOP=${COMPRESSION_GOP_LENGTH}, P=${COMPRESSION_FRAME_INTERVAL_P}, QG=${COMPRESSION_QUANT_GROUP_SIZE}"
else
    echo "Compression: ${COMPRESSION_CODEC} ${COMPRESSION_BITRATE}Mbps, GOP=${COMPRESSION_GOP_LENGTH}, P=${COMPRESSION_FRAME_INTERVAL_P}, QG=${COMPRESSION_QUANT_GROUP_SIZE}"
fi
echo ""

echo "[1/4] Running no-cache baseline..."
python scripts/run_flux_multi_gpu_optimized.py \
    "${COMMON_ARGS[@]}" \
    --output-dir "${BASELINE_DIR}"

echo ""
echo "[2/4] Running cache-only..."
python scripts/run_flux_multi_gpu_optimized.py \
    "${COMMON_ARGS[@]}" \
    --output-dir "${CACHE_ONLY_DIR}" \
    --use-cache \
    --cache-interval "${CACHE_INTERVAL}" \
    --threshold "${THRESHOLD}"

echo ""
echo "[3/4] Running cache + activation compression..."
python scripts/run_flux_multi_gpu_optimized.py \
    "${COMMON_ARGS[@]}" \
    --output-dir "${COMPRESSED_DIR}" \
    --use-cache \
    --use-cache-compression \
    --cache-interval "${CACHE_INTERVAL}" \
    --threshold "${THRESHOLD}" \
    --compression-bitrate "${COMPRESSION_BITRATE}" \
    --compression-codec "${COMPRESSION_CODEC}" \
    --compression-gop-length "${COMPRESSION_GOP_LENGTH}" \
    --compression-frame-interval-p "${COMPRESSION_FRAME_INTERVAL_P}" \
    --compression-quant-group-size "${COMPRESSION_QUANT_GROUP_SIZE}"

echo ""
echo "[4/4] Evaluating image metrics..."
EVAL_ARGS=(
    --baseline-dir "${BASELINE_DIR}/generation"
    --cache-dir "${CACHE_ONLY_DIR}/generation"
    --compressed-dir "${COMPRESSED_DIR}/generation"
    --output "${METRICS_JSON}"
)
if [[ "${SKIP_LPIPS}" == "1" ]]; then
    EVAL_ARGS+=(--no-lpips)
fi
python scripts/evaluate_image_metrics.py "${EVAL_ARGS[@]}"

echo ""
echo "=========================================="
echo "Done"
echo "Timing reports:"
echo "  baseline:   ${BASELINE_DIR}/timings.json"
echo "  cache-only: ${CACHE_ONLY_DIR}/timings.json"
echo "  compressed: ${COMPRESSED_DIR}/timings.json"
echo "Quality report:"
echo "  ${METRICS_JSON}"
echo "=========================================="
