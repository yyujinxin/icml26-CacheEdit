#!/bin/bash
# Run the full-dataset paper experiment:
#   1) no_cache baseline
#   2) cache_only
#   3) cache_compressed
# Then compute PSNR/SSIM/LPIPS and write CSV/JSON/XLSX reports.

set -euo pipefail

source .venv/bin/activate

# -----------------------------
# Paths
# -----------------------------

# Local FLUX-Kontext model directory.
MODEL_PATH="/mnt/data/models/black-forest-labs/FLUX___1-Kontext-dev"

# Dataset root. The script runs every image id in metadata.jsonl.
DATA_ROOT="/mnt/data/datasets/test"

# Output root for this complete paper experiment.
OUTPUT_ROOT="./outputs/full_dataset_paper_28step"

# -----------------------------
# Generation parameters
# -----------------------------

# Number of GPUs used by scripts/run_flux_multi_gpu_optimized.py.
NUM_GPUS="4"

# Soft per-GPU memory limit used by the cache/offload manager.
GPU_MEMORY_LIMIT_GB="16.0"

# Free-memory buffer kept before placing tensors on a GPU.
GPU_MEMORY_BUFFER_GB="5.0"

# Diffusion denoising steps. Keep this at 28 for the paper setting.
NUM_INFERENCE_STEPS="28"

# Maximum edit rounds per image. The test metadata has 8 rounds per image.
MAX_ROUNDS="8"

# FLUX guidance scale. This must be identical across all three modes.
GUIDANCE_SCALE="3.5"

# Random seed used by all three modes.
SEED="42"

# -----------------------------
# Cache parameters
# -----------------------------

# Cache interval. For 28 steps, cache anchor steps are 0, 5, 10, 15, 20, 25.
CACHE_INTERVAL="5"

# Cache reuse threshold.
THRESHOLD="0.97"

# -----------------------------
# Compression parameters
# -----------------------------

# Codec path used for compressed-cache mode. lossless still uses NVENC/codec
# after FP16 activations are quantized to uint8 frames; it avoids the multi-round
# NaN/black-image failures seen with the lossy HEVC ConstQP probe.
COMPRESSION_CODEC="lossless"

# HEVC/H.264 rate-control mode. Ignored by COMPRESSION_CODEC=lossless.
COMPRESSION_RC_MODE="vbr"

# HEVC constant QP. Ignored by COMPRESSION_CODEC=lossless.
COMPRESSION_CONST_QP=""

# Nominal bitrate in Mbps. For constqp this is kept for config/report
# compatibility; QP controls the actual compression strength.
COMPRESSION_BITRATE="5.0"

# Max bitrate multiplier used by VBR/CBR paths. Kept explicit for reporting.
COMPRESSION_BITRATE_MAX_MULTIPLIER="10"

# Consecutive transformer layers are treated as one video GOP.
COMPRESSION_GOP_LENGTH="32"

# P-frame interval inside each GOP. 1 means IPPP...
COMPRESSION_FRAME_INTERVAL_P="1"

# Group size for FP16 activation -> uint8 quantization before codec.
COMPRESSION_QUANT_GROUP_SIZE="256"

# Optional residual outlier metadata ratio. 0 disables outlier side data.
COMPRESSION_QUANT_OUTLIER_RATIO="0"

# -----------------------------
# Runtime/report parameters
# -----------------------------

# PyTorch allocator setting to reduce fragmentation in long full-dataset runs.
PYTORCH_CUDA_ALLOC_CONF_VALUE="expandable_segments:True"

# Set to 1 to skip LPIPS and only write PSNR/SSIM.
SKIP_LPIPS="0"

export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF_VALUE}"
export TORCH_HOME="${PWD}/outputs/lpips_torch_home"
export MPLCONFIGDIR="${PWD}/outputs/matplotlib_cache"
mkdir -p "${OUTPUT_ROOT}/logs" "${TORCH_HOME}" "${MPLCONFIGDIR}"

NO_CACHE_DIR="${OUTPUT_ROOT}/no_cache"
CACHE_ONLY_DIR="${OUTPUT_ROOT}/cache_only"
COMPRESSED_DIR="${OUTPUT_ROOT}/cache_compressed"
METRICS_JSON="${OUTPUT_ROOT}/quality_metrics.json"
REPORT_DIR="${OUTPUT_ROOT}/paper_report"

COMMON_ARGS=(
    --model-path "${MODEL_PATH}"
    --data-root "${DATA_ROOT}"
    --image-idx all
    --num-gpus "${NUM_GPUS}"
    --gpu-memory-limit-gb "${GPU_MEMORY_LIMIT_GB}"
    --gpu-memory-buffer-gb "${GPU_MEMORY_BUFFER_GB}"
    --num-inference-steps "${NUM_INFERENCE_STEPS}"
    --guidance-scale "${GUIDANCE_SCALE}"
    --seed "${SEED}"
    --max-rounds "${MAX_ROUNDS}"
    --resume-skip-complete
)

echo "=========================================="
echo "Full dataset paper experiment"
echo "=========================================="
echo "Output root: ${OUTPUT_ROOT}"
echo "Dataset: ${DATA_ROOT}"
echo "Steps: ${NUM_INFERENCE_STEPS}, rounds: ${MAX_ROUNDS}"
echo "Cache interval: ${CACHE_INTERVAL}, threshold: ${THRESHOLD}"
echo "Compressed mode: codec=${COMPRESSION_CODEC}, rc=${COMPRESSION_RC_MODE}, qp=${COMPRESSION_CONST_QP:-none}, qg=${COMPRESSION_QUANT_GROUP_SIZE}, gop=${COMPRESSION_GOP_LENGTH}, p=${COMPRESSION_FRAME_INTERVAL_P}"
echo ""

echo "[1/5] no_cache baseline"
python -u scripts/run_flux_multi_gpu_optimized.py \
    "${COMMON_ARGS[@]}" \
    --output-dir "${NO_CACHE_DIR}" \
    > "${OUTPUT_ROOT}/logs/no_cache.log" 2>&1

echo "[2/5] cache_only"
python -u scripts/run_flux_multi_gpu_optimized.py \
    "${COMMON_ARGS[@]}" \
    --output-dir "${CACHE_ONLY_DIR}" \
    --use-cache \
    --cache-interval "${CACHE_INTERVAL}" \
    --threshold "${THRESHOLD}" \
    > "${OUTPUT_ROOT}/logs/cache_only.log" 2>&1

echo "[3/5] cache_compressed"
python -u scripts/run_flux_multi_gpu_optimized.py \
    "${COMMON_ARGS[@]}" \
    --output-dir "${COMPRESSED_DIR}" \
    --use-cache \
    --cache-interval "${CACHE_INTERVAL}" \
    --threshold "${THRESHOLD}" \
    --use-cache-compression \
    --compression-codec "${COMPRESSION_CODEC}" \
    --compression-rc-mode "${COMPRESSION_RC_MODE}" \
    --compression-bitrate "${COMPRESSION_BITRATE}" \
    --compression-bitrate-max-multiplier "${COMPRESSION_BITRATE_MAX_MULTIPLIER}" \
    --compression-gop-length "${COMPRESSION_GOP_LENGTH}" \
    --compression-frame-interval-p "${COMPRESSION_FRAME_INTERVAL_P}" \
    --compression-quant-group-size "${COMPRESSION_QUANT_GROUP_SIZE}" \
    --compression-quant-outlier-ratio "${COMPRESSION_QUANT_OUTLIER_RATIO}" \
    > "${OUTPUT_ROOT}/logs/cache_compressed.log" 2>&1

echo "[4/5] image quality metrics"
EVAL_ARGS=(
    --baseline-dir "${NO_CACHE_DIR}/generation"
    --cache-dir "${CACHE_ONLY_DIR}/generation"
    --compressed-dir "${COMPRESSED_DIR}/generation"
    --output "${METRICS_JSON}"
)
if [[ "${SKIP_LPIPS}" == "1" ]]; then
    EVAL_ARGS+=(--no-lpips)
fi
python -u scripts/evaluate_image_metrics.py "${EVAL_ARGS[@]}" \
    > "${OUTPUT_ROOT}/logs/quality_metrics.log" 2>&1

echo "[5/5] paper report"
python -u scripts/summarize_full_dataset_experiment.py \
    --output-root "${OUTPUT_ROOT}" \
    --baseline-name no_cache \
    --cache-only-name cache_only \
    --compressed-name cache_compressed \
    --metrics-json "${METRICS_JSON}" \
    --report-dir "${REPORT_DIR}" \
    > "${OUTPUT_ROOT}/logs/paper_report.log" 2>&1

echo "Done."
echo "Logs: ${OUTPUT_ROOT}/logs"
echo "Metrics: ${METRICS_JSON}"
echo "Excel: ${REPORT_DIR}/full_dataset_report.xlsx"
