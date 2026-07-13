#!/bin/bash
# Full-dataset paper experiment for single RTX Pro 6000 (96GB):
#   1) no_cache baseline
#   2) cache_only
#   3) cache_compressed (ConstQP=4, qg=3072, GOP=32)
# Then compute PSNR/SSIM/LPIPS and write CSV/JSON/XLSX reports.

set -euo pipefail

# Activate conda environment
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate cacheedit

# -----------------------------
# Paths
# -----------------------------

# Local FLUX-Kontext model directory.
MODEL_PATH="/home/yujinxin/model/black-forest-labs/FLUX___1-Kontext-dev"

# Dataset root. The script runs every image id in metadata.jsonl.
DATA_ROOT="/home/yujinxin/dataset/test"

# Output root for this complete paper experiment.
OUTPUT_ROOT="./outputs/full_dataset_paper_28step_a6000pro"

# -----------------------------
# Generation parameters
# -----------------------------

# Number of GPUs. Single RTX Pro 6000 (96GB).
NUM_GPUS="1"

# Soft GPU memory limit used by the cache/offload manager.
GPU_MEMORY_LIMIT_GB="90.0"

# Free-memory buffer kept before placing tensors on GPU.
GPU_MEMORY_BUFFER_GB="6.0"

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
# Compression parameters (ConstQP=4, qg=3072, GOP=32 from optimization)
# -----------------------------

# Codec path. hevc with ConstQP=4 provides 23.96x compression ratio with
# acceptable quality (PSNR~32, SSIM~0.96).
COMPRESSION_CODEC="hevc"

# HEVC rate-control mode. ConstQP for fine-grained quality control.
COMPRESSION_RC_MODE="constqp"

# HEVC constant QP. 4 provides the best balance of compression ratio and quality
# based on 4x4090 parameter search and single-GPU validation.
COMPRESSION_CONST_QP="4"

# Nominal bitrate in Mbps. For constqp this is kept for config/report
# compatibility; QP controls the actual compression strength.
COMPRESSION_BITRATE="5.0"

# Max bitrate multiplier used by VBR/CBR paths. Kept explicit for reporting.
COMPRESSION_BITRATE_MAX_MULTIPLIER="10"

# Consecutive transformer layers are treated as one video GOP.
# GOP=32 with async compression (max_pending=8) achieves 0.00s wait time.
COMPRESSION_GOP_LENGTH="32"

# P-frame interval inside each GOP. 1 means IPPP...
COMPRESSION_FRAME_INTERVAL_P="1"

# Group size for FP16 activation -> uint8 quantization before codec.
# qg=3072 matches hidden dimension for optimal compression ratio.
COMPRESSION_QUANT_GROUP_SIZE="3072"

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
echo "Full dataset paper experiment (RTX Pro 6000)"
echo "=========================================="
echo "Output root: ${OUTPUT_ROOT}"
echo "Dataset: ${DATA_ROOT}"
echo "GPU: 1x RTX Pro 6000 (96GB)"
echo "Steps: ${NUM_INFERENCE_STEPS}, rounds: ${MAX_ROUNDS}"
echo "Cache interval: ${CACHE_INTERVAL}, threshold: ${THRESHOLD}"
echo "Compressed mode: codec=${COMPRESSION_CODEC}, rc=${COMPRESSION_RC_MODE}, qp=${COMPRESSION_CONST_QP}, qg=${COMPRESSION_QUANT_GROUP_SIZE}, gop=${COMPRESSION_GOP_LENGTH}, p=${COMPRESSION_FRAME_INTERVAL_P}"
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
    --compression-const-qp "${COMPRESSION_CONST_QP}" \
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

echo ""
echo "=========================================="
echo "Done!"
echo "=========================================="
echo "Logs: ${OUTPUT_ROOT}/logs"
echo "Metrics: ${METRICS_JSON}"
echo "Excel: ${REPORT_DIR}/full_dataset_report.xlsx"
echo ""
echo "Summary:"
echo "  - no_cache timing: ${NO_CACHE_DIR}/timings.json"
echo "  - cache_only timing: ${CACHE_ONLY_DIR}/timings.json"
echo "  - cache_compressed timing: ${COMPRESSED_DIR}/timings.json"
echo "  - Quality metrics: ${METRICS_JSON}"
echo "  - Paper report: ${REPORT_DIR}/"
echo "=========================================="
