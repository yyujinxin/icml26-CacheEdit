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

# Output root for this complete paper experiment. This run uses the best
# compression-ratio config that passed the PSNR/SSIM/LPIPS quality gate in the
# 28-step 2-round sweep.
OUTPUT_ROOT="./outputs/full_dataset_paper_28step_h264_qp8_gop57_qg3072"

# Reuse already completed no-cache/cache-only modes when they were generated
# with the same model, data, seed, 28 steps, 8 rounds, cache interval, and
# threshold. Set this empty to force rerunning all three modes from scratch.
REFERENCE_OUTPUT_ROOT="./outputs/full_dataset_paper_28step"

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

# Codec path used for compressed-cache mode. Current setting is the highest
# compression-ratio config that passed the quality gate in the sweep:
# compressed_vs_cache PSNR>=30, SSIM>=0.66, LPIPS<=0.18, failures=0.
COMPRESSION_CODEC="h264"

# HEVC/H.264 rate-control mode. constqp makes QP directly control codec strength.
COMPRESSION_RC_MODE="constqp"

# Constant QP. Lower is higher quality and lower compression.
COMPRESSION_CONST_QP="8"

# Nominal bitrate in Mbps. For constqp this is kept for config/report
# compatibility; QP controls the actual compression strength.
COMPRESSION_BITRATE="5.0"

# Max bitrate multiplier used by VBR/CBR paths. Kept explicit for reporting.
COMPRESSION_BITRATE_MAX_MULTIPLIER="10"

# Consecutive transformer layers are treated as one video GOP. 57 covers the
# full double/single transformer layer sequence for inter-layer P-frame reuse.
COMPRESSION_GOP_LENGTH="57"

# Layers below this index are compressed as independent single frames. 0 means
# the GOP starts at layer 0, matching the selected best-ratio quality-passing
# sweep result.
COMPRESSION_GOP_START_LAYER="0"

# P-frame interval inside each GOP. 1 means IPPP...
COMPRESSION_FRAME_INTERVAL_P="1"

# Group size for FP16 activation -> uint8 quantization before codec. qg3072 was
# the highest-ratio setting that still passed the quality gate in the sweep.
COMPRESSION_QUANT_GROUP_SIZE="3072"

# Optional residual outlier metadata ratio. 0 disables outlier side data.
COMPRESSION_QUANT_OUTLIER_RATIO="0"

# NVENC codec knobs. p7/high_quality was used in the selected sweep result.
COMPRESSION_CODEC_PRESET="p7"
COMPRESSION_CODEC_TUNING="high_quality"
COMPRESSION_CODEC_TEMPORAL_AQ="auto"

# -----------------------------
# Runtime/report parameters
# -----------------------------

# PyTorch allocator setting to reduce fragmentation in long full-dataset runs.
PYTORCH_CUDA_ALLOC_CONF_VALUE="expandable_segments:True"

# Set to 1 to skip LPIPS and only write PSNR/SSIM.
SKIP_LPIPS="0"

export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF_VALUE}"
# The compressor preserves bf16 activations before uint8 quantization, avoiding
# fp16 overflow-induced NaN/Inf frames. Full decode validation is useful for
# debugging, but it roughly doubles cache-write overhead on the full dataset.
export CACHEEDIT_VALIDATE_COMPRESSED_CACHE=0
export TORCH_HOME="${PWD}/outputs/lpips_torch_home"
export MPLCONFIGDIR="${PWD}/outputs/matplotlib_cache"
mkdir -p "${OUTPUT_ROOT}/logs" "${TORCH_HOME}" "${MPLCONFIGDIR}"

NO_CACHE_DIR="${OUTPUT_ROOT}/no_cache"
CACHE_ONLY_DIR="${OUTPUT_ROOT}/cache_only"
COMPRESSED_DIR="${OUTPUT_ROOT}/cache_compressed"
METRICS_JSON="${OUTPUT_ROOT}/quality_metrics.json"
REPORT_DIR="${OUTPUT_ROOT}/paper_report"

mode_complete() {
    local timings_json="$1"
    python - "$timings_json" <<'PY'
import json
import sys
from pathlib import Path

path = Path(sys.argv[1])
if not path.is_file():
    raise SystemExit(1)
try:
    data = json.loads(path.read_text())
except Exception:
    raise SystemExit(1)
raise SystemExit(0 if data.get("complete") is True else 1)
PY
}

link_reference_mode_if_complete() {
    local mode_name="$1"
    local target_dir="$2"
    local source_dir="$3"

    if mode_complete "${target_dir}/timings.json"; then
        return 0
    fi
    if [[ -z "${REFERENCE_OUTPUT_ROOT}" || -e "${target_dir}" ]]; then
        return 0
    fi
    if [[ "${target_dir}" == "${source_dir}" ]]; then
        return 0
    fi
    if mode_complete "${source_dir}/timings.json"; then
        ln -s "$(realpath "${source_dir}")" "${target_dir}"
        echo "  reuse ${mode_name}: linked completed result from ${source_dir}"
    fi
}

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
echo "Compressed mode: codec=${COMPRESSION_CODEC}, rc=${COMPRESSION_RC_MODE}, qp=${COMPRESSION_CONST_QP:-none}, qg=${COMPRESSION_QUANT_GROUP_SIZE}, gop=${COMPRESSION_GOP_LENGTH}, gop_start=${COMPRESSION_GOP_START_LAYER}, p=${COMPRESSION_FRAME_INTERVAL_P}, preset=${COMPRESSION_CODEC_PRESET}, tuning=${COMPRESSION_CODEC_TUNING}"
echo ""

echo "[1/5] no_cache baseline"
link_reference_mode_if_complete "no_cache" "${NO_CACHE_DIR}" "${REFERENCE_OUTPUT_ROOT}/no_cache"
if mode_complete "${NO_CACHE_DIR}/timings.json"; then
    echo "  skip: ${NO_CACHE_DIR}/timings.json is already complete"
else
    python -u scripts/run_flux_multi_gpu_optimized.py \
        "${COMMON_ARGS[@]}" \
        --output-dir "${NO_CACHE_DIR}" \
        > "${OUTPUT_ROOT}/logs/no_cache.log" 2>&1
fi

echo "[2/5] cache_only"
link_reference_mode_if_complete "cache_only" "${CACHE_ONLY_DIR}" "${REFERENCE_OUTPUT_ROOT}/cache_only"
if mode_complete "${CACHE_ONLY_DIR}/timings.json"; then
    echo "  skip: ${CACHE_ONLY_DIR}/timings.json is already complete"
else
    python -u scripts/run_flux_multi_gpu_optimized.py \
        "${COMMON_ARGS[@]}" \
        --output-dir "${CACHE_ONLY_DIR}" \
        --use-cache \
        --cache-interval "${CACHE_INTERVAL}" \
        --threshold "${THRESHOLD}" \
        > "${OUTPUT_ROOT}/logs/cache_only.log" 2>&1
fi

echo "[3/5] cache_compressed"
if mode_complete "${COMPRESSED_DIR}/timings.json"; then
    echo "  skip: ${COMPRESSED_DIR}/timings.json is already complete"
else
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
        --compression-gop-start-layer "${COMPRESSION_GOP_START_LAYER}" \
        --compression-frame-interval-p "${COMPRESSION_FRAME_INTERVAL_P}" \
        --compression-quant-group-size "${COMPRESSION_QUANT_GROUP_SIZE}" \
        --compression-quant-outlier-ratio "${COMPRESSION_QUANT_OUTLIER_RATIO}" \
        --compression-codec-preset "${COMPRESSION_CODEC_PRESET}" \
        --compression-codec-tuning "${COMPRESSION_CODEC_TUNING}" \
        --compression-codec-temporal-aq "${COMPRESSION_CODEC_TEMPORAL_AQ}" \
        > "${OUTPUT_ROOT}/logs/cache_compressed.log" 2>&1
fi

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
