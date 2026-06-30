#!/bin/bash
# Search high-ratio compression candidates under a relaxed quality gate.

set -euo pipefail

source .venv/bin/activate

# Edit parameters in this block directly.

PYTORCH_CUDA_ALLOC_CONF_VALUE="expandable_segments:True"
MODEL_PATH="/mnt/data/models/black-forest-labs/FLUX___1-Kontext-dev"
DATA_ROOT="/mnt/data/datasets/test"
IMAGE_IDX="0000"
MAX_ROUNDS="3"
OUTPUT_ROOT="./outputs/best_ratio_relaxed_28step_3round"
NUM_GPUS="4"
GPU_MEMORY_LIMIT_GB="16.0"
GPU_MEMORY_BUFFER_GB="5.0"
NUM_INFERENCE_STEPS="28"
CACHE_INTERVAL="5"
GUIDANCE_SCALE="3.5"
THRESHOLD="0.97"
SEED="42"

# Quality gate requested for best-ratio selection.
QUALITY_MIN_PSNR="30"
QUALITY_MIN_SSIM="0.66"
QUALITY_MAX_LPIPS="0.18"

# Candidate format:
#   codec:bitrate_mbps:gop_length:frame_interval_p:quant_group_size:outlier_ratio
#
# qg0 forces channel-wise quantization and currently has the best verified
# ratio in older 3-round results. Larger qg values trade metadata for more
# quantization error. qg256_o0.001 is a residual-outlier control point.
CANDIDATES=(
    "lossless:5.0:16:16:0:0"
    "lossless:5.0:32:1:0:0"
    "lossless:5.0:32:16:0:0"
    "lossless:5.0:32:1:512:0"
    "lossless:5.0:32:1:1024:0"
    "lossless:5.0:32:1:3072:0"
    "lossless:5.0:32:1:256:0.001"
)

# Set to 1 to rerun baseline and cache-only. Set to 0 to reuse existing outputs.
RUN_BASELINE_AND_CACHE_ONLY="1"

# Set to 1 to skip LPIPS even if the optional lpips package is installed.
SKIP_LPIPS="0"

# Optional memory gate for recommendation selection. Empty means no hard cap.
MAX_PEAK_RESERVED_GIB=""

export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF_VALUE}"
export TORCH_HOME="${PWD}/outputs/lpips_torch_home"

BASELINE_DIR="${OUTPUT_ROOT}/baseline_no_cache"
CACHE_ONLY_DIR="${OUTPUT_ROOT}/cache_only"
METRICS_DIR="${OUTPUT_ROOT}/metrics"

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

slug_float() {
    local value="$1"
    echo "${value//./p}"
}

mkdir -p "${OUTPUT_ROOT}" "${METRICS_DIR}"
LOG_FILE="${OUTPUT_ROOT}/sweep.log"
exec > >(tee -a "${LOG_FILE}") 2>&1

echo "=========================================="
echo "Best-ratio relaxed quality sweep"
echo "=========================================="
echo "Output root: ${OUTPUT_ROOT}"
echo "Image idx: ${IMAGE_IDX}, rounds: ${MAX_ROUNDS}, steps: ${NUM_INFERENCE_STEPS}"
echo "Quality gate: PSNR>=${QUALITY_MIN_PSNR}, SSIM>=${QUALITY_MIN_SSIM}, LPIPS<=${QUALITY_MAX_LPIPS}"
echo "Candidates:"
printf '  %s\n' "${CANDIDATES[@]}"
echo ""

if [[ "${RUN_BASELINE_AND_CACHE_ONLY}" == "1" ]]; then
    echo "[baseline] Running no-cache..."
    if [[ -f "${BASELINE_DIR}/timings.json" ]]; then
        echo "[baseline] Reusing existing ${BASELINE_DIR}/timings.json"
    else
        python scripts/run_flux_multi_gpu_optimized.py \
            "${COMMON_ARGS[@]}" \
            --output-dir "${BASELINE_DIR}"
    fi

    echo ""
    echo "[cache-only] Running uncompressed cache..."
    if [[ -f "${CACHE_ONLY_DIR}/timings.json" ]]; then
        echo "[cache-only] Reusing existing ${CACHE_ONLY_DIR}/timings.json"
    else
        python scripts/run_flux_multi_gpu_optimized.py \
            "${COMMON_ARGS[@]}" \
            --output-dir "${CACHE_ONLY_DIR}" \
            --use-cache \
            --cache-interval "${CACHE_INTERVAL}" \
            --threshold "${THRESHOLD}"
    fi
else
    echo "[baseline/cache-only] Reusing existing outputs"
fi

for candidate in "${CANDIDATES[@]}"; do
    IFS=':' read -r codec bitrate gop_length frame_interval_p qg outlier_ratio <<< "${candidate}"
    bitrate_slug="$(slug_float "${bitrate}")"
    outlier_slug="$(slug_float "${outlier_ratio}")"
    run_name="codec_${codec}_br${bitrate_slug}_gop${gop_length}_p${frame_interval_p}_qg${qg}_o${outlier_slug}"
    compressed_dir="${OUTPUT_ROOT}/${run_name}"
    metrics_json="${METRICS_DIR}/${run_name}.json"

    echo ""
    echo "[compressed] ${run_name}"
    if [[ -f "${compressed_dir}/timings.json" ]]; then
        echo "[compressed] Reusing existing ${compressed_dir}/timings.json"
    else
        python scripts/run_flux_multi_gpu_optimized.py \
            "${COMMON_ARGS[@]}" \
            --output-dir "${compressed_dir}" \
            --use-cache \
            --use-cache-compression \
            --cache-interval "${CACHE_INTERVAL}" \
            --threshold "${THRESHOLD}" \
            --compression-bitrate "${bitrate}" \
            --compression-codec "${codec}" \
            --compression-gop-length "${gop_length}" \
            --compression-frame-interval-p "${frame_interval_p}" \
            --compression-quant-group-size "${qg}" \
            --compression-quant-outlier-ratio "${outlier_ratio}"
    fi

    echo "[metrics] ${run_name}"
    if [[ -f "${metrics_json}" ]]; then
        echo "[metrics] Reusing existing ${metrics_json}"
    else
        eval_args=(
            --baseline-dir "${BASELINE_DIR}/generation"
            --cache-dir "${CACHE_ONLY_DIR}/generation"
            --compressed-dir "${compressed_dir}/generation"
            --output "${metrics_json}"
        )
        if [[ "${SKIP_LPIPS}" == "1" ]]; then
            eval_args+=(--no-lpips)
        fi
        python scripts/evaluate_image_metrics.py "${eval_args[@]}"
    fi
done

summary_args=(
    --output-root "${OUTPUT_ROOT}"
    --metrics-dir "${METRICS_DIR}"
    --min-psnr "${QUALITY_MIN_PSNR}"
    --min-ssim "${QUALITY_MIN_SSIM}"
    --max-lpips "${QUALITY_MAX_LPIPS}"
)
if [[ -n "${MAX_PEAK_RESERVED_GIB}" ]]; then
    summary_args+=(--max-peak-reserved-gib "${MAX_PEAK_RESERVED_GIB}")
fi
python scripts/summarize_compression_sweep.py "${summary_args[@]}"

echo ""
echo "=========================================="
echo "Best-ratio relaxed sweep completed"
echo "Summary: ${OUTPUT_ROOT}/sweep_summary.csv"
echo "Recommendation: ${OUTPUT_ROOT}/recommended_config.json"
echo "Log: ${LOG_FILE}"
echo "=========================================="
