#!/bin/bash
# Sweep residual-outlier ratios with fixed qg256 and selected GOP/P settings.

set -euo pipefail

source .venv/bin/activate

# Edit parameters in this block directly.

PYTORCH_CUDA_ALLOC_CONF_VALUE="expandable_segments:True"
MODEL_PATH="/mnt/data/models/black-forest-labs/FLUX___1-Kontext-dev"
DATA_ROOT="/mnt/data/datasets/test"
IMAGE_IDX="0000"
MAX_ROUNDS="2"
OUTPUT_ROOT="./outputs/quant_outlier_sweep_qg256_gop32p1_28step_2round"
NUM_GPUS="4"
GPU_MEMORY_LIMIT_GB="16.0"
GPU_MEMORY_BUFFER_GB="5.0"
NUM_INFERENCE_STEPS="28"
CACHE_INTERVAL="5"
GUIDANCE_SCALE="3.5"
THRESHOLD="0.97"
SEED="42"

# Fixed compression settings. The sweep only changes residual-outlier ratio.
COMPRESSION_CODEC="lossless"
COMPRESSION_BITRATE="5.0"
COMPRESSION_GOP_LENGTH="32"
COMPRESSION_FRAME_INTERVAL_P="1"
COMPRESSION_QUANT_GROUP_SIZE="256"

# Fraction of the worst FP16->uint8 quantization residuals stored exactly as
# side metadata. 0 disables the residual path and is the qg256 baseline.
COMPRESSION_QUANT_OUTLIER_RATIOS=("0" "0.0005" "0.001")

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

mkdir -p "${OUTPUT_ROOT}" "${METRICS_DIR}"
LOG_FILE="${OUTPUT_ROOT}/sweep.log"
exec > >(tee -a "${LOG_FILE}") 2>&1

echo "=========================================="
echo "Quantization outlier-ratio sweep with qg256"
echo "=========================================="
echo "Output root: ${OUTPUT_ROOT}"
echo "Image idx: ${IMAGE_IDX}, rounds: ${MAX_ROUNDS}, steps: ${NUM_INFERENCE_STEPS}"
echo "Codec=${COMPRESSION_CODEC}, GOP=${COMPRESSION_GOP_LENGTH}, P=${COMPRESSION_FRAME_INTERVAL_P}, QG=${COMPRESSION_QUANT_GROUP_SIZE}"
echo "Outlier ratios: ${COMPRESSION_QUANT_OUTLIER_RATIOS[*]}"
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

for outlier_ratio in "${COMPRESSION_QUANT_OUTLIER_RATIOS[@]}"; do
    outlier_slug="${outlier_ratio//./p}"
    run_name="codec_${COMPRESSION_CODEC}_gop${COMPRESSION_GOP_LENGTH}_p${COMPRESSION_FRAME_INTERVAL_P}_qg${COMPRESSION_QUANT_GROUP_SIZE}_o${outlier_slug}"
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
            --compression-bitrate "${COMPRESSION_BITRATE}" \
            --compression-codec "${COMPRESSION_CODEC}" \
            --compression-gop-length "${COMPRESSION_GOP_LENGTH}" \
            --compression-frame-interval-p "${COMPRESSION_FRAME_INTERVAL_P}" \
            --compression-quant-group-size "${COMPRESSION_QUANT_GROUP_SIZE}" \
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
)
if [[ -n "${MAX_PEAK_RESERVED_GIB}" ]]; then
    summary_args+=(--max-peak-reserved-gib "${MAX_PEAK_RESERVED_GIB}")
fi
python scripts/summarize_compression_sweep.py "${summary_args[@]}"

echo ""
echo "=========================================="
echo "Quantization outlier-ratio sweep completed"
echo "Metrics: ${METRICS_DIR}"
echo "Summary: ${OUTPUT_ROOT}/sweep_summary.csv"
echo "Recommendation: ${OUTPUT_ROOT}/recommended_config.json"
echo "Log: ${LOG_FILE}"
echo "=========================================="
