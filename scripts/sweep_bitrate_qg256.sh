#!/bin/bash
# Sweep lossy codec bitrates with fixed qg256 and selected GOP/P settings.

set -euo pipefail

source .venv/bin/activate

# Edit parameters in this block directly.

PYTORCH_CUDA_ALLOC_CONF_VALUE="expandable_segments:True"
MODEL_PATH="/mnt/data/models/black-forest-labs/FLUX___1-Kontext-dev"
DATA_ROOT="/mnt/data/datasets/test"
IMAGE_IDX="0000"
MAX_ROUNDS="2"
OUTPUT_ROOT="./outputs/bitrate_sweep_qg256_hevc_gop32p1_28step_2round"
NUM_GPUS="4"
GPU_MEMORY_LIMIT_GB="16.0"
GPU_MEMORY_BUFFER_GB="5.0"
NUM_INFERENCE_STEPS="28"
CACHE_INTERVAL="5"
GUIDANCE_SCALE="3.5"
THRESHOLD="0.97"
SEED="42"

# Actual codec under test. Use hevc/h264 for bitrate sweeps. lossless ignores
# bitrate and is therefore only useful as a reference anchor.
COMPRESSION_CODEC="hevc"

# Fixed compression settings for this sweep.
COMPRESSION_GOP_LENGTH="32"
COMPRESSION_FRAME_INTERVAL_P="1"
COMPRESSION_QUANT_GROUP_SIZE="256"

# Bitrates in Mbps. Values are encoded into run names with "." replaced by "p".
COMPRESSION_BITRATES=("0.5" "1.0" "2.0" "5.0" "10.0")

# Set to 1 to also run a lossless codec reference with the same qg/GOP settings.
RUN_LOSSLESS_ANCHOR="1"

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
echo "Bitrate sweep with qg256"
echo "=========================================="
echo "Output root: ${OUTPUT_ROOT}"
echo "Image idx: ${IMAGE_IDX}, rounds: ${MAX_ROUNDS}, steps: ${NUM_INFERENCE_STEPS}"
echo "Codec: ${COMPRESSION_CODEC}"
echo "GOP=${COMPRESSION_GOP_LENGTH}, P=${COMPRESSION_FRAME_INTERVAL_P}, QG=${COMPRESSION_QUANT_GROUP_SIZE}"
echo "Bitrates: ${COMPRESSION_BITRATES[*]}"
echo ""

if [[ "${RUN_BASELINE_AND_CACHE_ONLY}" == "1" ]]; then
    echo "[baseline] Running no-cache..."
    python scripts/run_flux_multi_gpu_optimized.py \
        "${COMMON_ARGS[@]}" \
        --output-dir "${BASELINE_DIR}"

    echo ""
    echo "[cache-only] Running uncompressed cache..."
    python scripts/run_flux_multi_gpu_optimized.py \
        "${COMMON_ARGS[@]}" \
        --output-dir "${CACHE_ONLY_DIR}" \
        --use-cache \
        --cache-interval "${CACHE_INTERVAL}" \
        --threshold "${THRESHOLD}"
else
    echo "[baseline/cache-only] Reusing existing outputs"
fi

run_and_eval() {
    local codec="$1"
    local bitrate="$2"
    local bitrate_slug="${bitrate//./p}"
    local run_name="codec_${codec}_br${bitrate_slug}_gop${COMPRESSION_GOP_LENGTH}_p${COMPRESSION_FRAME_INTERVAL_P}_qg${COMPRESSION_QUANT_GROUP_SIZE}"
    local compressed_dir="${OUTPUT_ROOT}/${run_name}"
    local metrics_json="${METRICS_DIR}/${run_name}.json"

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
            --compression-gop-length "${COMPRESSION_GOP_LENGTH}" \
            --compression-frame-interval-p "${COMPRESSION_FRAME_INTERVAL_P}" \
            --compression-quant-group-size "${COMPRESSION_QUANT_GROUP_SIZE}"
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
}

if [[ "${RUN_LOSSLESS_ANCHOR}" == "1" ]]; then
    run_and_eval "lossless" "0"
fi

for bitrate in "${COMPRESSION_BITRATES[@]}"; do
    run_and_eval "${COMPRESSION_CODEC}" "${bitrate}"
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
echo "Bitrate sweep completed"
echo "Metrics: ${METRICS_DIR}"
echo "Summary: ${OUTPUT_ROOT}/sweep_summary.csv"
echo "Recommendation: ${OUTPUT_ROOT}/recommended_config.json"
echo "Log: ${LOG_FILE}"
echo "=========================================="
