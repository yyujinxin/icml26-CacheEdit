#!/bin/bash
# Focused no-B-frame sweep for GOP length and quantization group length.
#
# frame_interval_p is fixed to 1 for every candidate. In NVENC this gives an
# IPPP... prediction structure: no B frames, and GOP length controls how many P
# frames are available after each I frame.

set -euo pipefail

source .venv/bin/activate

# Edit parameters in this block directly.
PYTORCH_CUDA_ALLOC_CONF_VALUE="expandable_segments:True"
MODEL_PATH="/mnt/data/models/black-forest-labs/FLUX___1-Kontext-dev"
DATA_ROOT="/mnt/data/datasets/test"
IMAGE_IDX="0000"
MAX_ROUNDS="2"
OUTPUT_ROOT="./outputs/no_b_gop_qg_sweep_28step_2round"
REFERENCE_ROOT="./outputs/codec_strength_qg3072_hevc_28step_2round"
NUM_GPUS="4"
GPU_MEMORY_LIMIT_GB="16.0"
GPU_MEMORY_BUFFER_GB="5.0"
NUM_INFERENCE_STEPS="28"
CACHE_INTERVAL="5"
GUIDANCE_SCALE="3.5"
THRESHOLD="0.97"
SEED="42"

# Quality gate for best-ratio selection.
QUALITY_MIN_PSNR="30"
QUALITY_MIN_SSIM="0.66"
QUALITY_MAX_LPIPS="0.18"

# Candidate format:
#   codec:rc_mode:bitrate_mbps:bitrate_max_multiplier:const_qp:gop_length:qg
#
# All candidates use frame_interval_p=1, so no B frames are generated. Larger
# GOP length means more P frames per activation-layer sequence. qg is the
# quantization group length before codec encoding.
CANDIDATES=(
    "h264:constqp:5.0:10:8:16:3072"
    "h264:constqp:5.0:10:8:32:3072"
    "h264:constqp:5.0:10:8:57:3072"
    "h264:constqp:5.0:10:8:57:4096"
    "h264:constqp:5.0:10:8:57:6144"
    "h264:constqp:5.0:10:9:57:3072"
    "hevc:constqp:5.0:10:8:16:3072"
    "hevc:constqp:5.0:10:8:57:3072"
    "hevc:constqp:5.0:10:8:57:4096"
)

FRAME_INTERVAL_P="1"
COMPRESSION_QUANT_OUTLIER_RATIO="0"
COMPRESSION_CODEC_PRESET="p7"
COMPRESSION_CODEC_TUNING="high_quality"
COMPRESSION_CODEC_TEMPORAL_AQ="auto"
SKIP_LPIPS="0"

export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF_VALUE}"
export TORCH_HOME="${PWD}/outputs/lpips_torch_home"
export MPLCONFIGDIR="${PWD}/outputs/matplotlib_cache"
export PYTHONUNBUFFERED=1
export CACHEEDIT_VALIDATE_COMPRESSED_CACHE=0

BASELINE_DIR="${REFERENCE_ROOT}/baseline_no_cache"
CACHE_ONLY_DIR="${REFERENCE_ROOT}/cache_only"
METRICS_DIR="${OUTPUT_ROOT}/metrics"
LOG_DIR="${OUTPUT_ROOT}/logs"

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

mkdir -p "${OUTPUT_ROOT}" "${METRICS_DIR}" "${LOG_DIR}" "${TORCH_HOME}" "${MPLCONFIGDIR}"
LOG_FILE="${OUTPUT_ROOT}/sweep.log"
exec > >(tee -a "${LOG_FILE}") 2>&1

echo "=========================================="
echo "No-B GOP/qg compression sweep"
echo "=========================================="
echo "Output root: ${OUTPUT_ROOT}"
echo "Reference root: ${REFERENCE_ROOT}"
echo "Image idx: ${IMAGE_IDX}, rounds: ${MAX_ROUNDS}, steps: ${NUM_INFERENCE_STEPS}"
echo "Quality gate: PSNR>=${QUALITY_MIN_PSNR}, SSIM>=${QUALITY_MIN_SSIM}, LPIPS<=${QUALITY_MAX_LPIPS}"
echo "Fixed frame_interval_p=${FRAME_INTERVAL_P} (IPPP..., no B frames)"
echo "Candidates:"
printf '  %s\n' "${CANDIDATES[@]}"
echo ""

if [[ ! -f "${BASELINE_DIR}/timings.json" ]]; then
    echo "[baseline] Reference missing; generating ${BASELINE_DIR}"
    python -u scripts/run_flux_multi_gpu_optimized.py \
        "${COMMON_ARGS[@]}" \
        --output-dir "${BASELINE_DIR}" \
        > "${LOG_DIR}/baseline_no_cache.log" 2>&1
else
    echo "[baseline] Reusing ${BASELINE_DIR}"
fi

if [[ ! -f "${CACHE_ONLY_DIR}/timings.json" ]]; then
    echo "[cache-only] Reference missing; generating ${CACHE_ONLY_DIR}"
    python -u scripts/run_flux_multi_gpu_optimized.py \
        "${COMMON_ARGS[@]}" \
        --output-dir "${CACHE_ONLY_DIR}" \
        --use-cache \
        --cache-interval "${CACHE_INTERVAL}" \
        --threshold "${THRESHOLD}" \
        > "${LOG_DIR}/cache_only.log" 2>&1
else
    echo "[cache-only] Reusing ${CACHE_ONLY_DIR}"
fi

for candidate in "${CANDIDATES[@]}"; do
    IFS=':' read -r codec rc_mode bitrate bitrate_max_multiplier const_qp gop_length qg <<< "${candidate}"
    bitrate_slug="$(slug_float "${bitrate}")"
    run_name="codec_${codec}_rc${rc_mode}_qp${const_qp}_br${bitrate_slug}_gop${gop_length}_p${FRAME_INTERVAL_P}_qg${qg}_o${COMPRESSION_QUANT_OUTLIER_RATIO}_${COMPRESSION_CODEC_PRESET}_${COMPRESSION_CODEC_TUNING}"
    compressed_dir="${OUTPUT_ROOT}/${run_name}"
    metrics_json="${METRICS_DIR}/${run_name}.json"

    echo ""
    echo "[compressed] ${run_name}"
    if [[ -f "${compressed_dir}/timings.json" ]]; then
        echo "[compressed] Reusing existing ${compressed_dir}/timings.json"
    else
        python -u scripts/run_flux_multi_gpu_optimized.py \
            "${COMMON_ARGS[@]}" \
            --output-dir "${compressed_dir}" \
            --use-cache \
            --use-cache-compression \
            --cache-interval "${CACHE_INTERVAL}" \
            --threshold "${THRESHOLD}" \
            --compression-codec "${codec}" \
            --compression-rc-mode "${rc_mode}" \
            --compression-const-qp "${const_qp}" \
            --compression-bitrate "${bitrate}" \
            --compression-bitrate-max-multiplier "${bitrate_max_multiplier}" \
            --compression-gop-length "${gop_length}" \
            --compression-frame-interval-p "${FRAME_INTERVAL_P}" \
            --compression-quant-group-size "${qg}" \
            --compression-quant-outlier-ratio "${COMPRESSION_QUANT_OUTLIER_RATIO}" \
            --compression-codec-preset "${COMPRESSION_CODEC_PRESET}" \
            --compression-codec-tuning "${COMPRESSION_CODEC_TUNING}" \
            --compression-codec-temporal-aq "${COMPRESSION_CODEC_TEMPORAL_AQ}" \
            > "${LOG_DIR}/${run_name}.log" 2>&1
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
        python -u scripts/evaluate_image_metrics.py "${eval_args[@]}" \
            > "${LOG_DIR}/${run_name}.metrics.log" 2>&1
    fi
done

python scripts/summarize_compression_sweep.py \
    --output-root "${OUTPUT_ROOT}" \
    --metrics-dir "${METRICS_DIR}" \
    --min-psnr "${QUALITY_MIN_PSNR}" \
    --min-ssim "${QUALITY_MIN_SSIM}" \
    --max-lpips "${QUALITY_MAX_LPIPS}"

echo ""
echo "=========================================="
echo "No-B GOP/qg sweep completed"
echo "Summary: ${OUTPUT_ROOT}/sweep_summary.csv"
echo "Recommendation: ${OUTPUT_ROOT}/recommended_config.json"
echo "Log: ${LOG_FILE}"
echo "=========================================="
