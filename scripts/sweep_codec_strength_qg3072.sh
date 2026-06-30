#!/bin/bash
# Sweep codec compression strength with fixed quantization/GOP settings.
#
# This script keeps quantization mostly fixed and focuses on NVENC strength
# knobs: rate-control mode, bitrate, bitrate max multiplier, and ConstQP.

set -euo pipefail

source .venv/bin/activate

# Edit parameters in this block directly.

PYTORCH_CUDA_ALLOC_CONF_VALUE="expandable_segments:True"
MODEL_PATH="/mnt/data/models/black-forest-labs/FLUX___1-Kontext-dev"
DATA_ROOT="/mnt/data/datasets/test"
IMAGE_IDX="0000"
MAX_ROUNDS="2"
OUTPUT_ROOT="./outputs/codec_strength_qg3072_hevc_28step_2round"
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

# Fixed quantization/GOP point. qg3072/GOP32/P1 was the best completed
# high-ratio lossless point under the 30/0.66/0.18 gate in the previous sweep.
COMPRESSION_QUANT_GROUP_SIZE="3072"
COMPRESSION_QUANT_OUTLIER_RATIO="0"
COMPRESSION_GOP_LENGTH="32"
COMPRESSION_FRAME_INTERVAL_P="1"

# Candidate format:
#   codec:rc_mode:bitrate_mbps:bitrate_max_multiplier:const_qp
#
# rc_mode:
#   constqp: use const_qp; bitrate fields are ignored by NVENC.
#   vbr/cbr: use bitrate_mbps and bitrate_max_multiplier; const_qp is ignored.
#
# ConstQP: smaller QP means higher quality and lower compression.
# VBR high-bitrate points are included as controls after fixing averageBitRate.
# 100Mbps was tested separately and hit NVENC/OOM fallback on this workload, so
# it is not part of the default sweep.
CANDIDATES=(
    "hevc:constqp:5.0:10:0"
    "hevc:constqp:5.0:10:4"
    "hevc:constqp:5.0:10:8"
    "hevc:vbr:20.0:10:none"
    "hevc:vbr:50.0:10:none"
)

# Set to 1 to also run a lossless anchor with the same qg/GOP settings.
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

slug_float() {
    local value="$1"
    echo "${value//./p}"
}

mkdir -p "${OUTPUT_ROOT}" "${METRICS_DIR}"
LOG_FILE="${OUTPUT_ROOT}/sweep.log"
exec > >(tee -a "${LOG_FILE}") 2>&1

echo "=========================================="
echo "Codec strength sweep"
echo "=========================================="
echo "Output root: ${OUTPUT_ROOT}"
echo "Image idx: ${IMAGE_IDX}, rounds: ${MAX_ROUNDS}, steps: ${NUM_INFERENCE_STEPS}"
echo "Quality gate: PSNR>=${QUALITY_MIN_PSNR}, SSIM>=${QUALITY_MIN_SSIM}, LPIPS<=${QUALITY_MAX_LPIPS}"
echo "Fixed: GOP=${COMPRESSION_GOP_LENGTH}, P=${COMPRESSION_FRAME_INTERVAL_P}, QG=${COMPRESSION_QUANT_GROUP_SIZE}, outlier=${COMPRESSION_QUANT_OUTLIER_RATIO}"
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

run_and_eval() {
    local codec="$1"
    local rc_mode="$2"
    local bitrate="$3"
    local bitrate_max_multiplier="$4"
    local const_qp="$5"

    local bitrate_slug
    bitrate_slug="$(slug_float "${bitrate}")"
    local run_name
    if [[ "${rc_mode}" == "constqp" ]]; then
        run_name="codec_${codec}_rcconstqp_qp${const_qp}_gop${COMPRESSION_GOP_LENGTH}_p${COMPRESSION_FRAME_INTERVAL_P}_qg${COMPRESSION_QUANT_GROUP_SIZE}"
    else
        run_name="codec_${codec}_rc${rc_mode}_br${bitrate_slug}_max${bitrate_max_multiplier}_gop${COMPRESSION_GOP_LENGTH}_p${COMPRESSION_FRAME_INTERVAL_P}_qg${COMPRESSION_QUANT_GROUP_SIZE}"
    fi
    local compressed_dir="${OUTPUT_ROOT}/${run_name}"
    local metrics_json="${METRICS_DIR}/${run_name}.json"

    echo ""
    echo "[compressed] ${run_name}"
    if [[ -f "${compressed_dir}/timings.json" ]]; then
        echo "[compressed] Reusing existing ${compressed_dir}/timings.json"
    else
        run_args=(
            "${COMMON_ARGS[@]}"
            --output-dir "${compressed_dir}"
            --use-cache
            --use-cache-compression
            --cache-interval "${CACHE_INTERVAL}"
            --threshold "${THRESHOLD}"
            --compression-bitrate "${bitrate}"
            --compression-codec "${codec}"
            --compression-rc-mode "${rc_mode}"
            --compression-bitrate-max-multiplier "${bitrate_max_multiplier}"
            --compression-gop-length "${COMPRESSION_GOP_LENGTH}"
            --compression-frame-interval-p "${COMPRESSION_FRAME_INTERVAL_P}"
            --compression-quant-group-size "${COMPRESSION_QUANT_GROUP_SIZE}"
            --compression-quant-outlier-ratio "${COMPRESSION_QUANT_OUTLIER_RATIO}"
        )
        if [[ "${rc_mode}" == "constqp" ]]; then
            run_args+=(--compression-const-qp "${const_qp}")
        fi
        python scripts/run_flux_multi_gpu_optimized.py "${run_args[@]}"
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
    run_and_eval "lossless" "constqp" "5.0" "10" "0"
fi

for candidate in "${CANDIDATES[@]}"; do
    IFS=':' read -r codec rc_mode bitrate bitrate_max_multiplier const_qp <<< "${candidate}"
    run_and_eval "${codec}" "${rc_mode}" "${bitrate}" "${bitrate_max_multiplier}" "${const_qp}"
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
echo "Codec strength sweep completed"
echo "Summary: ${OUTPUT_ROOT}/sweep_summary.csv"
echo "Recommendation: ${OUTPUT_ROOT}/recommended_config.json"
echo "Log: ${LOG_FILE}"
echo "=========================================="
