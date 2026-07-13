#!/bin/bash
# Focused high-ratio compression sweep.
#
# Goal:
#   Find cache_compressed settings with total compression ratio >= 20x while
#   keeping compressed_vs_cache quality inside the relaxed gate:
#     PSNR >= 30, SSIM >= 0.66, LPIPS <= 0.18, failure_count == 0.
#
# This is a probe script, not the full-dataset paper run. It reuses an existing
# 28-step cache-only/baseline reference when available, and writes all candidate
# runs under OUTPUT_ROOT.

set -euo pipefail

source .venv/bin/activate

PYTORCH_CUDA_ALLOC_CONF_VALUE="expandable_segments:True"
MODEL_PATH="/mnt/data/models/black-forest-labs/FLUX___1-Kontext-dev"
DATA_ROOT="/mnt/data/datasets/test"
IMAGE_IDX="0000"
MAX_ROUNDS="2"
OUTPUT_ROOT="./outputs/high_ratio_20x_28step_2round"
REFERENCE_ROOT="./outputs/codec_strength_qg3072_hevc_28step_2round"
NUM_GPUS="4"
GPU_MEMORY_LIMIT_GB="16.0"
GPU_MEMORY_BUFFER_GB="5.0"
NUM_INFERENCE_STEPS="28"
CACHE_INTERVAL="5"
GUIDANCE_SCALE="3.5"
THRESHOLD="0.97"
SEED="42"

QUALITY_MIN_PSNR="30"
QUALITY_MIN_SSIM="0.66"
QUALITY_MAX_LPIPS="0.18"
TARGET_RATIO="20"

# Candidate format:
#   codec:rc_mode:bitrate_mbps:bitrate_max_multiplier:const_qp:gop:p:qg:outlier:preset:tuning:spatial_aq:temporal_aq:target_quality:qp_intra:qp_inter_p:qp_inter_b
#
# This pass focuses on codec-side knobs:
# - HEVC vs H.264.
# - ConstQP around the known quality boundary.
# - VBR/CBR around the 50-120Mbps region where ratio can exceed 20x.
# - NVENC preset/tuning plus spatial/temporal AQ and VBR targetQuality.
# - GOP/P-frame structure with frame_interval_p fixed to 1. This is IPPP...,
#   so no B frames are used and longer GOPs mean more P frames per codec group.
# - Quantization group length around the current quality boundary. Larger qg
#   reduces scale/zero metadata and can improve total ratio, but increases
#   quantization error.
CANDIDATES=(
    "hevc:constqp:5.0:10:8:32:1:3072:0:p7:high_quality:none:auto:none"
    "hevc:constqp:5.0:10:8:32:1:3072:0:p7:high_quality:8:on:none"
    "hevc:constqp:5.0:10:8:32:1:3072:0:p7:high_quality:12:on:none"
    "hevc:constqp:5.0:10:9:32:1:3072:0:p7:high_quality:8:on:none"
    "hevc:constqp:5.0:10:9:32:1:3072:0:p7:high_quality:12:on:none"
    "hevc:constqp:5.0:10:10:32:1:3072:0:p7:high_quality:8:on:none"
    "hevc:constqp:5.0:10:10:32:1:3072:0:p7:high_quality:12:on:none"
    "hevc:constqp:5.0:10:10:32:1:3072:0:p7:high_quality:none:auto:none:4:10:10"
    "hevc:constqp:5.0:10:11:32:1:3072:0:p7:high_quality:none:auto:none:4:11:11"
    "hevc:constqp:5.0:10:8:16:1:3072:0:p7:high_quality:12:on:none"
    "hevc:constqp:5.0:10:8:57:1:3072:0:p7:high_quality:12:on:none"
    "hevc:constqp:5.0:10:8:57:1:4096:0:p7:high_quality:12:on:none"
    "hevc:constqp:5.0:10:8:57:1:6144:0:p7:high_quality:12:on:none"
    "hevc:constqp:5.0:10:8:32:1:3072:0:p6:high_quality:12:on:none"
    "hevc:constqp:5.0:10:8:32:1:3072:0:p5:high_quality:12:on:none"
    "hevc:constqp:5.0:10:8:32:1:3072:0:p7:low_latency:12:on:none"
    "hevc:vbr:75.0:10:none:32:1:3072:0:p7:high_quality:8:on:1"
    "hevc:vbr:90.0:10:none:32:1:3072:0:p7:high_quality:8:on:1"
    "hevc:vbr:120.0:10:none:32:1:3072:0:p7:high_quality:8:on:1"
    "hevc:vbr:75.0:2:none:32:1:3072:0:p7:high_quality:12:on:1"
    "hevc:vbr:90.0:2:none:32:1:3072:0:p7:high_quality:12:on:1"
    "hevc:cbr:75.0:1:none:32:1:3072:0:p7:high_quality:8:on:none"
    "hevc:cbr:90.0:1:none:32:1:3072:0:p7:high_quality:8:on:none"
    "hevc:cbr:100.0:1:none:32:1:3072:0:p7:high_quality:12:on:none"
    "h264:constqp:5.0:10:6:32:1:3072:0:p7:high_quality:8:on:none"
    "h264:constqp:5.0:10:8:32:1:3072:0:p7:high_quality:8:on:none"
    "h264:constqp:5.0:10:8:16:1:3072:0:p7:high_quality:8:on:none"
    "h264:constqp:5.0:10:8:57:1:3072:0:p7:high_quality:8:on:none"
    "h264:constqp:5.0:10:8:57:1:4096:0:p7:high_quality:8:on:none"
    "h264:constqp:5.0:10:8:57:1:6144:0:p7:high_quality:8:on:none"
    "h264:constqp:5.0:10:9:57:1:3072:0:p7:high_quality:8:on:none"
    "h264:cbr:75.0:1:none:32:1:3072:0:p7:high_quality:8:on:none"
)

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
echo "High-ratio 20x compression sweep"
echo "=========================================="
echo "Output root: ${OUTPUT_ROOT}"
echo "Reference root: ${REFERENCE_ROOT}"
echo "Image idx: ${IMAGE_IDX}, rounds: ${MAX_ROUNDS}, steps: ${NUM_INFERENCE_STEPS}"
echo "Target: total compression ratio >= ${TARGET_RATIO}x"
echo "Quality gate: PSNR>=${QUALITY_MIN_PSNR}, SSIM>=${QUALITY_MIN_SSIM}, LPIPS<=${QUALITY_MAX_LPIPS}"
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
    IFS=':' read -r codec rc_mode bitrate bitrate_max_multiplier const_qp gop_length frame_interval_p qg outlier_ratio preset tuning spatial_aq temporal_aq target_quality qp_intra qp_inter_p qp_inter_b <<< "${candidate}"
    qp_intra="${qp_intra:-none}"
    qp_inter_p="${qp_inter_p:-none}"
    qp_inter_b="${qp_inter_b:-none}"
    bitrate_slug="$(slug_float "${bitrate}")"
    outlier_slug="$(slug_float "${outlier_ratio}")"
    spatial_aq_slug="${spatial_aq}"
    target_quality_slug="${target_quality}"
    if [[ "${rc_mode}" == "constqp" ]]; then
        run_name="codec_${codec}_rcconstqp_qp${const_qp}_i${qp_intra}_pqp${qp_inter_p}_b${qp_inter_b}_gop${gop_length}_p${frame_interval_p}_qg${qg}_o${outlier_slug}_${preset}_${tuning}_saq${spatial_aq_slug}_taq${temporal_aq}"
    else
        run_name="codec_${codec}_rc${rc_mode}_br${bitrate_slug}_max${bitrate_max_multiplier}_gop${gop_length}_p${frame_interval_p}_qg${qg}_o${outlier_slug}_${preset}_${tuning}_saq${spatial_aq_slug}_taq${temporal_aq}_tq${target_quality_slug}"
    fi
    compressed_dir="${OUTPUT_ROOT}/${run_name}"
    metrics_json="${METRICS_DIR}/${run_name}.json"

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
            --compression-codec "${codec}"
            --compression-rc-mode "${rc_mode}"
            --compression-bitrate "${bitrate}"
            --compression-bitrate-max-multiplier "${bitrate_max_multiplier}"
            --compression-gop-length "${gop_length}"
            --compression-frame-interval-p "${frame_interval_p}"
            --compression-quant-group-size "${qg}"
            --compression-quant-outlier-ratio "${outlier_ratio}"
            --compression-codec-preset "${preset}"
            --compression-codec-tuning "${tuning}"
            --compression-codec-temporal-aq "${temporal_aq}"
        )
        if [[ "${spatial_aq}" != "none" ]]; then
            run_args+=(--compression-codec-spatial-aq "${spatial_aq}")
        fi
        if [[ "${target_quality}" != "none" ]]; then
            run_args+=(--compression-codec-target-quality "${target_quality}")
        fi
        if [[ "${rc_mode}" == "constqp" ]]; then
            run_args+=(--compression-const-qp "${const_qp}")
            if [[ "${qp_intra}" != "none" ]]; then
                run_args+=(--compression-const-qp-intra "${qp_intra}")
            fi
            if [[ "${qp_inter_p}" != "none" ]]; then
                run_args+=(--compression-const-qp-inter-p "${qp_inter_p}")
            fi
            if [[ "${qp_inter_b}" != "none" ]]; then
                run_args+=(--compression-const-qp-inter-b "${qp_inter_b}")
            fi
        fi
        python -u scripts/run_flux_multi_gpu_optimized.py "${run_args[@]}" \
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

python - "${OUTPUT_ROOT}/sweep_summary.json" "${TARGET_RATIO}" <<'PY'
import json
import sys
from pathlib import Path

rows = json.loads(Path(sys.argv[1]).read_text())
target = float(sys.argv[2])
passing = [
    r for r in rows
    if r.get("passes_quality_gate")
    and isinstance(r.get("total_compression_ratio"), (int, float))
]
passing_target = [r for r in passing if r["total_compression_ratio"] >= target]
print("")
print("Passing candidates:", len(passing))
print(f"Passing candidates with ratio >= {target}x:", len(passing_target))
for row in sorted(passing_target, key=lambda r: r["total_compression_ratio"], reverse=True):
    print(
        f"  {row['run_name']} ratio={row['total_compression_ratio']:.3f} "
        f"PSNR={row['compressed_vs_cache_psnr']:.3f} "
        f"SSIM={row['compressed_vs_cache_ssim']:.3f} "
        f"LPIPS={row['compressed_vs_cache_lpips']:.4f}"
    )
PY

echo ""
echo "Summary: ${OUTPUT_ROOT}/sweep_summary.csv"
echo "Recommendation: ${OUTPUT_ROOT}/recommended_config.json"
echo "Log: ${LOG_FILE}"
