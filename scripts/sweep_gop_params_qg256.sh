#!/bin/bash
# Sweep inter-layer GOP/P-frame settings with fixed qg256 quantization.

set -euo pipefail

source .venv/bin/activate

# Edit parameters in this block directly.

# PyTorch CUDA allocator setting. expandable_segments:True reduces fragmentation.
PYTORCH_CUDA_ALLOC_CONF_VALUE="expandable_segments:True"

# Local FLUX-Kontext model directory.
MODEL_PATH="/mnt/data/models/black-forest-labs/FLUX___1-Kontext-dev"

# Dataset root. The runner reads metadata and input images from this directory.
DATA_ROOT="/mnt/data/datasets/test"

# Image id from the dataset metadata. Use "all" only for the final full pass.
IMAGE_IDX="0000"

# Number of edit rounds per run. For GOP/P exploration, two rounds are enough
# to measure compression on round 0 and decompression/reuse on round 1. Validate
# the selected candidate later with 5 or 8 rounds.
MAX_ROUNDS="2"

# Shared output root. Each GOP/P setting writes into a subdirectory here.
OUTPUT_ROOT="./outputs/gop_param_sweep_qg256_28step_2round"

# Number of GPUs used by the optimized multi-GPU runner.
NUM_GPUS="4"

# Soft per-GPU memory limit in GiB used by the offload/cache manager.
GPU_MEMORY_LIMIT_GB="16.0"

# Extra GiB kept free as a safety buffer before placing tensors on a GPU.
GPU_MEMORY_BUFFER_GB="5.0"

# Number of denoising steps. Keep at 28 for full-step quality comparison.
NUM_INFERENCE_STEPS="28"

# Cache reuse interval. For 28 steps, cache anchor steps are 0, 5, 10, 15, 20, 25.
CACHE_INTERVAL="5"

# FLUX guidance scale. Keep aligned across all modes for fair comparison.
GUIDANCE_SCALE="3.5"

# Cache similarity/reuse threshold.
THRESHOLD="0.97"

# Random seed for deterministic generation when the runtime is otherwise stable.
SEED="42"

# Activation compression codec. lossless uses HEVC/NVENC lossless mode after
# FP16 activations are quantized to uint8 frames.
COMPRESSION_CODEC="lossless"

# NVENC bitrate in Mbps. Ignored by COMPRESSION_CODEC=lossless.
COMPRESSION_BITRATE="5.0"

# Fixed quantization group size requested for this GOP/P sweep.
COMPRESSION_QUANT_GROUP_SIZE="256"

# GOP:P pairs to test. GOP length controls how many consecutive layers are
# encoded together. frame_interval_p=1 means IPPP..., so every non-I frame is a
# P frame and no B frames are used. Longer GOPs increase the number of P frames
# per codec group.
# gop1/gop4/gop8 were measured as too slow on reuse rounds because they create
# too many small codec groups, so they are intentionally excluded by default.
GOP_CONFIGS=(
    "16:1"
    "24:1"
    "32:1"
    "48:1"
    "57:1"
)

# Set to 1 to rerun baseline and cache-only. Set to 0 to reuse existing outputs.
RUN_BASELINE_AND_CACHE_ONLY="0"

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
echo "GOP/P-frame sweep with qg256"
echo "=========================================="
echo "Output root: ${OUTPUT_ROOT}"
echo "Image idx: ${IMAGE_IDX}, rounds: ${MAX_ROUNDS}, steps: ${NUM_INFERENCE_STEPS}"
echo "GOP configs: ${GOP_CONFIGS[*]}"
echo "Quant group size: ${COMPRESSION_QUANT_GROUP_SIZE}"
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

for gop_pair in "${GOP_CONFIGS[@]}"; do
    gop_length="${gop_pair%%:*}"
    frame_interval_p="${gop_pair##*:}"
    run_name="codec_${COMPRESSION_CODEC}_gop${gop_length}_p${frame_interval_p}_qg${COMPRESSION_QUANT_GROUP_SIZE}"
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
            --compression-gop-length "${gop_length}" \
            --compression-frame-interval-p "${frame_interval_p}" \
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
echo "GOP/P sweep completed"
echo "Metrics: ${METRICS_DIR}"
echo "Summary: ${OUTPUT_ROOT}/sweep_summary.csv"
echo "Recommendation: ${OUTPUT_ROOT}/recommended_config.json"
echo "Log: ${LOG_FILE}"
echo "Timings, compression reports, and CUDA memory peaks are under each run directory."
echo "=========================================="
