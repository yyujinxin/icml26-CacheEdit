#!/bin/bash
# Sweep activation compression quantization/GOP parameters and evaluate quality.

set -euo pipefail

source .venv/bin/activate

# Edit parameters in this block directly.

# PyTorch CUDA allocator setting. expandable_segments:True reduces fragmentation.
PYTORCH_CUDA_ALLOC_CONF_VALUE="expandable_segments:True"

# Local FLUX-Kontext model directory.
MODEL_PATH="/mnt/data/models/black-forest-labs/FLUX___1-Kontext-dev"

# Dataset root. The runner reads metadata and input images from this directory.
DATA_ROOT="/mnt/data/datasets/test"

# Image id from the dataset metadata. Use a small representative case first.
IMAGE_IDX="0000"

# Number of edit rounds for each run. Increase after narrowing the parameter set.
MAX_ROUNDS="5"

# Shared output root. Each parameter set writes into a subdirectory here.
OUTPUT_ROOT="./outputs/compression_quant_sweep_28step"

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

# Quantization group sizes to test before codec encoding. Smaller values usually
# improve activation fidelity but increase scale/offset metadata. 0 forces
# channel-wise quantization.
QUANT_GROUP_SIZES=(128 256 512 64 32 16 0)

# GOP:P pairs to test. Example "16:16" means gop_length=16 and P-frame interval=16.
GOP_CONFIGS=("8:8" "16:16")

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
echo "Compression quantization/GOP sweep"
echo "=========================================="
echo "Output root: ${OUTPUT_ROOT}"
echo "Image idx: ${IMAGE_IDX}, rounds: ${MAX_ROUNDS}, steps: ${NUM_INFERENCE_STEPS}"
echo "Quant group sizes: ${QUANT_GROUP_SIZES[*]}"
echo "GOP configs: ${GOP_CONFIGS[*]}"
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
    for quant_group_size in "${QUANT_GROUP_SIZES[@]}"; do
        run_name="codec_${COMPRESSION_CODEC}_gop${gop_length}_p${frame_interval_p}_qg${quant_group_size}"
        compressed_dir="${OUTPUT_ROOT}/${run_name}"
        metrics_json="${METRICS_DIR}/${run_name}.json"

        echo ""
        echo "[compressed] ${run_name}"
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
            --compression-quant-group-size "${quant_group_size}"

        echo "[metrics] ${run_name}"
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
    done
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
echo "Sweep completed"
echo "Metrics: ${METRICS_DIR}"
echo "Summary: ${OUTPUT_ROOT}/sweep_summary.csv"
echo "Recommendation: ${OUTPUT_ROOT}/recommended_config.json"
echo "Log: ${LOG_FILE}"
echo "Timings, compression reports, and CUDA memory peaks are under each run directory."
echo "=========================================="
