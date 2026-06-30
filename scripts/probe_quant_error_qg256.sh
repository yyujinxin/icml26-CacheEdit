#!/bin/bash
# Probe FP16->uint8 quantization error on real Flux activations.

set -euo pipefail

source .venv/bin/activate

# Edit parameters in this block directly.

PYTORCH_CUDA_ALLOC_CONF_VALUE="expandable_segments:True"
MODEL_PATH="/mnt/data/models/black-forest-labs/FLUX___1-Kontext-dev"
DATA_ROOT="/mnt/data/datasets/test"
IMAGE_IDX="0000"
MAX_ROUNDS="2"
OUTPUT_DIR="./outputs/quant_error_probe_qg256_28step_2round"
NUM_GPUS="4"
GPU_MEMORY_LIMIT_GB="16.0"
GPU_MEMORY_BUFFER_GB="5.0"
NUM_INFERENCE_STEPS="28"
CACHE_INTERVAL="5"
GUIDANCE_SCALE="3.5"
THRESHOLD="0.97"
SEED="42"

# Actual compression setting used by the run.
COMPRESSION_CODEC="lossless"
COMPRESSION_BITRATE="5.0"
COMPRESSION_GOP_LENGTH="32"
COMPRESSION_FRAME_INTERVAL_P="1"
COMPRESSION_QUANT_GROUP_SIZE="256"
COMPRESSION_QUANT_OUTLIER_RATIO="0.0"

# Extra candidates evaluated on the same real activations. 0 means channel-wise.
COMPRESSION_QUANT_ERROR_PROBE_GROUPS="64,128,256,512,0"

# Residual-outlier ratios crossed with each probe group. Values >0 store the
# largest quantization residuals exactly as side metadata.
COMPRESSION_QUANT_ERROR_PROBE_OUTLIER_RATIOS="0,0.0005,0.001"

# Number of activation rows sampled per tensor for the probe. Increase this for
# stronger estimates; set to 0 for all rows.
COMPRESSION_QUANT_ERROR_PROBE_MAX_ROWS="512"

export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF_VALUE}"

mkdir -p "${OUTPUT_DIR}"
LOG_FILE="${OUTPUT_DIR}/probe.log"
exec > >(tee -a "${LOG_FILE}") 2>&1

echo "=========================================="
echo "Quantization error probe"
echo "=========================================="
echo "Output dir: ${OUTPUT_DIR}"
echo "Image idx: ${IMAGE_IDX}, rounds: ${MAX_ROUNDS}, steps: ${NUM_INFERENCE_STEPS}"
echo "Actual compression: codec=${COMPRESSION_CODEC}, GOP=${COMPRESSION_GOP_LENGTH}, P=${COMPRESSION_FRAME_INTERVAL_P}, QG=${COMPRESSION_QUANT_GROUP_SIZE}, outlier_ratio=${COMPRESSION_QUANT_OUTLIER_RATIO}"
echo "Probe groups: ${COMPRESSION_QUANT_ERROR_PROBE_GROUPS}, outlier_ratios=${COMPRESSION_QUANT_ERROR_PROBE_OUTLIER_RATIOS}, max_rows=${COMPRESSION_QUANT_ERROR_PROBE_MAX_ROWS}"
echo ""

python scripts/run_flux_multi_gpu_optimized.py \
    --model-path "${MODEL_PATH}" \
    --data-root "${DATA_ROOT}" \
    --image-idx "${IMAGE_IDX}" \
    --num-gpus "${NUM_GPUS}" \
    --gpu-memory-limit-gb "${GPU_MEMORY_LIMIT_GB}" \
    --gpu-memory-buffer-gb "${GPU_MEMORY_BUFFER_GB}" \
    --num-inference-steps "${NUM_INFERENCE_STEPS}" \
    --guidance-scale "${GUIDANCE_SCALE}" \
    --seed "${SEED}" \
    --max-rounds "${MAX_ROUNDS}" \
    --output-dir "${OUTPUT_DIR}" \
    --use-cache \
    --use-cache-compression \
    --cache-interval "${CACHE_INTERVAL}" \
    --threshold "${THRESHOLD}" \
    --compression-bitrate "${COMPRESSION_BITRATE}" \
    --compression-codec "${COMPRESSION_CODEC}" \
    --compression-gop-length "${COMPRESSION_GOP_LENGTH}" \
    --compression-frame-interval-p "${COMPRESSION_FRAME_INTERVAL_P}" \
    --compression-quant-group-size "${COMPRESSION_QUANT_GROUP_SIZE}" \
    --compression-quant-outlier-ratio "${COMPRESSION_QUANT_OUTLIER_RATIO}" \
    --compression-quant-error-probe-groups "${COMPRESSION_QUANT_ERROR_PROBE_GROUPS}" \
    --compression-quant-error-probe-outlier-ratios "${COMPRESSION_QUANT_ERROR_PROBE_OUTLIER_RATIOS}" \
    --compression-quant-error-probe-max-rows "${COMPRESSION_QUANT_ERROR_PROBE_MAX_ROWS}"

python scripts/summarize_quant_error_probe.py \
    --timings "${OUTPUT_DIR}/timings.json" \
    --csv-output "${OUTPUT_DIR}/quant_error_summary.csv" \
    --json-output "${OUTPUT_DIR}/quant_error_summary.json"

echo ""
echo "=========================================="
echo "Quantization error probe completed"
echo "Report: ${OUTPUT_DIR}/timings.json"
echo "Summary: ${OUTPUT_DIR}/quant_error_summary.csv"
echo "Log: ${LOG_FILE}"
echo "=========================================="
