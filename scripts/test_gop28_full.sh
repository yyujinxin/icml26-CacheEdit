#!/bin/bash
# Full 28-step Flux multi-round test with inter-layer GOP compression.

set -euo pipefail

source .venv/bin/activate

export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

MODEL_PATH="${MODEL_PATH:-/mnt/data/models/black-forest-labs/FLUX___1-Kontext-dev}"
DATA_ROOT="${DATA_ROOT:-/mnt/data/datasets/test}"
OUTPUT_DIR="${OUTPUT_DIR:-./outputs/flux_gop8_28step_full}"
IMAGE_IDX="${IMAGE_IDX:-0000}"
NUM_GPUS="${NUM_GPUS:-4}"
GPU_MEMORY_LIMIT_GB="${GPU_MEMORY_LIMIT_GB:-16.0}"
GPU_MEMORY_BUFFER_GB="${GPU_MEMORY_BUFFER_GB:-5.0}"
CACHE_INTERVAL="${CACHE_INTERVAL:-5}"
NUM_INFERENCE_STEPS="${NUM_INFERENCE_STEPS:-28}"
COMPRESSION_BITRATE="${COMPRESSION_BITRATE:-5.0}"
COMPRESSION_CODEC="${COMPRESSION_CODEC:-hevc}"
COMPRESSION_GOP_LENGTH="${COMPRESSION_GOP_LENGTH:-8}"
COMPRESSION_FRAME_INTERVAL_P="${COMPRESSION_FRAME_INTERVAL_P:-1}"
GUIDANCE_SCALE="${GUIDANCE_SCALE:-3.5}"
THRESHOLD="${THRESHOLD:-0.97}"
SEED="${SEED:-42}"

echo "=========================================="
echo "Full 28-step GOP compression test"
echo "=========================================="
echo ""
echo "Configuration:"
echo "  - Image idx: ${IMAGE_IDX}"
echo "  - Output dir: ${OUTPUT_DIR}"
echo "  - Inference steps: ${NUM_INFERENCE_STEPS}"
echo "  - Cache interval: ${CACHE_INTERVAL} (28-step cache steps: 0, 5, 10, 15, 20, 25)"
echo "  - Compression: ${COMPRESSION_CODEC} ${COMPRESSION_BITRATE}Mbps"
echo "  - Inter-layer GOP: length=${COMPRESSION_GOP_LENGTH}, frame_interval_p=${COMPRESSION_FRAME_INTERVAL_P}"
echo "  - GPUs: ${NUM_GPUS}, memory limit=${GPU_MEMORY_LIMIT_GB}GB, buffer=${GPU_MEMORY_BUFFER_GB}GB"
echo ""

python scripts/run_flux_multi_gpu_optimized.py \
    --model-path "${MODEL_PATH}" \
    --data-root "${DATA_ROOT}" \
    --output-dir "${OUTPUT_DIR}" \
    --image-idx "${IMAGE_IDX}" \
    --num-gpus "${NUM_GPUS}" \
    --gpu-memory-limit-gb "${GPU_MEMORY_LIMIT_GB}" \
    --gpu-memory-buffer-gb "${GPU_MEMORY_BUFFER_GB}" \
    --use-cache \
    --use-cache-compression \
    --compression-bitrate "${COMPRESSION_BITRATE}" \
    --compression-codec "${COMPRESSION_CODEC}" \
    --compression-gop-length "${COMPRESSION_GOP_LENGTH}" \
    --compression-frame-interval-p "${COMPRESSION_FRAME_INTERVAL_P}" \
    --num-inference-steps "${NUM_INFERENCE_STEPS}" \
    --cache-interval "${CACHE_INTERVAL}" \
    --guidance-scale "${GUIDANCE_SCALE}" \
    --threshold "${THRESHOLD}" \
    --seed "${SEED}"

echo ""
echo "=========================================="
echo "Full GOP test completed"
echo "Report: ${OUTPUT_DIR}/timings.json"
echo "Images: ${OUTPUT_DIR}/generation"
echo "=========================================="
