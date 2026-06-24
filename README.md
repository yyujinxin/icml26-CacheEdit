# CacheEdit

CacheEdit is an experimental system for multi-round diffusion image editing with
activation caching. The current active path in this repository is the
FLUX.1-Kontext multi-round runner with:

- activation cache reuse across edit rounds;
- key-token based partial transformer computation;
- multi-GPU transformer/offload placement;
- optional activation compression through NVIDIA NVENC/NVDEC;
- inter-layer GOP compression where consecutive transformer layers are encoded
  as video frames;
- PSNR / SSIM / LPIPS evaluation scripts for cache quality comparison.

The optimized and maintained workflow is centered on
`scripts/run_flux_multi_gpu_optimized.py`. Older Qwen scripts and CLI entrypoints
still exist in the tree, but they are not the main path used by the current
compression and quality-evaluation work.

## Current Stable Setup

The current stable FLUX configuration is:

```text
num_inference_steps = 28
cache_interval = 5
compression_codec = lossless
compression_gop_length = 16
compression_frame_interval_p = 16
```

Important constraints:

- Do not manually pass `--width` or `--height`.
- Input image size should be handled by the pipeline's internal `_auto_resize`
  path.
- `lossless` still uses HEVC/NVENC codec. It does not bypass codec storage.
  FP16 activations are quantized to uint8 frames first; the codec then encodes
  those frames in lossless mode.
- P frames are supported. The native decoder handles flush calls that return
  multiple frames, so GOP/P-frame activation recovery remains aligned.

## Environment

Typical local setup:

```bash
cd /home/yujinxin/icml26-CacheEdit
python3.10 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install -e .
```

The compression path also needs a working NVIDIA driver, CUDA, NVENC/NVDEC, and
the local LLM.265 / NVIDIA Video Codec SDK sources expected by
`scripts/build_cacheedit_ops.py`.

If the native extension has changed or is missing, rebuild it:

```bash
source .venv/bin/activate
python scripts/build_cacheedit_ops.py
```

## Data And Model Layout

The default scripts assume:

```text
model:   /mnt/data/models/black-forest-labs/FLUX___1-Kontext-dev
dataset: /mnt/data/datasets/test
```

The dataset root should contain metadata plus input images. The optimized runner
reads `metadata_multi_round.jsonl` by default unless `--metadata` is provided.

## Main Commands

Run the full 28-step cache + codec compression test:

```bash
bash scripts/test_gop28_full.sh
```

Run the full 28-step no-cache baseline:

```bash
bash scripts/test_no_cache_28_full.sh
```

Run the three-way quality comparison:

```bash
bash scripts/test_cache_quality_metrics.sh
```

This compares:

- no-cache baseline;
- cache-only;
- cache + activation compression/recovery.

It then evaluates generated images with PSNR, SSIM, and LPIPS.

Run cache + compression over the full dataset with resume support:

```bash
bash scripts/run_cache_compressed_full_dataset_resume.sh
```

Script parameters are edited directly at the top of each `.sh` file. The current
scripts intentionally do not rely on environment-variable parameter injection.

## Direct Runner Example

```bash
source .venv/bin/activate

PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
python scripts/run_flux_multi_gpu_optimized.py \
    --model-path /mnt/data/models/black-forest-labs/FLUX___1-Kontext-dev \
    --data-root /mnt/data/datasets/test \
    --output-dir ./outputs/flux_28step_lossless_gop16 \
    --image-idx 0000 \
    --num-gpus 4 \
    --gpu-memory-limit-gb 16.0 \
    --gpu-memory-buffer-gb 5.0 \
    --num-inference-steps 28 \
    --guidance-scale 3.5 \
    --seed 42 \
    --use-cache \
    --cache-interval 5 \
    --threshold 0.97 \
    --use-cache-compression \
    --compression-codec lossless \
    --compression-bitrate 5.0 \
    --compression-gop-length 16 \
    --compression-frame-interval-p 16
```

`--compression-bitrate` is ignored by `lossless`; it is only meaningful for
lossy `hevc` / `h264` modes.

## Outputs

The runner writes:

- generated images under `<output-dir>/generation`;
- `timings.partial.json` during execution and on interruptions;
- `timings.json` after successful completion.

When cache compression is enabled, `timings.json` includes a
`compression.summary` section. Useful fields include:

- `success_count` / `failure_count`;
- `success_count_by_mode`;
- `success_count_by_quantization`;
- `payload_compression_ratio`;
- `total_compression_ratio`;
- `decompression_failure_count`;
- `gop_decode_cache_hit_count`;
- `gop_decode_cache_miss_count`.

`payload_compression_ratio` counts codec bitstream bytes only.
`total_compression_ratio` also includes quantization metadata and packet-size
metadata, so it better reflects real cache memory use.

## How Cache Compression Works

For each cache step, the cache manager groups consecutive transformer layers
within the same diffusion step:

1. FP16 activation is reshaped to `[tokens, hidden]`.
2. It is quantized to uint8 frames. In `lossless` mode the preferred quantizer
   is `GWQuantization(groupsize=64)`.
3. `FixedTiling` pads and tiles the frame into NVENC-compatible chunks.
4. `MonoNVEncodeSequence` encodes the layer sequence as HEVC/H.264 video.
5. The compressed payload and quantization metadata are stored in cache entries.

On reuse, NVDEC decodes the GOP, the original layer frame is selected by
`frame_index`, padding is removed, and quantization metadata restores the FP16
activation before the transformer continues.

See [README_COMPRESSION.md](README_COMPRESSION.md) for the detailed compression
data flow and troubleshooting notes.

## Important Implementation Files

- `scripts/run_flux_multi_gpu_optimized.py`: main FLUX runner.
- `cache_edit/models/flux/cache_manager.py`: cache ownership, GOP grouping,
  compression/decompression reporting.
- `cache_edit/models/flux/transformer_forward.py`: key-token reuse and partial
  transformer execution.
- `cache_edit/compression/activation_compressor.py`: activation quantization,
  tiling, NVENC/NVDEC orchestration.
- `cache_edit/compression/pipeline/nvenc.py`: Python codec pipeline wrappers.
- `cache_edit/compression/csrc/`: native NVENC/NVDEC extension.
- `scripts/evaluate_image_metrics.py`: PSNR / SSIM / LPIPS evaluation.

## Profiling And Optimization Notes

Detailed optimization history, Nsight findings, OOM notes, and current tradeoffs
are documented in:

- [docs/OPTIMIZATION_GUIDE.md](docs/OPTIMIZATION_GUIDE.md)
- [README_COMPRESSION.md](README_COMPRESSION.md)

The current stable implementation keeps GOP compression/decompression
synchronous. Async compression and GOP prefetch code paths are retained for
future experiments, but default settings keep them disabled because overlapping
native NVENC/NVDEC calls from Python worker threads previously caused CUDA
context and resource-stability problems.

## Development Checks

Fast syntax checks:

```bash
source .venv/bin/activate
python -m py_compile \
    cache_edit/compression/activation_compressor.py \
    cache_edit/compression/pipeline/nvenc.py \
    cache_edit/models/flux/cache_manager.py \
    cache_edit/models/flux/pipeline.py \
    cache_edit/models/flux/transformer_forward.py \
    scripts/run_flux_multi_gpu_optimized.py \
    scripts/evaluate_image_metrics.py

bash -n \
    scripts/test_gop28_full.sh \
    scripts/test_no_cache_28_full.sh \
    scripts/test_cache_quality_metrics.sh \
    scripts/run_cache_compressed_full_dataset_resume.sh
```

GPU/NVENC validation requires running on a machine with visible NVIDIA GPUs.

## License

See [LICENSE](LICENSE).
