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

The current qg256 GOP/P-frame exploration starts from:

```text
num_inference_steps = 28
cache_interval = 5
compression_codec = lossless
compression_gop_length = 32
compression_frame_interval_p = 1
compression_quant_group_size = 256
compression_quant_outlier_ratio = 0.0
```

Important constraints:

- Do not manually pass `--width` or `--height`.
- Input image size should be handled by the pipeline's internal `_auto_resize`
  path.
- `lossless` still uses HEVC/NVENC codec. It does not bypass codec storage.
  FP16 activations are quantized to uint8 frames first; the codec then encodes
  those frames in lossless mode.
- `compression_quant_group_size` controls the group-wise FP16-to-uint8
  quantizer used before codec encoding. Smaller groups usually improve
  activation fidelity but increase scale/offset metadata; larger groups can
  improve total compression ratio at higher quantization error.
- `compression_quant_outlier_ratio` optionally stores a tiny fraction of the
  largest quantization residuals as side metadata. The main activation still
  goes through the uint8 codec path; `0` preserves the current qg behavior.
- P frames are supported. The native decoder handles flush calls that return
  multiple frames, so GOP/P-frame activation recovery remains aligned.
- GOP/P-frame settings are still being swept with qg256. Use
  `scripts/sweep_gop_params_qg256.sh` to pick the best setting for the target
  round count instead of assuming one setting is always optimal. In the current
  28-step, 2-round probe, `gop32,p1,qg256` gave the best compression ratio while
  still passing the strict cache-vs-compressed quality gate.

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

Sweep compression quantization/GOP settings:

```bash
bash scripts/sweep_compression_quant_params.sh
```

This runs no-cache and cache-only references once, then evaluates each
configured cache+compression setting. It writes `sweep_summary.csv`,
`sweep_summary.json`, and `recommended_config.json`; the last file contains both
the quality-first setting and the best compression-ratio setting that passes the
configured quality gate.

Sweep only GOP/P-frame settings while keeping qg256 fixed:

```bash
bash scripts/sweep_gop_params_qg256.sh
```

This is the preferred script for the current GOP search. It defaults to a
2-round coarse sweep: round 0 builds/compresses cache, and round 1 decodes and
reuses it. It reports PSNR, SSIM, LPIPS, compression ratio, and torch CUDA peak
memory for each candidate. Validate the selected candidate later with the target
round count.

Probe quantization error on real activations while keeping the actual run on
qg256:

```bash
bash scripts/probe_quant_error_qg256.sh
```

This writes `quant_error_summary.csv` and `quant_error_summary.json` next to the
run report. It compares candidate qg values on the same activation tensors
without changing the actual compression setting.

Sweep residual-outlier ratios while keeping `qg256,gop32,p1` fixed:

```bash
bash scripts/sweep_quant_outlier_qg256.sh
```

The activation-level probe found that storing `0.1%` of the worst residuals
cuts qg256 max activation error from roughly `110` to `25` with metadata ratio
rising from `0.098%` to `0.117%` of original activation bytes. This needs
image-level validation before becoming the default.

Sweep lossy HEVC bitrate while keeping qg256 and the selected GOP/P setting:

```bash
bash scripts/sweep_bitrate_qg256.sh
```

The current 28-step, 2-round probe uses `qg256,gop32,p1`. In that run, the
lossless codec anchor was the only setting that passed the strict quality gate
(`PSNR>=41`, `SSIM>=0.994`, `LPIPS<=0.004` against cache-only). HEVC bitrates
from `0.5` to `10.0` Mbps improved compression ratio but caused much larger
image drift; 5 Mbps also produced an invalid-value warning in image postprocess.

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
    --compression-gop-length 32 \
    --compression-frame-interval-p 1 \
    --compression-quant-group-size 256 \
    --compression-quant-outlier-ratio 0.0
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
- `quant_group_size`;
- `payload_compression_ratio`;
- `total_compression_ratio`;
- `decompression_failure_count`;
- `gop_decode_cache_hit_count`;
- `gop_decode_cache_miss_count`.

Every `timings.json` also includes `cuda_memory`, with per-GPU
`peak_allocated_gib` and `peak_reserved_gib` measured by `torch.cuda` after the
pipeline is initialized. This is useful for comparing GOP settings under the
same model placement.

When `--compression-quant-error-probe-groups` is set, `compression.summary`
also includes `quant_error_probe_by_quantization`. This reports activation-level
RMSE, MAE, relative RMSE, max error, and scale/offset metadata overhead for each
candidate qg.

`payload_compression_ratio` counts codec bitstream bytes only.
`total_compression_ratio` also includes quantization metadata and packet-size
metadata, so it better reflects real cache memory use.

The current 28-step, 3-round probe on `image_idx=0000` found two useful qg points
among tested `64/128/256/512/0`:

- `compression_quant_group_size=128` is the quality-first setting and gave the
  strongest compressed-vs-cache quality.
- `compression_quant_group_size=256` is the current optimized default and the
  best ratio under the default
  quality gate used by `scripts/summarize_compression_sweep.py`
  (`PSNR>=41`, `SSIM>=0.994`, `LPIPS<=0.004`, no compression failures), raising
  total compression ratio from `2.75x` to `3.19x`.

`512` is useful when prioritizing compression ratio more aggressively; `0`
forces channel-wise quantization and compressed much more aggressively in the
probe, but with visibly larger metric loss.

The group-wise quantizer uses a rounded zero-point restore path. A min-offset
group-wise variant was tested because it slightly reduced synthetic activation
error, but the real 28-step probe produced worse image metrics, so it is not the
current default.

For longer edit chains, quality drift can accumulate. A 5-round probe did not
pass the earlier strict 3-round quality gate, so GOP/P-frame selection should be
validated at the same round count that will be used in the target experiment.

## How Cache Compression Works

For each cache step, the cache manager groups consecutive transformer layers
within the same diffusion step:

1. FP16 activation is reshaped to `[tokens, hidden]`.
2. It is quantized to uint8 frames. In `lossless` mode the preferred quantizer
   is `GWQuantization(groupsize=<compression_quant_group_size>)`; the default
   is 256. Group-wise quantization uses rounded zero-points; channel-wise
   fallback uses min-offset restore.
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
- `scripts/sweep_compression_quant_params.sh`: parameter sweep for GOP and
  quantization group size.
- `scripts/sweep_gop_params_qg256.sh`: GOP/P-frame sweep with qg256 fixed.
- `scripts/probe_quant_error_qg256.sh`: real-activation quantization error
  probe for qg candidates.

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
    scripts/run_cache_compressed_full_dataset_resume.sh \
    scripts/sweep_compression_quant_params.sh \
    scripts/sweep_gop_params_qg256.sh \
    scripts/probe_quant_error_qg256.sh
```

GPU/NVENC validation requires running on a machine with visible NVIDIA GPUs.

## License

See [LICENSE](LICENSE).
