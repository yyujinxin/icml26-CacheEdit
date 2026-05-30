# CacheEdit

Efficient activation caching for multi-round diffusion-based image editing.

CacheEdit accelerates **Qwen-Image-Edit** and **FLUX.1-Kontext** in multi-round
editing scenarios by caching transformer-block activations across rounds and
reusing them for tokens whose hidden state hasn't meaningfully changed
(Flux-style key-token mechanism with **dynamic per-step refresh**).

## Features

- **Two backends**: Qwen-Image-Edit and FLUX.1-Kontext pipelines
- **Activation cache + dynamic key-token refresh**: at every cache step in
  Round 1+, the manager compares the current hidden state against the cached
  reference from Round 0, and **re-computes `key_token_indices` on the fly** —
  so the "edit region" mask adapts to each round's prompt and to denoising
  progress within the round
- **Partial computation at reuse steps**: image tokens are rearranged so key
  tokens sit at the front, sliced down to K, and only K go through the 60
  transformer blocks; non-key positions are restitched from the per-layer cache
- **CFG-aware cache (Qwen)**: separate cond / uncond caches with correct step
  counting under true CFG (`true_cfg_scale > 1`)
- **Multi-GPU placement**: cache tensors are distributed across cards based on
  measured free memory, never on the model device
- **Unified config**: dataclass + YAML/JSON, with environment-variable overrides

## Installation

```bash
git clone https://github.com/yyujinxin/icml26-CacheEdit.git
cd icml26-CacheEdit
pip install -e .
```

Runtime: `torch`, `diffusers >= 0.35`, `transformers`, `Pillow`, `pyyaml`.

## Quick start

### Multi-round Qwen (CFG enabled, dynamic key-token cache)

```bash
python scripts/run_multi_round_qwen.py \
    --model-path /path/to/Qwen-Image-Edit \
    --image-idx 0000 \
    --num-inference-steps 30 \
    --cache-interval 5 \
    --num-gpus 4 \
    --use-cache \
    --true-cfg-scale 4.0 \
    --negative-prompt " "
```

Compared to the no-cache baseline (same CFG settings), the cache run is
roughly **3.4× faster on subsequent rounds** at the default `cache-interval=5`.

### Multi-round Flux Kontext

```bash
python scripts/run_multi_round_flux.py \
    --model-path /path/to/FLUX.1-Kontext-dev \
    --image-idx 0000 \
    --num-inference-steps 28 \
    --cache-interval 5 \
    --num-gpus 4 \
    --use-cache
```

### Python API

```python
import torch
from PIL import Image
from cache_edit.models.qwen import (
    create_default_cache_manager,
    init_qwen_pipeline,
)

cache_manager = create_default_cache_manager(
    num_inference_steps=30,
    threshold=0.99,
    cache_interval=5,
    cache_device=torch.device("cuda:0"),
    num_gpus=4,
)
# CFG doubles transformer calls per step (cond + uncond) — tell the manager.
cache_manager.calls_per_step = 2

pipeline = init_qwen_pipeline(
    model_path="/path/to/Qwen-Image-Edit",
    device="cuda:0",
    dtype=torch.bfloat16,
    cache_manager=cache_manager,
)

img = Image.open("input.png").convert("RGB")
for prompt in ["make the cat fluffier", "add a red collar", "change the bg"]:
    cache_manager.on_round_start()
    out = pipeline(
        image=img,
        prompt=prompt,
        num_inference_steps=30,
        true_cfg_scale=4.0,
        negative_prompt=" ",
    ).images[0]
    img = out  # feed into next round
    cache_manager.flush_activation_cache()
```

## How the cache works

1. **Round 0** — full computation; every cache step stores per-layer image and
   text activations to `prev_activation_cache` (under `(mode, stream, layer,
   step)` key).
2. **Round 1+ cache step** — full computation again, but after the last block
   it compares the current hidden state with the Round 0 cache at the same
   step. Tokens whose cosine similarity falls below `threshold` are recorded as
   `key_token_indices` for subsequent reuse steps.
3. **Round 1+ reuse step** — image tokens are rearranged so key positions sit
   at the front, then sliced to K tokens. All 60 transformer blocks run on the
   short sequence; after each block the cached non-key positions are pulled
   from the previous round's per-layer cache and concatenated back to the full
   length. Order is restored before `proj_out`.

Higher `cache_interval` → more reuse steps → bigger speedup, but the key-token
set is computed less often. The right setting depends on how local your edits
are.

## Config

Default configs live in `configs/`:

- `configs/qwen_default.yaml`
- `configs/flux_default.yaml`

Override any field via environment variables (prefix `CACHEEDIT_`, nested with
double underscores):

```bash
export CACHEEDIT_CACHE__THRESHOLD=0.99
export CACHEEDIT_MODEL__DEVICE=cuda:0
```

## CLI

| Command | Description |
| --- | --- |
| `cache-edit edit` | Edit a single image with a text instruction |
| `cache-edit benchmark` | Compare latency with/without the cache |
| `cache-edit --version` | Print version |

See `cache-edit edit --help` for the full parameter list.

## Docs

- [API reference](docs/API.md)
- [Tutorial](docs/TUTORIAL.md)

## Development

```bash
pytest tests/ -v
```

CI runs the test suite on Python 3.8–3.11 (see `.github/workflows/test.yml`).

## License

See `LICENSE`.
