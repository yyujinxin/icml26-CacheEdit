"""Real-GPU multi-round image editing test for Flux Kontext.

Each metadata row carries instruction0..instructionN. We edit one image
sequentially: round 0 uses the original image, and every later round uses the
previous round's output as input. Activation cache (when enabled) is reused
across rounds for the same image, then reset before the next image.

Built on the refactored ``cache_edit`` package (FluxCacheManager +
init_flux_pipeline), referencing the original
``Flux-kontext/Flux_diffuser/Flux_cache_multi_round.py``.

Example:
    python scripts/run_multi_round_flux.py \
        --model-path /home/dataset-local/chenxueqing/model/black-forest-labs/FLUX.1-Kontext-dev \
        --data-root /home/dataset-local/chenxueqing/datasets/test \
        --image-idx 0000 \
        --output-dir ./outputs/flux_multi_round \
        --use-cache
"""

import argparse
import json
import os
import re
import time
from pathlib import Path

import torch
from PIL import Image


def sanitize(prompt: str, max_len: int = 60) -> str:
    s = re.sub(r"[^a-z0-9]+", "_", prompt.strip().lower()).strip("_")
    return (s[:max_len] or "noprompt")


def load_metadata(path: Path):
    rows = []
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def extract_instructions(row: dict):
    """Return prompts ordered by the integer suffix of instruction* keys."""
    items = []
    for k, v in row.items():
        if k.startswith("instruction") and k[len("instruction"):].isdigit():
            items.append((int(k[len("instruction"):]), v))
    items.sort(key=lambda x: x[0])
    return [p for _, p in items]


def build_pipeline(args):
    from cache_edit.models.flux import (
        create_default_cache_manager,
        init_flux_pipeline,
    )

    if not args.use_cache:
        # No-cache baseline: use CacheFluxKontextPipeline but without cache_context
        # to preserve resolution auto-adjustment logic
        pipeline = init_flux_pipeline(
            model_path=args.model_path,
            device=args.device,
            dtype=torch.bfloat16,
            cache_manager=None,  # No caching
        )
        pipeline.set_progress_bar_config(disable=False)
        return pipeline, None

    cache_manager = create_default_cache_manager(
        num_inference_steps=args.num_inference_steps,
        threshold=args.threshold,
        cache_interval=args.cache_interval,
        cache_device=torch.device(args.device),
        num_gpus=args.num_gpus,
        gpu_memory_limit_gb=args.gpu_memory_limit_gb,
        gpu_memory_buffer_gb=args.gpu_memory_buffer_gb,
    )

    # 多卡时使用 device_map="balanced" 平衡分配模型到多张卡
    if args.num_gpus > 1:
        pipeline = init_flux_pipeline(
            model_path=args.model_path,
            device=args.device,
            dtype=torch.bfloat16,
            cache_manager=cache_manager,
            device_map="balanced",
        )
    else:
        pipeline = init_flux_pipeline(
            model_path=args.model_path,
            device=args.device,
            dtype=torch.bfloat16,
            cache_manager=cache_manager,
        )
    pipeline.set_progress_bar_config(disable=False)
    return pipeline, cache_manager


def run_image(pipeline, cache_manager, row, args):
    prompts = extract_instructions(row)
    if not prompts:
        print(f"[skip] no instruction* fields for {row.get('image_idx')}")
        return []

    key = row.get("image_idx") or row["file_name"]
    img_path = Path(args.data_root) / row["file_name"]
    if not img_path.is_file():
        print(f"[skip] missing image: {img_path}")
        return []

    gen_dir = Path(args.output_dir) / "generation"
    gen_dir.mkdir(parents=True, exist_ok=True)

    current_image = Image.open(img_path).convert("RGB")

    # 缩放图像以适应显存限制（4090 24GB）
    max_size = 512  # 降低到512以适应多卡时激活值仍在主卡的限制
    if max(current_image.size) > max_size:
        w, h = current_image.size
        if w > h:
            new_w, new_h = max_size, int(h * max_size / w)
        else:
            new_w, new_h = int(w * max_size / h), max_size
        # 调整到8的倍数（Flux要求）
        new_w = (new_w // 8) * 8
        new_h = (new_h // 8) * 8
        current_image = current_image.resize((new_w, new_h), Image.Resampling.LANCZOS)
        print(f"[image {key}] Resized from {w}x{h} to {new_w}x{new_h}")

    print(f"[image {key}] {len(prompts)} rounds, src={img_path}, size={current_image.size}")

    timings = []
    for r, prompt in enumerate(prompts):
        generator = torch.Generator(device="cpu").manual_seed(args.seed)
        inputs = dict(
            image=current_image,
            prompt=prompt,
            negative_prompt=" ",
            true_cfg_scale=args.true_cfg_scale,
            num_inference_steps=args.num_inference_steps,
            guidance_scale=args.guidance_scale,
            generator=generator,
        )

        if torch.cuda.is_available():
            torch.cuda.synchronize()
        t0 = time.time()
        with torch.inference_mode():
            output = pipeline(**inputs)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        cost = time.time() - t0

        out_image = output.images[0]
        save_path = gen_dir / f"{key}_r{r}_{sanitize(prompt)}.png"
        out_image.save(save_path)
        timings.append(cost)
        print(
            f"  round {r}: round={getattr(cache_manager, 'current_round', 'n/a')} "
            f"prompt='{prompt[:50]}' -> {save_path.name} ({cost:.2f}s)"
        )

        # feed this round's output into the next round
        current_image = out_image

    if cache_manager is not None:
        cache_manager.reset()

    return timings


def get_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--model-path",
        default="/home/dataset-local/chenxueqing/model/black-forest-labs/FLUX.1-Kontext-dev",
    )
    p.add_argument(
        "--data-root", default="/home/dataset-local/chenxueqing/datasets/test"
    )
    p.add_argument(
        "--metadata",
        default=None,
        help="Path to metadata jsonl (default: <data-root>/metadata_multi_round.jsonl)",
    )
    p.add_argument(
        "--image-idx",
        default=None,
        help="Which image_idx to run (default: first row). Use 'all' for the whole set.",
    )
    p.add_argument("--output-dir", default="./outputs/flux_multi_round")
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--num-gpus", type=int, default=1, help="Number of GPUs for multi-GPU cache placement")
    p.add_argument("--num-inference-steps", type=int, default=28)
    p.add_argument("--guidance-scale", type=float, default=2.5)
    p.add_argument("--true-cfg-scale", type=float, default=1.0)
    p.add_argument("--threshold", type=float, default=0.97, help="Key-token similarity threshold")
    p.add_argument("--cache-interval", type=int, default=5, help="Cache interval (steps); smaller = denser cache, more memory")
    p.add_argument("--gpu-memory-limit-gb", type=float, default=None, help="GPU memory limit (GB); None = auto-detect")
    p.add_argument("--gpu-memory-buffer-gb", type=float, default=1.0, help="GPU memory buffer (GB) to prevent OOM")
    p.add_argument("--seed", type=int, default=110)
    p.add_argument("--use-cache", action="store_true", help="Enable activation cache")
    return p.parse_args()


def main():
    args = get_args()
    meta_path = Path(args.metadata) if args.metadata else (
        Path(args.data_root) / "metadata_multi_round.jsonl"
    )
    rows = load_metadata(meta_path)
    if not rows:
        print(f"empty metadata: {meta_path}")
        return 1

    if args.image_idx == "all":
        selected = rows
    elif args.image_idx is not None:
        selected = [r for r in rows if str(r.get("image_idx")) == args.image_idx]
        if not selected:
            print(f"image_idx={args.image_idx} not found in {meta_path}")
            return 1
    else:
        selected = rows[:1]

    print(
        f"model={args.model_path}\ncache={'ON' if args.use_cache else 'OFF'} "
        f"steps={args.num_inference_steps} guidance={args.guidance_scale} "
        f"threshold={args.threshold} interval={args.cache_interval}"
    )
    pipeline, cache_manager = build_pipeline(args)

    all_timings = {}
    for row in selected:
        key = row.get("image_idx") or row["file_name"]
        all_timings[key] = run_image(pipeline, cache_manager, row, args)

    os.makedirs(args.output_dir, exist_ok=True)
    flat = [t for ts in all_timings.values() for t in ts]
    summary = {
        "use_cache": args.use_cache,
        "num_images": len(all_timings),
        "avg_round_time": sum(flat) / len(flat) if flat else 0.0,
        "per_image_round_times": all_timings,
    }
    with open(Path(args.output_dir) / "timings.json", "w") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print(f"\navg round time: {summary['avg_round_time']:.2f}s")
    print(f"results -> {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
