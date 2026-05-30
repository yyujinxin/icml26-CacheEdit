"""Real-GPU multi-round image editing test for Qwen2-VL.

Each metadata row carries instruction0..instructionN. We edit one image
sequentially: round 0 uses the original image, and every later round uses the
previous round's output as input. Activation cache (when enabled) is reused
across rounds for the same image, then reset before the next image.

Built on the refactored ``cache_edit`` package (QwenCacheManager +
init_qwen_pipeline).

Example:
    python scripts/run_multi_round_qwen.py \
        --model-path Qwen/Qwen2-VL-7B-Instruct \
        --data-root /home/dataset-local/chenxueqing/datasets/test \
        --image-idx 0000 \
        --output-dir ./outputs/qwen_multi_round \
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
    from cache_edit.models.qwen import (
        create_default_cache_manager,
        init_qwen_pipeline,
    )

    if not args.use_cache:
        # No-cache baseline: use pipeline without cache_manager
        pipeline = init_qwen_pipeline(
            model_path=args.model_path,
            device=args.device,
            dtype=torch.bfloat16,
            cache_manager=None,  # No caching
        )
        return pipeline, None

    cache_manager = create_default_cache_manager(
        num_inference_steps=args.num_inference_steps,
        threshold=args.threshold,
        cache_interval=args.cache_interval,
        cache_device=torch.device(args.device),
        num_gpus=args.num_gpus,
    )

    # CFG doubles transformer calls per step (cond + uncond)
    cfg_on = args.true_cfg_scale > 1.0 and args.negative_prompt is not None
    cache_manager.calls_per_step = 2 if cfg_on else 1

    pipeline = init_qwen_pipeline(
        model_path=args.model_path,
        device=args.device,
        dtype=torch.bfloat16,
        cache_manager=cache_manager,
    )
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

    original_image = Image.open(img_path).convert("RGB")
    print(f"[image {key}] {len(prompts)} rounds, src={img_path} ({original_image.width}x{original_image.height})")

    # No manual resize: pipeline will internally compute calculated_dimensions
    # (~1M pixels, preserving aspect ratio). From round 1 onwards the input is
    # the previous round's output, which the pipeline produced at the same
    # calculated_dimensions, so token count stays consistent automatically.
    current_image = original_image

    timings = []
    for r, prompt in enumerate(prompts):
        generator = torch.Generator(device="cpu").manual_seed(args.seed)
        if cache_manager is not None:
            cache_manager.on_round_start()
        inputs = dict(
            image=current_image,
            prompt=prompt,
            num_inference_steps=args.num_inference_steps,
            guidance_scale=args.guidance_scale,
            true_cfg_scale=args.true_cfg_scale,
            negative_prompt=args.negative_prompt,
            max_sequence_length=args.max_sequence_length,
            generator=generator,
        )
        if args.height is not None:
            inputs["height"] = args.height
        if args.width is not None:
            inputs["width"] = args.width

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

        round_info = f"round={cache_manager.current_round}" if cache_manager else "round=n/a"
        print(
            f"  round {r}: {round_info} "
            f"prompt='{prompt[:50]}' -> {save_path.name} "
            f"({out_image.width}x{out_image.height}, {cost:.2f}s)"
        )

        # Feed output directly to next round; pipeline preserves dimensions.
        current_image = out_image

        # Flush activation cache after each round (for Flux-style caching)
        if cache_manager is not None:
            cache_manager.flush_activation_cache()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    if cache_manager is not None:
        cache_manager.clear_cache()

    return timings


def get_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--model-path",
        default="Qwen/Qwen2-VL-7B-Instruct",
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
    p.add_argument("--output-dir", default="./outputs/qwen_multi_round")
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--num-gpus", type=int, default=1, help="Number of GPUs for multi-GPU cache placement")
    p.add_argument("--num-inference-steps", type=int, default=50)
    p.add_argument("--guidance-scale", type=float, default=1.0)
    p.add_argument("--height", type=int, default=None, help="Output height (default: auto from input)")
    p.add_argument("--width", type=int, default=None, help="Output width (default: auto from input)")
    p.add_argument("--max-sequence-length", type=int, default=512, help="Max sequence length")
    p.add_argument("--threshold", type=float, default=0.99, help="Key-token similarity threshold")
    p.add_argument("--cache-interval", type=int, default=3, help="Cache interval (steps); smaller = denser cache, higher quality, less speedup")
    p.add_argument("--seed", type=int, default=110)
    p.add_argument("--use-cache", action="store_true", help="Enable activation cache")
    p.add_argument("--true-cfg-scale", type=float, default=4.0, help="CFG scale (set >1 to enable CFG)")
    p.add_argument("--negative-prompt", type=str, default=" ", help="Negative prompt for CFG")
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
