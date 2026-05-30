"""benchmark subcommand: measure pipeline timing with vs without cache."""

import argparse
import time
from pathlib import Path
from typing import List

import torch

from cache_edit.cli.utils import load_config, resolve_dtype


def add_benchmark_parser(subparsers) -> argparse.ArgumentParser:
    """注册 `benchmark` 子命令。"""
    p = subparsers.add_parser(
        "benchmark",
        help="Benchmark inference latency with vs without activation cache",
        description=(
            "Run the same edit prompt N times with and without cache, "
            "reporting wall-clock latency and speedup."
        ),
    )
    p.add_argument(
        "--model", choices=["qwen", "flux"], required=True,
        help="Which model backend to use",
    )
    p.add_argument(
        "--image", type=Path, required=True,
        help="Path to the input image used for all rounds",
    )
    p.add_argument(
        "--prompt", type=str, required=True,
        help="Text instruction reused across rounds",
    )
    p.add_argument(
        "--config", type=Path, default=None,
        help="Optional config file. Defaults to configs/{model}_default.yaml",
    )
    p.add_argument(
        "--rounds", type=int, default=3,
        help="How many rounds to run with cache enabled (after round 0)",
    )
    p.add_argument(
        "--warmup", type=int, default=1,
        help="Warmup iterations before measurement (default 1)",
    )
    p.add_argument(
        "--output-dir", type=Path, default=Path("./bench_outputs"),
        help="Directory to save per-round outputs",
    )
    p.add_argument(
        "--seed", type=int, default=0,
        help="Fixed seed for fair comparison (default 0)",
    )
    p.add_argument(
        "--no-env", action="store_true",
        help="Do not apply CACHEEDIT_* environment variable overrides",
    )
    return p


def _build_pipeline(model: str, cfg, no_cache: bool):
    if model == "qwen":
        from cache_edit.models.qwen import (
            create_default_cache_manager,
            init_qwen_pipeline,
        )
        cache_manager = None
        if not no_cache and cfg.cache.use_activation_cache:
            cache_manager = create_default_cache_manager(
                num_inference_steps=cfg.pipeline.num_inference_steps,
                threshold=cfg.cache.threshold,
                cache_interval=cfg.cache.cache_interval,
            )
        pipeline = init_qwen_pipeline(
            model_path=cfg.model.model_path,
            device=cfg.model.device,
            dtype=resolve_dtype(cfg.model.dtype),
            cache_manager=cache_manager,
        )
        return pipeline, cache_manager

    from cache_edit.models.flux import (
        create_default_cache_manager,
        init_flux_pipeline,
    )
    cache_manager = None
    if not no_cache and cfg.cache.use_activation_cache:
        cache_manager = create_default_cache_manager(
            num_inference_steps=cfg.pipeline.num_inference_steps,
            threshold=cfg.cache.threshold,
            cache_interval=cfg.cache.cache_interval,
        )
        if hasattr(cache_manager, "num_gpus"):
            cache_manager.num_gpus = cfg.cache.num_gpus
    pipeline = init_flux_pipeline(
        model_path=cfg.model.model_path,
        device=cfg.model.device,
        dtype=resolve_dtype(cfg.model.dtype),
        cache_manager=cache_manager,
        device_map=cfg.model.device_map,
    )
    return pipeline, cache_manager


def _time_one(pipeline, image, prompt, cfg, seed: int, device: str) -> float:
    generator = torch.Generator(device=device).manual_seed(seed)
    call_kwargs = dict(
        image=image, prompt=prompt,
        num_inference_steps=cfg.pipeline.num_inference_steps,
        guidance_scale=cfg.pipeline.guidance_scale,
        generator=generator,
    )
    if cfg.pipeline.height:
        call_kwargs["height"] = cfg.pipeline.height
    if cfg.pipeline.width:
        call_kwargs["width"] = cfg.pipeline.width

    if torch.cuda.is_available():
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    result = pipeline(**call_kwargs)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - t0
    return elapsed, (result.images[0] if hasattr(result, "images") else result[0])


def run_benchmark(args: argparse.Namespace) -> int:
    """执行 benchmark 子命令。"""
    from PIL import Image

    cfg = load_config(args.model, args.config, apply_env=not args.no_env)
    if not args.image.exists():
        raise FileNotFoundError(f"Input image not found: {args.image}")
    image = Image.open(args.image).convert("RGB")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    # ---- no-cache baseline ----
    print("[benchmark] building no-cache pipeline...")
    base_pipeline, _ = _build_pipeline(args.model, cfg, no_cache=True)

    print(f"[benchmark] warmup x{args.warmup} (no-cache)")
    for _ in range(args.warmup):
        _time_one(base_pipeline, image, args.prompt, cfg, args.seed,
                  cfg.model.device)
    no_cache_t, no_cache_img = _time_one(
        base_pipeline, image, args.prompt, cfg, args.seed, cfg.model.device
    )
    no_cache_img.save(args.output_dir / "no_cache.png")
    print(f"[benchmark] no-cache: {no_cache_t:.3f}s")

    del base_pipeline
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # ---- cache pipeline, multi-round ----
    print("[benchmark] building cache pipeline...")
    pipeline, mgr = _build_pipeline(args.model, cfg, no_cache=False)

    print(f"[benchmark] warmup x{args.warmup} (cache, round 0)")
    for _ in range(args.warmup):
        _time_one(pipeline, image, args.prompt, cfg, args.seed, cfg.model.device)
        if mgr is not None and hasattr(mgr, "reset"):
            mgr.reset()

    times: List[float] = []
    for r in range(args.rounds):
        elapsed, out_img = _time_one(
            pipeline, image, args.prompt, cfg, args.seed, cfg.model.device
        )
        times.append(elapsed)
        out_img.save(args.output_dir / f"cache_round{r}.png")
        print(f"[benchmark] cache round {r}: {elapsed:.3f}s")

    avg_cache = sum(times) / max(len(times), 1)
    speedup = no_cache_t / avg_cache if avg_cache > 0 else 0.0

    print()
    print("=" * 48)
    print(f"  no-cache (baseline)  : {no_cache_t:.3f}s")
    print(f"  cache avg ({len(times)} runs) : {avg_cache:.3f}s")
    print(f"  speedup              : {speedup:.2f}x")
    print("=" * 48)
    return 0
