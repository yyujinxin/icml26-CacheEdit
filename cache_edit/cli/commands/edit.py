"""edit subcommand: run image editing on a single image."""

import argparse
from pathlib import Path

import torch

from cache_edit.cli.utils import load_config, resolve_dtype


def add_edit_parser(subparsers) -> argparse.ArgumentParser:
    """注册 `edit` 子命令。"""
    p = subparsers.add_parser(
        "edit",
        help="Edit a single image with a text prompt",
        description=(
            "Edit a single image using Qwen or Flux Kontext with activation caching."
        ),
    )
    p.add_argument(
        "--model",
        choices=["qwen", "flux"],
        required=True,
        help="Which model backend to use",
    )
    p.add_argument(
        "--image",
        type=Path,
        required=True,
        help="Path to the input image",
    )
    p.add_argument(
        "--prompt",
        type=str,
        required=True,
        help="Text instruction describing the desired edit",
    )
    p.add_argument(
        "--negative-prompt",
        type=str,
        default=None,
        help="Optional negative prompt (Flux only)",
    )
    p.add_argument(
        "--config",
        type=Path,
        default=None,
        help="Optional config file (YAML/JSON). Defaults to configs/{model}_default.yaml",
    )
    p.add_argument(
        "--output",
        type=Path,
        default=Path("output.png"),
        help="Output image path (default: output.png)",
    )
    p.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed for reproducibility",
    )
    p.add_argument(
        "--no-cache",
        action="store_true",
        help="Disable activation cache (run vanilla pipeline)",
    )
    p.add_argument(
        "--no-env",
        action="store_true",
        help="Do not apply CACHEEDIT_* environment variable overrides",
    )
    return p


def _build_qwen(cfg, no_cache: bool):
    from cache_edit.models.qwen import (
        QwenCacheManager,
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


def _build_flux(cfg, no_cache: bool):
    from cache_edit.models.flux import (
        FluxCacheVizConfig,
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

    viz_config = None
    if cfg.viz.enable:
        viz_config = FluxCacheVizConfig(
            enable=True,
            gen_dir=cfg.viz.gen_dir,
            viz_out_dir=cfg.viz.viz_out_dir,
            csv_out_path=cfg.viz.csv_out_path,
            edit_ratio_summary_candidates=list(
                cfg.viz.edit_ratio_summary_candidates
            ),
            rounds_per_image=cfg.viz.rounds_per_image,
            ref_layer_idx=cfg.viz.ref_layer_idx,
            ref_stream=cfg.viz.ref_stream,
        )

    pipeline = init_flux_pipeline(
        model_path=cfg.model.model_path,
        device=cfg.model.device,
        dtype=resolve_dtype(cfg.model.dtype),
        cache_manager=cache_manager,
        device_map=cfg.model.device_map,
        viz_config=viz_config,
    )
    return pipeline, cache_manager


def run_edit(args: argparse.Namespace) -> int:
    """执行 edit 子命令。"""
    from PIL import Image

    cfg = load_config(args.model, args.config, apply_env=not args.no_env)

    if not args.image.exists():
        raise FileNotFoundError(f"Input image not found: {args.image}")
    image = Image.open(args.image).convert("RGB")

    if args.model == "qwen":
        pipeline, cache_manager = _build_qwen(cfg, args.no_cache)
    else:
        pipeline, cache_manager = _build_flux(cfg, args.no_cache)

    generator = None
    if args.seed is not None:
        generator = torch.Generator(device=cfg.model.device).manual_seed(
            args.seed
        )

    if cache_manager is not None and hasattr(cache_manager, "on_step_start"):
        # 单图编辑视为新一轮
        cache_manager.current_round = -1

    call_kwargs = dict(
        image=image,
        prompt=args.prompt,
        num_inference_steps=cfg.pipeline.num_inference_steps,
        guidance_scale=cfg.pipeline.guidance_scale,
        generator=generator,
    )
    if cfg.pipeline.height:
        call_kwargs["height"] = cfg.pipeline.height
    if cfg.pipeline.width:
        call_kwargs["width"] = cfg.pipeline.width
    if args.model == "flux" and args.negative_prompt is not None:
        call_kwargs["negative_prompt"] = args.negative_prompt

    result = pipeline(**call_kwargs)
    out_image = result.images[0] if hasattr(result, "images") else result[0]

    args.output.parent.mkdir(parents=True, exist_ok=True)
    out_image.save(args.output)
    print(f"[edit] saved → {args.output}")
    return 0
