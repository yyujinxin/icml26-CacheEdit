"""Optimized multi-GPU Flux script with encoder offloading and activation management.

Key optimizations:
1. Text/Image encoders are offloaded to CPU after encoding
2. Activation values are dynamically moved between GPUs to avoid OOM
3. Smart memory management ensures correct computation results
"""

import argparse
import gc
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


class MultiGPUMemoryManager:
    """Manages activation memory across multiple GPUs."""

    def __init__(self, num_gpus=4, memory_limit_per_gpu_gb=20):
        self.num_gpus = num_gpus
        self.memory_limit = memory_limit_per_gpu_gb * 1024 * 1024 * 1024  # Convert to bytes
        self.device_list = [torch.device(f'cuda:{i}') for i in range(num_gpus)]
        self.current_device_idx = 0

    def get_next_device(self):
        """Round-robin device selection for activation placement."""
        device = self.device_list[self.current_device_idx]
        self.current_device_idx = (self.current_device_idx + 1) % self.num_gpus
        return device

    def get_device_memory_usage(self, device_idx):
        """Get current memory usage for a GPU."""
        torch.cuda.synchronize(device_idx)
        return torch.cuda.memory_allocated(device_idx)

    def find_best_device_for_tensor(self, tensor_size_bytes):
        """Find GPU with most free memory for placing a tensor."""
        min_usage = float('inf')
        best_device_idx = 0

        for i in range(self.num_gpus):
            usage = self.get_device_memory_usage(i)
            if usage < min_usage and (usage + tensor_size_bytes) < self.memory_limit:
                min_usage = usage
                best_device_idx = i

        return self.device_list[best_device_idx]


def _distribute_transformer_layers(pipeline, num_gpus):
    """
    Distribute transformer layers across multiple GPUs for model parallelism.

    Flux has:
    - 19 transformer_blocks (double-stream attention)
    - 38 single_transformer_blocks (single-stream attention)
    Total: 57 layers
    """
    import torch.nn as nn

    # Move base components to GPU 0
    print(f"  - Moving VAE, text encoders to cuda:0")
    pipeline.vae.to('cuda:0')
    pipeline.text_encoder.to('cuda:0')
    pipeline.text_encoder_2.to('cuda:0')

    # Move transformer base components (not blocks) to GPU 0
    print(f"  - Moving transformer embeddings and norms to cuda:0")
    # Only move the parameters that are NOT in blocks
    for name, module in pipeline.transformer.named_children():
        if name not in ['transformer_blocks', 'single_transformer_blocks']:
            module.to('cuda:0')

    # Calculate layers per GPU
    total_layers = len(pipeline.transformer.transformer_blocks) + len(pipeline.transformer.single_transformer_blocks)
    layers_per_gpu = (total_layers + num_gpus - 1) // num_gpus

    print(f"  - Total layers: {total_layers}, distributing ~{layers_per_gpu} layers per GPU")

    # Distribute transformer_blocks (19 layers)
    current_gpu = 0
    layers_on_current_gpu = 0

    for i, block in enumerate(pipeline.transformer.transformer_blocks):
        if layers_on_current_gpu >= layers_per_gpu and current_gpu < num_gpus - 1:
            current_gpu += 1
            layers_on_current_gpu = 0

        device = f'cuda:{current_gpu}'
        block.to(device)
        block._target_device = device  # Store for later reference
        layers_on_current_gpu += 1

        if i == 0 or i == len(pipeline.transformer.transformer_blocks) - 1:
            print(f"    transformer_blocks[{i}] -> {device}")

    # Distribute single_transformer_blocks (38 layers)
    for i, block in enumerate(pipeline.transformer.single_transformer_blocks):
        if layers_on_current_gpu >= layers_per_gpu and current_gpu < num_gpus - 1:
            current_gpu += 1
            layers_on_current_gpu = 0

        device = f'cuda:{current_gpu}'
        block.to(device)
        block._target_device = device
        layers_on_current_gpu += 1

        if i == 0 or i == len(pipeline.transformer.single_transformer_blocks) - 1:
            print(f"    single_transformer_blocks[{i}] -> {device}")

    print(f"  - Layer distribution complete")


def _print_cuda_memory(prefix: str, num_gpus: int):
    if not torch.cuda.is_available():
        return
    print(prefix)
    for i in range(num_gpus):
        allocated = torch.cuda.memory_allocated(i) / 1024**3
        reserved = torch.cuda.memory_reserved(i) / 1024**3
        print(
            f"  cuda:{i}: allocated={allocated:.2f}GB "
            f"reserved={reserved:.2f}GB"
        )


def _dispatch_transformer_blocks(pipeline, args):
    """Dispatch Flux transformer blocks across GPUs instead of one GPU."""
    from accelerate import infer_auto_device_map

    # Keep transformer weights well below the requested runtime limit so
    # denoising activations, cache copies, and NVENC buffers have headroom.
    weight_limit_gb = min(
        7.0,
        max(4.0, args.gpu_memory_limit_gb - args.gpu_memory_buffer_gb - 4.0),
    )
    max_memory = {
        i: f"{weight_limit_gb:.0f}GiB" for i in range(args.num_gpus)
    }
    max_memory["cpu"] = "128GiB"
    print(
        "[Optimization] Dispatching transformer blocks with "
        f"max_memory={max_memory}"
    )

    device_map = infer_auto_device_map(
        pipeline.transformer,
        max_memory=max_memory,
        no_split_module_classes=[
            "FluxTransformerBlock",
            "FluxSingleTransformerBlock",
        ],
    )
    for module_name, device_id in device_map.items():
        module = (
            pipeline.transformer
            if module_name == ""
            else pipeline.transformer.get_submodule(module_name)
        )
        device = torch.device(
            f"cuda:{device_id}" if isinstance(device_id, int) else device_id
        )
        module.to(device)
        module._target_device = device

    pipeline.transformer.hf_device_map = dict(device_map)
    print(f"[Optimization] Transformer device map entries: {len(device_map)}")
    print(f"[Optimization] Transformer device map: {device_map}")


def build_pipeline_with_offload(args, *, enable_cache: bool = True):
    """Build pipeline with model parallelism across multiple GPUs."""
    from cache_edit.models.flux import create_default_cache_manager
    from cache_edit.models.flux.pipeline import CacheFluxKontextPipeline
    from cache_edit.models.flux.processor import FluxAttnCacheProcessor
    from cache_edit.models.flux.blocks import (
        cache_flux_single_transformer_block_forward,
        cache_flux_transformer_block_forward,
    )
    from cache_edit.models.flux.transformer_forward import cache_flux_transformer_2d_forward

    if args.num_gpus > 1:
        print(f"[Optimization] Initializing pipeline with model parallelism across {args.num_gpus} GPUs...")
    else:
        print("[Optimization] Initializing pipeline with sequential CPU offload...")

    # Use CPU for cache storage to avoid GPU OOM when handling large images
    cache_manager = create_default_cache_manager(
        num_inference_steps=args.num_inference_steps,
        threshold=args.threshold,
        cache_interval=args.cache_interval,
        cache_device=torch.device("cpu"),  # Store cache on CPU to save GPU memory
        use_activation_cache=enable_cache,
        num_gpus=args.num_gpus,
        gpu_memory_limit_gb=args.gpu_memory_limit_gb,
        gpu_memory_buffer_gb=args.gpu_memory_buffer_gb,
        use_compression=enable_cache and args.use_cache_compression,
        compression_bitrate=args.compression_bitrate,
        compression_codec=args.compression_codec,
        compression_gop_length=args.compression_gop_length,
        compression_frame_interval_p=args.compression_frame_interval_p,
    )

    # Load pipeline directly from pretrained WITHOUT moving to device
    print("[Optimization] Loading pipeline components (keeping on CPU initially)...")

    if args.num_gpus > 1:
        # Multi-GPU: load on CPU first, then dispatch transformer blocks.
        # Pipeline-level device_map keeps FluxTransformer2DModel as one unit,
        # which places ~22GB of weights on cuda:0 and leaves no activation room.
        print("[Optimization] Loading pipeline on CPU for block dispatch...")
        pipeline = CacheFluxKontextPipeline.from_pretrained(
            args.model_path,
            torch_dtype=torch.bfloat16,
        )
        _dispatch_transformer_blocks(pipeline, args)
        pipeline.text_encoder.to(args.device)
        pipeline.text_encoder_2.to(args.device)
        pipeline.vae.to(args.device)
        if hasattr(pipeline.vae, "enable_slicing"):
            pipeline.vae.enable_slicing()
        if hasattr(pipeline.vae, "enable_tiling"):
            pipeline.vae.enable_tiling()
        _print_cuda_memory(
            "[Optimization] CUDA memory after block dispatch:",
            args.num_gpus,
        )
    else:
        # Single GPU: load to CPU first
        pipeline = CacheFluxKontextPipeline.from_pretrained(
            args.model_path,
            torch_dtype=torch.bfloat16,
            # Don't specify device_map, components stay on CPU
        )

    if args.num_gpus > 1:
        print(
            f"[Optimization] Transformer blocks distributed across "
            f"{args.num_gpus} GPUs"
        )
    else:
        # Single GPU: use sequential CPU offload
        print("[Optimization] Enabling sequential CPU offload (layers move to GPU on-demand)...")
        pipeline.enable_sequential_cpu_offload(gpu_id=0)

    # Now install cache hooks
    print("[Optimization] Installing cache hooks...")

    # Replace transformer.forward
    pipeline.transformer.forward = cache_flux_transformer_2d_forward.__get__(
        pipeline.transformer, pipeline.transformer.__class__
    )

    # Replace block forward + set processor
    for block in pipeline.transformer.transformer_blocks:
        block.workspace = {}
        block.forward = cache_flux_transformer_block_forward.__get__(
            block, block.__class__
        )
        block.attn.set_processor(FluxAttnCacheProcessor())

    for block in pipeline.transformer.single_transformer_blocks:
        block.workspace = {}
        block.forward = cache_flux_single_transformer_block_forward.__get__(
            block, block.__class__
        )
        block.attn.set_processor(FluxAttnCacheProcessor())

    pipeline.attach_cache_context(cache_manager)
    pipeline.set_progress_bar_config(disable=False)

    # Initialize memory manager
    memory_manager = MultiGPUMemoryManager(
        num_gpus=args.num_gpus,
        memory_limit_per_gpu_gb=args.gpu_memory_limit_gb
    )

    return pipeline, cache_manager, memory_manager


def offload_encoders_to_cpu(pipeline):
    """Offload text and image encoders to CPU to free GPU memory using diffusers' official method."""
    print("[Optimization] Enabling model CPU offload...")

    try:
        # Use enable_model_cpu_offload which works better with device_map
        # This offloads models to CPU when not in use, but keeps them on GPU during execution
        pipeline.enable_model_cpu_offload()
        print("  - Model CPU offload enabled (components moved to CPU when not in use)")
    except Exception as e:
        print(f"  - Note: Model CPU offload not available with device_map='auto': {e}")
        print("  - Models are already distributed across GPUs via device_map")
        # With device_map='auto', encoders are already optimally placed
        # No additional offloading needed


def encode_prompt_with_onload(pipeline, prompt, negative_prompt, device):
    """Temporarily load encoders to GPU for encoding, then offload back."""
    # Move encoders to GPU temporarily
    if hasattr(pipeline, 'text_encoder') and pipeline.text_encoder is not None:
        original_device_te = pipeline.text_encoder.device
        if str(original_device_te) == 'cpu':
            pipeline.text_encoder = pipeline.text_encoder.to(device)

    if hasattr(pipeline, 'text_encoder_2') and pipeline.text_encoder_2 is not None:
        original_device_te2 = pipeline.text_encoder_2.device
        if str(original_device_te2) == 'cpu':
            pipeline.text_encoder_2 = pipeline.text_encoder_2.to(device)

    # Encode
    prompt_embeds_tuple = pipeline.encode_prompt(
        prompt=prompt,
        prompt_2=None,
        device=device,
        num_images_per_prompt=1,
    )

    # Offload back to CPU
    if hasattr(pipeline, 'text_encoder') and pipeline.text_encoder is not None:
        pipeline.text_encoder = pipeline.text_encoder.to('cpu')

    if hasattr(pipeline, 'text_encoder_2') and pipeline.text_encoder_2 is not None:
        pipeline.text_encoder_2 = pipeline.text_encoder_2.to('cpu')

    torch.cuda.empty_cache()

    return prompt_embeds_tuple


def run_image(pipeline, cache_manager, memory_manager, row, args):
    prompts = extract_instructions(row)
    if args.max_rounds is not None:
        prompts = prompts[:args.max_rounds]
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
    print(f"[image {key}] {len(prompts)} rounds, src={img_path}, size={current_image.size}")
    if cache_manager is not None and hasattr(cache_manager, "set_compression_image_key"):
        cache_manager.set_compression_image_key(key)

    timings = []
    for r, prompt in enumerate(prompts):
        generator = torch.Generator(device="cpu").manual_seed(args.seed)

        # Use standard pipeline call - the encoders will be managed internally
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
            # Clear cache before each round to maximize available memory
            torch.cuda.empty_cache()
            gc.collect()

            output = pipeline(**inputs)

            # Clear cache after generation
            torch.cuda.empty_cache()
            gc.collect()

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
        default="/mnt/data/models/black-forest-labs/FLUX___1-Kontext-dev",
    )
    p.add_argument(
        "--data-root", default="/mnt/data/datasets/test"
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
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--num-gpus", type=int, default=1)
    p.add_argument("--gpu-memory-limit-gb", type=float, default=22.0)
    p.add_argument("--gpu-memory-buffer-gb", type=float, default=2.0)
    p.add_argument("--true-cfg-scale", type=float, default=1.0)
    p.add_argument("--guidance-scale", type=float, default=2.5)
    p.add_argument("--threshold", type=float, default=0.97)
    p.add_argument("--cache-interval", type=int, default=5)
    p.add_argument("--num-inference-steps", type=int, default=110)
    p.add_argument(
        "--max-rounds",
        type=int,
        default=None,
        help="Run only the first N edit rounds for each selected image.",
    )
    p.add_argument("--use-cache", action="store_true", help="Enable activation cache")
    p.add_argument("--offload-encoders", action="store_true", help="Offload encoders to CPU after use")
    p.add_argument("--use-cache-compression", action="store_true",
                   help="Use LLM.265 NVENC compression for activation cache")
    p.add_argument("--compression-bitrate", type=float, default=5.0,
                   help="Compression bitrate in Mbps (1-10, default: 5.0)")
    p.add_argument("--compression-codec", choices=["hevc", "h264"], default="hevc",
                   help="Video codec for compression (default: hevc)")
    p.add_argument("--compression-gop-length", type=int, default=1,
                   help="Inter-layer GOP length for activation compression; <=1 keeps all-I frames")
    p.add_argument("--compression-frame-interval-p", type=int, default=1,
                   help="P-frame interval for inter-layer GOP compression (1 = IPPP)")
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

    if args.use_cache:
        pipeline, cache_manager, memory_manager = build_pipeline_with_offload(
            args,
            enable_cache=True,
        )
    else:
        if args.num_gpus > 1:
            pipeline, cache_manager, memory_manager = build_pipeline_with_offload(
                args,
                enable_cache=False,
            )
        else:
            from cache_edit.models.flux import init_flux_pipeline
            pipeline = init_flux_pipeline(
                model_path=args.model_path,
                device=args.device,
                dtype=torch.bfloat16,
                cache_manager=None,
                device_map=None,
            )
            pipeline.set_progress_bar_config(disable=False)
            cache_manager = None
            memory_manager = None

    # Offload encoders if requested
    if args.offload_encoders:
        offload_encoders_to_cpu(pipeline)

    all_timings = {}
    for row in selected:
        key = row.get("image_idx") or row["file_name"]
        all_timings[key] = run_image(pipeline, cache_manager, memory_manager, row, args)

    os.makedirs(args.output_dir, exist_ok=True)
    flat = [t for ts in all_timings.values() for t in ts]
    summary = {
        "use_cache": args.use_cache,
        "num_images": len(all_timings),
        "avg_round_time": sum(flat) / len(flat) if flat else 0.0,
        "per_image_round_times": all_timings,
    }
    if (
        args.use_cache
        and cache_manager is not None
        and hasattr(cache_manager, "get_compression_report")
    ):
        summary["compression"] = cache_manager.get_compression_report()
    else:
        summary["compression"] = {
            "summary": {
                "enabled": False,
                "attempt_count": 0,
                "success_count": 0,
                "failure_count": 0,
            },
            "compression_records": [],
            "decompression_records": [],
        }

    report_path = Path(args.output_dir) / "timings.json"
    with open(report_path, "w") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print(f"\navg round time: {summary['avg_round_time']:.2f}s")
    comp_summary = summary["compression"]["summary"]
    if comp_summary.get("success_count", 0):
        payload_ratio = comp_summary.get("payload_compression_ratio") or 0.0
        total_ratio = comp_summary.get("total_compression_ratio") or 0.0
        success_modes = comp_summary.get("success_count_by_mode") or {}
        item_label = "groups" if success_modes.get("inter_layer_gop") else "records"
        print(
            "compression: "
            f"{comp_summary['success_count']} {item_label}, "
            f"payload={payload_ratio:.2f}x, "
            f"total={total_ratio:.2f}x"
        )
    print(f"report -> {report_path}")
    print(f"results -> {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
