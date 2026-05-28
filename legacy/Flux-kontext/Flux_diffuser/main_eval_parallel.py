import os
import json
import time
from copy import deepcopy
from PIL import Image
import torch
import argparse
from dataclasses import dataclass
from PIL import Image

from Flux_utils import ActivationCacheManager, FluxKontextPipeline, pipeline_call
from Flux_cache import cache_edit_init

def batch_evaluation(pipeline, args, metadata_slice=None, rank=0):
    """
    适配目录结构：
    args.image_path/
        images/
            0000.jpg
            0001.jpg
            ...
        metadata.jsonl   # 每行是 {"file_name": "...", "instruction": "...", "key": "...", ...}

    如果 metadata_slice 不为 None，则直接用该列表作为要跑的条目；
    否则，从 args.image_path/metadata.jsonl 读取全量。
    """

    root_dir = args.image_path    # e.g. /path/to/kontext-bench/test
    output_dir = args.output_dir
    os.makedirs(os.path.join(output_dir, "generation"), exist_ok=True)

    # 1. 读取 / 确定 metadata
    if metadata_slice is None:
        metadata_file = os.path.join(root_dir, "metadata.jsonl")
        if not os.path.isfile(metadata_file):
            print(f"[RANK {rank}] No metadata.jsonl found at {metadata_file}")
            return

        metadata = []
        with open(metadata_file, "r") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                metadata.append(json.loads(line))
    else:
        metadata = metadata_slice

    if len(metadata) == 0:
        print(f"[RANK {rank}] empty metadata, nothing to do.")
        return

    # 2. warmup：每个进程各自预热几次（可选，你也可以只让 rank 0 预热）
    print(f"[RANK {rank}] Warmup...")
    with torch.inference_mode():
        warmup_image = None
        for d in metadata:
            candidate_path = os.path.join(root_dir, d["file_name"])
            if os.path.isfile(candidate_path):
                warmup_image = Image.open(candidate_path).convert("RGB")
                break

        if warmup_image is not None:
            for _ in range(1):
                generator = torch.Generator(device="cpu").manual_seed(args.seed)
                warmup_inputs = {
                    "image": [warmup_image],
                    "prompt": "just warmup!",
                    "generator": generator,
                    "true_cfg_scale": 1.0,
                    "negative_prompt": " ",
                    "num_inference_steps": 6,
                    "guidance_scale": args.guidance_scale,
                    "num_images_per_prompt": 1,
                }
                _ = pipeline(**warmup_inputs)
        else:
            print(f"[RANK {rank}] No image found for warmup, skip warmup.")

    prefix_prompt = {}
    time_consuming = []

    print(f"[RANK {rank}] Start generation, num_items={len(metadata)}")

    for idx, data in enumerate(metadata):
        key = data["key"]                # e.g. "0000_00"
        prompt = data["instruction"]     # 指令文本
        rel_file = data["file_name"]     # e.g. "images/0000.jpg"

        input_img_path = os.path.join(root_dir, rel_file)
        if not os.path.isfile(input_img_path):
            print(f"[RANK {rank} {idx+1}/{len(metadata)}] missing image: {input_img_path}, skip.")
            continue

        try:
            input_image = Image.open(input_img_path).convert("RGB")
        except Exception as e:
            print(f"[RANK {rank} {idx+1}/{len(metadata)}] failed to open image {input_img_path}: {e}, skip.")
            continue

        print(f"[RANK {rank} {idx+1}/{len(metadata)}] file='{rel_file}', key='{key}', prompt: {prompt}")

        generator = torch.Generator(device="cpu").manual_seed(args.seed)

        inputs = {
            "image": [input_image],
            "prompt": prompt,
            "generator": generator,
            "true_cfg_scale": 1.0,
            "negative_prompt": " ",
            "num_inference_steps": args.num_inference_steps,
            "guidance_scale": args.guidance_scale,
            "num_images_per_prompt": 1,
        }

        if torch.cuda.is_available():
            torch.cuda.synchronize()
        t0 = time.time()
        with torch.inference_mode():
            output = pipeline(**inputs)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        t1 = time.time()

        output_image = output.images[0]
        save_path = os.path.join(output_dir, "generation", f"{key}.png")
        output_image.save(save_path)

        cost = t1 - t0
        prefix_prompt[key] = prompt
        time_consuming.append(cost)

        print(f"[RANK {rank} {idx+1}/{len(metadata)}] {save_path}, saved! consuming: {cost:.4f}s")

    # 3. 统计时间信息与 metadata 映射
    #   多进程写同一个 json 不安全，这里建议：
    #     - 每个 rank 写一个独立 json
    #     - 或只让 rank 0 汇总（下面示例用“每 rank 独立 json”的方式）
    time_info = {
        "rank": rank,
        "num_item": len(time_consuming),
        "ave_time_consuming": sum(time_consuming) / len(time_consuming) if len(time_consuming) > 0 else 0.0,
        "time_consuming_list": time_consuming,
    }

    with open(os.path.join(output_dir, f"time_consuming_rank{rank}.json"), "w") as f:
        json.dump(time_info, f, indent=4)

    with open(os.path.join(output_dir, f"metadata_rank{rank}.json"), "w") as f:
        json.dump(prefix_prompt, f, indent=4)

    print(f"[RANK {rank}] Done. Partial results saved to {output_dir}")
    
import torch.multiprocessing as mp

def load_full_metadata(root_dir):
    """读取全量 metadata.jsonl"""
    metadata_file = os.path.join(root_dir, "metadata.jsonl")
    if not os.path.isfile(metadata_file):
        raise FileNotFoundError(f"metadata.jsonl not found at {metadata_file}")

    metadata = []
    with open(metadata_file, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            metadata.append(json.loads(line))
    return metadata


def split_metadata(metadata, num_shards):
    """将 metadata 列表平均切成 num_shards 份，返回 list[ list[dict] ]"""
    n = len(metadata)
    shard_size = (n + num_shards - 1) // num_shards  # 向上取整
    shards = []
    for i in range(num_shards):
        start = i * shard_size
        end = min((i + 1) * shard_size, n)
        if start >= end:
            break
        shards.append(metadata[start:end])
    return shards


def worker_process(rank, gpu_id, args, metadata_slice):
    """
    每个进程执行的函数：
    - 设置当前 CUDA 设备
    - 初始化 pipeline 到对应 GPU
    - 调用 batch_evaluation 只跑自己的切片
    """
    print(f"[RANK {rank}] starting on GPU {gpu_id}")

    # 设置当前进程使用的 GPU
    torch.cuda.set_device(gpu_id)

    # 复制一份 args，避免多进程之间交叉修改
    local_args = deepcopy(args)
    # 强制 device 为当前 GPU
    local_args.device = f"cuda:{gpu_id}"

    # 初始化 pipeline（和你原来 main 里逻辑类似）
    if getattr(local_args, "use_cache", False):
        pipeline = cache_edit_init(local_args.model_path, local_args.device)
    else:
        FluxKontextPipeline.__call__ = pipeline_call
        pipeline = FluxKontextPipeline.from_pretrained(
            local_args.model_path,
            torch_dtype=torch.bfloat16,
            ).to(local_args.device)
        
    pipeline.set_progress_bar_config(disable=None)

    # 调用单卡评估，但只跑本 rank 对应的那部分 metadata
    batch_evaluation(pipeline, local_args, metadata_slice=metadata_slice, rank=rank)

    # 显式清理（可选）
    del pipeline
    torch.cuda.empty_cache()
    print(f"[RANK {rank}] finished on GPU {gpu_id}")

import argparse

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=110,
                        help="Initial random seed (can be changed in interactive mode)")
    parser.add_argument("--device", type=str, default="cuda",
                        help="Device to run the model on (e.g., 'cuda' or 'cpu')")
    parser.add_argument("--cache_device", type=str, default="cuda",
                        help="Device for cache")
    parser.add_argument("--num_inference_steps", type=int, default=30,
                        help="Default number of inference steps")
    parser.add_argument("--guidance_scale", type=float, default=16.0,
                        help="Default guidance scale")

    # cacheE /cache
    parser.add_argument("--use_cache", action='store_true',
                        help="Whether to use activation cache / RegionE")
    parser.add_argument("--warmup_step", type=int, default=6,
                        help="Step of the stablization stage")
    parser.add_argument("--threshold", type=float, default=0.97,
                        help="Threshold for adaptive region partition")

    # 模式选择：是否做批量评估
    parser.add_argument("--evaluation", action="store_true",
                        help="If set, run batch evaluation over a dataset folder instead of interactive editing")

    # 路径相关
    parser.add_argument("--model_path", type=str,
                        default="/data1/model/FLUX.1-Kontext-dev",
                        help="Path to the pre-trained model")
    parser.add_argument("--image_path", type=str,
                        default="/home/chenxueqing/my-flux-activation_cache/datasets/test",
                        help="Initial input image path (or root folder when --evaluation)")
    parser.add_argument("--output_dir", type=str,
                        default="/home/chenxueqing/image-edit-round-reuse/result/FluxKontext/kontext-bench-test/original",
                        help="Directory to save the output images / evaluation results")
    parser.add_argument("--prompt", type=str,
                        default="give the cat a tophat",
                        help="Initial prompt (can be changed in interactive mode)")
    parser.add_argument("--num_gpus", type=int, default=1,
                        help="并行使用的 GPU 数量（>1 时启用多进程多卡）")
    # 视你原来的参数再补充
    return parser.parse_args()

if __name__ == "__main__":
    args = get_args()

    if not args.evaluation:
        # 交互模式按原来的来（单卡即可）
        if args.use_cache:
            pipeline = cache_edit_init(args.model_path, args.device)
        else:
            FluxKontextPipeline.__call__ = pipeline_call
            pipeline = FluxKontextPipeline.from_pretrained(
                args.model_path,
                torch_dtype=torch.bfloat16,
                device_map="balanced",
            )
        pipeline.set_progress_bar_config(disable=None)
        interactive_edit_loop(pipeline, args)

    else:
        # 评估模式：根据 num_gpus 选择单卡 / 多卡
        if args.num_gpus <= 1:
            # 单卡：沿用你之前的逻辑
            if args.use_cache:
                pipeline = cache_edit_init(args.model_path, args.device)
            else:
                FluxKontextPipeline.__call__ = pipeline_call
                pipeline = FluxKontextPipeline.from_pretrained(
                    args.model_path,
                    torch_dtype=torch.bfloat16,
                    device_map=args.device,
                )
            pipeline.set_progress_bar_config(disable=None)
            batch_evaluation(pipeline, args, metadata_slice=None, rank=0)

        else:
            # 多卡：使用 torch.multiprocessing 多进程
            all_metadata = load_full_metadata(args.image_path)
            shards = split_metadata(all_metadata, args.num_gpus)

            # 确定要使用的 GPU id 列表，这里简单用 0..num_gpus-1
            gpu_ids = list(range(args.num_gpus))

            ctx = mp.get_context("spawn")
            processes = []
            for rank, (gpu_id, shard) in enumerate(zip(gpu_ids, shards)):
                p = ctx.Process(
                    target=worker_process,
                    args=(rank, gpu_id, args, shard),
                )
                p.start()
                processes.append(p)

            # 等所有进程结束
            for p in processes:
                p.join()

            print("All ranks finished. You can now merge metadata_rank*.json and time_consuming_rank*.json if needed.")

