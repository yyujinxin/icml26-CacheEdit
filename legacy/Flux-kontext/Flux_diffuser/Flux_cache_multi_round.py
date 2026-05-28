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
import re

def sanitize_prompt_for_filename(prompt, max_len=80):
    """
    将 prompt 转成适合文件名的一小段字符串：
    - 小写
    - 非字母数字字符替换为下划线
    - 合并连续下划线
    - 截断到 max_len
    """
    s = prompt.strip().lower()
    # 替换非 [a-z0-9]+ 为 _
    s = re.sub(r'[^a-z0-9]+', '_', s)
    # 去掉首尾下划线
    s = s.strip('_')
    # 截断
    if len(s) > max_len:
        s = s[:max_len]
    # 避免空字符串
    if not s:
        s = "noprompt"
    return s

def batch_evaluation_multi_round(pipeline, args, metadata_slice=None, rank=0):
    """
    多轮次图像编辑评估：
    - 每条 metadata 对应一张初始图片和多个轮次的 prompts（字段名: "prompts"，列表）
    - 每轮使用上一轮的输出图像作为输入
    - 输出命名: {key}_r{round_idx}.png
    - 记录每张图每轮的用时

    目录结构仍为:
    args.image_path/
        images/
            0000.jpg
            ...
        metadata.jsonl   # 每行示例:
                         # {"file_name": "images/0000.jpg",
                         #  "key": "0000",
                         #  "prompts": ["prompt for round 0", "prompt for round 1", ...]}
    """

    root_dir = args.image_path
    output_dir = args.output_dir
    os.makedirs(os.path.join(output_dir, "generation"), exist_ok=True)

    # 1. 读取 / 确定 metadata
    if metadata_slice is None:
        metadata_file = os.path.join(root_dir, "metadata_multi_round.jsonl")
        if not os.path.isfile(metadata_file):
            print(f"[RANK {rank}] No metadata_multi_round.jsonl found at {metadata_file}")
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

    # # 2. warmup：从任意一条样本中拿图片做一次预热
    # print(f"[RANK {rank}] Warmup...")
    # with torch.inference_mode():
    #     warmup_image = None
    #     for d in metadata:
    #         candidate_path = os.path.join(root_dir, d["file_name"])
    #         if os.path.isfile(candidate_path):
    #             warmup_image = Image.open(candidate_path).convert("RGB")
    #             break

    #     if warmup_image is not None:
    #         generator = torch.Generator(device="cpu").manual_seed(args.seed)
    #         warmup_inputs = {
    #             "image": [warmup_image],
    #             "prompt": "just warmup!",
    #             "generator": generator,
    #             "true_cfg_scale": 1.0,
    #             "negative_prompt": " ",
    #             "num_inference_steps": 6,
    #             "guidance_scale": args.guidance_scale,
    #             "num_images_per_prompt": 1,
    #         }
    #         _ = pipeline_call(**warmup_inputs)
    #     else:
    #         print(f"[RANK {rank}] No image found for warmup, skip warmup.")
    
    # ActivationCacheManager.__init__()
    # 3. 正式多轮次生成
    # prefix_prompt: 记录每个 key 在每个 round 的 prompt
    # time_consuming_detail: 记录每个 key 每一轮的耗时
    prefix_prompt = {}
    time_consuming_detail = {}

    print(f"[RANK {rank}] Start multi-round generation, num_items={len(metadata)}")

    for idx, data in enumerate(metadata):
        # 使用 image_idx 作为 key
        key = data.get("image_idx", None)
        if key is None:
            # 保险起见，若未来某行没有 image_idx，就退化用 file_name 当 key
            key = data["file_name"]

        rel_file = data["file_name"]  # e.g. "images/0000.jpg"

        # 从 data 中提取所有 instructionX 字段，并按 X 排序
        round_prompts = []
        for k, v in data.items():
            if not k.startswith("instruction"):
                continue
            # k 形如 "instruction0", "instruction1", ...
            suffix = k[len("instruction"):]
            if suffix.isdigit():
                round_idx = int(suffix)
                round_prompts.append((round_idx, v))

        if not round_prompts:
            print(f"[RANK {rank} {idx+1}/{len(metadata)}] no instruction* fields for key={key}, skip.")
            continue

        # 按 round_idx 排序，得到按轮次顺序排列的 prompts
        round_prompts.sort(key=lambda x: x[0])
        prompts = [p for _, p in round_prompts]

        input_img_path = os.path.join(root_dir, rel_file)
        if not os.path.isfile(input_img_path):
            print(f"[RANK {rank} {idx+1}/{len(metadata)}] missing image: {input_img_path}, skip.")
            continue

        try:
            # 初始轮次的输入图是原始图
            current_image = Image.open(input_img_path).convert("RGB")
        except Exception as e:
            print(f"[RANK {rank} {idx+1}/{len(metadata)}] failed to open image {input_img_path}: {e}, skip.")
            continue

        num_rounds = len(prompts)
        print(f"[RANK {rank} {idx+1}/{len(metadata)}] file='{rel_file}', key='{key}', num_rounds={num_rounds}")

        prefix_prompt[key] = {}
        time_consuming_detail[key] = {}

        # 逐轮编辑
        for local_round_idx, prompt in enumerate(prompts):
            # local_round_idx 就是 0,1,2,... 对应 instruction0,1,2,... 排序后的顺序
            print(
                f"[RANK {rank} {idx+1}/{len(metadata)}]  "
                f"Round {local_round_idx}, prompt: {prompt}"
            )

            generator = torch.Generator(device="cpu").manual_seed(args.seed)

            inputs = {
                "image": [current_image],
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
            
            # 新：对 prompt 做一层清理，用于文件名
            prompt_tag = sanitize_prompt_for_filename(prompt)
            # 文件命名：image_idx_r{round}_{prompt_tag}.png
            filename = f"{key}_r{local_round_idx}_{prompt_tag}.png"
            save_path = os.path.join(output_dir, "generation", filename)
            output_image.save(save_path)

            cost = t1 - t0
            prefix_prompt[key][str(local_round_idx)] = prompt
            time_consuming_detail[key][str(local_round_idx)] = cost

            print(
                f"[RANK {rank} {idx+1}/{len(metadata)}]  "
                f"Round {local_round_idx} saved to {save_path}, consuming: {cost:.4f}s"
            )

            current_image = output_image
        ActivationCacheManager.reset()
    # 4. 保存时间信息与 prompt 映射
    all_times = [
        float(t)
        for key_times in time_consuming_detail.values()
        for t in key_times.values()
    ]
    ave_time = sum(all_times) / len(all_times) if all_times else 0.0

    time_info = {
        "rank": rank,
        "num_item": len(metadata),
        "ave_time_consuming": ave_time,
        "time_consuming_detail": time_consuming_detail,
    }

    with open(os.path.join(output_dir, f"time_consuming_rank{rank}.json"), "w") as f:
        json.dump(time_info, f, indent=4, ensure_ascii=False)

    with open(os.path.join(output_dir, f"metadata_rank{rank}.json"), "w") as f:
        json.dump(prefix_prompt, f, indent=4, ensure_ascii=False)

    print(f"[RANK {rank}] Done. Multi-round results saved to {output_dir}")

    
import torch.multiprocessing as mp

def load_full_metadata(root_dir):
    """读取全量 metadata_multi_round.jsonl"""
    metadata_file = os.path.join(root_dir, "metadata_multi_round.jsonl")
    if not os.path.isfile(metadata_file):
        raise FileNotFoundError(f"metadata_multi_round.jsonl not found at {metadata_file}")

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
    - 调用 batch_evaluation_multi_round 只跑自己的切片
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
    batch_evaluation_multi_round(pipeline, local_args, metadata_slice=metadata_slice, rank=rank)

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
    parser.add_argument("--num_inference_steps", type=int, default=28,
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
    parser.add_argument("--cache_interval", type=int, default=5,
                        help="Interval steps for caching activations")

    # 模式选择：是否做批量评估
    parser.add_argument("--evaluation", action="store_true",
                        help="If set, run batch evaluation over a dataset folder instead of interactive editing")

    # 路径相关
    parser.add_argument("--model_path", type=str,
                        default="/home/dataset-local/chenxueqing/model/black-forest-labs/FLUX.1-Kontext-dev",
                        help="Path to the pre-trained model")
    parser.add_argument("--image_path", type=str,
                        default="/home/dataset-local/chenxueqing/datasets/test",
                        help="Initial input image path (or root folder when --evaluation)")
    parser.add_argument("--output_dir", type=str,
                        default="/home/dataset-local/chenxueqing/result/Flux/kontext-bench-test/CacheEdit/multi-round-cache_interval_5-threshold_0.95",
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

    if args.evaluation:
        # 交互模式按原来的来（单卡即可）
        if args.use_cache:
            all_metadata = load_full_metadata(args.image_path)
            pipeline = cache_edit_init(args.model_path, args.device)
            ActivationCacheManager.set_parameters(args=args)
            batch_evaluation_multi_round(pipeline, args, metadata_slice=all_metadata, rank=0)
        else:
            FluxKontextPipeline.__call__ = pipeline_call
            pipeline = FluxKontextPipeline.from_pretrained(
                args.model_path,
                torch_dtype=torch.bfloat16,
                device_map="balanced",
            )
        # pipeline.set_progress_bar_config(disable=None)
        # interactive_edit_loop(pipeline, args)
        print("All ranks finished. You can now merge metadata_rank*.json and time_consuming_rank*.json if needed.")
    else:
        pipeline = cache_edit_init(args.model_path, args.device)
        batch_evaluation_multi_round(pipeline, args, metadata_slice=None, rank=0)