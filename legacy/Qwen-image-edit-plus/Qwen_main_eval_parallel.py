# Qwen_eval_balanced_2gpu.py
import os
import argparse
import json
from copy import deepcopy
from tqdm import tqdm
import time
from PIL import Image

import torch
from diffusers import QwenImageEditPlusPipeline   # 如果你的 import 路径不同，在这里改

# =============== 一些工具函数（根据你的原代码适当改名/对接） =================

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--seed", type=int, default=1,
                        help="Initial random seed (can be changed in interactive mode)")
    parser.add_argument("--device", type=str, default="cuda",
                        help="Device to run the model on (e.g., 'cuda' or 'cpu')")
    parser.add_argument("--cache_device", type=str, default="cuda",
                        help="Device for cache")
    parser.add_argument("--num_inference_steps", type=int, default=30,
                        help="Default number of inference steps")
    parser.add_argument("--guidance_scale", type=float, default=1.0,
                        help="Default guidance scale")
    parser.add_argument("--true_cfg_scale", type=float, default=4.0)

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
                        default="/data1/model/Qwen-Image-Edit-2511",
                        help="Path to the pre-trained model")
    parser.add_argument("--image_path", type=str,
                        default="/home/chenxueqing/my-flux-activation_cache/datasets/test",
                        help="Initial input image path (or root folder when --evaluation)")
    parser.add_argument("--output_dir", type=str,
                        default="/home/chenxueqing/image-edit-round-reuse/result/QwenImageEdit/kontext-bench-test/original",
                        help="Directory to save the output images / evaluation results")
    parser.add_argument("--prompt", type=str,
                        default="give the cat a tophat",
                        help="Initial prompt (can be changed in interactive mode)")

    # 分片相关参数：总共有多少片，本进程是第几片（从 0 开始）
    parser.add_argument("--num_shards", type=int, default=1,
                        help="把整个数据集平均切成多少份")
    parser.add_argument("--shard_index", type=int, default=0,
                        help="本进程处理第几份 [0, num_shards-1]")

    return parser.parse_args()


def load_full_metadata(image_path):
    """
    这里写成一个示例，你需要按照你原来的逻辑改：
    - 有的人是 image_path 下有一个 metadata.json
    - 有的人是 image_path 下每张图一个 json
    下面给一个最简单的示例：假设 image_path/metadata.json 里是一整个 list。
    """
    metadata_file = os.path.join(image_path, "metadata.jsonl")
    if not os.path.isfile(metadata_file):
        print(f"No metadata.jsonl found at {metadata_file}")
        return

    metadata = []
    with open(metadata_file, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            metadata.append(json.loads(line))
    return metadata


def shard_metadata(metadata, num_shards, shard_index):
    """
    把 metadata 列表平均切成 num_shards 份，返回第 shard_index 这份。
    """
    n = len(metadata)
    shard_size = (n + num_shards - 1) // num_shards
    start = shard_index * shard_size
    end = min((shard_index + 1) * shard_size, n)
    return metadata[start:end]


def batch_evaluation_loop(pipeline, args, metadata_slice, rank=0):
    """
    带有单次生成时间统计的 batch loop。
    """
    os.makedirs(args.output_dir, exist_ok=True)
    root_dir = args.image_path  
    output_dir = args.output_dir
    os.makedirs(os.path.join(output_dir, "generation"), exist_ok=True)
    
    results = []
    per_sample_times = []  # 记录每条样本的生成时间（秒）

    for i, data in enumerate(tqdm(
        metadata_slice,
        desc=f"Processing shard {rank}",
        total=len(metadata_slice)
    )):
        key = data["key"]                # e.g. "0000_00"
        prompt = data["instruction"]     # 指令文本
        rel_file = data["file_name"]     # e.g. "images/0000.jpg"
        image_path = os.path.join(root_dir, rel_file)

        # 加载图像
        image = Image.open(image_path).convert("RGB")

        # 统计 pipeline 调用时间
        t_start = time.time()
        out = pipeline(
            prompt=prompt,
            image=image,
            num_inference_steps=args.num_inference_steps,
            guidance_scale=args.guidance_scale,
            true_cfg_scale=args.true_cfg_scale,
        )
        t_end = time.time()
        elapsed = t_end - t_start
        per_sample_times.append(elapsed)

        # 保存生成图
        save_path = os.path.join(output_dir, "generation", f"{key}.png")
        out.images[0].save(save_path)

        # 收集信息 + 本次耗时
        results.append({
            "key": key,
            "input_image": image_path,
            "prompt": prompt,
            "output_image": save_path,
            "gen_time_sec": elapsed,  # 本条样本生成耗时（秒）
            "num_inference_steps": getattr(args, "num_inference_steps", None),
        })

    # 把本进程的结果写一个 json
    result_file = os.path.join(args.output_dir, f"results_rank{rank}.json")
    with open(result_file, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    # 一些统计信息
    total_samples = len(metadata_slice)
    total_time = sum(per_sample_times) if per_sample_times else 0.0
    avg_time = total_time / total_samples if total_samples > 0 else 0.0
    max_time = max(per_sample_times) if per_sample_times else 0.0
    min_time = min(per_sample_times) if per_sample_times else 0.0

    # 也可以把统计信息单独写入一个 json
    summary = {
        "rank": rank,
        "num_samples": total_samples,
        "total_time_sec": total_time,
        "avg_time_sec": avg_time,
        "max_time_sec": max_time,
        "min_time_sec": min_time,
    }
    summary_file = os.path.join(args.output_dir, f"time_summary_rank{rank}.json")
    with open(summary_file, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print(
        f"[RANK {rank}] done {total_samples} samples, "
        f"results saved to {result_file}. "
        f"Time - total: {total_time:.2f}s, avg: {avg_time:.3f}s, "
        f"min: {min_time:.3f}s, max: {max_time:.3f}s; "
        f"summary saved to {summary_file}"
    )


# =============== 主程序：单进程，使用 balanced 在本进程可见的卡上多卡切分 ===============

def main():
    args = parse_args()

    # 这一步非常关键：
    #   每个进程外部用 CUDA_VISIBLE_DEVICES 控制“本进程可见的 2 张卡”
    #   在进程内部，我们只需要告诉 diffusers：device_map="balanced"
    #   它会在“本进程可见的所有 GPU”之间自动切分模型。
    #
    # 因此，这里不要再改 CUDA_VISIBLE_DEVICES，也不要自己做 mp.Process。

    # 1) 加载完整 metadata，然后按 num_shards / shard_index 做切片
    full_metadata = load_full_metadata(args.image_path)
    my_metadata = shard_metadata(full_metadata, args.num_shards, args.shard_index)

    print(f"[SHARD {args.shard_index}/{args.num_shards}] "
          f"total {len(full_metadata)}, this shard {len(my_metadata)}")

    if len(my_metadata) == 0:
        print("No data in this shard, exit.")
        return

    # 2) 加载 QwenImageEditPlusPipeline，用 balanced 把权重放到本进程可见的多张卡
    if args.use_cache:
        raise NotImplementedError("use_cache + balanced 多卡暂未实现缓存版本。")
    else:
        pipeline = QwenImageEditPlusPipeline.from_pretrained(
            args.model_path,
            torch_dtype=torch.bfloat16,
            device_map="balanced",  # 关键：只要本进程 CUDA_VISIBLE_DEVICES 里有 2 张卡，模型会自动切到这 2 张卡
        )

    pipeline.set_progress_bar_config(disable=None)

    # 注意：这里千万不要再调用 pipeline.to("cuda:x")
    # 否则会报你之前碰到的那个错误

    # 3) 跑本分片数据
    batch_evaluation_loop(pipeline, args, metadata_slice=my_metadata, rank=args.shard_index)

    del pipeline
    torch.cuda.empty_cache()
    print(f"[SHARD {args.shard_index}] finished.")


if __name__ == "__main__":
    main()
