import os
import time
import json
import torch
import argparse
from dataclasses import dataclass
from PIL import Image

from Flux_utils import ActivationCacheManager, FluxKontextPipeline, pipeline_call
from Flux_cache import cache_edit_init


# ========= 选项数据结构 =========
@dataclass
class EditOptions:
    prompt: str
    image_path: str
    num_inference_steps: int
    guidance_scale: float
    seed: int | None


# ========= 交互式解析 prompt =========
def parse_prompt(opts: EditOptions | None) -> EditOptions | None:
    """
    交互式输入 prompt：
    - 直接输入文本：作为新 prompt
    - 空行：重复上一次的 prompt
    - '/g <guidance>'：改 guidance
    - '/n <steps>'：改采样步数
    - '/s <seed>'：改 seed
    - '/q'：退出
    """
    if opts is None:
        return None

    user_question = "Next prompt (write /h for help, /q to quit and leave empty to repeat):\n"
    usage = (
        "Usage: Either write your prompt directly, leave this field empty "
        "to repeat the last prompt or write a command starting with a slash:\n"
        "- '/g <guidance>' sets the guidance\n"
        "- '/s <seed>' sets the seed\n"
        "- '/n <steps>' sets the number of inference steps\n"
        "- '/q' to quit"
    )

    while (prompt := input(user_question)).startswith("/"):
        # guidance
        if prompt.startswith("/g"):
            if prompt.count(" ") != 1:
                print(f"Got invalid command '{prompt}'\n{usage}")
                continue
            _, guidance = prompt.split()
            opts.guidance_scale = float(guidance)
            print(f"Setting guidance to {opts.guidance_scale}")
        # seed
        elif prompt.startswith("/s"):
            if prompt.count(" ") != 1:
                print(f"Got invalid command '{prompt}'\n{usage}")
                continue
            _, seed = prompt.split()
            opts.seed = int(seed)
            print(f"Setting seed to {opts.seed}")
        # steps
        elif prompt.startswith("/n"):
            if prompt.count(" ") != 1:
                print(f"Got invalid command '{prompt}'\n{usage}")
                continue
            _, steps = prompt.split()
            opts.num_inference_steps = int(steps)
            print(f"Setting num_inference_steps to {opts.num_inference_steps}")
        # quit
        elif prompt.startswith("/q"):
            print("Quitting")
            return None
        else:
            print(f"Got invalid command '{prompt}'\n{usage}")
            print(usage)

    # 真正更新 prompt：非空才覆盖，空就沿用旧的
    if prompt != "":
        opts.prompt = prompt
    return opts


# ========= 交互式解析图像路径 =========
def parse_image_path(opts: EditOptions | None) -> EditOptions | None:
    """
    交互式输入图像路径：
    - 输入完整路径（包括以 /home 开头的绝对路径）：更换待编辑图片
    - 空行：重复上一次图片
    - '/q'：退出
    其余以 '/' 开头的视为无效命令并给出帮助信息。
    """
    if opts is None:
        return None

    user_question = "Next input image path (write /h for help, /q to quit and leave empty to repeat):\n"
    usage = (
        "Usage: Either write a path to an image directly, leave this field empty "
        "to repeat the last input image or write a command starting with a slash:\n"
        "- '/q' to quit\n\n"
        "The input image will be edited by FLUX.1 Kontext based on your instruction prompt."
    )

    while True:
        img_path = input(user_question).strip()
        print("img_path:", img_path)

        # 处理命令
        if img_path.startswith("/"):
            # 1. 退出命令
            if img_path == "/q":
                print("Quitting")
                return None

            # 2. 绝对路径（例如 /home/...）：按文件路径处理
            if img_path.startswith("/home"):
                if not os.path.isfile(img_path) or not img_path.lower().endswith((".jpg", ".jpeg", ".png", ".webp")):
                    print(f"File '{img_path}' does not exist or is not a valid image file")
                    continue
                opts.image_path = img_path
                break

            # 3. 其他以 / 开头的，视为无效命令
            print(f"Got invalid command '{img_path}'\n{usage}")
            print(usage)
            continue

        # 空行：沿用上一次的图片
        if img_path == "":
            break

        # 相对路径或不以 / 开头的普通路径
        if not os.path.isfile(img_path) or not img_path.lower().endswith((".jpg", ".jpeg", ".png", ".webp")):
            print(f"File '{img_path}' does not exist or is not a valid image file")
            continue

        opts.image_path = img_path
        break

    return opts


# ========= 一次加载，多轮交互推理 =========
def interactive_edit_loop(pipeline, args):
    os.makedirs(args.output_dir, exist_ok=True)

    # 初始选项（可以通过命令行传入基础值）
    opts = EditOptions(
        prompt=args.prompt,
        image_path=args.image_path,
        num_inference_steps=args.num_inference_steps,
        guidance_scale=args.guidance_scale,
        seed=args.seed,
    )

    idx = 0  # 用于输出文件命名累积

    # 主循环：直到 parse_* 返回 None（用户 /q 退出）
    while opts is not None:
        # 准备随机数生成器
        generator = torch.Generator(device="cpu")
        # 如果希望完全沿用命令行 seed 就不用强制设为 1；这里保持你的写法：
        opts.seed = 1
        if opts.seed is None:
            # 如果没指定 seed，用当前时间生成一个
            seed_this_round = int(time.time() * 1000) % 2**31
        else:
            seed_this_round = opts.seed
        generator.manual_seed(seed_this_round)

        # 读入当前选定的图片
        try:
            image = Image.open(opts.image_path).convert("RGB")
        except Exception as e:
            print(f"Failed to open image '{opts.image_path}': {e}")
            # 让用户重新选图
            opts = parse_image_path(opts)
            continue

        inputs = {
            "image": [image],
            "prompt": opts.prompt,
            "generator": generator,
            "true_cfg_scale": 1.0,
            "negative_prompt": " ",
            "num_inference_steps": opts.num_inference_steps,
            "guidance_scale": opts.guidance_scale,
            "num_images_per_prompt": 1,
        }

        print(
            f"Generating: prompt='{opts.prompt}', image='{opts.image_path}', "
            f"seed={seed_this_round}, steps={opts.num_inference_steps}, guidance={opts.guidance_scale}"
        )
        t0 = time.time()
        with torch.inference_mode():
            output = pipeline(**inputs)
        output_image = output.images[0]

        # 保存输出
        save_name = os.path.join(args.output_dir, f"edit_{idx:03d}.png")
        output_image.save(save_name)
        t1 = time.time()
        print(f"Saved to {os.path.abspath(save_name)}, time={t1 - t0:.2f}s")

        idx += 1

        # 每轮结束后，再询问下一轮的 prompt / image
        print("-" * 80)
        opts = parse_prompt(opts)
        opts = parse_image_path(opts)


# ========= 批量评估模式（仿照你给的 evaluation 逻辑） =========
def batch_evaluation_loop(pipeline, args):
    """
    适配目录结构：
    args.image_path/
        images/
            0000.jpg
            0001.jpg
            ...
        metadata.jsonl   # 每行是你给出的结构

    输出：
    args.output_dir/
        generation/
            0000_00.png
            0000_01.png
            ...
        time_consuming.json
        metadata.json      # key -> instruction
    """
    root_dir = args.image_path           # e.g. /path/to/kontext-bench/test
    output_dir = args.output_dir

    os.makedirs(os.path.join(output_dir, "generation"), exist_ok=True)

    metadata_file = os.path.join(root_dir, "metadata.jsonl")
    if not os.path.isfile(metadata_file):
        print(f"No metadata.jsonl found at {metadata_file}")
        return

    # 读取 metadata
    metadata = []
    with open(metadata_file, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            metadata.append(json.loads(line))

    # warmup：用该数据集里的一张图预热几次
    print("Warmup...")
    with torch.inference_mode():
        warmup_image = None
        for d in metadata:
            # file_name 例如 "images/0000.jpg"
            candidate_path = os.path.join(root_dir, d["file_name"])
            if os.path.isfile(candidate_path):
                warmup_image = Image.open(candidate_path).convert("RGB")
                break

        if warmup_image is not None:
            for _ in range(3):
                generator = torch.Generator(device="cpu").manual_seed(args.seed)
                warmup_inputs = {
                    "image": [warmup_image],
                    "prompt": "just warmup!",
                    "generator": generator,
                    "true_cfg_scale": 1.0,
                    "negative_prompt": " ",
                    "num_inference_steps": args.num_inference_steps,
                    "guidance_scale": args.guidance_scale,
                    "num_images_per_prompt": 1,
                }
                _ = pipeline(**warmup_inputs)
        else:
            print("No image found for warmup, skip warmup.")

    prefix_prompt = {}
    time_consuming = []

    print(f"Start generation, num_items={len(metadata)}")
    for idx, data in enumerate(metadata):
        key = data["key"]                # 如 "0000_00"
        prompt = data["instruction"]     # 指令文本
        rel_file = data["file_name"]     # 如 "images/0000.jpg"

        input_img_path = os.path.join(root_dir, rel_file)
        if not os.path.isfile(input_img_path):
            print(f"[{idx+1}/{len(metadata)}] missing image: {input_img_path}, skip.")
            continue

        try:
            input_image = Image.open(input_img_path).convert("RGB")
        except Exception as e:
            print(f"[{idx+1}/{len(metadata)}] failed to open image {input_img_path}: {e}, skip.")
            continue

        print(f"[{idx+1}/{len(metadata)}] file='{rel_file}', key='{key}', prompt: {prompt}")

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
        # 用 key 命名输出，避免同一张图不同指令互相覆盖
        save_path = os.path.join(output_dir, "generation", f"{key}.png")
        output_image.save(save_path)

        cost = t1 - t0
        prefix_prompt[key] = prompt
        time_consuming.append(cost)

        print(f"[{idx+1}/{len(metadata)}] {save_path}, saved! consuming: {cost:.4f}s")

    # 统计时间信息
    if len(time_consuming) > 0:
        time_info = {
            "num_item": len(time_consuming),
            "ave_time_consuming": sum(time_consuming) / len(time_consuming),
            "time_consuming_list": time_consuming,
        }
    else:
        time_info = {
            "num_item": 0,
            "ave_time_consuming": 0.0,
            "time_consuming_list": [],
        }

    with open(os.path.join(output_dir, "time_consuming.json"), "w") as f:
        json.dump(time_info, f, indent=4)

    # 保存 key -> instruction 映射
    with open(os.path.join(output_dir, "metadata.json"), "w") as f:
        json.dump(prefix_prompt, f, indent=4)

    print(f"Done. Results saved to {output_dir}")




# ========= 主程序：加载一次模型，然后进入交互循环或批量评估 =========
if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    # 基本配置
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

    args = parser.parse_args()

    # 只加载一次模型
    if args.use_cache:
        pipeline = cache_edit_init(args.model_path, args.device)
    else:
        FluxKontextPipeline.__call__ = pipeline_call  # fix the resolution or other calling logic
        pipeline = FluxKontextPipeline.from_pretrained(
            args.model_path,
            torch_dtype=torch.bfloat16,
            device_map="balanced",  # 或者之后 .to(args.device)
        )

    pipeline.set_progress_bar_config(disable=None)

    # 根据是否 evaluation 选择模式
    if args.evaluation:
        batch_evaluation_loop(pipeline, args)
    else:
        interactive_edit_loop(pipeline, args)
