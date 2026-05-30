import os
import time
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

    # 第一次先让用户有机会修改 prompt / image
    # opts = parse_prompt(opts)
    # opts = parse_image_path(opts)

    idx = 0  # 用于输出文件命名累积

    # 主循环：直到 parse_* 返回 None（用户 /q 退出）
    while opts is not None:
        # 准备随机数生成器
        generator = torch.Generator(device="cpu")
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


# ========= 主程序：加载一次模型，然后进入交互循环 =========
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

    # RegionE / cache
    parser.add_argument("--use_cache", action='store_true',
                        help="Whether to use activation cache / RegionE")
    parser.add_argument("--warmup_step", type=int, default=6,
                        help="Step of the stablization stage")
    parser.add_argument("--threshold", type=float, default=0.97,
                        help="Threshold for adaptive region partition")

    # 路径相关
    parser.add_argument("--model_path", type=str,
                        default="/data1/model/FLUX.1-Kontext-dev",
                        help="Path to the pre-trained model")
    parser.add_argument("--image_path", type=str,
                        default="/home/chenxueqing/my-flux-activation_cache/datasets/test/images/0000.jpg",
                        help="Initial input image path")
    parser.add_argument("--output_dir", type=str,
                        default="result/FluxKontext/Demo/CacheEdit",
                        help="Directory to save the output images")
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

    # 进入交互式多轮编辑循环
    interactive_edit_loop(pipeline, args)
