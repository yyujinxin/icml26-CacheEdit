# main_qwen_loop.py
import os
import time
import torch
import argparse
from dataclasses import dataclass
from PIL import Image

from diffusers import QwenImageEditPlusPipeline
from Qwen_cache_plus import cache_edit_init
# from activation_cache_qwen import QWEN_ACTIVATION_CACHE

@dataclass
class EditOptions:
    prompt: str
    image_path: str
    num_inference_steps: int
    guidance_scale: float  # Qwen 的 guidance_scale 只在 guidance-distilled 时用
    true_cfg_scale: float  # 传统 CFG
    seed: int | None

def parse_prompt(opts: EditOptions | None) -> EditOptions | None:
    if opts is None:
        return None

    user_question = "Next prompt (write /h for help, /q to quit and leave empty to repeat):\n"
    usage = (
        "Usage: either write your prompt directly, leave empty to reuse last prompt, "
        "or use commands:\n"
        "- '/g <guidance>'  set guidance_scale\n"
        "- '/c <cfg>'       set true_cfg_scale\n"
        "- '/n <steps>'     set num_inference_steps\n"
        "- '/s <seed>'      set seed\n"
        "- '/q'             quit"
    )

    while (prompt := input(user_question)).startswith("/"):
        if prompt.startswith("/g"):
            try:
                _, g = prompt.split()
                opts.guidance_scale = float(g)
                print(f"Setting guidance_scale = {opts.guidance_scale}")
            except Exception:
                print(f"Invalid command '{prompt}'\n{usage}")
        elif prompt.startswith("/c"):
            try:
                _, c = prompt.split()
                opts.true_cfg_scale = float(c)
                print(f"Setting true_cfg_scale = {opts.true_cfg_scale}")
            except Exception:
                print(f"Invalid command '{prompt}'\n{usage}")
        elif prompt.startswith("/n"):
            try:
                _, n = prompt.split()
                opts.num_inference_steps = int(n)
                print(f"Setting num_inference_steps = {opts.num_inference_steps}")
            except Exception:
                print(f"Invalid command '{prompt}'\n{usage}")
        elif prompt.startswith("/s"):
            try:
                _, s = prompt.split()
                opts.seed = int(s)
                print(f"Setting seed = {opts.seed}")
            except Exception:
                print(f"Invalid command '{prompt}'\n{usage}")
        elif prompt.startswith("/q"):
            print("Quitting.")
            return None
        else:
            print(f"Invalid command '{prompt}'\n{usage}")

    if prompt != "":
        opts.prompt = prompt
    return opts

def parse_image_path(opts: EditOptions | None) -> EditOptions | None:
    if opts is None:
        return None

    user_question = "Next input image path (write /h for help, /q to quit and leave empty to repeat):\n"
    usage = (
        "Usage: write a path to an image, leave empty to reuse last image, or '/q' to quit.\n"
        "Image will be edited by Qwen-Image-Edit based on your prompt."
    )

    while True:
        img_path = input(user_question).strip()
        if img_path.startswith("/"):
            if img_path == "/q":
                print("Quitting.")
                return None
            if os.path.isfile(img_path) and img_path.lower().endswith((".jpg", ".jpeg", ".png", ".webp")):
                opts.image_path = img_path
                break
            print(f"Invalid command or file '{img_path}'\n{usage}")
            continue

        if img_path == "":
            break

        if not os.path.isfile(img_path) or not img_path.lower().endswith((".jpg", ".jpeg", ".png", ".webp")):
            print(f"File '{img_path}' does not exist or is not a valid image file")
            continue

        opts.image_path = img_path
        break

    return opts

def interactive_edit_loop(pipeline, args):
    os.makedirs(args.output_dir, exist_ok=True)

    opts = EditOptions(
        prompt=args.prompt,
        image_path=args.image_path,
        num_inference_steps=args.num_inference_steps,
        guidance_scale=args.guidance_scale,
        true_cfg_scale=args.true_cfg_scale,
        seed=args.seed,
    )

    # opts = parse_prompt(opts)
    # opts = parse_image_path(opts)

    idx = 0
    while opts is not None:
        generator = torch.Generator(device="cpu")
        if opts.seed is None:
            seed_this_round = int(time.time() * 1000) % 2**31
        else:
            seed_this_round = opts.seed
        seed_this_round = 1
        generator.manual_seed(seed_this_round)

        try:
            image = Image.open(opts.image_path).convert("RGB")
        except Exception as e:
            print(f"Failed to open image '{opts.image_path}': {e}")
            opts = parse_image_path(opts)
            continue

        # # 更新 cache manager 的步数和阈值（可基于 args）
        # QWEN_ACTIVATION_CACHE.set_parameters(
        #     num_inference_steps=opts.num_inference_steps,
        #     threshold=args.threshold,
        #     cache_device=torch.device(args.device),
        # )

        inputs = {
            "image": [image],
            "prompt": opts.prompt,
            "generator": generator,
            "true_cfg_scale": opts.true_cfg_scale,
            "negative_prompt": " ",
            "num_inference_steps": opts.num_inference_steps,
            "guidance_scale": opts.guidance_scale,
            "num_images_per_prompt": 1,
        }

        print(
            f"Generating: prompt='{opts.prompt}', image='{opts.image_path}', "
            f"seed={seed_this_round}, steps={opts.num_inference_steps}, "
            f"guidance={opts.guidance_scale}, true_cfg={opts.true_cfg_scale}"
        )

        t0 = time.time()
        with torch.inference_mode():
            output = pipeline(**inputs)
        out_img = output.images[0]

        save_name = os.path.join(args.output_dir, f"edit_1_{idx:03d}.png")
        out_img.save(save_name)
        t1 = time.time()
        print(f"Saved to {os.path.abspath(save_name)}, time={t1 - t0:.2f}s")

        idx += 1
        print("-" * 80)
        opts = parse_prompt(opts)
        opts = parse_image_path(opts)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=110)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--num_inference_steps", type=int, default=40)
    parser.add_argument("--guidance_scale", type=float, default=1.0)
    parser.add_argument("--true_cfg_scale", type=float, default=4.0)

    parser.add_argument("--use_cache", action="store_true")
    parser.add_argument("--threshold", type=float, default=0.97)

    parser.add_argument("--model_path", type=str, default="/data1/model/Qwen-Image-Edit")
    parser.add_argument("--image_path", type=str, default="/home/chenxueqing/my-flux-activation_cache/datasets/test/images/0000.jpg")
    parser.add_argument("--output_dir", type=str, default="result/QwenImageEdit/Demo/CacheEdit")
    parser.add_argument("--prompt", type=str, default="give the cat a tophat")

    args = parser.parse_args()

    if args.use_cache:
        pipeline = cache_edit_init(args.model_path, args.device)
    else:
        pipeline = QwenImageEditPlusPipeline.from_pretrained(
            args.model_path, torch_dtype=torch.bfloat16, device_map="balanced"
        )

    pipeline.set_progress_bar_config(disable=None)
    interactive_edit_loop(pipeline, args)
