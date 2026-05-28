import os
import time
import json
import torch
import argparse

from PIL import Image
# from utils import MANAGER
from diffusers import QwenImageEditPlusPipeline

# from pipeline_qwenimage_edit_plus import QwenImageEditPlusPipeline
from io import BytesIO
import requests
import accelerate
from Qwen_cache_plus import cache_edit_init

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()

    # config for Qwen-Image-Edit-2509
    parser.add_argument("--seed", type=int, default=110, help="Random seed for reproducibility")
    parser.add_argument("--device", type=str, default="cuda", help="Device to run the model on (e.g., 'cuda' or 'cpu')")
    parser.add_argument("--num_inference_steps", type=int, default=28, help="Number of inference steps for the model")
    parser.add_argument("--guidance_scale", type=float, default=4.0, help="Guidance scale for the model")
    # config for RegionE
    parser.add_argument("--use_cache", action='store_true', help="Whether to use regione")
    parser.add_argument("--warmup_step", type=int, default=6, help="Step of the stablization stage")
    parser.add_argument("--post_step", type=int, default=2, help="Step of the smooth stage")
    parser.add_argument("--refresh_step", type=str, default="16", help="Steps are forcibly updated during the region-aware generation stage, format(str):16,22")
    parser.add_argument("--threshold", type=float, default=0.80, help="Threshold for adaptive region partition")
    parser.add_argument("--cache_threshold", type=float, default=0.03, help="Threshold for adaptive velocity decacy cache")
    parser.add_argument("--erosion_dilation", action='store_true', help="Whether to use dilation and erosion")
    # config for path
    parser.add_argument("--model_path", type=str, default="/data1/model/Qwen-Image-Edit", help="Path to the pre-trained model")
    parser.add_argument("--evaluation", action='store_true', help="Whether to evaluate the model on the benchmark")
    parser.add_argument("--image_path", type=str, default="assets/data.jsonl", help="Path to the input data")
    parser.add_argument("--output_dir", type=str, default="result/Qwen-Image-Edit-2509/Demo/RegionE", help="Directory to save the output images")
    args = parser.parse_args()

    if args.use_cache:
        # MANAGER.set_parameters(args)
        pipeline = cache_edit_init(args.model_path, args.device)
    else:
        pipeline = QwenImageEditPlusPipeline.from_pretrained(args.model_path, torch_dtype=torch.bfloat16, device_map="balanced")

    # pipeline = QwenImageEditPlusPipeline.from_pretrained("/data1/model/Qwen-Image-Edit", torch_dtype=torch.bfloat16, device_map="")
    # print("pipeline loaded")

    # pipeline.to('cuda:3')
    pipeline.set_progress_bar_config(disable=None)
    # image1 = Image.open(BytesIO(requests.get("https://qianwen-res.oss-accelerate-overseas.aliyuncs.com/Qwen-Image/edit2511/edit2511input.png").content))
    # image1 = Image.open("/home/chenxueqing/Qwen-Image/cat_hat.png")
    # prompt = "Change the hat on the cat's head to a helmet"
    image1 = Image.open("/home/chenxueqing/my-flux-activation_cache/datasets/test/images/0000.jpg")
    prompt = "give the cat a tophat"
    inputs = {
        "image": [image1],
        "prompt": prompt,
        "generator": torch.manual_seed(0),
        "true_cfg_scale": 16.0,
        "negative_prompt": " ",
        "num_inference_steps": 40,
        "guidance_scale": 1.0,
        "num_images_per_prompt": 1,
    }
    with torch.inference_mode():
        output = pipeline(**inputs)
        output_image = output.images[0]
        # output_image.save("cat_helmet.png")
        output_image.save("cat_hat.png")
        print("image saved at", os.path.abspath("cat_hat.png"))