import os
import json
import time
import torch
import argparse
import accelerate
from PIL import Image

from diffusers.utils import load_image
from utils import ActivationCacheManager, FluxKontextPipeline, pipeline_call
from Flux_cache import cache_edit_init

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    # config for Flux.1 Kontext
    parser.add_argument("--seed", type=int, default=110, help="Random seed for reproducibility")
    parser.add_argument("--device", type=str, default="cuda", help="Device to run the model on (e.g., 'cuda' or 'cpu')")
    parser.add_argument("--cache_device", type=str, default="cuda", help="Device to run the model on (e.g., 'cuda' or 'cpu')")
    parser.add_argument("--num_inference_steps", type=int, default=28, help="Number of inference steps for the model")
    parser.add_argument("--guidance_scale", type=float, default=2.5, help="Guidance scale for the model")
    # config for RegionE
    parser.add_argument("--use_cache", action='store_true', help="Whether to use regione")
    parser.add_argument("--warmup_step", type=int, default=6, help="Step of the stablization stage")
    parser.add_argument("--threshold", type=float, default=0.97, help="Threshold for adaptive region partition")
    # parser.add_argument("--cache_threshold", type=float, default=0.04, help="Threshold for adaptive velocity decacy cache")
    # parser.add_argument("--erosion_dilation", action='store_true', help="Whether to use dilation and erosion")
    # config for path
    parser.add_argument("--model_path", type=str, default="/data1/model/FLUX.1-Kontext-dev", help="Path to the pre-trained model")
    parser.add_argument("--evaluation", action='store_true', help="Whether to evaluate the model on the benchmark")
    parser.add_argument("--image_path", type=str, default="/home/chenxueqing/my-flux-activation_cache/datasets/test/images/0000.jpg", help="Path to the input data")
    parser.add_argument("--output_dir", type=str, default="result/FluxKontext/Demo/RegionE", help="Directory to save the output images")
    args = parser.parse_args()

    if args.use_cache:
        # ActivationCacheManager.set_parameters(args)
        pipeline = cache_edit_init(args.model_path, args.device)
    else:
        FluxKontextPipeline.__call__ = pipeline_call    # fix the resolution
        pipeline = FluxKontextPipeline.from_pretrained(args.model_path, torch_dtype=torch.bfloat16, device_map="balanced")

    # pipeline.to('cuda:3')
    pipeline.set_progress_bar_config(disable=None)
    image1 = Image.open("/home/chenxueqing/my-flux-activation_cache/datasets/test/images/0000.jpg")
    prompt = "give the cat a tophat"
    inputs = {
        "image": [image1],
        "prompt": prompt,
        "generator": torch.manual_seed(0),
        "true_cfg_scale": 1.0,
        "negative_prompt": " ",
        "num_inference_steps": 30,
        "guidance_scale": 16.0,
        "num_images_per_prompt": 1,
    }
    with torch.inference_mode():
        output = pipeline(**inputs)
        output_image = output.images[0]
        # output_image.save("cat_helmet.png")
        output_image.save("cat_hat_1.png")
        print("image saved at", os.path.abspath("cat_hat_1.png"))



   