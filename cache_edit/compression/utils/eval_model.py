import torch
from transformers import GPTNeoForCausalLM, GPT2Tokenizer, AutoModelForCausalLM, AutoTokenizer
from lm_eval.models.huggingface import HFLM
from lm_eval import evaluator, utils
from lm_eval.utils import make_table
from ..ops import *
from ..pipeline import *
from tqdm import tqdm
import safetensors as st
from lm_eval.models.vllm_causallms import VLLM
import argparse
import json


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_id", type=str, default="meta-llama/Llama-2-7b-hf")
    parser.add_argument("--output_json", type=str)
    parser.add_argument("--num_gpus", type=int, default=1)
    parser.add_argument("--batchsize", type=int, default=16)
    args = parser.parse_args()

    output_json_path = args.output_json

    batchsize = args.batchsize
    model_id = args.model_id
    print(f"Using model {model_id}")
    model = AutoModelForCausalLM.from_pretrained(args.model_id, device_map="cuda", torch_dtype=torch.float16)
    lm_model = HFLM(model, batch_size=batchsize)
    # lm_model.AUTO_MODEL_CLASS = AutoModelForCausalLM

    results = evaluator.simple_evaluate(
        model=lm_model,
        tasks=['hellaswag', 'winogrande', 'piqa', 'lambada', 'openbookqa', 'rte', 'copa', 'arc_challenge', 'arc_easy'],
        batch_size=batchsize,
    )

    # print results
    print(make_table(results))
    # dump results to json
    with open(output_json_path, 'w') as f:
        json.dump(results, f, indent=4)
