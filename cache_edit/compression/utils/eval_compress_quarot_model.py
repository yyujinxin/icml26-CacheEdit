import torch
from transformers import GPTNeoForCausalLM, GPT2Tokenizer, AutoModelForCausalLM, AutoTokenizer
from lm_eval.models.huggingface import HFLM
from lm_eval import evaluator, utils
from lm_eval.utils import make_table
from ..ops import *
from ..pipeline import *
from ..quarot import rotation_utils, hadamard_utils, gptq_utils
from ..quarot import utils as quarot_utils
from tqdm import tqdm
import safetensors as st
from lm_eval.models.vllm_causallms import VLLM
import argparse
import json


def rotate(model):
    rotation_utils.fuse_layer_norms(model)
    rotation_utils.rotate_model(model)
    quarot_utils.cleanup_memory(verbos=True)
    rotation_utils.add_actrot(model) #Add Activation Wrapper to the model
    qlayers = rotation_utils.find_rlayers(model)
    for name in qlayers:
        #if 'down_proj' in name:
        #    had_K, K = hadamard_utils.get_hadK(model.config.intermediate_size)
        #    qlayers[name].online_full_had = True
        #    qlayers[name].had_K = had_K
        #    qlayers[name].K = K
        #    qlayers[name].fp32_had = True
        if 'o_proj' in name:
            had_K, K = hadamard_utils.get_hadK(model.config.num_attention_heads)
            qlayers[name].online_partial_had = True
            qlayers[name].had_K = had_K
            qlayers[name].K = K
            qlayers[name].had_dim = model.config.hidden_size//model.config.num_attention_heads
            qlayers[name].fp32_had = True

def process(model, bitrate, bitrate_max_multiplier, codec='hevc'):
    torch.cuda.set_device(0)
    config = TensorEncodeConfig()
    config.input_format = InputFormat.NV12
    # 100 Mbps
    config.average_bit_rate = int(bitrate * 1000000)
    # 200 Mbps
    config.max_bit_rate = int(bitrate * 1000000 * bitrate_max_multiplier)
    if codec == 'hevc':
        config.codec_type = CodecType.HEVC
    elif codec == 'h264':
        config.codec_type = CodecType.H264
    else:
        raise ValueError(f"Invalid codec {codec}")
    config.rc_mode = RateControlMode.VBR
    config.preset = PresetType.P7
    config.tuning_info = TuningInfo.HighQuality
    config.monochrome = True
    q_pipeline = Pipeline([
            Transpose(),
            CheckShape([4096, 4096]),
            CWQuantization(),
            CheckShape([4096, 4096]),
            FixedTiling([4096, 4096], [1, 1, 4096, 4096], [1, 1, 2048, 2048]),
            # PadUVChannel(),
            MonoNVEncode(config, 2048, 2048),
        ])
    kv_pipeline = Pipeline([
        Transpose(),
        CheckShape([4096, 4096]),
        CWQuantization(),
        CheckShape([4096, 4096]),
        FixedTiling([4096, 4096], [1, 1, 4096, 4096], [1, 1, 2048, 2048]),
        # PadUVChannel(),
        MonoNVEncode(config, 2048, 2048),
        ])
    o_pipeline = Pipeline([
        CheckShape([4096, 4096]),
        CWQuantization(),
        CheckShape([4096, 4096]),
        FixedTiling([4096, 4096], [1, 1, 4096, 4096], [1, 1, 2048, 2048]),
        # PadUVChannel(),
        MonoNVEncode(config, 2048, 2048),
    ])
    mlp_down_pipeline = Pipeline([
        CheckShape([11008, 4096]),
        CWQuantization(),
        CheckShape([11008, 4096]),
        FixedTiling([11008, 4096], [1, 1, 11008, 4096], [1, 1, 2752, 2048]),
        # PadUVChannel(),
        MonoNVEncode(config, 2752, 4096),
    ])
    mlp_up_pipeline = Pipeline([
        Transpose(),
        CheckShape([11008, 4096]),
        CWQuantization(),
        CheckShape([11008, 4096]),
        FixedTiling([11008, 4096], [1, 1, 11008, 4096], [1, 1, 2752, 2048]),
        # PadUVChannel(),
        MonoNVEncode(config, 2752, 4096),
        ])
    original_size, encoded_size = gptq_utils.rtn_nvenc_fwrd(model, quarot_utils.DEV, kv_pipeline, q_pipeline, o_pipeline, mlp_up_pipeline, mlp_down_pipeline)
    print(model.model.layers[0].self_attn.q_proj.module.weight.sum())
    return model, original_size, encoded_size


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_id", type=str, default="meta-llama/Llama-2-7b-hf")
    parser.add_argument("--bitrate", type=float, default=10)
    parser.add_argument("--codec", type=str, default='hevc', choices=['hevc', 'h264'])
    parser.add_argument("--bitrate_max_multiplier", type=float, default=10)
    parser.add_argument("--output_json", type=str)
    args = parser.parse_args()

    model_id = args.model_id
    model = AutoModelForCausalLM.from_pretrained(model_id, device_map="cuda", torch_dtype=torch.float16)
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = model.eval()
    rotate(model)
    model.seqlen = 2048
    with torch.inference_mode():
        model, original_size, encoded_size = process(model, args.bitrate, args.bitrate_max_multiplier, codec=args.codec)
    model = model.cuda()
    print(model.model.layers[0].self_attn.q_proj.module.weight.sum())
    print("Model replaced with NVENC compressed weights")
    batchsize = 4
    lm_model = HFLM(model, batch_size=batchsize)
    results = evaluator.simple_evaluate(
        model=lm_model,
        tasks=['hellaswag', 'winogrande', 'piqa', 'lambada', 'openbookqa', 'rte', 'copa', 'arc_challenge', 'arc_easy'],
        batch_size=batchsize,
    )

    results['original_size'] = original_size
    results['encoded_size'] = encoded_size
    results['bitrate'] = args.bitrate
    results['bitrate_max_multiplier'] = args.bitrate_max_multiplier
    results['args'] = vars(args)

    # print results
    output_json_path = args.output_json
    print(make_table(results))
    # dump results to json
    with open(output_json_path, 'w') as f:
        json.dump(results, f, indent=4)
