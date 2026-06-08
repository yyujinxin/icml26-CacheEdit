import torch
from transformers import GPTNeoForCausalLM, GPT2Tokenizer, AutoModelForCausalLM, AutoTokenizer
from lm_eval.models.huggingface import HFLM
from lm_eval import evaluator
from lm_eval.utils import make_table
from ..ops import *
from ..pipeline import *
from ..quarot import rotation_utils, hadamard_utils, gptq_utils, quant_utils
from ..quarot import utils as quarot_utils
from tqdm import tqdm
import safetensors as st
from lm_eval.models.vllm_causallms import VLLM
import argparse
import json
import numpy as np
from ..quarot import utils


def rotate(model):
    rotation_utils.fuse_layer_norms(model)
    rotation_utils.rotate_model(model)
    quarot_utils.cleanup_memory(verbos=True)
    rotation_utils.add_actrot(model) #Add Activation Wrapper to the model
    qlayers = rotation_utils.find_rlayers(model)
    for name in qlayers:
        if 'o_proj' in name:
            had_K, K = hadamard_utils.get_hadK(model.config.num_attention_heads)
            qlayers[name].online_partial_had = True
            qlayers[name].had_K = had_K
            qlayers[name].K = K
            qlayers[name].had_dim = model.config.hidden_size//model.config.num_attention_heads
            qlayers[name].fp32_had = True

def get_pipeline(base_name, bitrate, bitrate_max_multiplier, codec='hevc', gop_length=None, frame_interval_p=None):
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

    # GOP configuration for inter-frame prediction
    if gop_length is not None:
        config.gop_length = gop_length
    if frame_interval_p is not None:
        config.frame_interval_p = frame_interval_p

    print(f"Bitrate: {bitrate} Mbps for {base_name}")
    if gop_length is not None or frame_interval_p is not None:
        print(f"  GOP: length={gop_length}, frame_interval_p={frame_interval_p}")
    if base_name == 'self_attn.k_proj.module' or base_name == 'self_attn.v_proj.module':
        kv_pipeline = Pipeline([
            Transpose(),
            CheckShape([4096, 1024]),
            CWQuantization(),
            CheckShape([4096, 1024]),
            FixedTiling([4096, 1024], [1, 1, 4096, 1024], [1, 1, 2048, 1024]),
            # PadUVChannel(),
            MonoNVEncode(config, 2048, 2048),
            ])
        return kv_pipeline
    elif base_name == 'self_attn.q_proj.module':
        q_pipeline = Pipeline([
                Transpose(),
                CheckShape([4096, 4096]),
                CWQuantization(),
                CheckShape([4096, 4096]),
                FixedTiling([4096, 4096], [1, 1, 4096, 4096], [1, 1, 2048, 2048]),
                # PadUVChannel(),
                MonoNVEncode(config, 2048, 2048),
            ])
        return q_pipeline
    elif base_name == 'self_attn.o_proj.module':
        o_pipeline = Pipeline([
            CheckShape([4096, 4096]),
            CWQuantization(),
            CheckShape([4096, 4096]),
            FixedTiling([4096, 4096], [1, 1, 4096, 4096], [1, 1, 2048, 2048]),
            # PadUVChannel(),
            MonoNVEncode(config, 2048, 2048),
        ])
        return o_pipeline
    elif base_name == 'mlp.down_proj.module':
        mlp_down_pipeline = Pipeline([
            Transpose(),
            CheckShape([14336, 4096]),
            CWQuantization(),
            CheckShape([14336, 4096]),
            FixedTiling([14336, 4096], [1, 1, 14336, 4096], [1, 1, 2048, 2048]),
            # PadUVChannel(),
            MonoNVEncode(config, 2048, 2048),
        ])
        return mlp_down_pipeline
    elif base_name == 'mlp.up_proj.module' or base_name == 'mlp.gate_proj.module':
        mlp_up_pipeline = Pipeline([
            CheckShape([14336, 4096]),
            CWQuantization(),
            CheckShape([14336, 4096]),
            FixedTiling([14336, 4096], [1, 1, 14336, 4096], [1, 1, 2048, 2048]),
            # PadUVChannel(),
            MonoNVEncode(config, 2048, 2048),
            ])
        return mlp_up_pipeline


@torch.no_grad
def rtn_nvenc_fwrd(model, dev, mlp_layer_bitrates, attn_layer_bitrates, bitrate_max_multiplier, gop_length=None, frame_interval_p=None):
    '''
    From GPTQ repo
    TODO: Make this function general to support both OPT and LLaMA models
    '''
    layers = model.model.layers
    torch.cuda.empty_cache()

    original_size = 0
    encoded_size = 1
    per_layer_encoded_size = []
    per_layer_original_size = []

    sequential = [
                ['self_attn.k_proj.module', 'self_attn.v_proj.module', ],
                ['self_attn.q_proj.module'],
                ['self_attn.o_proj.module'],
                ['mlp.up_proj.module', 'mlp.gate_proj.module'],
                ['mlp.down_proj.module']
            ]
    for i in tqdm(range(len(layers)), desc="(RtN Quant.) Layers"):
        layer = layers[i].to(dev)
        full = quant_utils.find_qlayers(layer, layers=[torch.nn.Linear])
        for names in sequential:
            subset = {n: full[n] for n in names}
            for name in subset:
                original_weight = subset[name].weight
                print(f'{name}', end='  ', flush=True)
                if 'lm_head' in name:
                    continue
                if "attn" in name:
                    bitrate = attn_layer_bitrates[i]
                elif "mlp" in name:
                    bitrate = mlp_layer_bitrates[i]
                else:
                    raise ValueError(f"Unknown layer type: {name}")
                pipeline = get_pipeline(name, bitrate, bitrate_max_multiplier, gop_length=gop_length, frame_interval_p=frame_interval_p)
                print(name)
                print(original_weight.shape)
                encoded = pipeline.forward(original_weight, name=name+f"_{i}")
                recovered_weight = pipeline.backward(encoded)
                original_size += original_weight.numel() * 2
                encoded_size += encoded["code_size"]
                per_layer_encoded_size.append(encoded["code_size"])
                per_layer_original_size.append(original_weight.numel() * 2)
                original_weight.data.copy_(recovered_weight)
        layers[i] = layer.cpu()
        torch.cuda.empty_cache()
        del layer

    # print original size and encoded size in GB
    print("Original Size: {} GB".format(original_size / 1024 / 1024 / 1024))
    print("Encoded Size: {} GB".format(encoded_size / 1024 / 1024 / 1024))
    # print compression ratio
    print("Compression Ratio: {}".format(original_size / encoded_size))

    utils.cleanup_memory(verbos=True)
    return original_size, encoded_size, per_layer_encoded_size, per_layer_original_size


def process(model, mlp_layer_bitrates, attn_layer_bitrates, bitrate_max_multiplier, gop_length=None, frame_interval_p=None):
    original_size, encoded_size, per_layer_encoded_size, per_layer_original_size = rtn_nvenc_fwrd(
        model, quarot_utils.DEV, mlp_layer_bitrates, attn_layer_bitrates, bitrate_max_multiplier,
        gop_length=gop_length, frame_interval_p=frame_interval_p
    )
    print(model.model.layers[0].self_attn.q_proj.module.weight.sum())
    return model, original_size, encoded_size, per_layer_encoded_size, per_layer_original_size


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # parser.add_argument("--model_id", type=str, default="meta-llama/Meta-Llama-3-8B")
    parser.add_argument("--model_id", type=str, default="meta-llama/Meta-Llama-3-8B")
    # parser.add_argument("--model_id", type=str, default="meta-llama/Llama-2-7b-hf")
    parser.add_argument("--mlp_bitrate_start", type=float, default=1)
    parser.add_argument("--mlp_bitrate_end", type=float, default=10)
    parser.add_argument("--attn_bitrate_start", type=float, default=1)
    parser.add_argument("--attn_bitrate_end", type=float, default=10)
    parser.add_argument("--bitrate_max_multiplier", type=float, default=10)
    parser.add_argument("--codec", type=str, default='hevc', choices=['hevc', 'h264'])
    parser.add_argument("--output_json", type=str)
    parser.add_argument("--gop_length", type=int, default=None,
                        help='GOP length (frames between I-frames). Default: None (all I-frames)')
    parser.add_argument("--frame_interval_p", type=int, default=None,
                        help='P-frame interval (1=IPPP, 2=IBPBP). Default: None (all I-frames)')
    args = parser.parse_args()

    mlp_layer_bitrates = np.linspace(args.mlp_bitrate_start, args.mlp_bitrate_end, 32)
    attn_layer_bitrates = np.linspace(args.attn_bitrate_start, args.attn_bitrate_end, 32)
    print(f"MLP Layer Bitrates: {mlp_layer_bitrates}")
    print(f"Attention Layer Bitrates: {attn_layer_bitrates}")

    # Print GOP configuration
    if args.gop_length is not None or args.frame_interval_p is not None:
        print(f"GOP Configuration: length={args.gop_length}, frame_interval_p={args.frame_interval_p}")
        print("  Inter-frame prediction ENABLED")
    else:
        print("GOP Configuration: All I-frames (intra-only encoding)")

    model_id = args.model_id
    model = AutoModelForCausalLM.from_pretrained(model_id, device_map="cuda", torch_dtype=torch.float16)
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = model.eval()
    rotate(model)
    model.seqlen = 2048
    with torch.inference_mode():
        model, original_size, encoded_size, per_layer_encoded_size, per_layer_original_size = process(
            model, mlp_layer_bitrates, attn_layer_bitrates, args.bitrate_max_multiplier,
            gop_length=args.gop_length, frame_interval_p=args.frame_interval_p
        )
    model = model.cuda()
    # print(model.model.layers[0].self_attn.q_proj.module.weight.sum())
    print("Model replaced with NVENC compressed weights")
    batchsize = 4
    lm_model = HFLM(model, batch_size=batchsize)
    results = evaluator.simple_evaluate(
        model=lm_model,
        tasks=['piqa'],
        batch_size=batchsize,
    )

    results['original_size'] = original_size
    results['encoded_size'] = encoded_size
    results['mlp_layer_bitrates'] = list(mlp_layer_bitrates)
    results['attn_layer_bitrates'] = list(attn_layer_bitrates)
    results['bitrate_max_multiplier'] = args.bitrate_max_multiplier
    results['gop_length'] = args.gop_length
    results['frame_interval_p'] = args.frame_interval_p
    results['per_layer_encoded_size'] = per_layer_encoded_size
    results['per_layer_original_size'] = per_layer_original_size
    results['args'] = vars(args)

    # print results
    output_json_path = args.output_json
    print(make_table(results))
    # dump results to json
    with open(output_json_path, 'w') as f:
        json.dump(results, f, indent=4)

def main():
    """Entry point for the CLI command"""
    import sys
    # Run the main script logic
    globals()['__name__'] = '__main__'
    exec(open(__file__).read())
