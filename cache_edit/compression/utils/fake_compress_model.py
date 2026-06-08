import argparse
import transformers
from ..ops import *
from ..pipeline import *
import ray
from tqdm import tqdm
import json


@torch.no_grad()
def replace_linear_weights(model, config, outlier, blacklist=[]):
    # init CUDA context
    tensor = torch.zeros(0, dtype=torch.uint8, device='cuda')

    outlier_vars_to_center = outlier
    clip_vars_to_center = 10000
    ffn_pipeline = Pipeline([
        CheckShape([4096, 4096]),
        CWOQuantization(outlier_vars_to_center=outlier_vars_to_center, clip_vars_to_center=clip_vars_to_center),
        CheckShape([4096, 4096]),
        FixedTiling([4096, 4096], [1, 1, 4096, 4096], [1, 1, 4096, 4096]),
        PadUVChannel(),
        NVEncode(config, 4096, 4096),
    ])
    mlp_down_pipeline = Pipeline([
        CheckShape([11008, 4096]),
        CWOQuantization(outlier_vars_to_center=outlier_vars_to_center, clip_vars_to_center=clip_vars_to_center),
        CheckShape([11008, 4096]),
        FixedTiling([11008, 4096], [1, 1, 11008, 4096], [1, 1, 2752, 2048]),
        PadUVChannel(),
        NVEncode(config, 2752, 4096),
    ])
    mlp_up_pipeline = Pipeline([
        CheckShape([4096, 11008]),
        CWOQuantization(outlier_vars_to_center=outlier_vars_to_center, clip_vars_to_center=clip_vars_to_center),
        CheckShape([4096, 11008]),
        FixedTiling([4096, 11008], [1, 1, 4096, 11008], [1, 1, 2048, 2752]),
        PadUVChannel(),
        NVEncode(config, 4096, 2752),
    ])

    # iterate through all Linear layers in the model
    original_size = 0
    encoded_size = 1
    outlier_size = 0
    for name, module in tqdm(list(model.named_modules())):
        if isinstance(module, torch.nn.Linear):
            to_continue = False
            for black in blacklist:
                if black in name:
                    to_continue = True
            if to_continue:
                print("Skipping layer {}".format(name))
                continue
            # torch clear cache
            torch.cuda.empty_cache()
            original_weight = module.weight
            # check original_weight shape == [4096, 4096]
            if list(original_weight.shape) == [4096, 4096]:
                encoded = ffn_pipeline.forward(original_weight, name=name)
                recovered_weight = ffn_pipeline.backward(encoded)
            elif list(original_weight.shape) == [11008, 4096]:
                encoded = mlp_down_pipeline.forward(original_weight, name=name)
                recovered_weight = mlp_down_pipeline.backward(encoded)
            elif list(original_weight.shape) == [4096, 11008]:
                encoded = mlp_up_pipeline.forward(original_weight, name=name)
                recovered_weight = mlp_up_pipeline.backward(encoded)
            else:
                raise ValueError("Unsupported weight shape {} for layer {}".format(original_weight.shape, name))

            original_size += original_weight.numel() * 2
            sparse_nnz = encoded["outliers"]._nnz()
            encoded_size += encoded["code_size"] if "code_size" in encoded else 0
            outlier_size += sparse_nnz * 3
            original_weight.data.copy_(recovered_weight)
    # print original size and encoded size in GB
    print("Original Size: {} GB".format(original_size / 1024 / 1024 / 1024))
    print("Encoded Size: {} GB".format(encoded_size / 1024 / 1024 / 1024))
    print("Outlier Size: {} GB".format(outlier_size / 1024 / 1024 / 1024))
    print("Compression Ratio: {}".format(original_size / (encoded_size + outlier_size)))
    result_dict = {
        "original_size": original_size,
        "encoded_size": encoded_size,
        "outlier_size": outlier_size,
    }
    return model, result_dict


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_id", type=str, default="meta-llama/Llama-2-7b-hf")
    parser.add_argument("--bitrate", type=float, default=10)
    parser.add_argument("--outlier", type=float, default=0.01)
    parser.add_argument("--output_model_id", type=str)
    args = parser.parse_args()

    model = transformers.AutoModelForCausalLM.from_pretrained(args.model_id, device_map="cuda", torch_dtype=torch.float16)
    tokenizer = transformers.AutoTokenizer.from_pretrained(args.model_id)

    config = TensorEncodeConfig()
    config.input_format = InputFormat.YUV444
    # 100 Mbps
    config.average_bit_rate = int(args.bitrate * 1000000)
    # 200 Mbps
    config.max_bit_rate = int(args.bitrate * 1000000 * 5)
    config.codec_type = CodecType.HEVC
    config.rc_mode = RateControlMode.VBR
    config.preset = PresetType.P7
    config.tuning_info = TuningInfo.HighQuality

    model = model.eval()
    model, result_dict = replace_linear_weights(model, config, args.outlier, blacklist=["lm_head", "o_proj", "down_proj"])
    print("Model replaced with NVENC compressed weights")
    model.save_pretrained(args.output_model_id)
    tokenizer.save_pretrained(args.output_model_id)
    # dump result_dict to file
    with open(f"{args.output_model_id}/compression_result.json", "w") as f:
        json.dump(result_dict, f)
    print(f"Model saved to {args.output_model_id}")
    print("Done")
