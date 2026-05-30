import torch
import torch.nn.functional as F
from einops import rearrange
from torch import Tensor
import math
import sys

def attention(q: Tensor, k: Tensor, v: Tensor, pe: Tensor, pe_q: Tensor, layer: int, is_double: bool) -> Tensor:
    q, k = apply_rope(q, k, pe, pe_q)

    # for attention_score visualization
    # if is_double and layer == 18:  # only visualize the attention scores of the 3rd double layer
    #     attn_scores = torch.matmul(q, k.transpose(-2, -1))
    #     scale_factor = 1 / math.sqrt(k.size(-1))
    #     attn_scores = attn_scores * scale_factor
    #     attn_weights = F.softmax(attn_scores, dim=-1)
    #     analyze_and_visualize_attention(attn_weights, output_dir="/home/chenxueqing/my-flux-activation_cache/src/flux/tools/visualize/my_attention_analysis_results", threshold = 0.01, head_to_analyze=4, top_n_to_print=15, save_csv=True)
    #     sys.exit()  # Exit after visualization to avoid further processing
    
    # 计算注意力输出
    x = torch.nn.functional.scaled_dot_product_attention(q, k, v)
    x = rearrange(x, "B H L D -> B L (H D)")

    return x


def rope(pos: Tensor, dim: int, theta: int) -> Tensor:
    assert dim % 2 == 0
    scale = torch.arange(0, dim, 2, dtype=pos.dtype, device=pos.device) / dim
    omega = 1.0 / (theta**scale)
    out = torch.einsum("...n,d->...nd", pos, omega)
    out = torch.stack([torch.cos(out), -torch.sin(out), torch.sin(out), torch.cos(out)], dim=-1)
    out = rearrange(out, "b n d (i j) -> b n d i j", i=2, j=2)
    return out.float()


def apply_rope(xq: Tensor, xk: Tensor, freqs_cis: Tensor, freqs_cis_q: Tensor) -> tuple[Tensor, Tensor]:
    xq_ = xq.float().reshape(*xq.shape[:-1], -1, 1, 2)
    xk_ = xk.float().reshape(*xk.shape[:-1], -1, 1, 2)
    xq_out = freqs_cis_q[..., 0] * xq_[..., 0] + freqs_cis_q[..., 1] * xq_[..., 1]
    xk_out = freqs_cis[..., 0] * xk_[..., 0] + freqs_cis[..., 1] * xk_[..., 1]
    return xq_out.reshape(*xq.shape).type_as(xq), xk_out.reshape(*xk.shape).type_as(xk)
