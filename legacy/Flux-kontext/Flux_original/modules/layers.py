import math
from dataclasses import dataclass

import torch
from einops import rearrange
from torch import Tensor, nn

from attention import attention, rope
import os,sys
import torch.nn.functional as F

class EmbedND(nn.Module):
    def __init__(self, dim: int, theta: int, axes_dim: list[int]):
        super().__init__()
        self.dim = dim
        self.theta = theta
        self.axes_dim = axes_dim

    def forward(self, ids: Tensor) -> Tensor:
        n_axes = ids.shape[-1]
        emb = torch.cat(
            [rope(ids[..., i], self.axes_dim[i], self.theta) for i in range(n_axes)],
            dim=-3,
        )

        return emb.unsqueeze(1)


def timestep_embedding(t: Tensor, dim, max_period=10000, time_factor: float = 1000.0):
    """
    Create sinusoidal timestep embeddings.
    :param t: a 1-D Tensor of N indices, one per batch element.
                      These may be fractional.
    :param dim: the dimension of the output.
    :param max_period: controls the minimum frequency of the embeddings.
    :return: an (N, D) Tensor of positional embeddings.
    """
    t = time_factor * t
    half = dim // 2
    freqs = torch.exp(-math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half).to(
        t.device
    )

    args = t[:, None].float() * freqs[None]
    embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    if dim % 2:
        embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
    if torch.is_floating_point(t):
        embedding = embedding.to(t)
    return embedding


class MLPEmbedder(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int):
        super().__init__()
        self.in_layer = nn.Linear(in_dim, hidden_dim, bias=True)
        self.silu = nn.SiLU()
        self.out_layer = nn.Linear(hidden_dim, hidden_dim, bias=True)

    def forward(self, x: Tensor) -> Tensor:
        return self.out_layer(self.silu(self.in_layer(x)))


class RMSNorm(torch.nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.scale = nn.Parameter(torch.ones(dim))

    def forward(self, x: Tensor):
        x_dtype = x.dtype
        x = x.float()
        rrms = torch.rsqrt(torch.mean(x**2, dim=-1, keepdim=True) + 1e-6)
        return (x * rrms).to(dtype=x_dtype) * self.scale


class QKNorm(torch.nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.query_norm = RMSNorm(dim)
        self.key_norm = RMSNorm(dim)

    def forward(self, q: Tensor, k: Tensor, v: Tensor) -> tuple[Tensor, Tensor]:
        q = self.query_norm(q)
        k = self.key_norm(k)
        return q.to(v), k.to(v)


class SelfAttention(nn.Module):
    def __init__(self, dim: int, num_heads: int = 8, qkv_bias: bool = False):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.norm = QKNorm(head_dim)
        self.proj = nn.Linear(dim, dim)

    def forward(self, x: Tensor, pe: Tensor) -> Tensor:
        qkv = self.qkv(x)
        q, k, v = rearrange(qkv, "B L (K H D) -> K B H L D", K=3, H=self.num_heads)
        q, k = self.norm(q, k, v)
        x = attention(q, k, v, pe=pe)
        x = self.proj(x)
        return x


@dataclass
class ModulationOut:
    shift: Tensor
    scale: Tensor
    gate: Tensor


class Modulation(nn.Module):
    def __init__(self, dim: int, double: bool):
        super().__init__()
        self.is_double = double
        self.multiplier = 6 if double else 3
        self.lin = nn.Linear(dim, self.multiplier * dim, bias=True)

    def forward(self, vec: Tensor) -> tuple[ModulationOut, ModulationOut | None]:
        out = self.lin(nn.functional.silu(vec))[:, None, :].chunk(self.multiplier, dim=-1)

        return (
            ModulationOut(*out[:3]),
            ModulationOut(*out[3:]) if self.is_double else None,
        )


class DoubleStreamBlock(nn.Module):
    def __init__(self, hidden_size: int, num_heads: int, mlp_ratio: float, qkv_bias: bool = False):
        super().__init__()

        mlp_hidden_dim = int(hidden_size * mlp_ratio)
        self.num_heads = num_heads
        self.hidden_size = hidden_size
        self.img_mod = Modulation(hidden_size, double=True)
        self.img_norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.img_attn = SelfAttention(dim=hidden_size, num_heads=num_heads, qkv_bias=qkv_bias)

        self.img_norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.img_mlp = nn.Sequential(
            nn.Linear(hidden_size, mlp_hidden_dim, bias=True),
            nn.GELU(approximate="tanh"),
            nn.Linear(mlp_hidden_dim, hidden_size, bias=True),
        )

        self.txt_mod = Modulation(hidden_size, double=True)
        self.txt_norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.txt_attn = SelfAttention(dim=hidden_size, num_heads=num_heads, qkv_bias=qkv_bias)

        self.txt_norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.txt_mlp = nn.Sequential(
            nn.Linear(hidden_size, mlp_hidden_dim, bias=True),
            nn.GELU(approximate="tanh"),
            nn.Linear(mlp_hidden_dim, hidden_size, bias=True),
        )
        self.workspace: dict | None = None

    def forward(self, step: int, double_layer: int, img: Tensor, txt: Tensor, vec: Tensor, pe: Tensor, key_token_num: int, should_reuse: bool) -> tuple[Tensor, Tensor]:
        img_mod1, img_mod2 = self.img_mod(vec)
        txt_mod1, txt_mod2 = self.txt_mod(vec)

        # prepare image for attention
        img_modulated = self.img_norm1(img)
        img_modulated = (1 + img_mod1.scale) * img_modulated + img_mod1.shift
        # 拆 img_qkv，只算 key 的 Q，KV 仍算全量
        if should_reuse:
            H = self.hidden_size
            W = self.img_attn.qkv.weight
            b = self.img_attn.qkv.bias
            # Q on prefix only (img 已经在 model.py 里变成 [Key|BG]，所以前缀就是 key)
            q_len = key_token_num
            img_mod_prefix = img_modulated[:, :q_len, :]          # [B, q_len, H]
            W_q = W[0:H, :]
            b_q = None if b is None else b[0:H]
            q_proj = F.linear(img_mod_prefix, W_q, b_q)           # [B, q_len, H]

            # KV on full length
            W_kv = W[H:3 * H, :]
            b_kv = None if b is None else b[H:3 * H]
            kv_full = F.linear(img_modulated, W_kv, b_kv)         # [B, L_img, 2H]
            k_proj, v_proj = kv_full.split(H, dim=-1)             # each [B, L_img, H]

            # reshape to attention layout: [B, Heads, L, HeadDim]
            img_q = rearrange(q_proj, "B L (H D) -> B H L D", H=self.num_heads)
            img_k = rearrange(k_proj, "B L (H D) -> B H L D", H=self.num_heads)
            img_v = rearrange(v_proj, "B L (H D) -> B H L D", H=self.num_heads)

            img_q, img_k = self.img_attn.norm(img_q, img_k, img_v)

        else:
            img_qkv = self.img_attn.qkv(img_modulated)
            img_q, img_k, img_v = rearrange(img_qkv, "B L (K H D) -> K B H L D", K=3, H=self.num_heads)
            img_q, img_k = self.img_attn.norm(img_q, img_k, img_v)

        # prepare txt for attention
        txt_modulated = self.txt_norm1(txt)
        txt_modulated = (1 + txt_mod1.scale) * txt_modulated + txt_mod1.shift
        txt_qkv = self.txt_attn.qkv(txt_modulated)
        txt_q, txt_k, txt_v = rearrange(txt_qkv, "B L (K H D) -> K B H L D", K=3, H=self.num_heads)
        
        txt_q, txt_k = self.txt_attn.norm(txt_q, txt_k, txt_v)

       
    
        # extract important tensor part by key_token_indices
        # img_q = torch.index_select(img_q, 2, key_token_indices)
        if should_reuse:
            img_q = img_q[:, :, :key_token_num, :]
            # img = torch.index_select(img, 1, key_token_indices)
            img = img[:, :key_token_num, :]
            pe_q = pe[:, :, :key_token_num + 512, ...]
            # pe_q = torch.index_select(pe, 2, key_token_indices + txt.shape[1])
            # pe_q = torch.cat(( pe[:, :, :txt.shape[1], ...], pe_q,), dim=2)
        else:
            pe_q = pe
        # run actual attention
        if self.workspace is not None:
            # txt_q/txt_k/txt_v: [B, H, Lt, D]
            # img_q:            [B, H, Li_q, D]  (reuse 时 Li_q=key_token_num)
            # img_k/img_v:      [B, H, Li_kv, D] (通常是 full image tokens)
            B, H, Lt, D = txt_q.shape
            Li_q = img_q.shape[2]
            Li_kv = img_k.shape[2]

            Lq = Lt + Li_q
            Lkv = Lt + Li_kv

            # ---- q buffer ----
            qbuf = self.workspace.get("double_q")
            if (
                qbuf is None
                or qbuf.device != txt_q.device
                or qbuf.dtype != txt_q.dtype
                or qbuf.shape != (B, H, Lq, D)
            ):
                qbuf = torch.empty((B, H, Lq, D), device=txt_q.device, dtype=txt_q.dtype)
                self.workspace["double_q"] = qbuf
            qbuf[:, :, :Lt, :].copy_(txt_q)
            qbuf[:, :, Lt:, :].copy_(img_q)
            q = qbuf

            # ---- k buffer ----
            kbuf = self.workspace.get("double_k")
            if (
                kbuf is None
                or kbuf.device != txt_k.device
                or kbuf.dtype != txt_k.dtype
                or kbuf.shape != (B, H, Lkv, D)
            ):
                kbuf = torch.empty((B, H, Lkv, D), device=txt_k.device, dtype=txt_k.dtype)
                self.workspace["double_k"] = kbuf
            kbuf[:, :, :Lt, :].copy_(txt_k)
            kbuf[:, :, Lt:, :].copy_(img_k)
            k = kbuf

            # ---- v buffer ----
            vbuf = self.workspace.get("double_v")
            if (
                vbuf is None
                or vbuf.device != txt_v.device
                or vbuf.dtype != txt_v.dtype
                or vbuf.shape != (B, H, Lkv, D)
            ):
                vbuf = torch.empty((B, H, Lkv, D), device=txt_v.device, dtype=txt_v.dtype)
                self.workspace["double_v"] = vbuf
            vbuf[:, :, :Lt, :].copy_(txt_v)
            vbuf[:, :, Lt:, :].copy_(img_v)
            v = vbuf
        else:
            q = torch.cat((txt_q, img_q), dim=2)
            k = torch.cat((txt_k, img_k), dim=2)
            v = torch.cat((txt_v, img_v), dim=2)
        # #print("k.shape: ", k.shape, "v.shape: ", v.shape)    k.shape:  torch.Size([1, 24, 8597, 128]) v.shape:  torch.Size([1, 24, 8597, 128])
        #print("double 3: k.shape: ", k.shape, "v.shape: ", v.shape)
        
        # print("is_round0: ",is_round0, "q.shepe: ", q.shape, "pe.shape: ", "pe_q.shape: ", pe_q.shape, "img_qkv.shape :", img_qkv.shape)
        attn = attention(q, k, v, pe=pe, pe_q=pe_q, layer=double_layer, is_double=True)
        
        txt_attn, img_attn = attn[:, : txt.shape[1]], attn[:, txt.shape[1] :]

        # calculate the img blocks
        img = img + img_mod1.gate * self.img_attn.proj(img_attn)
        img = img + img_mod2.gate * self.img_mlp((1 + img_mod2.scale) * self.img_norm2(img) + img_mod2.shift)

        # calculate the txt blocks
        txt = txt + txt_mod1.gate * self.txt_attn.proj(txt_attn)
        txt = txt + txt_mod2.gate * self.txt_mlp((1 + txt_mod2.scale) * self.txt_norm2(txt) + txt_mod2.shift)
        return img, txt


class SingleStreamBlock(nn.Module):
    """
    A DiT block with parallel linear layers as described in
    https://arxiv.org/abs/2302.05442 and adapted modulation interface.
    """

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        qk_scale: float | None = None,
    ):
        super().__init__()
        self.hidden_dim = hidden_size
        self.num_heads = num_heads
        head_dim = hidden_size // num_heads
        self.scale = qk_scale or head_dim**-0.5

        self.mlp_hidden_dim = int(hidden_size * mlp_ratio)
        # qkv and mlp_in
        self.linear1 = nn.Linear(hidden_size, hidden_size * 3 + self.mlp_hidden_dim)
        # proj and mlp_out
        self.linear2 = nn.Linear(hidden_size + self.mlp_hidden_dim, hidden_size)

        self.norm = QKNorm(head_dim)

        self.hidden_size = hidden_size
        self.pre_norm = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)

        self.mlp_act = nn.GELU(approximate="tanh")
        self.modulation = Modulation(hidden_size, double=False)
        self.workspace: dict | None = None

    def forward(self, x: Tensor, vec: Tensor, pe: Tensor, step: int, single_layer: int, key_token_num: int, should_reuse : bool) -> Tensor:
        mod, _ = self.modulation(vec)
        x_mod_full = (1 + mod.scale) * self.pre_norm(x) + mod.shift
        H = self.hidden_size
        MLPH = self.mlp_hidden_dim
        W = self.linear1.weight
        b = self.linear1.bias
        
        if should_reuse:
            Lq = key_token_num + 512  # txt(512) + key_token
            x_mod_prefix = x_mod_full[:, :Lq, :]

            # ---- (A) KV on full length ----
            W_kv = W[H:3*H, :]
            b_kv = None if b is None else b[H:3*H]
            kv_full = F.linear(x_mod_full, W_kv, b_kv)
            k_proj, v_proj = kv_full.split(H, dim=-1)                 # each [B, L, H]

            # ---- (B) Q + MLP on prefix only ----
            W_q = W[0:H, :]
            b_q = None if b is None else b[0:H]
            q_proj = F.linear(x_mod_prefix, W_q, b_q)                 # [B, Lq, H]

            W_mlp = W[3*H:3*H+MLPH, :]
            b_mlp = None if b is None else b[3*H:3*H+MLPH]
            mlp = F.linear(x_mod_prefix, W_mlp, b_mlp)                # [B, Lq, MLPH]

            # reshape to attention layout
            q = rearrange(q_proj, "B L (H D) -> B H L D", H=self.num_heads)
            k = rearrange(k_proj, "B L (H D) -> B H L D", H=self.num_heads)
            v = rearrange(v_proj, "B L (H D) -> B H L D", H=self.num_heads)

            q, k = self.norm(q, k, v)

            # 注意：x 也要裁成 prefix 长度，因为 residual 输出只会有 prefix
            x = x[:, :Lq, :]

            pe_q = pe[:, :, :Lq, ...]
        else:
            qkv, mlp = torch.split(self.linear1(x_mod_full), [3 * H, MLPH], dim=-1)
            q, k, v = rearrange(qkv, "B L (K H D) -> K B H L D", K=3, H=self.num_heads)
            q, k = self.norm(q, k, v)
            pe_q = pe

        # compute attention
        attn = attention(q, k, v, pe=pe, pe_q=pe_q, layer=single_layer, is_double=False)
      
        # compute activation in mlp stream, cat again and run second linear layer
        # output = self.linear2(torch.cat((attn, self.mlp_act(mlp)), 2))

        # compute activation in mlp stream
        mlp_act = self.mlp_act(mlp)

        # non-cache step(should_reuse=True) 用 workspace buffer + copy_ 替代 cat
        # 这样不会每层都分配一个巨大的 cat 临时张量
        if should_reuse and (self.workspace is not None):
            key = "single_linear2_in"  # 新key
            B, L, _ = attn.shape
            in_dim = self.hidden_size + self.mlp_hidden_dim

            buf = self.workspace.get(key)
            if (
                buf is None
                or buf.device != attn.device
                or buf.dtype != attn.dtype
                or buf.shape != (B, L, in_dim)
            ):
                buf = torch.empty((B, L, in_dim), device=attn.device, dtype=attn.dtype)
                self.workspace[key] = buf

            # 写入 [attn | mlp_act]
            buf[:, :, : self.hidden_size].copy_(attn)
            buf[:, :, self.hidden_size :].copy_(mlp_act)
            output = self.linear2(buf)
        else:
            # cache step 或者未提供 workspace：保持原逻辑，避免常驻超大 buffer
            output = self.linear2(torch.cat((attn, mlp_act), dim=2))
        
        return x + mod.gate * output


class LastLayer(nn.Module):
    def __init__(self, hidden_size: int, patch_size: int, out_channels: int):
        super().__init__()
        self.norm_final = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.linear = nn.Linear(hidden_size, patch_size * patch_size * out_channels, bias=True)
        self.adaLN_modulation = nn.Sequential(nn.SiLU(), nn.Linear(hidden_size, 2 * hidden_size, bias=True))

    def forward(self, x: Tensor, vec: Tensor) -> Tensor:
        shift, scale = self.adaLN_modulation(vec).chunk(2, dim=1)
        x = (1 + scale[:, None, :]) * self.norm_final(x) + shift[:, None, :]
        x = self.linear(x)
        return x
