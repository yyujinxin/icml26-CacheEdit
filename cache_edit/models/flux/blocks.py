"""Flux transformer block forward functions with caching support.

These forward functions are designed to be **bound to** existing diffusers
`FluxTransformerBlock` / `FluxSingleTransformerBlock` instances. They behave
like the original block forwards but:

- read the cache context from `block.cache_context` (set during pipeline init)
- support the key-token-only residual and MLP optimizations
- maintain a per-block ``workspace`` dict for buffer reuse

Usage::

    block.cache_context = cache_manager
    block.workspace = {}
    block.forward = cache_flux_transformer_block_forward.__get__(
        block, block.__class__
    )
"""

from typing import Any, Dict, Optional, Tuple

import torch


def _get_ctx(block):
    return getattr(block, "cache_context", None)


def _should_reuse_with_indices(
    ctx,
) -> Tuple[bool, Optional[torch.Tensor]]:
    if ctx is None:
        return False, None
    should_reuse = bool(ctx.should_reuse(ctx.current_step))
    kti = getattr(ctx, "key_token_indices", None)
    return should_reuse and (kti is not None), kti


def cache_flux_transformer_block_forward(
    self,
    hidden_states: torch.Tensor,
    encoder_hidden_states: torch.Tensor,
    temb: torch.Tensor,
    image_rotary_emb: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    joint_attention_kwargs: Optional[Dict[str, Any]] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Double-stream transformer block forward with caching support.

    Compared to the stock FluxTransformerBlock forward, this:
    1. After attention, if the block is in a reuse step with key_token_indices,
       slices `hidden_states` down to `[:, :key_token_num]` so that the residual
       add aligns with the partial-Q attn_output.
    """
    norm_hidden_states, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.norm1(
        hidden_states, emb=temb
    )

    norm_encoder_hidden_states, c_gate_msa, c_shift_mlp, c_scale_mlp, c_gate_mlp = (
        self.norm1_context(encoder_hidden_states, emb=temb)
    )
    joint_attention_kwargs = joint_attention_kwargs or {}

    attention_outputs = self.attn(
        hidden_states=norm_hidden_states,
        encoder_hidden_states=norm_encoder_hidden_states,
        image_rotary_emb=image_rotary_emb,
        **joint_attention_kwargs,
    )

    ip_attn_output = None
    if len(attention_outputs) == 2:
        attn_output, context_attn_output = attention_outputs
    elif len(attention_outputs) == 3:
        attn_output, context_attn_output, ip_attn_output = attention_outputs
    else:
        raise ValueError(
            f"Unexpected attention output length: {len(attention_outputs)}"
        )

    attn_output = gate_msa.unsqueeze(1) * attn_output

    ctx = _get_ctx(self)
    do_partial, kti = _should_reuse_with_indices(ctx)
    if do_partial and kti is not None:
        key_token_num = kti.shape[0]
        hidden_states = hidden_states[:, :key_token_num]

    hidden_states = hidden_states + attn_output

    norm_hidden_states = self.norm2(hidden_states)
    norm_hidden_states = (
        norm_hidden_states * (1 + scale_mlp[:, None]) + shift_mlp[:, None]
    )

    ff_output = self.ff(norm_hidden_states)
    ff_output = gate_mlp.unsqueeze(1) * ff_output

    hidden_states = hidden_states + ff_output
    if ip_attn_output is not None:
        hidden_states = hidden_states + ip_attn_output

    context_attn_output = c_gate_msa.unsqueeze(1) * context_attn_output
    encoder_hidden_states = encoder_hidden_states + context_attn_output

    norm_encoder_hidden_states = self.norm2_context(encoder_hidden_states)
    norm_encoder_hidden_states = (
        norm_encoder_hidden_states * (1 + c_scale_mlp[:, None]) + c_shift_mlp[:, None]
    )

    context_ff_output = self.ff_context(norm_encoder_hidden_states)
    encoder_hidden_states = (
        encoder_hidden_states + c_gate_mlp.unsqueeze(1) * context_ff_output
    )
    if encoder_hidden_states.dtype == torch.float16:
        encoder_hidden_states = encoder_hidden_states.clip(-65504, 65504)

    return encoder_hidden_states, hidden_states


def cache_flux_single_transformer_block_forward(
    self,
    hidden_states: torch.Tensor,
    encoder_hidden_states: torch.Tensor,
    temb: torch.Tensor,
    image_rotary_emb: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    joint_attention_kwargs: Optional[Dict[str, Any]] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Single-stream transformer block forward with caching support.

    Optimizations:
    - Buffer reuse via ``self.workspace`` to avoid repeated allocation for the
      concat/fused tensors.
    - MLP partial-compute: in reuse steps only computes MLP for the first
      ``text_seq_len + key_token_num`` positions instead of the full sequence.
    """
    text_seq_len = encoder_hidden_states.shape[1]

    if not hasattr(self, "workspace"):
        self.workspace = {}
    concat_key = "single_concat_in"
    B, Lt, C = encoder_hidden_states.shape
    Li = hidden_states.shape[1]
    buf = self.workspace.get(concat_key)
    if (
        buf is None
        or buf.shape != (B, Lt + Li, C)
        or buf.dtype != encoder_hidden_states.dtype
        or buf.device != encoder_hidden_states.device
    ):
        buf = torch.empty(
            (B, Lt + Li, C),
            device=encoder_hidden_states.device,
            dtype=encoder_hidden_states.dtype,
        )
        self.workspace[concat_key] = buf
    buf[:, :Lt, :].copy_(encoder_hidden_states)
    buf[:, Lt:, :].copy_(hidden_states)
    hidden_states = buf

    residual = hidden_states
    norm_hidden_states, gate = self.norm(hidden_states, emb=temb)

    ctx = _get_ctx(self)
    do_partial, kti = _should_reuse_with_indices(ctx)

    if do_partial and kti is not None:
        key_token_num = kti.shape[0]
        calc_len = text_seq_len + key_token_num
        input_slice = norm_hidden_states[:, :calc_len]
        mlp_hidden_states = self.act_mlp(self.proj_mlp(input_slice))
    else:
        calc_len = None
        mlp_hidden_states = self.act_mlp(self.proj_mlp(norm_hidden_states))

    joint_attention_kwargs = joint_attention_kwargs or {}
    attn_output = self.attn(
        hidden_states=norm_hidden_states,
        image_rotary_emb=image_rotary_emb,
        **joint_attention_kwargs,
    )

    if do_partial and kti is not None:
        residual = residual[:, :calc_len]

    fused_key = "single_fused"
    B, L, D_attn = attn_output.shape
    D_mlp = mlp_hidden_states.shape[2]
    D_fused = D_attn + D_mlp
    buf = self.workspace.get(fused_key)
    if (
        buf is None
        or buf.shape != (B, L, D_fused)
        or buf.dtype != attn_output.dtype
        or buf.device != attn_output.device
    ):
        buf = torch.empty(
            (B, L, D_fused),
            device=attn_output.device,
            dtype=attn_output.dtype,
        )
        self.workspace[fused_key] = buf
    buf[:, :, :D_attn].copy_(attn_output)
    buf[:, :, D_attn:].copy_(mlp_hidden_states)
    hidden_states = buf

    gate = gate.unsqueeze(1)
    hidden_states = gate * self.proj_out(hidden_states)
    hidden_states = residual + hidden_states
    if hidden_states.dtype == torch.float16:
        hidden_states = hidden_states.clip(-65504, 65504)

    encoder_hidden_states, hidden_states = (
        hidden_states[:, :text_seq_len],
        hidden_states[:, text_seq_len:],
    )

    return encoder_hidden_states, hidden_states
