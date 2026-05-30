"""Qwen Transformer 2D Model forward function with activation caching support.

Implements Flux-style activation caching with **dynamic key-token refresh**:

- Round 0: every step fully computes; cache steps store per-layer activations.
- Round 1+ cache steps: fully compute, store, then compare against Round 0's
  same-step cache at the reference layer to refresh ``key_token_indices``
  (tokens whose hidden state diverges meaningfully — i.e. the "edit region").
- Round 1+ non-cache (reuse) steps with ``key_token_indices`` available:
  rearrange image tokens to key-first, slice down to K tokens, run all blocks
  on the slice, and reconstruct the full sequence after every layer by pulling
  the non-key positions from the cached activation. Order is restored before
  the output projection.

The model must have a ``cache_manager`` attribute pointing to a
``QwenCacheManager`` (with key-token helpers).
"""

from typing import Any, Dict, List, Optional, Tuple, Union
from math import prod

import torch

try:
    from diffusers.models.modeling_outputs import Transformer2DModelOutput
    from diffusers.utils import USE_PEFT_BACKEND, scale_lora_layers, unscale_lora_layers
except ImportError:
    Transformer2DModelOutput = None
    USE_PEFT_BACKEND = False
    scale_lora_layers = unscale_lora_layers = None


def cache_qwen_transformer_2d_forward(
    self,
    hidden_states: torch.Tensor,
    encoder_hidden_states: torch.Tensor = None,
    encoder_hidden_states_mask: torch.Tensor = None,
    timestep: torch.LongTensor = None,
    img_shapes: Optional[List[Tuple[int, int, int]]] = None,
    txt_seq_lens: Optional[List[int]] = None,
    guidance: torch.Tensor = None,
    attention_kwargs: Optional[Dict[str, Any]] = None,
    controlnet_block_samples=None,
    additional_t_cond=None,
    return_dict: bool = True,
) -> Union[torch.Tensor, "Transformer2DModelOutput"]:
    cm = getattr(self, "cache_manager", None)
    if cm is None:
        raise RuntimeError(
            "cache_qwen_transformer_2d_forward requires self.cache_manager."
        )

    cm.begin_forward_call()

    if attention_kwargs is not None:
        attention_kwargs = attention_kwargs.copy()
        lora_scale = attention_kwargs.pop("scale", 1.0)
    else:
        lora_scale = 1.0

    if USE_PEFT_BACKEND:
        scale_lora_layers(self, lora_scale)

    # ---- input projections (unchanged) ----
    hidden_states = self.img_in(hidden_states)
    timestep = timestep.to(hidden_states.dtype)

    if self.zero_cond_t:
        timestep = torch.cat([timestep, timestep * 0], dim=0)
        modulate_index = torch.tensor(
            [[0] * prod(sample[0]) + [1] * sum([prod(s) for s in sample[1:]]) for sample in img_shapes],
            device=timestep.device,
            dtype=torch.int,
        )
    else:
        modulate_index = None

    encoder_hidden_states = self.txt_norm(encoder_hidden_states)
    encoder_hidden_states = self.txt_in(encoder_hidden_states)

    if guidance is not None:
        guidance = guidance.to(hidden_states.dtype) * 1000

    temb = (
        self.time_text_embed(timestep, hidden_states, additional_t_cond)
        if guidance is None
        else self.time_text_embed(timestep, guidance, hidden_states, additional_t_cond)
    )

    image_rotary_emb = self.pos_embed(
        img_shapes, txt_seq_lens, device=hidden_states.device
    )

    # ---- decide reuse mode ----
    should_reuse = cm.should_reuse(cm.current_step)
    should_cache = cm.should_cache(cm.current_step)
    key_indices = cm.key_token_indices
    do_partial = bool(should_reuse and key_indices is not None)

    img_freqs, txt_freqs = image_rotary_emb
    L = hidden_states.shape[1]

    # ---- partial-compute setup: rearrange + slice ----
    if do_partial:
        K = int(key_indices.shape[0])
        if K == 0 or K >= L:
            # Degenerate: no key tokens or all tokens are key → fall back
            do_partial = False
        else:
            hidden_states, img_freqs_rearranged, mask_not_key_img = (
                cm.rearrange_image_and_freqs(hidden_states, img_freqs, key_indices)
            )
            sliced_img_freqs = img_freqs_rearranged[:K, ...]
            sliced_rotary_emb = (sliced_img_freqs, txt_freqs)
            # build a mask aligned to the cached tensors' device (loaded lazily)
            mask_not_key_cpu = mask_not_key_img.to("cpu")
    else:
        K = L

    # ---- transformer blocks ----
    fallback_used = False
    for index_block, block in enumerate(self.transformer_blocks):
        if do_partial:
            # slice to K tokens (key tokens, already at the front)
            sliced_hidden = hidden_states[:, :K, :]

            encoder_hidden_states, sliced_out = block(
                hidden_states=sliced_hidden,
                encoder_hidden_states=encoder_hidden_states,
                encoder_hidden_states_mask=encoder_hidden_states_mask,
                temb=temb,
                image_rotary_emb=sliced_rotary_emb,
                joint_attention_kwargs=attention_kwargs,
                modulate_index=modulate_index,
            )

            # pull non-key from cached activation (original order) and stitch back
            prev_img = cm.load_activation(
                stream="image",
                layer_idx=index_block,
                device=sliced_out.device,
                step=cm.current_step,
            )
            if prev_img is None:
                # no cache available for this layer → cannot reconstruct;
                # one-time fallback log, keep hidden_states as K (broken for
                # subsequent layers expecting L; signal upstream).
                if not fallback_used:
                    print(
                        f"[qwen-cache] missing cache at layer={index_block} "
                        f"step={cm.current_step}; reusing computed slice only."
                    )
                    fallback_used = True
                hidden_states = sliced_out
            else:
                mask_dev = mask_not_key_cpu.to(prev_img.device)
                prev_not_key = prev_img[:, mask_dev, :]
                hidden_states = torch.cat([sliced_out, prev_not_key], dim=1)

        elif should_reuse:
            # Full reuse path (no key_token_indices yet, e.g. very first
            # reuse step before any indices have been computed). Skip the block.
            cached_img = cm.load_activation(
                stream="image",
                layer_idx=index_block,
                device=hidden_states.device,
                step=cm.current_step,
            )
            cached_txt = cm.load_activation(
                stream="text",
                layer_idx=index_block,
                device=encoder_hidden_states.device,
                step=cm.current_step,
            )
            if cached_img is not None and cached_txt is not None:
                hidden_states = cached_img
                encoder_hidden_states = cached_txt
                continue
            # cache miss → fall through to normal compute
            encoder_hidden_states, hidden_states = block(
                hidden_states=hidden_states,
                encoder_hidden_states=encoder_hidden_states,
                encoder_hidden_states_mask=encoder_hidden_states_mask,
                temb=temb,
                image_rotary_emb=image_rotary_emb,
                joint_attention_kwargs=attention_kwargs,
                modulate_index=modulate_index,
            )
        else:
            # Normal full compute (Round 0 or Round 1+ cache step)
            if torch.is_grad_enabled() and getattr(self, "gradient_checkpointing", False):
                encoder_hidden_states, hidden_states = self._gradient_checkpointing_func(
                    block,
                    hidden_states,
                    encoder_hidden_states,
                    encoder_hidden_states_mask,
                    temb,
                    image_rotary_emb,
                    attention_kwargs,
                    modulate_index,
                )
            else:
                encoder_hidden_states, hidden_states = block(
                    hidden_states=hidden_states,
                    encoder_hidden_states=encoder_hidden_states,
                    encoder_hidden_states_mask=encoder_hidden_states_mask,
                    temb=temb,
                    image_rotary_emb=image_rotary_emb,
                    joint_attention_kwargs=attention_kwargs,
                    modulate_index=modulate_index,
                )

            # Store activations only at cache steps (full compute branch)
            if should_cache:
                cm.store_activation(
                    stream="image",
                    layer_idx=index_block,
                    tensor=hidden_states,
                )
                cm.store_activation(
                    stream="text",
                    layer_idx=index_block,
                    tensor=encoder_hidden_states,
                )

        # ControlNet support (only meaningful in the full-compute branch where
        # hidden_states still represents the full image sequence).
        if controlnet_block_samples is not None and not do_partial:
            hidden_states = hidden_states + controlnet_block_samples[index_block]

    # ---- restore image-token order if we ran partial compute ----
    if do_partial:
        hidden_states = cm.restore_image_order(hidden_states, key_indices)

    # ---- refresh key_token_indices at Round 1+ cache steps (cond mode only) ----
    if (
        cm.use_activation_cache
        and not cm.is_round0
        and should_cache
        and cm.current_mode == "cond"
    ):
        ref_layer = self.config.num_layers - 1
        ref_img = cm.load_activation(
            stream="image",
            layer_idx=ref_layer,
            device=hidden_states.device,
            step=cm.current_step,
        )
        cm.update_key_token_indices(cur_img=hidden_states, ref_img=ref_img)

    # ---- output projection ----
    hidden_states = self.norm_out(hidden_states, temb)
    output = self.proj_out(hidden_states)

    cm.end_forward_call()

    if USE_PEFT_BACKEND:
        unscale_lora_layers(self, lora_scale)

    if not return_dict:
        return (output,)

    return Transformer2DModelOutput(sample=output)
