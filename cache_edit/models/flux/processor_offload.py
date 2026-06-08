"""Memory-optimized FluxAttnCacheProcessor with dynamic device offloading.

This version offloads large intermediate tensors to secondary GPUs during computation
to avoid OOM on the primary GPU.
"""

import torch
from typing import Optional, Tuple
from cache_edit.models.flux.processor import FluxAttnCacheProcessor


class FluxAttnCacheProcessorWithOffload(FluxAttnCacheProcessor):
    """
    Extended processor that offloads large computations to secondary GPUs.

    Key optimization: When computing rotary embeddings and attention for large
    activations, temporarily move tensors to a secondary GPU with more free memory.
    """

    def __init__(self, offload_device: Optional[torch.device] = None):
        super().__init__()
        self.offload_device = offload_device or torch.device("cuda:1")

    def _find_best_offload_device(self, tensor_size_bytes: int, num_gpus: int = 4):
        """Find GPU with most free memory for offloading computation."""
        max_free = 0
        best_device = self.offload_device

        for i in range(num_gpus):
            try:
                torch.cuda.synchronize(i)
                props = torch.cuda.get_device_properties(i)
                allocated = torch.cuda.memory_allocated(i)
                free = props.total_memory - allocated

                if free > max_free and free > tensor_size_bytes * 1.5:  # 1.5x safety margin
                    max_free = free
                    best_device = torch.device(f"cuda:{i}")
            except:
                continue

        return best_device

    def _offload_computation(self, func, *tensors, target_device=None):
        """
        Execute a computation on an offload device to save memory on primary GPU.

        Args:
            func: Function to execute
            *tensors: Input tensors
            target_device: Device to offload to (None = auto-select)

        Returns:
            Result moved back to original device
        """
        if target_device is None:
            # Estimate tensor size
            total_size = sum(t.numel() * t.element_size() for t in tensors if torch.is_tensor(t))
            target_device = self._find_best_offload_device(total_size)

        original_device = tensors[0].device if tensors and torch.is_tensor(tensors[0]) else None

        # Move to offload device
        offloaded_tensors = []
        for t in tensors:
            if torch.is_tensor(t):
                offloaded_tensors.append(t.to(target_device))
            else:
                offloaded_tensors.append(t)

        # Compute on offload device
        result = func(*offloaded_tensors)

        # Move back to original device
        if original_device is not None and torch.is_tensor(result):
            result = result.to(original_device)
        elif isinstance(result, (tuple, list)):
            result = type(result)(
                r.to(original_device) if torch.is_tensor(r) and original_device else r
                for r in result
            )

        return result

    def _double_stream(
        self,
        attn,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor],
        image_rotary_emb: Optional[Tuple[torch.Tensor, torch.Tensor]],
        do_partial_q: bool,
        key_token_indices: Optional[torch.Tensor],
    ):
        """Double stream with offloaded rotary embedding computation."""

        # Original device
        orig_device = hidden_states.device

        # QKV projections on original device (small operations)
        if do_partial_q and key_token_indices is not None:
            key_token_num = key_token_indices.shape[0]
            img_calc_len = key_token_num
            txt_len = encoder_hidden_states.shape[1]
            calc_len = txt_len + img_calc_len

            input_q = torch.cat(
                [encoder_hidden_states, hidden_states[:, :img_calc_len, :]], dim=1
            )
            input_q = input_q[:, :calc_len, :]

            encoder_query = attn.to_q(encoder_hidden_states)
            query = attn.to_q(hidden_states[:, :img_calc_len, :])
            encoder_key = attn.to_k(encoder_hidden_states)
            key = attn.to_k(hidden_states)
            encoder_value = attn.to_v(encoder_hidden_states)
            value = attn.to_v(hidden_states)
        else:
            encoder_query = attn.to_q(encoder_hidden_states)
            query = attn.to_q(hidden_states)
            encoder_key = attn.to_k(encoder_hidden_states)
            key = attn.to_k(hidden_states)
            encoder_value = attn.to_v(encoder_hidden_states)
            value = attn.to_v(hidden_states)

        # Unflatten and normalize (small operations, keep on original device)
        encoder_query = encoder_query.unflatten(-1, (attn.heads, -1))
        query = query.unflatten(-1, (attn.heads, -1))
        encoder_key = encoder_key.unflatten(-1, (attn.heads, -1))
        key = key.unflatten(-1, (attn.heads, -1))
        encoder_value = encoder_value.unflatten(-1, (attn.heads, -1))
        value = value.unflatten(-1, (attn.heads, -1))

        encoder_query = attn.norm_q(encoder_query)
        query = attn.norm_q(query)
        encoder_key = attn.norm_k(encoder_key)
        key = attn.norm_k(key)

        # Concatenate
        full_query = torch.cat([encoder_query, query], dim=1)
        full_key = torch.cat([encoder_key, key], dim=1)
        full_value = torch.cat([encoder_value, value], dim=1)

        # Offload rotary embedding computation to secondary GPU
        if image_rotary_emb is not None:
            from diffusers.models.embeddings import apply_rotary_emb

            cos, sin = image_rotary_emb
            cos_q = cos[: full_query.shape[1], :]
            sin_q = sin[: full_query.shape[1], :]

            # Offload query rotary embedding
            def apply_query_rope(q, c, s):
                return apply_rotary_emb(q, (c, s), sequence_dim=1)

            full_query = self._offload_computation(
                apply_query_rope, full_query, cos_q, sin_q
            )

            # Offload key rotary embedding
            def apply_key_rope(k, cos_sin):
                return apply_rotary_emb(k, cos_sin, sequence_dim=1)

            full_key = self._offload_computation(
                apply_key_rope, full_key, image_rotary_emb
            )

        # Attention computation (large, keep on original device but could offload if needed)
        from diffusers.models.attention_dispatch import dispatch_attention_fn

        attn_output = dispatch_attention_fn(
            full_query,
            full_key,
            full_value,
            attn_mask=attention_mask,
            backend=self._attention_backend,
            parallel_config=self._parallel_config,
        )
        attn_output = attn_output.flatten(2, 3).to(query.dtype)

        # Split and project output
        txt_len = encoder_hidden_states.shape[1]
        txt_out = attn_output[:, :txt_len, :]
        img_out = attn_output[:, txt_len:, :]

        img_out = attn.to_out[0](img_out)
        img_out = attn.to_out[1](img_out)
        txt_out = attn.to_add_out(txt_out)

        return img_out, txt_out

    def _single_stream(
        self,
        attn,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor],
        image_rotary_emb: Optional[Tuple[torch.Tensor, torch.Tensor]],
        do_partial_q: bool,
        key_token_indices: Optional[torch.Tensor],
    ):
        """Single stream with offloaded rotary embedding computation."""

        if do_partial_q and key_token_indices is not None:
            key_token_num = key_token_indices.shape[0]
            txt_len = self.single_stream_txt_len
            calc_len = txt_len + key_token_num
            input_q = hidden_states[:, :calc_len, :]
            query = attn.to_q(input_q)
            key = attn.to_k(hidden_states)
            value = attn.to_v(hidden_states)
        else:
            query = attn.to_q(hidden_states)
            key = attn.to_k(hidden_states)
            value = attn.to_v(hidden_states)

        query = query.unflatten(-1, (attn.heads, -1))
        key = key.unflatten(-1, (attn.heads, -1))
        value = value.unflatten(-1, (attn.heads, -1))
        query = attn.norm_q(query)
        key = attn.norm_k(key)

        # Offload rotary embedding computation
        if image_rotary_emb is not None:
            from diffusers.models.embeddings import apply_rotary_emb

            cos, sin = image_rotary_emb
            q_len = query.shape[1]
            cos_q = cos[:q_len, :]
            sin_q = sin[:q_len, :]

            def apply_query_rope(q, c, s):
                return apply_rotary_emb(q, (c, s), sequence_dim=1)

            query = self._offload_computation(
                apply_query_rope, query, cos_q, sin_q
            )

            def apply_key_rope(k, cos_sin):
                return apply_rotary_emb(k, cos_sin, sequence_dim=1)

            key = self._offload_computation(
                apply_key_rope, key, image_rotary_emb
            )

        from diffusers.models.attention_dispatch import dispatch_attention_fn

        attn_output = dispatch_attention_fn(
            query,
            key,
            value,
            attn_mask=attention_mask,
            backend=self._attention_backend,
            parallel_config=self._parallel_config,
        )
        attn_output = attn_output.flatten(2, 3).to(query.dtype)
        return attn_output
