"""Flux attention processor implementation with caching support."""

from typing import Optional, Tuple

import torch
import torch.nn.functional as F

try:
    from diffusers.models.embeddings import apply_rotary_emb
except ImportError:  # pragma: no cover - diffusers must be present in env
    apply_rotary_emb = None  # type: ignore

try:
    from diffusers.models.attention_dispatch import dispatch_attention_fn
except ImportError:  # pragma: no cover
    dispatch_attention_fn = None  # type: ignore

try:
    import flash_attn  # noqa: F401
    from flash_attn import flash_attn_func  # noqa: F401
except ImportError:
    flash_attn = None  # type: ignore
    flash_attn_func = None  # type: ignore


class FluxAttnCacheProcessor:
    """
    Flux 缓存感知注意力处理器（支持 double / single 两种模式）。

    与原始 `CacheFluxAttnProcessor2_0` 的关键区别：
    - 不依赖全局 MANAGER，改为通过 cache_context 注入
    - 由 cache_context.stream_type 决定走 double / single 分支
    - 由 cache_context.should_reuse / key_token_indices 决定是否启用 key-token-only Q 优化

    优化:
    - **Q 部分计算**：复用轮次只对 key tokens 计算 Q，K/V 仍全量计算以保证注意力完整性
    - **RoPE 截断**：Q 用前 q_len 个位置编码，K 用全长 RoPE
    """

    _attention_backend = None
    _parallel_config = None

    def __init__(self, cache_context=None, single_stream_txt_len: int = 512):
        """
        Args:
            cache_context: 缓存上下文（FluxCacheManager 实例或兼容对象）
            single_stream_txt_len: single 模式下文本 token 数（Flux 默认 512）
        """
        if not hasattr(F, "scaled_dot_product_attention"):
            raise ImportError(
                f"{self.__class__.__name__} requires PyTorch 2.0+. "
                "Please upgrade your PyTorch version."
            )
        if apply_rotary_emb is None or dispatch_attention_fn is None:
            raise ImportError(
                f"{self.__class__.__name__} requires diffusers with FluxAttention. "
                "Please install diffusers."
            )
        self.cache_context = cache_context
        self.single_stream_txt_len = single_stream_txt_len

    def attach_cache_context(self, cache_context) -> None:
        """附加缓存上下文。"""
        self.cache_context = cache_context

    # ---------- Helpers ----------

    def _resolve_reuse(self) -> Tuple[bool, bool, Optional[torch.Tensor]]:
        ctx = self.cache_context
        if ctx is None:
            return False, False, None
        should_reuse = bool(ctx.should_reuse(ctx.current_step))
        kti = getattr(ctx, "key_token_indices", None)
        has_indices = kti is not None
        return should_reuse, has_indices, kti

    def _stream_type(self) -> str:
        ctx = self.cache_context
        if ctx is None:
            return "double"
        return getattr(ctx, "stream_type", "double")

    # ---------- Main entry ----------

    def __call__(
        self,
        attn,
        hidden_states: torch.Tensor,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        image_rotary_emb: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ):
        should_reuse, has_indices, key_token_indices = self._resolve_reuse()
        stream = self._stream_type()

        if stream == "double":
            return self._double_stream(
                attn,
                hidden_states,
                encoder_hidden_states,
                attention_mask,
                image_rotary_emb,
                should_reuse and has_indices,
                key_token_indices,
            )
        return self._single_stream(
            attn,
            hidden_states,
            attention_mask,
            image_rotary_emb,
            should_reuse and has_indices,
            key_token_indices,
        )

    # ---------- Double stream ----------

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
        if encoder_hidden_states is None:
            raise ValueError(
                "FluxAttnCacheProcessor double-stream requires encoder_hidden_states"
            )

        encoder_query = attn.add_q_proj(encoder_hidden_states)
        encoder_key = attn.add_k_proj(encoder_hidden_states)
        encoder_value = attn.add_v_proj(encoder_hidden_states)

        encoder_query = encoder_query.unflatten(-1, (attn.heads, -1))
        encoder_key = encoder_key.unflatten(-1, (attn.heads, -1))
        encoder_value = encoder_value.unflatten(-1, (attn.heads, -1))

        encoder_query = attn.norm_added_q(encoder_query)
        encoder_key = attn.norm_added_k(encoder_key)

        img_len = hidden_states.shape[1]
        if do_partial_q and key_token_indices is not None:
            key_token_num = min(key_token_indices.shape[0], img_len)
            hidden_states_q = hidden_states[:, :key_token_num, :]
            query = attn.to_q(hidden_states_q)
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

        full_query = torch.cat([encoder_query, query], dim=1)
        full_key = torch.cat([encoder_key, key], dim=1)
        full_value = torch.cat([encoder_value, value], dim=1)

        if image_rotary_emb is not None:
            cos, sin = image_rotary_emb
            cos_q = cos[: full_query.shape[1], :]
            sin_q = sin[: full_query.shape[1], :]
            full_query = apply_rotary_emb(
                full_query, (cos_q, sin_q), sequence_dim=1
            )
            full_key = apply_rotary_emb(full_key, image_rotary_emb, sequence_dim=1)

        attn_output = dispatch_attention_fn(
            full_query,
            full_key,
            full_value,
            attn_mask=attention_mask,
            backend=self._attention_backend,
            parallel_config=self._parallel_config,
        )
        attn_output = attn_output.flatten(2, 3).to(query.dtype)

        txt_len = encoder_hidden_states.shape[1]
        txt_out = attn_output[:, :txt_len, :]
        img_out = attn_output[:, txt_len:, :]

        img_out = attn.to_out[0](img_out)
        img_out = attn.to_out[1](img_out)
        txt_out = attn.to_add_out(txt_out)

        return img_out, txt_out

    # ---------- Single stream ----------

    def _single_stream(
        self,
        attn,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor],
        image_rotary_emb: Optional[Tuple[torch.Tensor, torch.Tensor]],
        do_partial_q: bool,
        key_token_indices: Optional[torch.Tensor],
    ):
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

        if image_rotary_emb is not None:
            cos, sin = image_rotary_emb
            q_len = query.shape[1]
            cos_q = cos[:q_len, :]
            sin_q = sin[:q_len, :]
            query = apply_rotary_emb(query, (cos_q, sin_q), sequence_dim=1)
            key = apply_rotary_emb(key, image_rotary_emb, sequence_dim=1)

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
