"""Qwen attention processor implementation with caching support."""

from typing import Optional, Tuple

import torch
import torch.nn.functional as F
from diffusers.models.attention_processor import Attention
from diffusers.models.transformers.transformer_qwenimage import apply_rotary_emb_qwen

try:
    import flash_attn
    from flash_attn import flash_attn_func
except ImportError:
    flash_attn = None
    flash_attn_func = None


class QwenDoubleStreamCacheAttnProcessor:
    """
    Qwen 双流注意力处理器（带缓存支持）。

    支持双流（图像 + 文本）联合注意力计算，并通过外部 cache_context
    提供激活缓存功能。

    与原始 `RegionEQwenDoubleStreamAttnProcessor2_0` 的区别：
    - 解耦全局 MANAGER，改为通过 cache_context 注入
    - 不再硬编码 warmup/post step 逻辑
    - 接口清晰，易于测试和扩展
    - 缓存策略由 cache_context 决定

    Attributes:
        single: 是否为单流模式
        cache_context: 缓存上下文（可选），包含缓存策略和数据
    """

    _attention_backend = None

    def __init__(self, single: bool = False, cache_context=None):
        """
        初始化注意力处理器。

        Args:
            single: 是否为单流模式
            cache_context: 缓存上下文对象，提供缓存策略
        """
        if not hasattr(F, "scaled_dot_product_attention"):
            raise ImportError(
                "QwenDoubleStreamCacheAttnProcessor requires PyTorch 2.0+. "
                "Please upgrade PyTorch."
            )

        self.single = single
        self.cache_context = cache_context

        # 本地缓存（按 cond/uncond 区分）
        self.k_cache: dict = {"cond": None, "uncond": None}
        self.v_cache: dict = {"cond": None, "uncond": None}

        # Debug counters
        self.compute_count = 0
        self.reuse_count = 0

    def attach_cache_context(self, cache_context) -> None:
        """
        附加缓存上下文。

        Args:
            cache_context: 缓存上下文对象，需要提供：
                - should_compute_kv() -> bool
                - should_store_kv() -> bool
                - should_reuse_kv() -> bool
                - get_selection_ids() -> Tensor
        """
        self.cache_context = cache_context

    def clear_local_cache(self) -> None:
        """清空本地 KV 缓存。"""
        self.k_cache = {"cond": None, "uncond": None}
        self.v_cache = {"cond": None, "uncond": None}

    def _compute_image_kv(
        self,
        attn: Attention,
        hidden_states: torch.FloatTensor,
        tag: str,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        计算或复用图像流的 K/V。

        Args:
            attn: 注意力模块
            hidden_states: 图像隐藏状态
            tag: "cond" 或 "uncond"

        Returns:
            tuple: (img_key, img_value)
        """
        ctx = self.cache_context

        # 无缓存上下文：直接计算
        if ctx is None:
            img_key = attn.to_k(hidden_states)
            img_value = attn.to_v(hidden_states)
            return img_key, img_value

        # 通过缓存上下文决定行为
        if ctx.should_compute_kv():
            self.compute_count += 1
            img_key = attn.to_k(hidden_states)
            img_value = attn.to_v(hidden_states)

            # 是否需要存储到缓存
            if ctx.should_store_kv():
                self.k_cache[tag] = img_key
                self.v_cache[tag] = img_value

            return img_key, img_value

        elif ctx.should_reuse_kv() and self.k_cache[tag] is not None:
            self.reuse_count += 1
            # 复用缓存（可能需要部分更新）
            selection_ids = ctx.get_selection_ids()
            if selection_ids is not None:
                # 部分更新：只更新被选中的 token
                img_key, img_value = self._partial_update_kv(
                    attn, hidden_states, tag, selection_ids
                )
            else:
                # 完全复用缓存（跳过 to_k/to_v 计算）
                img_key = self.k_cache[tag]
                img_value = self.v_cache[tag]
            return img_key, img_value

        # 默认：重新计算
        img_key = attn.to_k(hidden_states)
        img_value = attn.to_v(hidden_states)
        return img_key, img_value

    def _partial_update_kv(
        self,
        attn: Attention,
        hidden_states: torch.Tensor,
        tag: str,
        selection_ids: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        部分更新 K/V 缓存（仅更新被选中的 token）。

        Args:
            attn: 注意力模块
            hidden_states: 图像隐藏状态
            tag: 模式标签
            selection_ids: 需要更新的 token 索引

        Returns:
            tuple: (img_key, img_value)
        """
        # 尝试使用 fused_kernels._partially_linear（如果可用）
        try:
            from fused_kernels import _partially_linear

            batch_size = hidden_states.shape[0]
            selection = selection_ids.squeeze(0)

            k_cache_view = self.k_cache[tag].view(
                batch_size, self.k_cache[tag].shape[1], -1
            )
            v_cache_view = self.v_cache[tag].view(
                batch_size, self.v_cache[tag].shape[1], -1
            )

            _partially_linear(
                hidden_states,
                attn.to_k.weight,
                attn.to_k.bias,
                selection,
                k_cache_view,
            )
            _partially_linear(
                hidden_states,
                attn.to_v.weight,
                attn.to_v.bias,
                selection,
                v_cache_view,
            )

            return self.k_cache[tag], self.v_cache[tag]
        except ImportError:
            # Fallback：完整重新计算（性能下降但功能正确）
            img_key = attn.to_k(hidden_states)
            img_value = attn.to_v(hidden_states)
            return img_key, img_value

    def __call__(
        self,
        attn: Attention,
        hidden_states: torch.FloatTensor,
        encoder_hidden_states: Optional[torch.FloatTensor] = None,
        encoder_hidden_states_mask: Optional[torch.FloatTensor] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        image_rotary_emb: Optional[torch.Tensor] = None,
        tag: str = "cond",
    ) -> Tuple[torch.FloatTensor, torch.FloatTensor]:
        """
        执行双流联合注意力。

        Args:
            attn: 注意力模块
            hidden_states: 图像流隐藏状态
            encoder_hidden_states: 文本流隐藏状态
            encoder_hidden_states_mask: 文本流掩码
            attention_mask: 注意力掩码
            image_rotary_emb: 图像旋转位置编码
            tag: 模式标签（"cond" 或 "uncond"）

        Returns:
            tuple: (img_attn_output, txt_attn_output)
        """
        if encoder_hidden_states is None:
            raise ValueError(
                "QwenDoubleStreamCacheAttnProcessor requires encoder_hidden_states"
            )

        batch_size = encoder_hidden_states.shape[0]
        seq_txt = encoder_hidden_states.shape[1]

        # 图像流：Q 始终计算，K/V 通过缓存上下文决定
        img_query = attn.to_q(hidden_states)
        img_key, img_value = self._compute_image_kv(attn, hidden_states, tag)

        # 文本流：QKV
        txt_query = attn.add_q_proj(encoder_hidden_states)
        txt_key = attn.add_k_proj(encoder_hidden_states)
        txt_value = attn.add_v_proj(encoder_hidden_states)

        # 重塑为多头格式
        img_query = img_query.unflatten(-1, (attn.heads, -1))
        img_key = img_key.unflatten(-1, (attn.heads, -1))
        img_value = img_value.unflatten(-1, (attn.heads, -1))
        head_dim = img_query.shape[-1]

        txt_query = txt_query.unflatten(-1, (attn.heads, -1))
        txt_key = txt_key.unflatten(-1, (attn.heads, -1))
        txt_value = txt_value.unflatten(-1, (attn.heads, -1))

        # QK 归一化
        if attn.norm_q is not None:
            img_query = attn.norm_q(img_query)
        if attn.norm_k is not None:
            img_key = attn.norm_k(img_key)
        if attn.norm_added_q is not None:
            txt_query = attn.norm_added_q(txt_query)
        if attn.norm_added_k is not None:
            txt_key = attn.norm_added_k(txt_key)

        # 应用 RoPE
        if image_rotary_emb is not None:
            img_freqs, txt_freqs = image_rotary_emb

            # 从 cache_context 获取自定义的 img_freqs（如果可用）
            ctx_img_freqs = (
                self.cache_context.get_image_rotary_emb()
                if self.cache_context is not None
                and hasattr(self.cache_context, "get_image_rotary_emb")
                else img_freqs
            )

            if img_query.shape[1] != 0:
                img_query = apply_rotary_emb_qwen(img_query, ctx_img_freqs, use_real=False)
            img_key = apply_rotary_emb_qwen(img_key, img_freqs, use_real=False)
            txt_query = apply_rotary_emb_qwen(txt_query, txt_freqs, use_real=False)
            txt_key = apply_rotary_emb_qwen(txt_key, txt_freqs, use_real=False)

        # 拼接做联合注意力 [text, image]
        joint_query = torch.cat([txt_query, img_query], dim=1)
        joint_key = torch.cat([txt_key, img_key], dim=1)
        joint_value = torch.cat([txt_value, img_value], dim=1)

        # 计算注意力
        joint_hidden_states = self._compute_attention(
            joint_query, joint_key, joint_value, attention_mask, batch_size, head_dim, attn
        )

        # 拆分输出
        txt_attn_output = joint_hidden_states[:, :seq_txt, :]
        img_attn_output = joint_hidden_states[:, seq_txt:, :]

        # 输出投影
        img_attn_output = attn.to_out[0](img_attn_output)
        if len(attn.to_out) > 1:
            img_attn_output = attn.to_out[1](img_attn_output)  # dropout
        txt_attn_output = attn.to_add_out(txt_attn_output)

        return img_attn_output, txt_attn_output

    def _compute_attention(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attention_mask: Optional[torch.Tensor],
        batch_size: int,
        head_dim: int,
        attn: Attention,
    ) -> torch.Tensor:
        """
        计算注意力（支持 flash_attn 或 PyTorch SDPA）。

        Args:
            query, key, value: QKV 张量
            attention_mask: 注意力掩码
            batch_size: 批次大小
            head_dim: 头维度
            attn: 注意力模块（用于获取 heads 数）

        Returns:
            torch.Tensor: 注意力输出
        """
        if flash_attn is not None:
            out = flash_attn_func(query, key, value, dropout_p=0.0, causal=False)
            out = out.reshape(batch_size, -1, attn.heads * head_dim)
        else:
            q = query.transpose(1, 2)
            k = key.transpose(1, 2)
            v = value.transpose(1, 2)
            out = F.scaled_dot_product_attention(
                q, k, v, attn_mask=attention_mask, dropout_p=0.0, is_causal=False
            )
            out = out.transpose(1, 2).reshape(batch_size, -1, attn.heads * head_dim)

        return out.to(query.dtype)
