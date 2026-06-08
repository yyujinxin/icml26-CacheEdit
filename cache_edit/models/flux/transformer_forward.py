"""Flux Transformer 2D Model forward function with caching support.

Designed to be bound to an existing `FluxTransformer2DModel` instance from
diffusers. The model must have a ``cache_context`` attribute pointing to a
``FluxCacheManager`` (or compatible) instance. Optionally:

- ``cache_viz_config``: a ``FluxCacheVizConfig`` controlling key-token
  visualization and CSV logging at step 0 of reuse rounds.

Usage::

    transformer.cache_context = cache_manager
    transformer.forward = cache_flux_transformer_2d_forward.__get__(
        transformer, transformer.__class__
    )
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Union

import os
import re

import numpy as np
import torch

try:
    from diffusers.models.modeling_outputs import Transformer2DModelOutput
    from diffusers.utils import (
        USE_PEFT_BACKEND,
        logging,
        scale_lora_layers,
        unscale_lora_layers,
    )
except ImportError:  # pragma: no cover
    Transformer2DModelOutput = None  # type: ignore
    USE_PEFT_BACKEND = False  # type: ignore
    scale_lora_layers = unscale_lora_layers = None  # type: ignore
    logging = None  # type: ignore

try:
    from PIL import Image
    _HAS_PIL = True
except ImportError:
    _HAS_PIL = False

from cache_edit.models.flux.stats import (
    append_key_token_ratio_with_edit_ratio,
    infer_image_id_from_csv_by_round,
    visualize_key_tokens_on_image,
)


logger = (
    logging.get_logger(__name__) if logging is not None else None  # type: ignore
)


def _module_device(module) -> Optional[torch.device]:
    """Return the device of the first tensor owned by a module."""
    for param in module.parameters(recurse=True):
        return param.device
    for buffer in module.buffers(recurse=True):
        return buffer.device
    return None


def _move_tensor_tree(obj, device: torch.device):
    """Move tensors inside common container types to a device."""
    if obj is None:
        return None
    if torch.is_tensor(obj):
        return obj.to(device)
    if isinstance(obj, tuple):
        return tuple(_move_tensor_tree(x, device) for x in obj)
    if isinstance(obj, list):
        return [_move_tensor_tree(x, device) for x in obj]
    if isinstance(obj, dict):
        return {k: _move_tensor_tree(v, device) for k, v in obj.items()}
    return obj


def _move_if_needed(tensor: Optional[torch.Tensor], device: torch.device):
    if tensor is None or tensor.device == device:
        return tensor
    return tensor.to(device)


@dataclass
class FluxCacheVizConfig:
    """
    控制 Flux Transformer forward 中 key-token 可视化与 CSV 记录的配置。

    所有字段默认为 None / False，表示禁用该功能；启用时需要给出绝对路径。

    Attributes:
        enable: 总开关
        gen_dir: 用于查找原始生成图的目录（按 image_id 与 round 匹配文件名）
        viz_out_dir: key_token 可视化结果输出目录
        csv_out_path: key_token_ratio vs edit_ratio CSV 输出路径
        edit_ratio_summary_candidates: 参考的 edit_ratio summary CSV 候选路径
        rounds_per_image: 每张图最多多少轮（用于 image_id 推断）
        ref_layer_idx: 选作参考帧的 single 层索引（原代码硬编码为 37）
        ref_stream: "single" 或 "double"，参考帧从哪个 stream 加载
    """

    enable: bool = False
    gen_dir: Optional[str] = None
    viz_out_dir: Optional[str] = None
    csv_out_path: Optional[str] = None
    edit_ratio_summary_candidates: List[str] = field(default_factory=list)
    rounds_per_image: int = 7
    ref_layer_idx: int = 37
    ref_stream: str = "single"


def _resolve_ref_for_key_token_update(
    ctx,
    hidden_states: torch.Tensor,
    viz_config: Optional[FluxCacheVizConfig],
):
    """加载用于 key_token 更新的参考帧激活。"""
    stream = "single"
    layer_idx = 37
    if viz_config is not None:
        stream = viz_config.ref_stream or "single"
        layer_idx = viz_config.ref_layer_idx
    return ctx.load_activation(
        stream=stream,
        layer_idx=layer_idx,
        device=hidden_states.device,
        step=ctx.current_step,
    )


def _record_key_token_stats(
    ctx,
    viz_config: Optional[FluxCacheVizConfig],
    hidden_states: torch.Tensor,
) -> None:
    """在 step==0 时执行 key-token 可视化与 CSV 记录。"""
    if viz_config is None or not viz_config.enable:
        return
    if ctx.key_token_indices is None:
        return

    cur_round = int(ctx.current_round)
    cur_step = int(ctx.current_step)
    if cur_step != 0:
        return

    img_token_len = hidden_states.shape[1] // 2

    try:
        image_filename = f"r{cur_round}_unknown.png"
        vis_image = None

        if viz_config.csv_out_path:
            image_id = infer_image_id_from_csv_by_round(
                viz_config.csv_out_path, viz_config.rounds_per_image
            )
        else:
            image_id = "0000"

        if viz_config.gen_dir and _HAS_PIL and os.path.isdir(viz_config.gen_dir):
            pat = re.compile(
                rf"^{re.escape(image_id)}_r{cur_round}_.*\.(png|jpg|jpeg|webp|bmp)$",
                re.IGNORECASE,
            )
            files = [
                fn for fn in os.listdir(viz_config.gen_dir) if pat.match(fn)
            ]
            files.sort(
                key=lambda x: (0 if x.lower().endswith(".png") else 1, x)
            )
            if files:
                img_path = os.path.join(viz_config.gen_dir, files[0])
                image_filename = os.path.basename(img_path)
                vis_image = Image.open(img_path).convert("RGB")
            else:
                image_filename = f"{image_id}_r{cur_round}_unknown.png"

        if viz_config.viz_out_dir and vis_image is not None:
            vis_path = os.path.join(
                viz_config.viz_out_dir,
                f"{image_id}_round{cur_round}_step{cur_step}.png",
            )
            visualize_key_tokens_on_image(
                key_token_indices=ctx.key_token_indices,
                image=vis_image,
                img_token_len=img_token_len,
                save_path=vis_path,
            )

        indices = ctx.key_token_indices.detach().flatten().to("cpu").long()
        valid = indices[(indices >= 0) & (indices < img_token_len)]
        key_token_num = int(valid.shape[0])

        if viz_config.csv_out_path:
            append_key_token_ratio_with_edit_ratio(
                image_filename=image_filename,
                cur_round=cur_round,
                cur_step=cur_step,
                key_token_num=key_token_num,
                img_token_len=int(img_token_len),
                output_csv=viz_config.csv_out_path,
                edit_ratio_summary_candidates=viz_config.edit_ratio_summary_candidates,
            )

        print(
            f"[key-ratio@step0] logged: id={image_id}, round={cur_round}, "
            f"step={cur_step}, key_img_tokens={key_token_num}/{img_token_len}, "
            f"ratio={key_token_num / max(1, img_token_len):.6f}"
        )
    except Exception as e:  # 不让统计错误影响主流程
        print(
            f"[Flux] visualization/stat record failed at step "
            f"{ctx.current_step}, err={e}"
        )


def cache_flux_transformer_2d_forward(
    self,
    hidden_states: torch.Tensor,
    encoder_hidden_states: torch.Tensor = None,
    pooled_projections: torch.Tensor = None,
    timestep: torch.LongTensor = None,
    img_ids: torch.Tensor = None,
    txt_ids: torch.Tensor = None,
    guidance: torch.Tensor = None,
    joint_attention_kwargs: Optional[Dict[str, Any]] = None,
    controlnet_block_samples=None,
    controlnet_single_block_samples=None,
    return_dict: bool = True,
    controlnet_blocks_repeat: bool = False,
) -> Union[torch.Tensor, "Transformer2DModelOutput"]:
    """
    Cache-aware forward for ``FluxTransformer2DModel``.

    Requires ``self.cache_context`` set to a ``FluxCacheManager`` instance.
    Optional ``self.cache_viz_config`` of type :class:`FluxCacheVizConfig`
    enables key-token visualization and CSV logging at step 0 of reuse rounds.
    """
    ctx = getattr(self, "cache_context", None)
    if ctx is None:
        raise RuntimeError(
            "cache_flux_transformer_2d_forward requires self.cache_context "
            "to be set (e.g., to a FluxCacheManager instance)."
        )
    viz_config: Optional[FluxCacheVizConfig] = getattr(
        self, "cache_viz_config", None
    )

    if joint_attention_kwargs is not None:
        joint_attention_kwargs = joint_attention_kwargs.copy()
        lora_scale = joint_attention_kwargs.pop("scale", 1.0)
    else:
        lora_scale = 1.0

    if USE_PEFT_BACKEND:
        scale_lora_layers(self, lora_scale)
    else:
        if (
            joint_attention_kwargs is not None
            and joint_attention_kwargs.get("scale", None) is not None
            and logger is not None
        ):
            logger.warning(
                "Passing `scale` via `joint_attention_kwargs` when not using "
                "the PEFT backend is ineffective."
            )

    embed_device = _module_device(self.x_embedder) or hidden_states.device
    hidden_states = _move_if_needed(hidden_states, embed_device)
    encoder_hidden_states = _move_if_needed(encoder_hidden_states, embed_device)
    pooled_projections = _move_if_needed(pooled_projections, embed_device)
    timestep = _move_if_needed(timestep, embed_device)
    guidance = _move_if_needed(guidance, embed_device)

    hidden_states = self.x_embedder(hidden_states)

    timestep = timestep.to(hidden_states.dtype) * 1000
    if guidance is not None:
        guidance = guidance.to(hidden_states.dtype) * 1000

    temb = (
        self.time_text_embed(timestep, pooled_projections)
        if guidance is None
        else self.time_text_embed(timestep, guidance, pooled_projections)
    )
    encoder_hidden_states = self.context_embedder(encoder_hidden_states)

    pos_device = _module_device(self.pos_embed) or hidden_states.device
    txt_ids = _move_if_needed(txt_ids, pos_device)
    img_ids = _move_if_needed(img_ids, pos_device)

    if txt_ids.ndim == 3:
        if logger is not None:
            logger.warning(
                "Passing `txt_ids` 3d torch.Tensor is deprecated. "
                "Please remove the batch dimension and pass it as a 2d torch Tensor."
            )
        txt_ids = txt_ids[0]
    if img_ids.ndim == 3:
        if logger is not None:
            logger.warning(
                "Passing `img_ids` 3d torch.Tensor is deprecated. "
                "Please remove the batch dimension and pass it as a 2d torch Tensor."
            )
        img_ids = img_ids[0]

    ids = torch.cat((txt_ids, img_ids), dim=0)
    image_rotary_emb = self.pos_embed(ids)
    txt_len = txt_ids.shape[0]

    hidden_states, image_rotary_emb, _key_token_num, should_reuse = (
        ctx.maybe_rearrange_img_and_pe(
            img=hidden_states,
            pe=image_rotary_emb,
            txt_len=txt_len,
        )
    )

    if (
        joint_attention_kwargs is not None
        and "ip_adapter_image_embeds" in joint_attention_kwargs
    ):
        ip_adapter_image_embeds = joint_attention_kwargs.pop(
            "ip_adapter_image_embeds"
        )
        ip_hidden_states = self.encoder_hid_proj(ip_adapter_image_embeds)
        joint_attention_kwargs.update({"ip_hidden_states": ip_hidden_states})

    # ---------------- double-stream blocks ----------------
    ctx.stream_type = "double"
    for index_block, block in enumerate(self.transformer_blocks):
        block_device = _module_device(block) or hidden_states.device
        hidden_states = _move_if_needed(hidden_states, block_device)
        encoder_hidden_states = _move_if_needed(
            encoder_hidden_states, block_device
        )
        temb = _move_if_needed(temb, block_device)
        image_rotary_emb_for_block = _move_tensor_tree(
            image_rotary_emb, block_device
        )
        joint_attention_kwargs_for_block = _move_tensor_tree(
            joint_attention_kwargs, block_device
        )

        if torch.is_grad_enabled() and getattr(self, "gradient_checkpointing", False):
            encoder_hidden_states, hidden_states = self._gradient_checkpointing_func(
                block,
                hidden_states,
                encoder_hidden_states,
                temb,
                image_rotary_emb_for_block,
                joint_attention_kwargs_for_block,
            )
        else:
            encoder_hidden_states, hidden_states = block(
                hidden_states=hidden_states,
                encoder_hidden_states=encoder_hidden_states,
                temb=temb,
                image_rotary_emb=image_rotary_emb_for_block,
                joint_attention_kwargs=joint_attention_kwargs_for_block,
            )

        if should_reuse:
            prev = ctx.load_activation(
                stream="double",
                layer_idx=index_block,
                device=hidden_states.device,
            )
            if prev is not None and ctx.key_token_indices is not None:
                mask_prev = torch.ones(
                    prev.size(1), dtype=torch.bool, device=prev.device
                )
                kti = ctx.key_token_indices
                if kti.device != prev.device:
                    kti = kti.to(prev.device)
                mask_prev[kti] = False
                prev = prev[:, mask_prev, :]
                hidden_states = torch.cat((hidden_states, prev), dim=1)

        ctx.maby_store_activation(
            stream="double",
            layer_idx=index_block,
            tensor=hidden_states,
        )

        if controlnet_block_samples is not None:
            interval_control = len(self.transformer_blocks) / len(
                controlnet_block_samples
            )
            interval_control = int(np.ceil(interval_control))
            if controlnet_blocks_repeat:
                hidden_states = (
                    hidden_states
                    + controlnet_block_samples[
                        index_block % len(controlnet_block_samples)
                    ].to(hidden_states.device)
                )
            else:
                hidden_states = (
                    hidden_states
                    + controlnet_block_samples[
                        index_block // interval_control
                    ].to(hidden_states.device)
                )

    # ---------------- single-stream blocks ----------------
    ctx.stream_type = "single"
    for index_block, block in enumerate(self.single_transformer_blocks):
        block_device = _module_device(block) or hidden_states.device
        hidden_states = _move_if_needed(hidden_states, block_device)
        encoder_hidden_states = _move_if_needed(
            encoder_hidden_states, block_device
        )
        temb = _move_if_needed(temb, block_device)
        image_rotary_emb_for_block = _move_tensor_tree(
            image_rotary_emb, block_device
        )
        joint_attention_kwargs_for_block = _move_tensor_tree(
            joint_attention_kwargs, block_device
        )

        if torch.is_grad_enabled() and getattr(self, "gradient_checkpointing", False):
            encoder_hidden_states, hidden_states = self._gradient_checkpointing_func(
                block,
                hidden_states,
                encoder_hidden_states,
                temb,
                image_rotary_emb_for_block,
                joint_attention_kwargs_for_block,
            )
        else:
            encoder_hidden_states, hidden_states = block(
                hidden_states=hidden_states,
                encoder_hidden_states=encoder_hidden_states,
                temb=temb,
                image_rotary_emb=image_rotary_emb_for_block,
                joint_attention_kwargs=joint_attention_kwargs_for_block,
            )

        if should_reuse:
            prev = ctx.load_activation(
                stream="single",
                layer_idx=index_block,
                device=hidden_states.device,
            )
            if prev is not None and ctx.key_token_indices is not None:
                mask_prev = torch.ones(
                    prev.size(1), dtype=torch.bool, device=prev.device
                )
                kti = ctx.key_token_indices
                if kti.device != prev.device:
                    kti = kti.to(prev.device)
                mask_prev[kti] = False
                prev = prev[:, mask_prev, :]
                hidden_states = torch.cat((hidden_states, prev), dim=1)

        ctx.maby_store_activation(
            stream="single",
            layer_idx=index_block,
            tensor=hidden_states,
        )

        if controlnet_single_block_samples is not None:
            interval_control = len(self.single_transformer_blocks) / len(
                controlnet_single_block_samples
            )
            interval_control = int(np.ceil(interval_control))
            hidden_states = (
                hidden_states
                + controlnet_single_block_samples[
                    index_block // interval_control
                ].to(hidden_states.device)
            )

    # ---- update key_token_indices ----
    if (
        ctx.use_activation_cache
        and not ctx.is_round0
        and ctx.cache_steps is not None
        and ctx.current_step in ctx.cache_steps
    ):
        ref_img = _resolve_ref_for_key_token_update(ctx, hidden_states, viz_config)
        ctx.update_key_token_indices(cur_img=hidden_states, ref_img=ref_img)
        _record_key_token_stats(ctx, viz_config, hidden_states)

    # ---- flush new_cache → prev_cache by policy ----
    ctx.flush_new_cache_after_step()

    # ---- restore token order if reused ----
    hidden_states = ctx.maybe_restore_img_order(img=hidden_states)

    out_device = _module_device(self.norm_out) or hidden_states.device
    hidden_states = _move_if_needed(hidden_states, out_device)
    temb = _move_if_needed(temb, out_device)

    hidden_states = self.norm_out(hidden_states, temb)
    output = self.proj_out(hidden_states)

    if USE_PEFT_BACKEND:
        unscale_lora_layers(self, lora_scale)

    if not return_dict:
        return (output,)

    return Transformer2DModelOutput(sample=output)
