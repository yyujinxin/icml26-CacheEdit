"""Flux Kontext pipeline with caching support."""

from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import torch

try:
    from diffusers import FluxKontextPipeline
    from diffusers.image_processor import PipelineImageInput
    from diffusers.pipelines.flux import FluxPipelineOutput
    from diffusers.utils import is_torch_xla_available
    _HAS_DIFFUSERS = True
except ImportError:  # pragma: no cover
    FluxKontextPipeline = object  # type: ignore
    PipelineImageInput = None  # type: ignore
    FluxPipelineOutput = None  # type: ignore
    is_torch_xla_available = lambda: False  # type: ignore
    _HAS_DIFFUSERS = False

if _HAS_DIFFUSERS and is_torch_xla_available():
    import torch_xla.core.xla_model as xm  # type: ignore

    XLA_AVAILABLE = True
else:
    XLA_AVAILABLE = False

from cache_edit.utils.scheduler_utils import (
    calculate_shift,
    retrieve_timesteps,
)
from cache_edit.models.flux.cache_manager import FluxCacheManager
from cache_edit.models.flux.processor import FluxAttnCacheProcessor
from cache_edit.models.flux.blocks import (
    cache_flux_single_transformer_block_forward,
    cache_flux_transformer_block_forward,
)
from cache_edit.models.flux.transformer_forward import (
    FluxCacheVizConfig,
    cache_flux_transformer_2d_forward,
)


PREFERRED_KONTEXT_RESOLUTIONS: List[Tuple[int, int]] = [
    (672, 1568),
    (688, 1504),
    (720, 1456),
    (752, 1392),
    (800, 1328),
    (832, 1248),
    (880, 1184),
    (944, 1104),
    (1024, 1024),
    (1104, 944),
    (1184, 880),
    (1248, 832),
    (1328, 800),
    (1392, 752),
    (1456, 720),
    (1504, 688),
    (1568, 672),
]


if _HAS_DIFFUSERS:

    class CacheFluxKontextPipeline(FluxKontextPipeline):
        """
        FluxKontextPipeline 子类，在去噪循环中调用 cache_context.on_step_start。

        通过 ``self.cache_context``（FluxCacheManager 实例）协调 transformer
        / blocks / attn processors 共享同一份缓存上下文。
        """

        cache_context: Optional[FluxCacheManager] = None

        def attach_cache_context(self, cache_context: FluxCacheManager) -> None:
            """附加缓存上下文，并广播到 transformer / blocks / processors。"""
            self.cache_context = cache_context
            self.transformer.cache_context = cache_context
            for block in self.transformer.transformer_blocks:
                block.cache_context = cache_context
                proc = block.attn.processor
                if hasattr(proc, "attach_cache_context"):
                    proc.attach_cache_context(cache_context)
            for block in self.transformer.single_transformer_blocks:
                block.cache_context = cache_context
                proc = block.attn.processor
                if hasattr(proc, "attach_cache_context"):
                    proc.attach_cache_context(cache_context)

        def attach_viz_config(self, viz_config: FluxCacheVizConfig) -> None:
            """附加 key-token 可视化与 CSV 记录配置。"""
            self.transformer.cache_viz_config = viz_config

        @torch.no_grad()
        def __call__(
            self,
            image: Optional[PipelineImageInput] = None,
            prompt: Union[str, List[str]] = None,
            prompt_2: Optional[Union[str, List[str]]] = None,
            negative_prompt: Union[str, List[str]] = None,
            negative_prompt_2: Optional[Union[str, List[str]]] = None,
            true_cfg_scale: float = 1.0,
            height: Optional[int] = None,
            width: Optional[int] = None,
            num_inference_steps: int = 28,
            sigmas: Optional[List[float]] = None,
            guidance_scale: float = 3.5,
            num_images_per_prompt: Optional[int] = 1,
            generator: Optional[
                Union[torch.Generator, List[torch.Generator]]
            ] = None,
            latents: Optional[torch.FloatTensor] = None,
            prompt_embeds: Optional[torch.FloatTensor] = None,
            pooled_prompt_embeds: Optional[torch.FloatTensor] = None,
            ip_adapter_image: Optional[PipelineImageInput] = None,
            ip_adapter_image_embeds: Optional[List[torch.Tensor]] = None,
            negative_ip_adapter_image: Optional[PipelineImageInput] = None,
            negative_ip_adapter_image_embeds: Optional[List[torch.Tensor]] = None,
            negative_prompt_embeds: Optional[torch.FloatTensor] = None,
            negative_pooled_prompt_embeds: Optional[torch.FloatTensor] = None,
            output_type: Optional[str] = "pil",
            return_dict: bool = True,
            joint_attention_kwargs: Optional[Dict[str, Any]] = None,
            callback_on_step_end: Optional[
                Callable[[int, int, Dict], None]
            ] = None,
            callback_on_step_end_tensor_inputs: List[str] = ["latents"],
            max_sequence_length: int = 512,
            max_area: int = 1024**2,
            _auto_resize: bool = True,
        ):
            multiple_of = self.vae_scale_factor * 2

            # 1. Preprocess image (auto-resize to nearest preferred resolution)
            if image is not None and not (
                isinstance(image, torch.Tensor)
                and image.size(1) == self.latent_channels
            ):
                img = image[0] if isinstance(image, list) else image
                image_height, image_width = (
                    self.image_processor.get_default_height_width(img)
                )
                aspect_ratio = image_width / image_height
                if _auto_resize:
                    _, image_width, image_height = min(
                        (abs(aspect_ratio - w / h), w, h)
                        for w, h in PREFERRED_KONTEXT_RESOLUTIONS
                    )
                image_width = image_width // multiple_of * multiple_of
                image_height = image_height // multiple_of * multiple_of
                image = self.image_processor.resize(
                    image, image_height, image_width
                )
                image = self.image_processor.preprocess(
                    image, image_height, image_width
                )
                height, width = image.shape[-2], image.shape[-1]
            else:
                height = (
                    height or self.default_sample_size * self.vae_scale_factor
                )
                width = width or self.default_sample_size * self.vae_scale_factor

                aspect_ratio = width / height
                width = round((max_area * aspect_ratio) ** 0.5)
                height = round((max_area / aspect_ratio) ** 0.5)
                width = width // multiple_of * multiple_of
                height = height // multiple_of * multiple_of

            self.check_inputs(
                prompt,
                prompt_2,
                height,
                width,
                negative_prompt=negative_prompt,
                negative_prompt_2=negative_prompt_2,
                prompt_embeds=prompt_embeds,
                negative_prompt_embeds=negative_prompt_embeds,
                pooled_prompt_embeds=pooled_prompt_embeds,
                negative_pooled_prompt_embeds=negative_pooled_prompt_embeds,
                callback_on_step_end_tensor_inputs=callback_on_step_end_tensor_inputs,
                max_sequence_length=max_sequence_length,
            )

            self._guidance_scale = guidance_scale
            self._joint_attention_kwargs = joint_attention_kwargs
            self._current_timestep = None
            self._interrupt = False

            if prompt is not None and isinstance(prompt, str):
                batch_size = 1
            elif prompt is not None and isinstance(prompt, list):
                batch_size = len(prompt)
            else:
                batch_size = prompt_embeds.shape[0]

            device = self._execution_device

            lora_scale = (
                self.joint_attention_kwargs.get("scale", None)
                if self.joint_attention_kwargs is not None
                else None
            )
            has_neg_prompt = negative_prompt is not None or (
                negative_prompt_embeds is not None
                and negative_pooled_prompt_embeds is not None
            )
            do_true_cfg = true_cfg_scale > 1 and has_neg_prompt
            (
                prompt_embeds,
                pooled_prompt_embeds,
                text_ids,
            ) = self.encode_prompt(
                prompt=prompt,
                prompt_2=prompt_2,
                prompt_embeds=prompt_embeds,
                pooled_prompt_embeds=pooled_prompt_embeds,
                device=device,
                num_images_per_prompt=num_images_per_prompt,
                max_sequence_length=max_sequence_length,
                lora_scale=lora_scale,
            )
            if do_true_cfg:
                (
                    negative_prompt_embeds,
                    negative_pooled_prompt_embeds,
                    negative_text_ids,
                ) = self.encode_prompt(
                    prompt=negative_prompt,
                    prompt_2=negative_prompt_2,
                    prompt_embeds=negative_prompt_embeds,
                    pooled_prompt_embeds=negative_pooled_prompt_embeds,
                    device=device,
                    num_images_per_prompt=num_images_per_prompt,
                    max_sequence_length=max_sequence_length,
                    lora_scale=lora_scale,
                )

            num_channels_latents = self.transformer.config.in_channels // 4
            latents, image_latents, latent_ids, image_ids = self.prepare_latents(
                image,
                batch_size * num_images_per_prompt,
                num_channels_latents,
                height,
                width,
                prompt_embeds.dtype,
                device,
                generator,
                latents,
            )
            if image_ids is not None:
                latent_ids = torch.cat([latent_ids, image_ids], dim=0)

            sigmas = (
                np.linspace(1.0, 1 / num_inference_steps, num_inference_steps)
                if sigmas is None
                else sigmas
            )
            image_seq_len = latents.shape[1]
            mu = calculate_shift(
                image_seq_len,
                self.scheduler.config.get("base_image_seq_len", 256),
                self.scheduler.config.get("max_image_seq_len", 4096),
                self.scheduler.config.get("base_shift", 0.5),
                self.scheduler.config.get("max_shift", 1.15),
            )
            timesteps, num_inference_steps = retrieve_timesteps(
                self.scheduler,
                num_inference_steps,
                device,
                sigmas=sigmas,
                mu=mu,
            )
            num_warmup_steps = max(
                len(timesteps) - num_inference_steps * self.scheduler.order, 0
            )
            self._num_timesteps = len(timesteps)

            if self.transformer.config.guidance_embeds:
                guidance = torch.full(
                    [1], guidance_scale, device=device, dtype=torch.float32
                )
                guidance = guidance.expand(latents.shape[0])
            else:
                guidance = None

            if (
                ip_adapter_image is not None
                or ip_adapter_image_embeds is not None
            ) and (
                negative_ip_adapter_image is None
                and negative_ip_adapter_image_embeds is None
            ):
                negative_ip_adapter_image = np.zeros(
                    (width, height, 3), dtype=np.uint8
                )
                negative_ip_adapter_image = (
                    [negative_ip_adapter_image]
                    * self.transformer.encoder_hid_proj.num_ip_adapters
                )

            elif (
                ip_adapter_image is None and ip_adapter_image_embeds is None
            ) and (
                negative_ip_adapter_image is not None
                or negative_ip_adapter_image_embeds is not None
            ):
                ip_adapter_image = np.zeros((width, height, 3), dtype=np.uint8)
                ip_adapter_image = (
                    [ip_adapter_image]
                    * self.transformer.encoder_hid_proj.num_ip_adapters
                )

            if self.joint_attention_kwargs is None:
                self._joint_attention_kwargs = {}

            image_embeds = None
            negative_image_embeds = None
            if (
                ip_adapter_image is not None
                or ip_adapter_image_embeds is not None
            ):
                image_embeds = self.prepare_ip_adapter_image_embeds(
                    ip_adapter_image,
                    ip_adapter_image_embeds,
                    device,
                    batch_size * num_images_per_prompt,
                )
            if (
                negative_ip_adapter_image is not None
                or negative_ip_adapter_image_embeds is not None
            ):
                negative_image_embeds = self.prepare_ip_adapter_image_embeds(
                    negative_ip_adapter_image,
                    negative_ip_adapter_image_embeds,
                    device,
                    batch_size * num_images_per_prompt,
                )

            self.scheduler.set_begin_index(0)
            with self.progress_bar(total=num_inference_steps) as progress_bar:
                for i, t in enumerate(timesteps):
                    if self.interrupt:
                        continue

                    if self.cache_context is not None:
                        self.cache_context.on_step_start(i)

                    self._current_timestep = t
                    if image_embeds is not None:
                        self._joint_attention_kwargs[
                            "ip_adapter_image_embeds"
                        ] = image_embeds

                    latent_model_input = latents
                    if image_latents is not None:
                        latent_model_input = torch.cat(
                            [latents, image_latents], dim=1
                        )
                    timestep = t.expand(latents.shape[0]).to(latents.dtype)

                    noise_pred = self.transformer(
                        hidden_states=latent_model_input,
                        timestep=timestep / 1000,
                        guidance=guidance,
                        pooled_projections=pooled_prompt_embeds,
                        encoder_hidden_states=prompt_embeds,
                        txt_ids=text_ids,
                        img_ids=latent_ids,
                        joint_attention_kwargs=self.joint_attention_kwargs,
                        return_dict=False,
                    )[0]
                    noise_pred = noise_pred[:, : latents.size(1)]

                    if do_true_cfg:
                        if negative_image_embeds is not None:
                            self._joint_attention_kwargs[
                                "ip_adapter_image_embeds"
                            ] = negative_image_embeds
                        neg_noise_pred = self.transformer(
                            hidden_states=latent_model_input,
                            timestep=timestep / 1000,
                            guidance=guidance,
                            pooled_projections=negative_pooled_prompt_embeds,
                            encoder_hidden_states=negative_prompt_embeds,
                            txt_ids=negative_text_ids,
                            img_ids=latent_ids,
                            joint_attention_kwargs=self.joint_attention_kwargs,
                            return_dict=False,
                        )[0]
                        neg_noise_pred = neg_noise_pred[:, : latents.size(1)]
                        noise_pred = neg_noise_pred + true_cfg_scale * (
                            noise_pred - neg_noise_pred
                        )

                    latents_dtype = latents.dtype
                    noise_pred = noise_pred.to(latents.device)
                    latents = self.scheduler.step(
                        noise_pred, t, latents, return_dict=False
                    )[0]

                    if latents.dtype != latents_dtype:
                        if torch.backends.mps.is_available():
                            latents = latents.to(latents_dtype)

                    if callback_on_step_end is not None:
                        callback_kwargs = {}
                        for k in callback_on_step_end_tensor_inputs:
                            callback_kwargs[k] = locals()[k]
                        callback_outputs = callback_on_step_end(
                            self, i, t, callback_kwargs
                        )
                        latents = callback_outputs.pop("latents", latents)
                        prompt_embeds = callback_outputs.pop(
                            "prompt_embeds", prompt_embeds
                        )

                    if i == len(timesteps) - 1 or (
                        (i + 1) > num_warmup_steps
                        and (i + 1) % self.scheduler.order == 0
                    ):
                        progress_bar.update()

                    if XLA_AVAILABLE:
                        xm.mark_step()

            self._current_timestep = None

            if output_type == "latent":
                image = latents
            else:
                latents = self._unpack_latents(
                    latents, height, width, self.vae_scale_factor
                )
                latents = (
                    latents / self.vae.config.scaling_factor
                ) + self.vae.config.shift_factor
                image = self.vae.decode(latents, return_dict=False)[0]
                image = self.image_processor.postprocess(
                    image, output_type=output_type
                )

            self.maybe_free_model_hooks()

            if not return_dict:
                return (image,)

            return FluxPipelineOutput(images=image)

else:  # pragma: no cover - diffusers missing
    CacheFluxKontextPipeline = None  # type: ignore


def init_flux_pipeline(
    model_path: str,
    device: Union[str, torch.device] = "cuda",
    dtype: torch.dtype = torch.bfloat16,
    cache_manager: Optional[FluxCacheManager] = None,
    use_cache_processor: bool = True,
    install_block_forward_hooks: bool = True,
    device_map: Optional[str] = None,
    viz_config: Optional[FluxCacheVizConfig] = None,
):
    """
    创建并初始化 Flux Kontext 缓存 Pipeline。

    步骤：
    1. 从 ``model_path`` 加载 ``CacheFluxKontextPipeline``
    2. 替换 transformer.forward → cache_flux_transformer_2d_forward
    3. 替换每个 double / single block 的 forward
    4. 设置每个 attn 的 processor 为 FluxAttnCacheProcessor
    5. 附加 cache_context 到 transformer / blocks / processors

    Args:
        model_path: 预训练模型路径或 HF ID
        device: 目标设备
        dtype: 数据类型
        cache_manager: FluxCacheManager 实例；None 时不启用缓存
        use_cache_processor: 是否替换 attn 的 processor 为缓存版
        install_block_forward_hooks: 是否替换 block.forward 为缓存版
        device_map: 可选的 device_map（如 "balanced"）；若给出则忽略 ``device`` 的搬移
        viz_config: 可选的 key-token 可视化与 CSV 配置

    Returns:
        CacheFluxKontextPipeline 实例
    """
    if not _HAS_DIFFUSERS:
        raise ImportError(
            "diffusers is required to use init_flux_pipeline. "
            "Install with: pip install diffusers"
        )

    if device_map is not None:
        pipeline = CacheFluxKontextPipeline.from_pretrained(
            model_path, torch_dtype=dtype, device_map=device_map
        )
    else:
        pipeline = CacheFluxKontextPipeline.from_pretrained(
            model_path, torch_dtype=dtype
        )

    # Only install cache hooks if cache_manager is provided
    if cache_manager is not None:
        # 替换 transformer.forward
        pipeline.transformer.forward = cache_flux_transformer_2d_forward.__get__(
            pipeline.transformer, pipeline.transformer.__class__
        )

        # 替换 block forward + 设置 processor
        if install_block_forward_hooks:
            for block in pipeline.transformer.transformer_blocks:
                block.workspace = {}
                block.forward = cache_flux_transformer_block_forward.__get__(
                    block, block.__class__
                )
            for block in pipeline.transformer.single_transformer_blocks:
                block.workspace = {}
                block.forward = cache_flux_single_transformer_block_forward.__get__(
                    block, block.__class__
                )

        if use_cache_processor:
            for block in pipeline.transformer.transformer_blocks:
                block.attn.set_processor(FluxAttnCacheProcessor())
            for block in pipeline.transformer.single_transformer_blocks:
                block.attn.set_processor(FluxAttnCacheProcessor())

        pipeline.attach_cache_context(cache_manager)

        if viz_config is not None:
            pipeline.attach_viz_config(viz_config)

    if device_map is None:
        pipeline = pipeline.to(device)

    return pipeline


def create_default_cache_manager(
    num_inference_steps: int = 28,
    threshold: float = 0.97,
    cache_interval: int = 5,
    cache_device: Optional[torch.device] = None,
    num_gpus: int = 1,
    gpu_memory_limit_gb: Optional[float] = None,
    gpu_memory_buffer_gb: float = 1.0,
    use_compression: bool = False,
    compression_bitrate: float = 5.0,
    compression_codec: str = "hevc",
    compression_gop_length: int = 1,
    compression_frame_interval_p: int = 1,
) -> FluxCacheManager:
    """
    创建一组合理默认参数的 FluxCacheManager。

    Args:
        num_inference_steps: 推理步数
        threshold: key-token 相似度阈值
        cache_interval: 缓存间隔（步数），值越小缓存越密集
        cache_device: 起始缓存设备，None 时自动选择
        num_gpus: 可用 GPU 数量，用于多卡显存分配
        gpu_memory_limit_gb: 每张 GPU 显存上限（GB），None 时自动检测
        gpu_memory_buffer_gb: 显存预留 buffer（GB），防止 OOM
        use_compression: 是否使用 LLM.265 NVENC 压缩
        compression_bitrate: 压缩码率（Mbps），1-10 典型值
        compression_codec: 视频编解码器，'hevc' 或 'h264'
        compression_gop_length: 连续 layer 帧间压缩 GOP 长度；<=1 表示全 I 帧
        compression_frame_interval_p: P 帧间隔；1 表示 IPPP
    """
    if cache_device is None:
        cache_device = torch.device(
            "cuda:0" if torch.cuda.is_available() else "cpu"
        )

    return FluxCacheManager(
        use_activation_cache=True,
        cache_device=cache_device,
        total_step_num=num_inference_steps,
        threshold=threshold,
        cache_interval=cache_interval,
        num_gpus=num_gpus,
        gpu_memory_limit_gb=gpu_memory_limit_gb,
        gpu_memory_buffer_gb=gpu_memory_buffer_gb,
        use_compression=use_compression,
        compression_bitrate=compression_bitrate,
        compression_codec=compression_codec,
        compression_gop_length=compression_gop_length,
        compression_frame_interval_p=compression_frame_interval_p,
    )
