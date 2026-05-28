from turtle import width
import torch
import numpy as np
import torch.nn.functional as F
from typing import Optional, Union, List, Dict, Any, Callable, Tuple
from accelerate import infer_auto_device_map, dispatch_model
from math import prod

from diffusers import QwenImageEditPlusPipeline
from diffusers.image_processor import PipelineImageInput
from diffusers.models.transformers.transformer_qwenimage import apply_rotary_emb_qwen
from diffusers.models.attention_processor import Attention
from diffusers.models.attention_dispatch import dispatch_attention_fn
from diffusers.models.modeling_outputs import Transformer2DModelOutput
from diffusers.pipelines.qwenimage import QwenImagePipelineOutput
from diffusers.schedulers import FlowMatchEulerDiscreteScheduler
from diffusers.utils import (
    is_torch_xla_available,
    logging,
    USE_PEFT_BACKEND,
    scale_lora_layers,
    unscale_lora_layers,
)
from Qwen_utils import (
    calculate_dimensions,
    calculate_shift,
    retrieve_timesteps,
    QwenActivationCacheManager,
    KeyTokenStatsCollector,
    stats_collector,
    
    # MANAGER,
    # ids_gather,
    # ids_scatter,
    # token_selector,
    # FlowMatchEulerDiscreteSchedulerOutput
)
if is_torch_xla_available():
    import torch_xla.core.xla_model as xm
    XLA_AVAILABLE = True
else:
    XLA_AVAILABLE = False
try:
    import flash_attn
    from flash_attn import flash_attn_func
except ImportError:
    flash_attn = False
# from fused_kernels import _partially_linear


logger = logging.get_logger(__name__)  # pylint: disable=invalid-name
# gamma = torch.tensor([1.0186, 1.0241, 1.0236, 1.0205, 1.0298, 1.0221, 1.0248, 1.0246, 1.0269,
#         1.0275, 1.0323, 1.0311, 1.0298, 1.0353, 1.0343, 1.0397, 1.0387, 1.0393,
#         1.0404, 1.0458, 1.0507, 1.0418, 1.0518, 1.0426, 1.0311, 1.0068, 0.7628],
#        dtype=torch.float16)

CONDITION_IMAGE_SIZE = 384 * 384
VAE_IMAGE_SIZE = 1024 * 1024

# def cache_edit_init(model_path, device):

#     # 1. 先在 CPU 上加载
#     pipeline = CacheEQwenImageEditPlusPipeline.from_pretrained(
#         model_path,
#         torch_dtype=torch.bfloat16,
#         device_map=None,   # 不让 diffusers 自动分配
#     )

#     transformer = pipeline.transformer

#     # 2. 为 transformer 生成 device_map
#     max_memory = {
#         0: "24GiB",  # 0 卡少放点权重
#         1: "36GiB",
#         2: "36GiB",
#         3: "36GiB",
#     }

#     device_map = infer_auto_device_map(
#         transformer,
#         max_memory=max_memory,
#         # 这里类名要用你真实的 block 类名：
#         # 假设是 CacheEQwenImageTransformerBlock 或 QWenImageTransformerBlock 之类
#         no_split_module_classes=["CacheEQwenImageTransformerBlock", "QWenImageTransformerBlock"],
#     )
#     print("transformer device_map:", device_map)

#     # 3. 用 dispatch_model 把 transformer 切到多卡
#     transformer = dispatch_model(transformer, device_map=device_map)
#     pipeline.transformer = transformer

#     # 4. 其他大模块自己指定设备，避免占用 0 卡
#     if hasattr(pipeline, "vae"):
#         pipeline.vae.to("cuda:1")
#     if hasattr(pipeline, "text_encoder") and pipeline.text_encoder is not None:
#         pipeline.text_encoder.to("cuda:1")

#     # 5. 重新绑定自定义 forward 和 attention processor
#     pipeline.transformer.forward = CacheEQwenImageTransformer2DModelforward.__get__(
#         pipeline.transformer, pipeline.transformer.__class__
#     )

#     for block in pipeline.transformer.transformer_blocks:
#         block.forward = CacheEQwenImageTransformerBlockforward.__get__(block, block.__class__)
#         block.attn.set_processor(CacheEQwenDoubleStreamAttnProcessor2_0())

#     return pipeline

def cache_edit_init(model_path, device):

    pipeline = CacheEQwenImageEditPlusPipeline.from_pretrained(model_path, torch_dtype=torch.bfloat16, device_map="balanced")
    
    pipeline.transformer.forward = CacheEQwenImageTransformer2DModelforward.__get__(pipeline.transformer, pipeline.transformer.__class__)
   
    for block in pipeline.transformer.transformer_blocks:
       
        block.forward = CacheEQwenImageTransformerBlockforward.__get__(block, block.__class__)
        block.attn.set_processor(CacheEQwenDoubleStreamAttnProcessor2_0())
    return pipeline

class CacheEQwenImageEditPlusPipeline(QwenImageEditPlusPipeline):
    @torch.no_grad()
    def __call__(
        self,
        image: Optional[PipelineImageInput] = None,
        prompt: Union[str, List[str]] = None,
        negative_prompt: Union[str, List[str]] = None,
        true_cfg_scale: float = 4.0,
        height: Optional[int] = None,
        width: Optional[int] = None,
        num_inference_steps: int = 30,
        sigmas: Optional[List[float]] = None,
        guidance_scale: Optional[float] = None,
        num_images_per_prompt: int = 1,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        latents: Optional[torch.Tensor] = None,
        prompt_embeds: Optional[torch.Tensor] = None,
        prompt_embeds_mask: Optional[torch.Tensor] = None,
        negative_prompt_embeds: Optional[torch.Tensor] = None,
        negative_prompt_embeds_mask: Optional[torch.Tensor] = None,
        output_type: Optional[str] = "pil",
        return_dict: bool = True,
        attention_kwargs: Optional[Dict[str, Any]] = None,
        callback_on_step_end: Optional[Callable[[int, int, Dict], None]] = None,
        callback_on_step_end_tensor_inputs: List[str] = ["latents"],
        max_sequence_length: int = 512,
    ):
        r"""
        Function invoked when calling the pipeline for generation.

        Args:
            image (`torch.Tensor`, `PIL.Image.Image`, `np.ndarray`, `List[torch.Tensor]`, `List[PIL.Image.Image]`, or `List[np.ndarray]`):
                `Image`, numpy array or tensor representing an image batch to be used as the starting point. For both
                numpy array and pytorch tensor, the expected value range is between `[0, 1]` If it's a tensor or a list
                or tensors, the expected shape should be `(B, C, H, W)` or `(C, H, W)`. If it is a numpy array or a
                list of arrays, the expected shape should be `(B, H, W, C)` or `(H, W, C)` It can also accept image
                latents as `image`, but if passing latents directly it is not encoded again.
            prompt (`str` or `List[str]`, *optional*):
                The prompt or prompts to guide the image generation. If not defined, one has to pass `prompt_embeds`.
                instead.
            negative_prompt (`str` or `List[str]`, *optional*):
                The prompt or prompts not to guide the image generation. If not defined, one has to pass
                `negative_prompt_embeds` instead. Ignored when not using guidance (i.e., ignored if `true_cfg_scale` is
                not greater than `1`).
            true_cfg_scale (`float`, *optional*, defaults to 1.0):
                true_cfg_scale (`float`, *optional*, defaults to 1.0): Guidance scale as defined in [Classifier-Free
                Diffusion Guidance](https://huggingface.co/papers/2207.12598). `true_cfg_scale` is defined as `w` of
                equation 2. of [Imagen Paper](https://huggingface.co/papers/2205.11487). Classifier-free guidance is
                enabled by setting `true_cfg_scale > 1` and a provided `negative_prompt`. Higher guidance scale
                encourages to generate images that are closely linked to the text `prompt`, usually at the expense of
                lower image quality.
            height (`int`, *optional*, defaults to self.unet.config.sample_size * self.vae_scale_factor):
                The height in pixels of the generated image. This is set to 1024 by default for the best results.
            width (`int`, *optional*, defaults to self.unet.config.sample_size * self.vae_scale_factor):
                The width in pixels of the generated image. This is set to 1024 by default for the best results.
            num_inference_steps (`int`, *optional*, defaults to 50):
                The number of denoising steps. More denoising steps usually lead to a higher quality image at the
                expense of slower inference.
            sigmas (`List[float]`, *optional*):
                Custom sigmas to use for the denoising process with schedulers which support a `sigmas` argument in
                their `set_timesteps` method. If not defined, the default behavior when `num_inference_steps` is passed
                will be used.
            guidance_scale (`float`, *optional*, defaults to None):
                A guidance scale value for guidance distilled models. Unlike the traditional classifier-free guidance
                where the guidance scale is applied during inference through noise prediction rescaling, guidance
                distilled models take the guidance scale directly as an input parameter during forward pass. Guidance
                scale is enabled by setting `guidance_scale > 1`. Higher guidance scale encourages to generate images
                that are closely linked to the text `prompt`, usually at the expense of lower image quality. This
                parameter in the pipeline is there to support future guidance-distilled models when they come up. It is
                ignored when not using guidance distilled models. To enable traditional classifier-free guidance,
                please pass `true_cfg_scale > 1.0` and `negative_prompt` (even an empty negative prompt like " " should
                enable classifier-free guidance computations).
            num_images_per_prompt (`int`, *optional*, defaults to 1):
                The number of images to generate per prompt.
            generator (`torch.Generator` or `List[torch.Generator]`, *optional*):
                One or a list of [torch generator(s)](https://pytorch.org/docs/stable/generated/torch.Generator.html)
                to make generation deterministic.
            latents (`torch.Tensor`, *optional*):
                Pre-generated noisy latents, sampled from a Gaussian distribution, to be used as inputs for image
                generation. Can be used to tweak the same generation with different prompts. If not provided, a latents
                tensor will be generated by sampling using the supplied random `generator`.
            prompt_embeds (`torch.Tensor`, *optional*):
                Pre-generated text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt weighting. If not
                provided, text embeddings will be generated from `prompt` input argument.
            negative_prompt_embeds (`torch.Tensor`, *optional*):
                Pre-generated negative text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt
                weighting. If not provided, negative_prompt_embeds will be generated from `negative_prompt` input
                argument.
            output_type (`str`, *optional*, defaults to `"pil"`):
                The output format of the generate image. Choose between
                [PIL](https://pillow.readthedocs.io/en/stable/): `PIL.Image.Image` or `np.array`.
            return_dict (`bool`, *optional*, defaults to `True`):
                Whether or not to return a [`~pipelines.qwenimage.QwenImagePipelineOutput`] instead of a plain tuple.
            attention_kwargs (`dict`, *optional*):
                A kwargs dictionary that if specified is passed along to the `AttentionProcessor` as defined under
                `self.processor` in
                [diffusers.models.attention_processor](https://github.com/huggingface/diffusers/blob/main/src/diffusers/models/attention_processor.py).
            callback_on_step_end (`Callable`, *optional*):
                A function that calls at the end of each denoising steps during the inference. The function is called
                with the following arguments: `callback_on_step_end(self: DiffusionPipeline, step: int, timestep: int,
                callback_kwargs: Dict)`. `callback_kwargs` will include a list of all tensors as specified by
                `callback_on_step_end_tensor_inputs`.
            callback_on_step_end_tensor_inputs (`List`, *optional*):
                The list of tensor inputs for the `callback_on_step_end` function. The tensors specified in the list
                will be passed as `callback_kwargs` argument. You will only be able to include variables listed in the
                `._callback_tensor_inputs` attribute of your pipeline class.
            max_sequence_length (`int` defaults to 512): Maximum sequence length to use with the `prompt`.

        Examples:

        Returns:
            [`~pipelines.qwenimage.QwenImagePipelineOutput`] or `tuple`:
            [`~pipelines.qwenimage.QwenImagePipelineOutput`] if `return_dict` is True, otherwise a `tuple`. When
            returning a tuple, the first element is a list with the generated images.
        """
        QwenActivationCacheManager.num_inference_steps = num_inference_steps
        image_size = image[-1].size if isinstance(image, list) else image.size
        calculated_width, calculated_height = calculate_dimensions(1024 * 1024, image_size[0] / image_size[1])
        height = height or calculated_height
        width = width or calculated_width

        multiple_of = self.vae_scale_factor * 2
        width = width // multiple_of * multiple_of
        height = height // multiple_of * multiple_of

        # print("i'm QwenImageEditPlusPipeline")
        # 1. Check inputs. Raise error if not correct
        self.check_inputs(
            prompt,
            height,
            width,
            negative_prompt=negative_prompt,
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds,
            prompt_embeds_mask=prompt_embeds_mask,
            negative_prompt_embeds_mask=negative_prompt_embeds_mask,
            callback_on_step_end_tensor_inputs=callback_on_step_end_tensor_inputs,
            max_sequence_length=max_sequence_length,
        )

        self._guidance_scale = guidance_scale
        self._attention_kwargs = attention_kwargs
        self._current_timestep = None
        self._interrupt = False

        # 2. Define call parameters
        if prompt is not None and isinstance(prompt, str):
            batch_size = 1
        elif prompt is not None and isinstance(prompt, list):
            batch_size = len(prompt)
        else:
            batch_size = prompt_embeds.shape[0]

        device = self._execution_device
        if hasattr(self.transformer, "device"):
             device = self.transformer.device
        else:
             # 如果 .device 属性不可靠，尝试获取第一个参数的设备
             device = next(self.transformer.parameters()).device
        # 3. Preprocess image
        if image is not None and not (isinstance(image, torch.Tensor) and image.size(1) == self.latent_channels):
            if not isinstance(image, list):
                image = [image]
            condition_image_sizes = []
            condition_images = []
            vae_image_sizes = []
            vae_images = []
            for img in image:
                image_width, image_height = img.size
                condition_width, condition_height = calculate_dimensions(
                    CONDITION_IMAGE_SIZE, image_width / image_height
                )
                vae_width, vae_height = calculate_dimensions(VAE_IMAGE_SIZE, image_width / image_height)
                condition_image_sizes.append((condition_width, condition_height))
                vae_image_sizes.append((vae_width, vae_height))
                condition_images.append(self.image_processor.resize(img, condition_height, condition_width))
                vae_images.append(self.image_processor.preprocess(img, vae_height, vae_width).unsqueeze(2))

        has_neg_prompt = negative_prompt is not None or (
            negative_prompt_embeds is not None and negative_prompt_embeds_mask is not None
        )

        if true_cfg_scale > 1 and not has_neg_prompt:
            logger.warning(
                f"true_cfg_scale is passed as {true_cfg_scale}, but classifier-free guidance is not enabled since no negative_prompt is provided."
            )
        elif true_cfg_scale <= 1 and has_neg_prompt:
            logger.warning(
                " negative_prompt is passed but classifier-free guidance is not enabled since true_cfg_scale <= 1"
            )

        do_true_cfg = true_cfg_scale > 1 and has_neg_prompt
        print("do_true_cfg:", do_true_cfg, " true_cfg_scale:", true_cfg_scale)
        prompt_embeds, prompt_embeds_mask = self.encode_prompt(
            image=condition_images,
            prompt=prompt,
            prompt_embeds=prompt_embeds,
            prompt_embeds_mask=prompt_embeds_mask,
            device=device,
            num_images_per_prompt=num_images_per_prompt,
            max_sequence_length=max_sequence_length,
        )
        if do_true_cfg:
            negative_prompt_embeds, negative_prompt_embeds_mask = self.encode_prompt(
                image=condition_images,
                prompt=negative_prompt,
                prompt_embeds=negative_prompt_embeds,
                prompt_embeds_mask=negative_prompt_embeds_mask,
                device=device,
                num_images_per_prompt=num_images_per_prompt,
                max_sequence_length=max_sequence_length,
            )

        # 4. Prepare latent variables
        num_channels_latents = self.transformer.config.in_channels // 4
        latents, image_latents = self.prepare_latents(
            vae_images,
            batch_size * num_images_per_prompt,
            num_channels_latents,
            height,
            width,
            prompt_embeds.dtype,
            device,
            generator,
            latents,
        )
        img_shapes = [
            [
                (1, height // self.vae_scale_factor // 2, width // self.vae_scale_factor // 2),
                *[
                    (1, vae_height // self.vae_scale_factor // 2, vae_width // self.vae_scale_factor // 2)
                    for vae_width, vae_height in vae_image_sizes
                ],
            ]
        ] * batch_size

        # 5. Prepare timesteps
        sigmas = np.linspace(1.0, 1 / num_inference_steps, num_inference_steps) if sigmas is None else sigmas
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
        num_warmup_steps = max(len(timesteps) - num_inference_steps * self.scheduler.order, 0)
        self._num_timesteps = len(timesteps)

        # handle guidance
        if self.transformer.config.guidance_embeds and guidance_scale is None:
            raise ValueError("guidance_scale is required for guidance-distilled model.")
        elif self.transformer.config.guidance_embeds:
            guidance = torch.full([1], guidance_scale, device=device, dtype=torch.float32)
            guidance = guidance.expand(latents.shape[0])
        elif not self.transformer.config.guidance_embeds and guidance_scale is not None:
            logger.warning(
                f"guidance_scale is passed as {guidance_scale}, but ignored since the model is not guidance-distilled."
            )
            guidance = None
        elif not self.transformer.config.guidance_embeds and guidance_scale is None:
            guidance = None

        if self.attention_kwargs is None:
            self._attention_kwargs = {}

        txt_seq_lens = prompt_embeds_mask.sum(dim=1).tolist() if prompt_embeds_mask is not None else None
        negative_txt_seq_lens = (
            negative_prompt_embeds_mask.sum(dim=1).tolist() if negative_prompt_embeds_mask is not None else None
        )

        # 6. Denoising loop
        self.scheduler.set_begin_index(0)
        with self.progress_bar(total=num_inference_steps) as progress_bar:
            for i, t in enumerate(timesteps):
                
                QwenActivationCacheManager.on_step_start(i)
                print("current_round:", QwenActivationCacheManager.current_round, " current_step:", QwenActivationCacheManager.current_step)
                if self.interrupt:
                    continue

                self._current_timestep = t

                latent_model_input = latents
                if image_latents is not None:
                    image_latents = image_latents.to(latents.device)
                    latent_model_input = torch.cat([latents, image_latents], dim=1)

                # broadcast to batch dimension in a way that's compatible with ONNX/Core ML
                timestep = t.expand(latents.shape[0]).to(latents.dtype)
                with self.transformer.cache_context("cond"):
                    QwenActivationCacheManager.set_mode("cond")
                    noise_pred = self.transformer(
                        hidden_states=latent_model_input,
                        timestep=timestep / 1000,
                        guidance=guidance,
                        encoder_hidden_states_mask=prompt_embeds_mask,
                        encoder_hidden_states=prompt_embeds,
                        img_shapes=img_shapes,
                        txt_seq_lens=txt_seq_lens,
                        attention_kwargs=self.attention_kwargs,
                        return_dict=False,
                    )[0]
                    noise_pred = noise_pred[:, : latents.size(1)]

                if do_true_cfg:
                    print("Doing true classifier-free guidance...")
                    with self.transformer.cache_context("uncond"):
                        QwenActivationCacheManager.set_mode("uncond")
                        neg_noise_pred = self.transformer(
                            hidden_states=latent_model_input,
                            timestep=timestep / 1000,
                            guidance=guidance,
                            encoder_hidden_states_mask=negative_prompt_embeds_mask,
                            encoder_hidden_states=negative_prompt_embeds,
                            img_shapes=img_shapes,
                            txt_seq_lens=negative_txt_seq_lens,
                            attention_kwargs=self.attention_kwargs,
                            return_dict=False,
                        )[0]
                    neg_noise_pred = neg_noise_pred[:, : latents.size(1)]
                    comb_pred = neg_noise_pred + true_cfg_scale * (noise_pred - neg_noise_pred)

                    cond_norm = torch.norm(noise_pred, dim=-1, keepdim=True)
                    noise_norm = torch.norm(comb_pred, dim=-1, keepdim=True)
                    noise_pred = comb_pred * (cond_norm / noise_norm)

                # compute the previous noisy sample x_t -> x_t-1
                latents_dtype = latents.dtype
                latents = self.scheduler.step(noise_pred, t, latents, return_dict=False)[0]

                if latents.dtype != latents_dtype:
                    if torch.backends.mps.is_available():
                        # some platforms (eg. apple mps) misbehave due to a pytorch bug: https://github.com/pytorch/pytorch/pull/99272
                        latents = latents.to(latents_dtype)

                if callback_on_step_end is not None:
                    callback_kwargs = {}
                    for k in callback_on_step_end_tensor_inputs:
                        callback_kwargs[k] = locals()[k]
                    callback_outputs = callback_on_step_end(self, i, t, callback_kwargs)

                    latents = callback_outputs.pop("latents", latents)
                    prompt_embeds = callback_outputs.pop("prompt_embeds", prompt_embeds)

                # call the callback, if provided
                if i == len(timesteps) - 1 or ((i + 1) > num_warmup_steps and (i + 1) % self.scheduler.order == 0):
                    progress_bar.update()

                if XLA_AVAILABLE:
                    xm.mark_step()
                # 循环尾部，step 完成后 flush 缓存
                QwenActivationCacheManager.flush_new_cache_after_step()
        self._current_timestep = None
        if output_type == "latent":
            image = latents
        else:
            latents = self._unpack_latents(latents, height, width, self.vae_scale_factor)
            latents = latents.to(self.vae.dtype)
            latents_mean = (
                torch.tensor(self.vae.config.latents_mean)
                .view(1, self.vae.config.z_dim, 1, 1, 1)
                .to(latents.device, latents.dtype)
            )
            latents_std = 1.0 / torch.tensor(self.vae.config.latents_std).view(1, self.vae.config.z_dim, 1, 1, 1).to(
                latents.device, latents.dtype
            )
            latents = latents / latents_std + latents_mean
            image = self.vae.decode(latents, return_dict=False)[0][:, :, 0]
            image = self.image_processor.postprocess(image, output_type=output_type)

        # Offload all models
        self.maybe_free_model_hooks()

        if not return_dict:
            return (image,)

        return QwenImagePipelineOutput(images=image)
    
def CacheEQwenImageTransformer2DModelforward(
    self,
    hidden_states: torch.Tensor,
    encoder_hidden_states: torch.Tensor = None,
    encoder_hidden_states_mask: torch.Tensor = None,
    timestep: torch.LongTensor = None,
    img_shapes: Optional[List[Tuple[int, int, int]]] = None,
    txt_seq_lens: Optional[List[int]] = None,
    guidance: torch.Tensor = None,  # TODO: this should probably be removed
    attention_kwargs: Optional[Dict[str, Any]] = None,
    controlnet_block_samples=None,
    additional_t_cond=None,
    return_dict: bool = True,
) -> Union[torch.Tensor, Transformer2DModelOutput]:
    """
    The [`QwenTransformer2DModel`] forward method.

    Args:
        hidden_states (`torch.Tensor` of shape `(batch_size, image_sequence_length, in_channels)`):
            Input `hidden_states`.
        encoder_hidden_states (`torch.Tensor` of shape `(batch_size, text_sequence_length, joint_attention_dim)`):
            Conditional embeddings (embeddings computed from the input conditions such as prompts) to use.
        encoder_hidden_states_mask (`torch.Tensor` of shape `(batch_size, text_sequence_length)`):
            Mask of the input conditions.
        timestep ( `torch.LongTensor`):
            Used to indicate denoising step.
        attention_kwargs (`dict`, *optional*):
            A kwargs dictionary that if specified is passed along to the `AttentionProcessor` as defined under
            `self.processor` in
            [diffusers.models.attention_processor](https://github.com/huggingface/diffusers/blob/main/src/diffusers/models/attention_processor.py).
        return_dict (`bool`, *optional*, defaults to `True`):
            Whether or not to return a [`~models.transformer_2d.Transformer2DModelOutput`] instead of a plain
            tuple.

    Returns:
        If `return_dict` is True, an [`~models.transformer_2d.Transformer2DModelOutput`] is returned, otherwise a
        `tuple` where the first element is the sample tensor.
    """
    # print("i'm QwenImageTransformer2DModel forward")
    if attention_kwargs is not None:
        attention_kwargs = attention_kwargs.copy()
        lora_scale = attention_kwargs.pop("scale", 1.0)
    else:
        lora_scale = 1.0

    if USE_PEFT_BACKEND:
        # weight the lora layers by setting `lora_scale` for each PEFT layer
        scale_lora_layers(self, lora_scale)
    else:
        if attention_kwargs is not None and attention_kwargs.get("scale", None) is not None:
            logger.warning(
                "Passing `scale` via `joint_attention_kwargs` when not using the PEFT backend is ineffective."
            )

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
    # print("self.zero_cond_t:", self.zero_cond_t)
    encoder_hidden_states = self.txt_norm(encoder_hidden_states)
    encoder_hidden_states = self.txt_in(encoder_hidden_states)

    if guidance is not None:
        guidance = guidance.to(hidden_states.dtype) * 1000

    temb = (
        self.time_text_embed(timestep, hidden_states, additional_t_cond)
        if guidance is None
        else self.time_text_embed(timestep, guidance, hidden_states, additional_t_cond)
    )


    # pos_embed: 返回 (img_freqs, txt_freqs)
    image_rotary_emb = self.pos_embed(img_shapes, txt_seq_lens, device=hidden_states.device)
    img_freqs, txt_freqs = image_rotary_emb  # img_freqs: [L_img, D], txt_freqs: [L_txt, D]

    # ---- 1. image tokens + RoPE 重排（关键 token 前置） ----
    hidden_states, img_freqs, modulate_index, key_token_num, should_reuse = \
        QwenActivationCacheManager.maybe_rearrange_img_and_rope(
            img=hidden_states,
            img_freqs=img_freqs,
            modulate_index=modulate_index,
        )

    print("should_reuse:", should_reuse, "should_cache:", QwenActivationCacheManager.should_cache(QwenActivationCacheManager.current_step))
    image_rotary_emb = (img_freqs, txt_freqs)
    
    for index_block, block in enumerate(self.transformer_blocks):
        if torch.is_grad_enabled() and self.gradient_checkpointing:
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

        # ---- 复用上一轮缓存：把 prev 的非关键 token 接到当前 hidden_states 后面 ----
        if should_reuse and QwenActivationCacheManager.key_token_indices[QwenActivationCacheManager.current_mode] is not None:
            prev = QwenActivationCacheManager.load_activation(
                stream="double",
                layer_idx=index_block,
                device=hidden_states.device,
            )
            if prev is not None:
                # prev 当前是上一轮同 step 同层的 full image hidden_states（原顺序）
                mask_prev = torch.ones(prev.size(1), dtype=torch.bool, device=prev.device)
                mask_prev[QwenActivationCacheManager.key_token_indices[QwenActivationCacheManager.current_mode]] = False
                prev_non_key = prev[:, mask_prev, :]        # 只保留非关键 token
                hidden_states = torch.cat((hidden_states, prev_non_key), dim=1)
        
        # 如果需要写入 cache（已经重排）
        QwenActivationCacheManager.maby_store_activation(
            layer_idx=index_block,
            tensor=hidden_states,
            stream="double",
        )
        
        # controlnet residual
        if controlnet_block_samples is not None:
            interval_control = len(self.transformer_blocks) / len(controlnet_block_samples)
            interval_control = int(np.ceil(interval_control))
            hidden_states = hidden_states + controlnet_block_samples[index_block // interval_control]
    
    # ---- 2. 更新 key_token_indices（第二轮及以后、step==0） ----
    if (
        QwenActivationCacheManager.use_activation_cache
        and not QwenActivationCacheManager.is_round0
        and QwenActivationCacheManager.current_step in QwenActivationCacheManager.cache_steps
    ):
       
        total_layer_num = len(self.transformer_blocks)
        # print("total_layer_num:", total_layer_num)
        layer_idx_for_true_updating = 57
        ref_img = QwenActivationCacheManager.load_key_token_ref(
            stream="double",
            layer_idx=layer_idx_for_true_updating,
            device=hidden_states.device,
            step=QwenActivationCacheManager.current_step,
        )
        cur_img = QwenActivationCacheManager.load_key_token_cur(
            stream="double",
            layer_idx=layer_idx_for_true_updating,
            device=hidden_states.device,
            step=QwenActivationCacheManager.current_step,
        )
        # print("ref_img shape:", ref_img.shape)
        QwenActivationCacheManager.update_key_token_indices(
            cur_img=cur_img,
            ref_img=ref_img,
        )
        
        #  for print and stats collection
        # stats_collector = KeyTokenStatsCollector()
        # for layer_idx_for_updating in range(total_layer_num):
        #     # print(f"Updating key_token_indices at layer {layer_idx_for_updating}...")
        #     ref_img = QwenActivationCacheManager.load_key_token_ref(
        #         stream="double",
        #         layer_idx=layer_idx_for_updating,
        #         device=hidden_states.device,
        #         step=QwenActivationCacheManager.current_step,
        #     )
        #     cur_img = QwenActivationCacheManager.load_key_token_cur(
        #         stream="double",
        #         layer_idx=layer_idx_for_updating,
        #         device=hidden_states.device,
        #         step=QwenActivationCacheManager.current_step,
        #     )
        #     # print("ref_img shape:", ref_img.shape)
        #     QwenActivationCacheManager.update_key_token_indices(
        #         cur_img=cur_img,
        #         ref_img=ref_img,
        #     )
        #     # 把当前 step + layer 的信息记下来
        #     stats_collector.record(QwenActivationCacheManager, step=QwenActivationCacheManager.current_step, layer_idx=layer_idx_for_updating)
        # 3) 最后打印直观统计
        # if QwenActivationCacheManager.current_step == QwenActivationCacheManager.total_step_num -1:
        #     print("Final key token stats over all layers and steps:")
        #     # 所有 step 完成后，保存结果到 Excel（含图表）
        #     stats_collector.save_to_excel("/home/chenxueqing/image-edit-round-reuse/result/QwenImageEdit/analysis/key_token_stats.xlsx")
        #     stats_collector.report()
            
        
    # ---- 3. 恢复 token 原顺序 ----
    hidden_states = QwenActivationCacheManager.maybe_restore_img_order(hidden_states)
    
    if self.zero_cond_t:
        temb = temb.chunk(2, dim=0)[0]
    # Use only the image part (hidden_states) from the dual-stream blocks
    hidden_states = self.norm_out(hidden_states, temb)
    output = self.proj_out(hidden_states)

    if USE_PEFT_BACKEND:
        # remove `lora_scale` from each PEFT layer
        unscale_lora_layers(self, lora_scale)

    if not return_dict:
        return (output,)

    return Transformer2DModelOutput(sample=output)

def CacheEQwenImageTransformerBlockforward(
    self,
    hidden_states: torch.Tensor,
    encoder_hidden_states: torch.Tensor,
    encoder_hidden_states_mask: torch.Tensor,
    temb: torch.Tensor,
    image_rotary_emb: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    joint_attention_kwargs: Optional[Dict[str, Any]] = None,
    modulate_index: Optional[List[int]] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    # print("i'm QwenTransformerBlock forward")
    # Get modulation parameters for both streams
    img_mod_params = self.img_mod(temb)  # [B, 6*dim]

    if self.zero_cond_t:
        temb = torch.chunk(temb, 2, dim=0)[0]
    txt_mod_params = self.txt_mod(temb)  # [B, 6*dim]

    # Split modulation parameters for norm1 and norm2
    img_mod1, img_mod2 = img_mod_params.chunk(2, dim=-1)  # Each [B, 3*dim]
    txt_mod1, txt_mod2 = txt_mod_params.chunk(2, dim=-1)  # Each [B, 3*dim]
    
    del img_mod_params, txt_mod_params
    
    # Process image stream - norm1 + modulation
    img_normed = self.img_norm1(hidden_states)
    img_modulated, img_gate1 = self._modulate(img_normed, img_mod1, modulate_index)
    del img_normed, img_mod1
    # Process text stream - norm1 + modulation
    txt_normed = self.txt_norm1(encoder_hidden_states)
    txt_modulated, txt_gate1 = self._modulate(txt_normed, txt_mod1)
    del txt_normed, txt_mod1
    # Use QwenAttnProcessor2_0 for joint attention computation
    # This directly implements the DoubleStreamLayerMegatron logic:
    # 1. Computes QKV for both streams
    # 2. Applies QK normalization and RoPE
    # 3. Concatenates and runs joint attention
    # 4. Splits results back to separate streams
    joint_attention_kwargs = joint_attention_kwargs or {}
    attn_output = self.attn(
        hidden_states=img_modulated,  # Image stream (will be processed as "sample")
        encoder_hidden_states=txt_modulated,  # Text stream (will be processed as "context")
        encoder_hidden_states_mask=encoder_hidden_states_mask,
        image_rotary_emb=image_rotary_emb,
        **joint_attention_kwargs,
    )
    del img_modulated, txt_modulated
    if QwenActivationCacheManager.should_reuse(QwenActivationCacheManager.current_step) and QwenActivationCacheManager.key_token_indices[QwenActivationCacheManager.current_mode] is not None:
        key_token_num = QwenActivationCacheManager.key_token_indices[QwenActivationCacheManager.current_mode].shape[0]
        # print("key_token_num:", key_token_num)
        # print("attn_output[0] shape before:", attn_output[0].shape)
        hidden_states = hidden_states[:, :key_token_num]
        img_gate1 = img_gate1[:, :key_token_num]
        if modulate_index is not None:
            modulate_index = modulate_index[:, :key_token_num]
        # print("img_mod2.shape",img_mod2.shape)
        # print("img_mod_params.shape",img_mod_params.shape)
        # print("temb.shape",temb.shape)
        # print("modulate_index.shape",modulate_index.shape)
    
    # img_mod2 = img_mod2[:, :key_token_num]
        # print("hidden_states shape after:", hidden_states.shape)
        # print("attn_output[0] shape before:", attn_output[0].shape)
        # print("attn_output[1] shape before:", attn_output[1].shape)
        
    
    
    # QwenAttnProcessor2_0 returns (img_output, txt_output) when encoder_hidden_states is provided
    img_attn_output, txt_attn_output = attn_output

    # if QwenActivationCacheManager.current_round > 0 :
    #     print("hidden_states shape after:", hidden_states.shape)
    #     print("img_attn_output shape before:", img_attn_output.shape)
    #     print("image_gate1 shape before:", img_gate1.shape)

    # Apply attention gates and add residual (like in Megatron)
    hidden_states = hidden_states + img_gate1 * img_attn_output
    encoder_hidden_states = encoder_hidden_states + txt_gate1 * txt_attn_output
    del img_gate1, txt_gate1, img_attn_output, txt_attn_output
    
    # Process image stream - norm2 + MLP
    img_normed2 = self.img_norm2(hidden_states)
    img_modulated2, img_gate2 = self._modulate(img_normed2, img_mod2, modulate_index)
    del img_normed2, img_mod2
    
    img_mlp_output = self.img_mlp(img_modulated2)
    hidden_states = hidden_states + img_gate2 * img_mlp_output
    del img_gate2, img_modulated2, img_mlp_output


    # Process text stream - norm2 + MLP
    txt_normed2 = self.txt_norm2(encoder_hidden_states)
    txt_modulated2, txt_gate2 = self._modulate(txt_normed2, txt_mod2)
    del txt_normed2, txt_mod2
    
    txt_mlp_output = self.txt_mlp(txt_modulated2)
    encoder_hidden_states = encoder_hidden_states + txt_gate2 * txt_mlp_output
    del txt_gate2, txt_modulated2, txt_mlp_output
    
    # Clip to prevent overflow for fp16
    if encoder_hidden_states.dtype == torch.float16:
        encoder_hidden_states = encoder_hidden_states.clip(-65504, 65504)
    if hidden_states.dtype == torch.float16:
        hidden_states = hidden_states.clip(-65504, 65504)

    return encoder_hidden_states, hidden_states

# class CacheEQwenDoubleStreamAttnProcessor2_0:
#     """
#     Attention processor for Qwen double-stream architecture, matching DoubleStreamLayerMegatron logic.
#     Computes joint attention for text (encoder_hidden_states) and image (hidden_states) streams.
#     """

#     _attention_backend = None
#     _parallel_config = None

#     def __init__(self):
#         if not hasattr(F, "scaled_dot_product_attention"):
#             raise ImportError(
#                 "QwenDoubleStreamAttnProcessor2_0 requires PyTorch 2.0, "
#                 "please upgrade PyTorch to 2.0 or later."
#             )

#     def __call__(
#         self,
#         attn: Attention,
#         hidden_states: torch.FloatTensor,              # image stream: (B, L_img, D)
#         encoder_hidden_states: torch.FloatTensor = None,  # text stream: (B, L_txt, D)
#         encoder_hidden_states_mask: torch.FloatTensor = None,
#         attention_mask: Optional[torch.FloatTensor] = None,
#         image_rotary_emb: Optional[torch.Tensor] = None,
#     ) -> torch.FloatTensor:
#         # print("i'm QwenDoubleStreamAttnProcessor2_0")
#         if encoder_hidden_states is None:
#             raise ValueError(
#                 "QwenDoubleStreamAttnProcessor2_0 requires encoder_hidden_states (text stream)"
#             )

#         bsz, seq_img, _ = hidden_states.shape
#         _, seq_txt, _ = encoder_hidden_states.shape
#         n_heads = attn.heads

#         # ---- 1. QKV 投影 ----
#         # Image stream
#         img_query = attn.to_q(hidden_states)    # (B, L_img, D)
#         img_key   = attn.to_k(hidden_states)    # (B, L_img, D)
#         img_value = attn.to_v(hidden_states)    # (B, L_img, D)

#         # Text stream
#         txt_query = attn.add_q_proj(encoder_hidden_states)  # (B, L_txt, D)
#         txt_key   = attn.add_k_proj(encoder_hidden_states)  # (B, L_txt, D)
#         txt_value = attn.add_v_proj(encoder_hidden_states)  # (B, L_txt, D)

#         # ---- 2. 变形为 (B, L, H, Hd)，尽量用 view/reshape 避免额外 copy ----
#         # 这里假设最后一维可以被 n_heads 整除
#         head_dim = img_query.shape[-1] // n_heads

#         img_query = img_query.view(bsz, seq_img, n_heads, head_dim)
#         img_key   = img_key.view  (bsz, seq_img, n_heads, head_dim)
#         img_value = img_value.view(bsz, seq_img, n_heads, head_dim)

#         txt_query = txt_query.view(bsz, seq_txt, n_heads, head_dim)
#         txt_key   = txt_key.view  (bsz, seq_txt, n_heads, head_dim)
#         txt_value = txt_value.view(bsz, seq_txt, n_heads, head_dim)

#         # ---- 3. QK 归一化 ----
#         if attn.norm_q is not None:
#             img_query = attn.norm_q(img_query)
#         if attn.norm_k is not None:
#             img_key = attn.norm_k(img_key)
#         if attn.norm_added_q is not None:
#             txt_query = attn.norm_added_q(txt_query)
#         if attn.norm_added_k is not None:
#             txt_key = attn.norm_added_k(txt_key)

#         # ---- 4. RoPE：原位覆盖 Q/K，避免额外中间张量 ----
#         if image_rotary_emb is not None:
#             img_freqs, txt_freqs = image_rotary_emb  # (L_img, D_rot), (L_txt, D_rot)

#             img_query = apply_rotary_emb_qwen(img_query, img_freqs, use_real=False)
#             img_key   = apply_rotary_emb_qwen(img_key,   img_freqs, use_real=False)
#             txt_query = apply_rotary_emb_qwen(txt_query, txt_freqs, use_real=False)
#             txt_key   = apply_rotary_emb_qwen(txt_key,   txt_freqs, use_real=False)

#         # ---- 5. 拼接成 joint QKV，按 [text, image] 顺序 ----
#         # 这里直接复用变量名，减少引用数量
#         joint_query = torch.cat([txt_query, img_query], dim=1)   # (B, L_txt+L_img, H, Hd)
#         joint_key   = torch.cat([txt_key,   img_key],   dim=1)
#         joint_value = torch.cat([txt_value, img_value], dim=1)

#         # 旧的单独变量已经不会再用，可以丢掉引用，有助于 allocator 复用显存
#         del img_query, img_key, img_value, txt_query, txt_key, txt_value

#         # ---- 6. 计算联合注意力 ----
#         joint_hidden_states = dispatch_attention_fn(
#             joint_query,
#             joint_key,
#             joint_value,
#             attn_mask=attention_mask,
#             dropout_p=0.0,
#             is_causal=False,
#             backend=self._attention_backend,
#             parallel_config=self._parallel_config,
#         )
#         # joint_hidden_states: (B, L_txt+L_img, H, Hd)

#         # 不再需要 joint_key/joint_value
#         del joint_key, joint_value

#         # ---- 7. 合并头维，恢复到 (B, L, D) ----
#         # flatten(2, 3) 等价于 view(bsz, L, -1)，但明确写 view 更利于避免多余 copy
#         joint_hidden_states = joint_hidden_states.view(
#             bsz, seq_txt + seq_img, n_heads * head_dim
#         )

#         # 只有 dtype 不同才做一次 cast，避免多余拷贝
#         if joint_hidden_states.dtype != joint_query.dtype:
#             joint_hidden_states = joint_hidden_states.to(joint_query.dtype)

#         # 用完后 joint_query 可以释放引用
#         del joint_query

#         # ---- 8. 拆回 text / image 部分 ----
#         txt_attn_output = joint_hidden_states[:, :seq_txt, :]   # (B, L_txt, D)
#         img_attn_output = joint_hidden_states[:, seq_txt:, :]   # (B, L_img, D)

#         del joint_hidden_states

#         # ---- 9. 输出投影 ----
#         img_attn_output = attn.to_out[0](img_attn_output)
#         if len(attn.to_out) > 1:
#             img_attn_output = attn.to_out[1](img_attn_output)  # dropout

#         txt_attn_output = attn.to_add_out(txt_attn_output)

#         return img_attn_output, txt_attn_output


class CacheEQwenDoubleStreamAttnProcessor2_0:
    """
    Attention processor for Qwen double-stream architecture, matching DoubleStreamLayerMegatron logic. This processor
    implements joint attention computation where text and image streams are processed together.
    """

    _attention_backend = None
    _parallel_config = None

    def __init__(self):
        if not hasattr(F, "scaled_dot_product_attention"):
            raise ImportError(
                "QwenDoubleStreamAttnProcessor2_0 requires PyTorch 2.0, to use it, please upgrade PyTorch to 2.0."
            )
    def __call__(
        self,
        attn,
        hidden_states: torch.FloatTensor,  # Image stream
        encoder_hidden_states: torch.FloatTensor = None,  # Text stream
        encoder_hidden_states_mask: torch.FloatTensor = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        image_rotary_emb: Optional[torch.Tensor] = None,
    ) -> torch.FloatTensor:
        # if encoder_hidden_states is None: ... (保持原有检查)
        
        B, L_img, _ = hidden_states.shape
        Btxt, L_txt, _ = encoder_hidden_states.shape

        # 优化1：把 key_token_num 的判断提到最前面
        # 这样在计算 img_query 时就可以只计算需要的那一部分，避免浪费巨大显存
        if QwenActivationCacheManager.should_reuse(QwenActivationCacheManager.current_step) and \
           QwenActivationCacheManager.key_token_indices[QwenActivationCacheManager.current_mode] is not None:
            key_token_num = min(QwenActivationCacheManager.key_token_indices[QwenActivationCacheManager.current_mode].shape[0], L_img)
        else:
            key_token_num = L_img

        # 优化2：img_query 只投影需要的前 key_token_num 个 token
        # 原写法是 attn.to_q(hidden_states) 算全量再切片，这里直接切输入
        img_query = attn.to_q(hidden_states[:, :key_token_num, :])
        img_key = attn.to_k(hidden_states)
        img_value = attn.to_v(hidden_states)

        txt_query = attn.add_q_proj(encoder_hidden_states)
        txt_key = attn.add_k_proj(encoder_hidden_states)
        txt_value = attn.add_v_proj(encoder_hidden_states)

        # Reshape & Norm (逻辑保持不变)
        img_query = img_query.unflatten(-1, (attn.heads, -1))
        img_key = img_key.unflatten(-1, (attn.heads, -1))
        img_value = img_value.unflatten(-1, (attn.heads, -1))

        txt_query = txt_query.unflatten(-1, (attn.heads, -1))
        txt_key = txt_key.unflatten(-1, (attn.heads, -1))
        txt_value = txt_value.unflatten(-1, (attn.heads, -1))

        if attn.norm_q is not None: img_query = attn.norm_q(img_query)
        if attn.norm_k is not None: img_key = attn.norm_k(img_key)
        if attn.norm_added_q is not None: txt_query = attn.norm_added_q(txt_query)
        if attn.norm_added_k is not None: txt_key = attn.norm_added_k(txt_key)

        # 优化3：拼接后立刻 del 掉分量，防止显存翻倍
        # joint Q
        # 注意：img_query 已经是切过的形状了，直接 cat
        joint_query = torch.cat([txt_query, img_query], dim=1)
        del txt_query, img_query  # <--- 释放显存

        # joint K
        joint_key = torch.cat([txt_key, img_key], dim=1)
        del txt_key, img_key      # <--- 释放显存
        
        # joint V
        joint_value = torch.cat([txt_value, img_value], dim=1)
        del txt_value, img_value  # <--- 释放显存

        # RoPE (逻辑保持不变，只需适配新的 joint 变量名)
        if image_rotary_emb is not None:
            img_freqs, txt_freqs = image_rotary_emb
            
            # 准备 freqs
            freqs_q = torch.cat([txt_freqs[:L_txt], img_freqs[:key_token_num]], dim=0)
            freqs_k = torch.cat([txt_freqs[:L_txt], img_freqs[:L_img]], dim=0)

            # 直接对 joint 做 RoPE
            joint_query = apply_rotary_emb_qwen(joint_query, freqs_q, use_real=False)
            joint_key = apply_rotary_emb_qwen(joint_key, freqs_k, use_real=False)

        # Compute joint attention
        joint_hidden_states = dispatch_attention_fn(
            joint_query, joint_key, joint_value,
            attn_mask=attention_mask,
            dropout_p=0.0,
            is_causal=False,
            backend=self._attention_backend,
            parallel_config=self._parallel_config,
        )
        del joint_query, joint_key, joint_value  # <--- 释放显存
        # 后续处理保持不变...
        joint_hidden_states = joint_hidden_states.flatten(2, 3)
        joint_hidden_states = joint_hidden_states.to(hidden_states.dtype)

        txt_attn_output = joint_hidden_states[:, :L_txt, :]
        img_attn_output = joint_hidden_states[:, L_txt:, :]
        del joint_hidden_states # <--- 释放大张量

        img_attn_output = attn.to_out[0](img_attn_output)
        if len(attn.to_out) > 1:
            img_attn_output = attn.to_out[1](img_attn_output)

        txt_attn_output = attn.to_add_out(txt_attn_output)

        return img_attn_output, txt_attn_output


    # def __call__(
    #     self,
    #     attn: Attention,
    #     hidden_states: torch.FloatTensor,  # Image stream
    #     encoder_hidden_states: torch.FloatTensor = None,  # Text stream
    #     encoder_hidden_states_mask: torch.FloatTensor = None,
    #     attention_mask: Optional[torch.FloatTensor] = None,
    #     image_rotary_emb: Optional[torch.Tensor] = None,
    # ) -> torch.FloatTensor:
    #     # print("i'm QwenDoubleStreamAttnProcessor2_0")
    #     if encoder_hidden_states is None:
    #         raise ValueError("QwenDoubleStreamAttnProcessor2_0 requires encoder_hidden_states (text stream)")
    #     B, L_img, _ = hidden_states.shape
    #     Btxt, L_txt, _ = encoder_hidden_states.shape

    #     # Compute QKV for image stream (sample projections)
    #     img_query = attn.to_q(hidden_states)
    #     img_key = attn.to_k(hidden_states)
    #     img_value = attn.to_v(hidden_states)

    #     # Compute QKV for text stream (context projections)
    #     txt_query = attn.add_q_proj(encoder_hidden_states)
    #     txt_key = attn.add_k_proj(encoder_hidden_states)
    #     txt_value = attn.add_v_proj(encoder_hidden_states)

    #     # Reshape for multi-head attention
    #     img_query = img_query.unflatten(-1, (attn.heads, -1))
    #     img_key = img_key.unflatten(-1, (attn.heads, -1))
    #     img_value = img_value.unflatten(-1, (attn.heads, -1))

    #     txt_query = txt_query.unflatten(-1, (attn.heads, -1))
    #     txt_key = txt_key.unflatten(-1, (attn.heads, -1))
    #     txt_value = txt_value.unflatten(-1, (attn.heads, -1))

    #     # Apply QK normalization
    #     if attn.norm_q is not None:
    #         img_query = attn.norm_q(img_query)
    #     if attn.norm_k is not None:
    #         img_key = attn.norm_k(img_key)
    #     if attn.norm_added_q is not None:
    #         txt_query = attn.norm_added_q(txt_query)
    #     if attn.norm_added_k is not None:
    #         txt_key = attn.norm_added_k(txt_key)

    #     # ------ 关键：只对 image 前 key_token_num 个位置重算 Q ------
    #     if QwenActivationCacheManager.should_reuse(QwenActivationCacheManager.current_step) and \
    #        QwenActivationCacheManager.key_token_indices is not None:
    #         key_token_num = min(QwenActivationCacheManager.key_token_indices.shape[0], L_img)
    #     else:
    #         key_token_num = L_img

    #     # text Q 全部保留，image Q 只保留 prefix
    #     img_q_used = img_query[:, :key_token_num]  # [B, key_token_num, H, D]
    #     # joint Q: [text, img_prefix]
    #     joint_query = torch.cat([txt_query, img_q_used], dim=1)  # [B, L_txt + key_token_num, H, D]

    #     # joint K/V：text 全部 + image 全部
    #     joint_key = torch.cat([txt_key, img_key], dim=1)         # [B, L_txt + L_img, H, D]
    #     joint_value = torch.cat([txt_value, img_value], dim=1)

    #     # RoPE：image_rotary_emb = (img_freqs, txt_freqs)
    #     if image_rotary_emb is not None:
    #         img_freqs, txt_freqs = image_rotary_emb  # img_freqs: [L_img, D], txt_freqs: [L_txt, D]

    #         # 为 Q 准备 freqs: text + image_prefix
    #         freqs_q_txt = txt_freqs[:L_txt, :]                   # [L_txt, D]
    #         freqs_q_img = img_freqs[:key_token_num, :]           # [key_token_num, D]
    #         freqs_q = torch.cat([freqs_q_txt, freqs_q_img], dim=0)  # [L_txt + key_token_num, D]

    #         # 为 K 准备 freqs: text + full image
    #         freqs_k_txt = txt_freqs[:L_txt, :]
    #         freqs_k_img = img_freqs[:L_img, :]
    #         freqs_k = torch.cat([freqs_k_txt, freqs_k_img], dim=0)   # [L_txt + L_img, D]

    #         # apply_rotary_emb_qwen: 期望 [B, L, H, D_head], freqs [L, D_rope]
    #         joint_query = apply_rotary_emb_qwen(joint_query, freqs_q, use_real=False)
    #         joint_key = apply_rotary_emb_qwen(joint_key, freqs_k, use_real=False)
        
    #     # # Apply RoPE
    #     # if image_rotary_emb is not None:
    #     #     img_freqs, txt_freqs = image_rotary_emb
    #     #     img_query = apply_rotary_emb_qwen(img_query, img_freqs, use_real=False)
    #     #     img_key = apply_rotary_emb_qwen(img_key, img_freqs, use_real=False)
    #     #     txt_query = apply_rotary_emb_qwen(txt_query, txt_freqs, use_real=False)
    #     #     txt_key = apply_rotary_emb_qwen(txt_key, txt_freqs, use_real=False)

    #     # # Concatenate for joint attention
    #     # # Order: [text, image]
    #     # joint_query = torch.cat([txt_query, img_query], dim=1)
    #     # joint_key = torch.cat([txt_key, img_key], dim=1)
    #     # joint_value = torch.cat([txt_value, img_value], dim=1)

    #     # Compute joint attention
    #     joint_hidden_states = dispatch_attention_fn(
    #         joint_query,
    #         joint_key,
    #         joint_value,
    #         attn_mask=attention_mask,
    #         dropout_p=0.0,
    #         is_causal=False,
    #         backend=self._attention_backend,
    #         parallel_config=self._parallel_config,
    #     )

    #     # Reshape back
    #     joint_hidden_states = joint_hidden_states.flatten(2, 3)
    #     joint_hidden_states = joint_hidden_states.to(joint_query.dtype)

    #     # Split attention outputs back
    #     txt_attn_output = joint_hidden_states[:, :L_txt, :]  # Text part
    #     img_attn_output = joint_hidden_states[:, L_txt:, :]  # Image part

    #     # Apply output projections
    #     img_attn_output = attn.to_out[0](img_attn_output)
    #     if len(attn.to_out) > 1:
    #         img_attn_output = attn.to_out[1](img_attn_output)  # dropout

    #     txt_attn_output = attn.to_add_out(txt_attn_output)

    #     return img_attn_output, txt_attn_output


