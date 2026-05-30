import torch
import numpy as np
import torch.nn.functional as F
import os
import csv
import re
from typing import Optional, Union, List, Dict, Any, Callable, Tuple
import accelerate
import sys
from PIL import Image


from diffusers.models.embeddings import apply_rotary_emb
from diffusers.models.attention_processor import Attention
from diffusers.models.modeling_outputs import Transformer2DModelOutput
from diffusers.schedulers import FlowMatchEulerDiscreteScheduler
from diffusers.models.attention_dispatch import dispatch_attention_fn
from diffusers.models.transformers.transformer_flux import _get_qkv_projections
from diffusers.utils import (
    is_torch_xla_available,
    logging,
    USE_PEFT_BACKEND,
    scale_lora_layers,
    unscale_lora_layers,
)
from Flux_utils import (
    calculate_shift,
    retrieve_timesteps,
    ActivationCacheManager,
    stats_collector,
    visualize_key_tokens_on_image,
    PipelineImageInput,
    FluxPipelineOutput,
    FluxKontextPipeline,
    PREFERRED_KONTEXT_RESOLUTIONS,
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


def _append_key_token_ratio_with_edit_ratio(
    image_filename: Optional[str],
    cur_round: int,
    cur_step: int,
    key_token_num: int,
    img_token_len: int,
):
    if key_token_num < 0 or img_token_len <= 0:
        return

    key_token_ratio = float(key_token_num) / float(img_token_len)

    base_out_dir = "/home/dataset-local/chenxueqing/result/Flux/kontext-bench-test/CacheEdit"
    os.makedirs(base_out_dir, exist_ok=True)
    out_csv = os.path.join(base_out_dir, f"key_token_ratio_vs_edit_ratio_threshold_{ActivationCacheManager.threshold}.csv")

    image_id = None
    if image_filename:
        m = re.match(r"^(\d{4})", image_filename)
        if m:
            image_id = m.group(1)

    image_edit_ratio = ""
    ratio_gap = ""

    # 尝试读取你之前统计得到的图像编辑区域比例总表
    # 约定字段: id, round, changed_ratio (若无则尝试 changed_percent / 100)
    summary_candidates = [
        "/home/dataset-local/chenxueqing/result/Flux/kontext-bench-test/Flux_oringinal/multi-round/generation-sub-diff/summary_all.csv",
        "/home/dataset-local/chenxueqing/result/Flux/kontext-bench-test/Flux_oringinal/multi-round/generation-sub-diff-v2/summary_all.csv",
    ]
    summary_path = None
    for p in summary_candidates:
        if os.path.isfile(p):
            summary_path = p
            break

    if summary_path is not None and image_id is not None:
        try:
            with open(summary_path, "r", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    rid = str(row.get("id", "")).strip()
                    rround = str(row.get("round", "")).strip()
                    if rid == image_id and rround == str(cur_round):
                        if row.get("changed_ratio", "") not in (None, ""):
                            image_edit_ratio = float(row["changed_ratio"])
                        elif row.get("changed_percent", "") not in (None, ""):
                            image_edit_ratio = float(row["changed_percent"]) / 100.0
                        break
        except Exception:
            image_edit_ratio = ""

    if isinstance(image_edit_ratio, float):
        ratio_gap = key_token_ratio - image_edit_ratio

        # --- 去重: 同一 (image_id, round, step) 只保留一条 ---
    existing_keys = set()
    existing_rows = []
    if os.path.isfile(out_csv):
        try:
            with open(out_csv, "r", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                for r in reader:
                    k = (
                        str(r.get("image_id", "")).strip(),
                        str(r.get("round", "")).strip(),
                        str(r.get("step", "")).strip(),
                    )
                    existing_keys.add(k)
                    existing_rows.append(r)
        except Exception:
            existing_keys = set()
            existing_rows = []

    new_key = (
        image_id if image_id is not None else "",
        str(cur_round),
        str(cur_step),
    )
    if new_key in existing_keys:
        return  # 已存在，不重复写入

    fieldnames = [
        "image_id",
        "round",
        "step",
        "image_filename",
        "key_token_num",
        "img_token_len",
        "key_token_ratio",
        "image_edit_ratio",
        "ratio_gap_key_minus_edit",
        "edit_ratio_summary_path",
    ]

    row_obj = {
        "image_id": image_id if image_id is not None else "",
        "round": cur_round,
        "step": cur_step,
        "image_filename": image_filename if image_filename is not None else "",
        "key_token_num": key_token_num,
        "img_token_len": img_token_len,
        "key_token_ratio": key_token_ratio,
        "image_edit_ratio": image_edit_ratio,
        "ratio_gap_key_minus_edit": ratio_gap,
        "edit_ratio_summary_path": summary_path if summary_path is not None else "",
    }

    # 统一重写（避免并发乱序时越来越脏）
    merged = existing_rows + [row_obj]
    try:
        merged.sort(key=lambda r: (int(str(r.get("image_id", "0") or 0)),
                                   int(str(r.get("round", "0") or 0)),
                                   int(str(r.get("step", "0") or 0))))
    except Exception:
        pass

    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(merged)

    
def _infer_image_id_from_csv_by_round() -> str:
    """
    稳健规则（不依赖最后一行）：
    1) 扫描整个 CSV，统计每个 image_id 的最大 round
    2) 取数值最大的 image_id 作为 current_id
    3) 若 current_id 的 max_round >= 7，则返回 current_id + 1
       否则返回 current_id
    """
    out_csv = f"/home/dataset-local/chenxueqing/result/Flux/kontext-bench-test/CacheEdit/key_token_ratio_vs_edit_ratio_threshold_{ActivationCacheManager.threshold}.csv"

    if not os.path.isfile(out_csv):
        return "0000"

    id_to_max_round = {}
    try:
        with open(out_csv, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                rid = str(row.get("image_id", "")).strip()
                rround = str(row.get("round", "")).strip()
                if re.fullmatch(r"\d{4}", rid) and re.fullmatch(r"\d+", rround):
                    rr = int(rround)
                    if rid not in id_to_max_round or rr > id_to_max_round[rid]:
                        id_to_max_round[rid] = rr
    except Exception:
        return "0000"

    if not id_to_max_round:
        return "0000"

    current_id = max(id_to_max_round.keys(), key=lambda x: int(x))
    max_round_for_current = id_to_max_round[current_id]

    if max_round_for_current >= 7:
        return f"{int(current_id) + 1:04d}"
    return current_id


# gamma = torch.tensor([0.8352, 0.9986, 1.0090, 1.0097, 1.0161, 1.0152, 1.0160, 1.0173, 1.0177,
#         1.0199, 1.0213, 1.0203, 1.0257, 1.0236, 1.0235, 1.0278, 1.0302, 1.0311,
#         1.0352, 1.0371, 1.0391, 1.0459, 1.0498, 1.0581, 1.0693, 1.0866, 1.1090],
#        dtype=torch.float16)

def cache_edit_init(model_path, device):

    pipeline = CacheEFluxKontextPipeline.from_pretrained(model_path, torch_dtype=torch.bfloat16, device_map = "balanced")
    # pipeline.scheduler = RegionEFlowMatchEulerDiscreteScheduler.from_config(pipeline.scheduler.config)
    pipeline.transformer.forward = CacheEFluxTransformer2DModelforward.__get__(pipeline.transformer, pipeline.transformer.__class__)
    for block in pipeline.transformer.transformer_blocks:
        block.workspace = {}  # 【优化】初始化 workspace 用于 buffer 复用
        block.forward = CacheFluxTransformerBlockforward.__get__(block, block.__class__)
        block.attn.set_processor(CacheFluxAttnProcessor2_0())
    for block in pipeline.transformer.single_transformer_blocks:
        block.workspace = {}  # 【优化】初始化 workspace 用于 buffer 复用
        block.forward = CacheFluxSingleTransformerBlockforward.__get__(block, block.__class__)
        block.attn.set_processor(CacheFluxAttnProcessor2_0())
    return pipeline


class CacheEFluxKontextPipeline(FluxKontextPipeline):
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
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
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
        callback_on_step_end: Optional[Callable[[int, int, Dict], None]] = None,
        callback_on_step_end_tensor_inputs: List[str] = ["latents"],
        max_sequence_length: int = 512,
        max_area: int = 1024**2,
        _auto_resize: bool = True,
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
            prompt_2 (`str` or `List[str]`, *optional*):
                The prompt or prompts to be sent to `tokenizer_2` and `text_encoder_2`. If not defined, `prompt` is
                will be used instead.
            negative_prompt (`str` or `List[str]`, *optional*):
                The prompt or prompts not to guide the image generation. If not defined, one has to pass
                `negative_prompt_embeds` instead. Ignored when not using guidance (i.e., ignored if `true_cfg_scale` is
                not greater than `1`).
            negative_prompt_2 (`str` or `List[str]`, *optional*):
                The prompt or prompts not to guide the image generation to be sent to `tokenizer_2` and
                `text_encoder_2`. If not defined, `negative_prompt` is used in all the text-encoders.
            true_cfg_scale (`float`, *optional*, defaults to 1.0):
                When > 1.0 and a provided `negative_prompt`, enables true classifier-free guidance.
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
            guidance_scale (`float`, *optional*, defaults to 3.5):
                Embedded guidance scale is enabled by setting `guidance_scale` > 1. Higher `guidance_scale` encourages
                a model to generate images more aligned with prompt at the expense of lower image quality.

                Guidance-distilled models approximates true classifier-free guidance for `guidance_scale` > 1. Refer to
                the [paper](https://huggingface.co/papers/2210.03142) to learn more.
            num_images_per_prompt (`int`, *optional*, defaults to 1):
                The number of images to generate per prompt.
            generator (`torch.Generator` or `List[torch.Generator]`, *optional*):
                One or a list of [torch generator(s)](https://pytorch.org/docs/stable/generated/torch.Generator.html)
                to make generation deterministic.
            latents (`torch.FloatTensor`, *optional*):
                Pre-generated noisy latents, sampled from a Gaussian distribution, to be used as inputs for image
                generation. Can be used to tweak the same generation with different prompts. If not provided, a latents
                tensor will be generated by sampling using the supplied random `generator`.
            prompt_embeds (`torch.FloatTensor`, *optional*):
                Pre-generated text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt weighting. If not
                provided, text embeddings will be generated from `prompt` input argument.
            pooled_prompt_embeds (`torch.FloatTensor`, *optional*):
                Pre-generated pooled text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt weighting.
                If not provided, pooled text embeddings will be generated from `prompt` input argument.
            ip_adapter_image: (`PipelineImageInput`, *optional*):
                Optional image input to work with IP Adapters.
            ip_adapter_image_embeds (`List[torch.Tensor]`, *optional*):
                Pre-generated image embeddings for IP-Adapter. It should be a list of length same as number of
                IP-adapters. Each element should be a tensor of shape `(batch_size, num_images, emb_dim)`. If not
                provided, embeddings are computed from the `ip_adapter_image` input argument.
            negative_ip_adapter_image:
                (`PipelineImageInput`, *optional*): Optional image input to work with IP Adapters.
            negative_ip_adapter_image_embeds (`List[torch.Tensor]`, *optional*):
                Pre-generated image embeddings for IP-Adapter. It should be a list of length same as number of
                IP-adapters. Each element should be a tensor of shape `(batch_size, num_images, emb_dim)`. If not
                provided, embeddings are computed from the `ip_adapter_image` input argument.
            negative_prompt_embeds (`torch.FloatTensor`, *optional*):
                Pre-generated negative text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt
                weighting. If not provided, negative_prompt_embeds will be generated from `negative_prompt` input
                argument.
            negative_pooled_prompt_embeds (`torch.FloatTensor`, *optional*):
                Pre-generated negative pooled text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt
                weighting. If not provided, pooled negative_prompt_embeds will be generated from `negative_prompt`
                input argument.
            output_type (`str`, *optional*, defaults to `"pil"`):
                The output format of the generate image. Choose between
                [PIL](https://pillow.readthedocs.io/en/stable/): `PIL.Image.Image` or `np.array`.
            return_dict (`bool`, *optional*, defaults to `True`):
                Whether or not to return a [`~pipelines.flux.FluxPipelineOutput`] instead of a plain tuple.
            joint_attention_kwargs (`dict`, *optional*):
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
            max_sequence_length (`int` defaults to 512):
                Maximum sequence length to use with the `prompt`.
            max_area (`int`, defaults to `1024 ** 2`):
                The maximum area of the generated image in pixels. The height and width will be adjusted to fit this
                area while maintaining the aspect ratio.

        Examples:

        Returns:
            [`~pipelines.flux.FluxPipelineOutput`] or `tuple`: [`~pipelines.flux.FluxPipelineOutput`] if `return_dict`
            is True, otherwise a `tuple`. When returning a tuple, the first element is a list with the generated
            images.
        """
        
        multiple_of = self.vae_scale_factor * 2
        # 1. Preprocess image
        if image is not None and not (isinstance(image, torch.Tensor) and image.size(1) == self.latent_channels):
            img = image[0] if isinstance(image, list) else image
            image_height, image_width = self.image_processor.get_default_height_width(img)
            aspect_ratio = image_width / image_height
            if _auto_resize:
                # Kontext is trained on specific resolutions, using one of them is recommended
                _, image_width, image_height = min(
                    (abs(aspect_ratio - w / h), w, h) for w, h in PREFERRED_KONTEXT_RESOLUTIONS
                )
            image_width = image_width // multiple_of * multiple_of
            image_height = image_height // multiple_of * multiple_of
            image = self.image_processor.resize(image, image_height, image_width)
            image = self.image_processor.preprocess(image, image_height, image_width)
            height, width = image.shape[-2], image.shape[-1]
        else:
            height = height or self.default_sample_size * self.vae_scale_factor
            width = width  or self.default_sample_size * self.vae_scale_factor

            original_height, original_width = height, width
            aspect_ratio = width / height
            width = round((max_area * aspect_ratio) ** 0.5)
            height = round((max_area / aspect_ratio) ** 0.5)

            width = width // multiple_of * multiple_of
            height = height // multiple_of * multiple_of


        # if height != original_height or width != original_width:
        #     logger.warning(
        #         f"Generation `height` and `width` have been adjusted to {height} and {width} to fit the model requirements."
        #     )
        print("height and width used for generation:", height, width)

        
        # 1. Check inputs. Raise error if not correct
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

        # 2. Define call parameters
        if prompt is not None and isinstance(prompt, str):
            batch_size = 1
        elif prompt is not None and isinstance(prompt, list):
            batch_size = len(prompt)
        else:
            batch_size = prompt_embeds.shape[0]

        device = self._execution_device

        lora_scale = (
            self.joint_attention_kwargs.get("scale", None) if self.joint_attention_kwargs is not None else None
        )
        has_neg_prompt = negative_prompt is not None or (
            negative_prompt_embeds is not None and negative_pooled_prompt_embeds is not None
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

        # # 3. Preprocess image
        # if image is not None and not (isinstance(image, torch.Tensor) and image.size(1) == self.latent_channels):
        #     img = image[0] if isinstance(image, list) else image
        #     # img = img.resize((8 * width, 8 * height), Image.Resampling.LANCZOS)
        #     image_height, image_width = self.image_processor.get_default_height_width(img)
        #     aspect_ratio = image_width / image_height
        #     print("aspect_ratio of input image:", aspect_ratio)
        #     if _auto_resize:
        #         # Kontext is trained on specific resolutions, using one of them is recommended
        #         _, image_width, image_height = min(
        #             (abs(aspect_ratio - w / h), w, h) for w, h in PREFERRED_KONTEXT_RESOLUTIONS
        #         )
        #     image_width = image_width // multiple_of * multiple_of
        #     image_height = image_height // multiple_of * multiple_of
        #     print("input image resized to:", image_width, image_height)
        #     image = self.image_processor.resize(image, image_height, image_width)
        #     image = self.image_processor.preprocess(image, image_height, image_width)

        # 4. Prepare latent variables
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
            latent_ids = torch.cat([latent_ids, image_ids], dim=0)  # dim 0 is sequence dimension

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
        if self.transformer.config.guidance_embeds:
            guidance = torch.full([1], guidance_scale, device=device, dtype=torch.float32)
            guidance = guidance.expand(latents.shape[0])
        else:
            guidance = None

        if (ip_adapter_image is not None or ip_adapter_image_embeds is not None) and (
            negative_ip_adapter_image is None and negative_ip_adapter_image_embeds is None
        ):
            negative_ip_adapter_image = np.zeros((width, height, 3), dtype=np.uint8)
            negative_ip_adapter_image = [negative_ip_adapter_image] * self.transformer.encoder_hid_proj.num_ip_adapters

        elif (ip_adapter_image is None and ip_adapter_image_embeds is None) and (
            negative_ip_adapter_image is not None or negative_ip_adapter_image_embeds is not None
        ):
            ip_adapter_image = np.zeros((width, height, 3), dtype=np.uint8)
            ip_adapter_image = [ip_adapter_image] * self.transformer.encoder_hid_proj.num_ip_adapters

        if self.joint_attention_kwargs is None:
            self._joint_attention_kwargs = {}

        image_embeds = None
        negative_image_embeds = None
        if ip_adapter_image is not None or ip_adapter_image_embeds is not None:
            image_embeds = self.prepare_ip_adapter_image_embeds(
                ip_adapter_image,
                ip_adapter_image_embeds,
                device,
                batch_size * num_images_per_prompt,
            )
        if negative_ip_adapter_image is not None or negative_ip_adapter_image_embeds is not None:
            negative_image_embeds = self.prepare_ip_adapter_image_embeds(
                negative_ip_adapter_image,
                negative_ip_adapter_image_embeds,
                device,
                batch_size * num_images_per_prompt,
            )

        # 6. Denoising loop
        # We set the index here to remove DtoH sync, helpful especially during compilation.
        # Check out more details here: https://github.com/huggingface/diffusers/pull/11696
        self.scheduler.set_begin_index(0)
        with self.progress_bar(total=num_inference_steps) as progress_bar:
            for i, t in enumerate(timesteps):
                if self.interrupt:
                    continue
                
                # up date step/round info
                ActivationCacheManager.on_step_start(i)
                print("current_round: ", ActivationCacheManager.current_round, "current_step: ", ActivationCacheManager.current_step)
                self._current_timestep = t
                if image_embeds is not None:
                    self._joint_attention_kwargs["ip_adapter_image_embeds"] = image_embeds

                latent_model_input = latents
                if image_latents is not None:
                    latent_model_input = torch.cat([latents, image_latents], dim=1)
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
                        self._joint_attention_kwargs["ip_adapter_image_embeds"] = negative_image_embeds
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
                    noise_pred = neg_noise_pred + true_cfg_scale * (noise_pred - neg_noise_pred)

                # compute the previous noisy sample x_t -> x_t-1
                latents_dtype = latents.dtype
                noise_pred = noise_pred.to(latents.device)
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

        self._current_timestep = None

        if output_type == "latent":
            image = latents
        else:
            latents = self._unpack_latents(latents, height, width, self.vae_scale_factor)
            print("height and width", height, width)
            latents = (latents / self.vae.config.scaling_factor) + self.vae.config.shift_factor
            image = self.vae.decode(latents, return_dict=False)[0]
            image = self.image_processor.postprocess(image, output_type=output_type)
        image_height, image_width = self.image_processor.get_default_height_width(image[0])
        print("final aspect_ratio of output image:", image_width / image_height)
        print("output image size:", image_width, image_height)
        # Offload all models
        self.maybe_free_model_hooks()

        if not return_dict:
            print("output_type: not return_dict")
            return (image,)

        return FluxPipelineOutput(images=image)


def CacheEFluxTransformer2DModelforward(
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
) -> Union[torch.Tensor, Transformer2DModelOutput]:
    """
    The [`FluxTransformer2DModel`] forward method.

    Args:
        hidden_states (`torch.Tensor` of shape `(batch_size, image_sequence_length, in_channels)`):
            Input `hidden_states`.
        encoder_hidden_states (`torch.Tensor` of shape `(batch_size, text_sequence_length, joint_attention_dim)`):
            Conditional embeddings (embeddings computed from the input conditions such as prompts) to use.
        pooled_projections (`torch.Tensor` of shape `(batch_size, projection_dim)`): Embeddings projected
            from the embeddings of input conditions.
        timestep ( `torch.LongTensor`):
            Used to indicate denoising step.
        block_controlnet_hidden_states: (`list` of `torch.Tensor`):
            A list of tensors that if specified are added to the residuals of transformer blocks.
        joint_attention_kwargs (`dict`, *optional*):
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
    if joint_attention_kwargs is not None:
        joint_attention_kwargs = joint_attention_kwargs.copy()
        lora_scale = joint_attention_kwargs.pop("scale", 1.0)
    else:
        lora_scale = 1.0

    if USE_PEFT_BACKEND:
        # weight the lora layers by setting `lora_scale` for each PEFT layer
        scale_lora_layers(self, lora_scale)
    else:
        if joint_attention_kwargs is not None and joint_attention_kwargs.get("scale", None) is not None:
            logger.warning(
                "Passing `scale` via `joint_attention_kwargs` when not using the PEFT backend is ineffective."
            )

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

    if txt_ids.ndim == 3:
        logger.warning(
            "Passing `txt_ids` 3d torch.Tensor is deprecated."
            "Please remove the batch dimension and pass it as a 2d torch Tensor"
        )
        txt_ids = txt_ids[0]
    if img_ids.ndim == 3:
        logger.warning(
            "Passing `img_ids` 3d torch.Tensor is deprecated."
            "Please remove the batch dimension and pass it as a 2d torch Tensor"
        )
        img_ids = img_ids[0]

    ids = torch.cat((txt_ids, img_ids), dim=0)
    image_rotary_emb = self.pos_embed(ids)
    # cos, sin = image_rotary_emb
    txt_len = txt_ids.shape[0]
    
    hidden_states, image_rotary_emb, key_token_num, should_reuse = \
        ActivationCacheManager.maybe_rearrange_img_and_pe(
            img=hidden_states,                # 这里只包含 image tokens 的话就简单；如包含文本需要先切分
            pe=image_rotary_emb,              # (B, C, L_txt + L_img, ...)
            txt_len=txt_len,
        )
    print("should_reuse:", should_reuse)
    if joint_attention_kwargs is not None and "ip_adapter_image_embeds" in joint_attention_kwargs:
        ip_adapter_image_embeds = joint_attention_kwargs.pop("ip_adapter_image_embeds")
        ip_hidden_states = self.encoder_hid_proj(ip_adapter_image_embeds)
        joint_attention_kwargs.update({"ip_hidden_states": ip_hidden_states})

    # ---------------- transformer_blocks（对应 double_blocks） ----------------
    ActivationCacheManager.stream_type = "double"
    for index_block, block in enumerate(self.transformer_blocks):
        if torch.is_grad_enabled() and self.gradient_checkpointing:
            encoder_hidden_states, hidden_states = self._gradient_checkpointing_func(
                block,
                hidden_states,
                encoder_hidden_states,
                temb,
                image_rotary_emb,
                joint_attention_kwargs,
            )

        else:
            encoder_hidden_states, hidden_states = block(
                hidden_states=hidden_states,
                encoder_hidden_states=encoder_hidden_states,
                temb=temb,
                image_rotary_emb=image_rotary_emb,
                joint_attention_kwargs=joint_attention_kwargs,
            )
        # ---- 复用上一轮缓存 ----
        if should_reuse:
            prev = ActivationCacheManager.load_activation(
                stream="double",
                layer_idx=index_block,
                device=hidden_states.device,
            )
            
            if prev is not None and ActivationCacheManager.key_token_indices is not None:
                # 与你原始逻辑一致：去掉 prev 中的 key tokens，然后 concat 到当前 hidden_states
                mask_prev = ActivationCacheManager._prev_mask_cache
                # if (
                #     mask_prev is None
                #     or mask_prev.numel() != prev.size(1)
                #     or mask_prev.device != prev.device
                # ):
                #     key_token_indices = ActivationCacheManager.key_token_indices
                    
                #     if key_token_indices.device != prev.device:
                #         key_token_indices = key_token_indices.to(prev.device)
                #     mask_prev = torch.ones(prev.size(1), dtype=torch.bool, device=prev.device)
                #     mask_prev[key_token_indices] = False
                #     ActivationCacheManager._prev_mask_cache = mask_prev
                mask_prev = torch.ones(prev.size(1), dtype=torch.bool, device=prev.device)
                mask_prev[ActivationCacheManager.key_token_indices] = False
                
                prev = prev[:, mask_prev, :]
                hidden_states = torch.cat((hidden_states, prev), dim=1)
               
                # prev = prev[:, mask_prev, :]
                # cur_len = hidden_states.shape[1]
                # bg_len = prev.shape[1]
                # # 【优化】torch.cat 替换为 buffer copy
                # if not hasattr(self, "_reuse_double_buf"):
                #     self._reuse_double_buf = None
                # buf = self._reuse_double_buf
                # if (buf is None or buf.shape != (hidden_states.shape[0], cur_len + bg_len, hidden_states.shape[2])
                #         or buf.dtype != hidden_states.dtype or buf.device != hidden_states.device):
                #     buf = torch.empty((hidden_states.shape[0], cur_len + bg_len, hidden_states.shape[2]),
                #                     device=hidden_states.device, dtype=hidden_states.dtype)
                #     self._reuse_double_buf = buf
                # buf[:, :cur_len, :].copy_(hidden_states)
                # buf[:, cur_len:, :].copy_(prev)
                # hidden_states = buf
                

        # ---- 写入当前激活到 new_cache ----
        ActivationCacheManager.maby_store_activation(
            stream="double",
            layer_idx=index_block,
            tensor=hidden_states,  # 或只存 image 部分，看你想怎么用
        )

        # controlnet residual
        if controlnet_block_samples is not None:
            interval_control = len(self.transformer_blocks) / len(controlnet_block_samples)
            interval_control = int(np.ceil(interval_control))
            # For Xlabs ControlNet.
            if controlnet_blocks_repeat:
                hidden_states = (
                    hidden_states + controlnet_block_samples[index_block % len(controlnet_block_samples)]
                )
            else:
                hidden_states = hidden_states + controlnet_block_samples[index_block // interval_control]
    
    # ---------------- single_transformer_blocks（对应 single_blocks） ----------------
    ActivationCacheManager.stream_type = "single"
    for index_block, block in enumerate(self.single_transformer_blocks):
        if torch.is_grad_enabled() and self.gradient_checkpointing:
            encoder_hidden_states, hidden_states = self._gradient_checkpointing_func(
                block,
                hidden_states,
                encoder_hidden_states,
                temb,
                image_rotary_emb,
                joint_attention_kwargs,
            )

        else:
            encoder_hidden_states, hidden_states = block(
                hidden_states=hidden_states,
                encoder_hidden_states=encoder_hidden_states,
                temb=temb,
                image_rotary_emb=image_rotary_emb,
                joint_attention_kwargs=joint_attention_kwargs,
            )
        # ====== 复用上一轮 single-stream 缓存 ======
        if should_reuse:
            prev = ActivationCacheManager.load_activation(
                stream="single",
                layer_idx=index_block,
                device=hidden_states.device,
            )
            
            if prev is not None and ActivationCacheManager.key_token_indices is not None:
                mask_prev = torch.ones(prev.size(1), dtype=torch.bool, device=prev.device)
                mask_prev[ActivationCacheManager.key_token_indices] = False
                prev = prev[:, mask_prev, :]
                hidden_states = torch.cat((hidden_states, prev), dim=1)
            #     # 和你原来的逻辑一致：去掉 prev 中的 key tokens，再拼回当前 hidden_states
            #     mask_prev = None
            #     if (
            #         mask_prev is None
            #         or mask_prev.numel() != prev.size(1)
            #         or mask_prev.device != prev.device
            #     ):
            #         key_token_indices = ActivationCacheManager.key_token_indices
            #         if key_token_indices.device != prev.device:
            #             key_token_indices = key_token_indices.to(prev.device)
            #         mask_prev = torch.ones(prev.size(1), dtype=torch.bool, device=prev.device)
            #         mask_prev[key_token_indices] = False
            #         ActivationCacheManager._prev_mask_cache = mask_prev
            #     prev = prev[:, mask_prev, :]
            #     cur_len = hidden_states.shape[1]
            #     bg_len = prev.shape[1]
            #     total_len = cur_len + bg_len
            #     # 【优化】torch.cat 替换为 buffer copy
            #     if not hasattr(self, "_reuse_single_buf"):
            #         self._reuse_single_buf = None
            #     buf = self._reuse_single_buf
            #     if (buf is None or buf.shape != (hidden_states.shape[0], total_len, hidden_states.shape[2])
            #             or buf.dtype != hidden_states.dtype or buf.device != hidden_states.device):
            #         buf = torch.empty((hidden_states.shape[0], total_len, hidden_states.shape[2]),
            #                           device=hidden_states.device, dtype=hidden_states.dtype)
            #         self._reuse_single_buf = buf
            #     buf[:, :cur_len, :].copy_(hidden_states)
            #     buf[:, cur_len:, :].copy_(prev)
            #     hidden_states = buf
                

        # ====== 写入当前 single-block 激活到 new_cache ======
        ActivationCacheManager.maby_store_activation(
            stream="single",
            layer_idx=index_block,
            tensor=hidden_states,   
        )
        
        
        # controlnet residual
        if controlnet_single_block_samples is not None:
            interval_control = len(self.single_transformer_blocks) / len(controlnet_single_block_samples)
            interval_control = int(np.ceil(interval_control))
            hidden_states = hidden_states + controlnet_single_block_samples[index_block // interval_control]

    # ---- 根据需要更新 key_token_indices ----
    if (
        ActivationCacheManager.use_activation_cache
        and not ActivationCacheManager.is_round0
        and ActivationCacheManager.current_step in ActivationCacheManager.cache_steps
    ):
        ref_img = ActivationCacheManager.load_activation(
            stream="single",
            layer_idx=37,
            device=hidden_states.device,
            step=ActivationCacheManager.current_step,
        )
        ActivationCacheManager.update_key_token_indices(
            cur_img=hidden_states,
            ref_img=ref_img,
        )

        if ActivationCacheManager.key_token_indices is not None:
            img_token_len = hidden_states.shape[1] // 2
            cur_round = int(ActivationCacheManager.current_round)
            cur_step = int(ActivationCacheManager.current_step)

            try:
                if cur_step == 0:
                    gen_dir = "/home/dataset-local/chenxueqing/result/Flux/kontext-bench-test/CacheEdit/threshold_sweep_summary/threshold_0p97/generation"

                    # 按“round到7才换下一个id”规则推断当前image_id
                    image_id = _infer_image_id_from_csv_by_round()

                    # 仅匹配该image_id和当前round，避免总命中0000
                    pat = re.compile(
                        rf"^{re.escape(image_id)}_r{cur_round}_.*\.(png|jpg|jpeg|webp|bmp)$",
                        re.IGNORECASE,
                    )

                    files = []
                    if os.path.isdir(gen_dir):
                        for fn in os.listdir(gen_dir):
                            if pat.match(fn):
                                files.append(fn)

                    image_filename = f"{image_id}_r{cur_round}_unknown.png"
                    if files:
                        files.sort(key=lambda x: (0 if x.lower().endswith(".png") else 1, x))
                        img_path = os.path.join(gen_dir, files[0])
                        image_filename = os.path.basename(img_path)
                        vis_image = Image.open(img_path).convert("RGB")
                    else:
                        vis_image = None

                    vis_path = os.path.join(
                        "/home/dataset-local/chenxueqing/result/Flux/kontext-bench-test/CacheEdit",
                        f"key_token_vis_threshold_{ActivationCacheManager.threshold}",
                        f"{image_id}_round{cur_round}_step{cur_step}.png",
                    )
                    print("visualize key tokens on image, save to ", vis_path)

                    if vis_image is not None:
                        visualize_key_tokens_on_image(
                            key_token_indices=ActivationCacheManager.key_token_indices,
                            image=vis_image,
                            img_token_len=img_token_len,
                            save_path=vis_path,
                        )

                # 只统计 step 0，并且只统计图像token(index < img_token_len)
                
                    indices = ActivationCacheManager.key_token_indices.detach().flatten().to("cpu").long()
                    valid = indices[(indices >= 0) & (indices < img_token_len)]
                    key_token_num = int(valid.shape[0])

                    _append_key_token_ratio_with_edit_ratio(
                        image_filename=image_filename,
                        cur_round=cur_round,
                        cur_step=cur_step,
                        key_token_num=key_token_num,
                        img_token_len=int(img_token_len),
                    )
                    print(
                        f"[key-ratio@step0] logged: id={image_id}, "
                        f"round={cur_round}, step={cur_step}, "
                        f"key_img_tokens={key_token_num}/{img_token_len}, "
                        f"ratio={key_token_num / max(1, img_token_len):.6f}"
                    )

            except Exception as e:
                print("visualization/stat record failed for step ", ActivationCacheManager.current_step, " err=", e)
                pass

            
            
        # print("key_token_indices.shape: ", ActivationCacheManager.key_token_indices.shape)
        # print("updated key_token_indices: ", ActivationCacheManager.key_token_indices)

        
        # step = ActivationCacheManager.current_step
        # total_layer_num_double = len(self.transformer_blocks)
        # total_layer_num_single = len(self.single_transformer_blocks)
        # for layer_idx in range(total_layer_num_double):
        #     if ActivationCacheManager.use_activation_cache and not ActivationCacheManager.is_round0:
        #         # 加载 ref
        #         ref_img = ActivationCacheManager.load_key_token_ref(
        #             stream="double",
        #             layer_idx=layer_idx,
        #             device=hidden_states.device,
        #             step=step,
        #         )
        #         cur_img = ActivationCacheManager.load_key_token_cur(
        #             stream="double",
        #             layer_idx=layer_idx,
        #             device=hidden_states.device,
        #             step=step,
        #         )
        #         ActivationCacheManager.update_key_token_indices(
        #             cur_img=cur_img,
        #             ref_img=ref_img,
        #         )
        #         # 记录这个 double-block 的统计
        #         stats_collector.record(
        #             manager_cls=ActivationCacheManager,
        #             step=step,
        #             layer_idx=layer_idx,
        #             stream="double",
        #         )
        # # 2) 再跑 single 的所有 layer
        # for layer_idx in range(total_layer_num_single):
        #     if ActivationCacheManager.use_activation_cache and not ActivationCacheManager.is_round0:
        #         ref_img = ActivationCacheManager.load_key_token_ref(
        #             stream="single",
        #             layer_idx=layer_idx,
        #             device=hidden_states.device,
        #             step=step,
        #         )
        #         cur_img = ActivationCacheManager.load_key_token_cur(
        #             stream="single",
        #             layer_idx=layer_idx,
        #             device=hidden_states.device,
        #             step=step,
        #         )
        #         ActivationCacheManager.update_key_token_indices(
        #             cur_img=cur_img,
        #             ref_img=ref_img,
        #         )
        #         stats_collector.record(
        #             manager_cls=ActivationCacheManager,
        #             step=step,
        #             layer_idx=layer_idx,
        #             stream="single",
        #         )
        # if ActivationCacheManager.current_step == ActivationCacheManager.total_step_num -1:
        #     stats_collector.report()
        #     stats_collector.save_to_excel("/home/chenxueqing/image-edit-round-reuse/result/FluxKontext/analysis/flux_key_token_stats.xlsx")
    # ---- flush new_cache 到 prev_cache ----
    ActivationCacheManager.flush_new_cache_after_step()
    
    
    # ---- 恢复 token 原始顺序 ----
    hidden_states = ActivationCacheManager.maybe_restore_img_order(
        img=hidden_states,
    )
    # if ActivationCacheManager.key_token_indices is not None:
    #     print("key_token_num: ", ActivationCacheManager.key_token_indices.shape)
    hidden_states = self.norm_out(hidden_states, temb)
    output = self.proj_out(hidden_states)

    
    
    if USE_PEFT_BACKEND:
        # remove `lora_scale` from each PEFT layer
        unscale_lora_layers(self, lora_scale)

    if not return_dict:
        return (output,)
    

    return Transformer2DModelOutput(sample=output)



def CacheFluxTransformerBlockforward(
    self,
    hidden_states: torch.Tensor,
    encoder_hidden_states: torch.Tensor,
    temb: torch.Tensor,
    image_rotary_emb: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    joint_attention_kwargs: Optional[Dict[str, Any]] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    # print("i'm in CacheFluxTransformerBlockforward")
    norm_hidden_states, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.norm1(hidden_states, emb=temb)

    norm_encoder_hidden_states, c_gate_msa, c_shift_mlp, c_scale_mlp, c_gate_mlp = self.norm1_context(
        encoder_hidden_states, emb=temb
    )
    joint_attention_kwargs = joint_attention_kwargs or {}

    # Attention.
    attention_outputs = self.attn(
        hidden_states=norm_hidden_states,
        encoder_hidden_states=norm_encoder_hidden_states,
        image_rotary_emb=image_rotary_emb,
        **joint_attention_kwargs,
    )

    if len(attention_outputs) == 2:
        attn_output, context_attn_output = attention_outputs
    elif len(attention_outputs) == 3:
        attn_output, context_attn_output, ip_attn_output = attention_outputs

    # Process attention outputs for the `hidden_states`.
    attn_output = gate_msa.unsqueeze(1) * attn_output
    if ActivationCacheManager.should_reuse(ActivationCacheManager.current_step) and ActivationCacheManager.key_token_indices is not None:
        key_token_num = ActivationCacheManager.key_token_indices.shape[0]
        hidden_states = hidden_states[:, :key_token_num]
    hidden_states = hidden_states + attn_output

    norm_hidden_states = self.norm2(hidden_states)
    norm_hidden_states = norm_hidden_states * (1 + scale_mlp[:, None]) + shift_mlp[:, None]

    ff_output = self.ff(norm_hidden_states)
    ff_output = gate_mlp.unsqueeze(1) * ff_output

    hidden_states = hidden_states + ff_output
    if len(attention_outputs) == 3:
        hidden_states = hidden_states + ip_attn_output

    # Process attention outputs for the `encoder_hidden_states`.
    context_attn_output = c_gate_msa.unsqueeze(1) * context_attn_output
    encoder_hidden_states = encoder_hidden_states + context_attn_output

    norm_encoder_hidden_states = self.norm2_context(encoder_hidden_states)
    norm_encoder_hidden_states = norm_encoder_hidden_states * (1 + c_scale_mlp[:, None]) + c_shift_mlp[:, None]

    context_ff_output = self.ff_context(norm_encoder_hidden_states)
    encoder_hidden_states = encoder_hidden_states + c_gate_mlp.unsqueeze(1) * context_ff_output
    if encoder_hidden_states.dtype == torch.float16:
        encoder_hidden_states = encoder_hidden_states.clip(-65504, 65504)

    return encoder_hidden_states, hidden_states

def CacheFluxSingleTransformerBlockforward(
    self,
    hidden_states: torch.Tensor,
    encoder_hidden_states: torch.Tensor,
    temb: torch.Tensor,
    image_rotary_emb: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    joint_attention_kwargs: Optional[Dict[str, Any]] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    # print("i'm in CacheFluxSingleTransformerBlockforward")
    text_seq_len = encoder_hidden_states.shape[1]
    
    # 【优化】torch.cat 替换为 buffer copy
    if not hasattr(self, "workspace"):
        self.workspace = {}
    concat_key = "single_concat_in"
    B, Lt, C = encoder_hidden_states.shape
    Li = hidden_states.shape[1]
    buf = self.workspace.get(concat_key)
    if (buf is None or buf.shape != (B, Lt + Li, C)
            or buf.dtype != encoder_hidden_states.dtype
            or buf.device != encoder_hidden_states.device):
        buf = torch.empty((B, Lt + Li, C), device=encoder_hidden_states.device, dtype=encoder_hidden_states.dtype)
        self.workspace[concat_key] = buf
    buf[:, :Lt, :].copy_(encoder_hidden_states)
    buf[:, Lt:, :].copy_(hidden_states)
    hidden_states = buf
    
    residual = hidden_states
    norm_hidden_states, gate = self.norm(hidden_states, emb=temb)

    should_reuse = ActivationCacheManager.should_reuse(ActivationCacheManager.current_step)
    has_indices = ActivationCacheManager.key_token_indices is not None

    # 【优化】MLP 只计算关键 token，而不是先全量计算再裁剪
    if should_reuse and has_indices:
        key_token_num = ActivationCacheManager.key_token_indices.shape[0]
        calc_len = text_seq_len + key_token_num
        input_slice = norm_hidden_states[:, :calc_len]
        mlp_hidden_states = self.act_mlp(self.proj_mlp(input_slice))
    else:
        mlp_hidden_states = self.act_mlp(self.proj_mlp(norm_hidden_states))
    
    joint_attention_kwargs = joint_attention_kwargs or {}
    attn_output = self.attn(
        hidden_states=norm_hidden_states,
        image_rotary_emb=image_rotary_emb,
        **joint_attention_kwargs,
    )
    
    if should_reuse and has_indices:
        residual = residual[:, :calc_len]
    
    # 【优化】torch.cat 替换为 buffer copy
    fused_key = "single_fused"
    B, L, D_attn = attn_output.shape
    D_mlp = mlp_hidden_states.shape[2]
    D_fused = D_attn + D_mlp
    buf = self.workspace.get(fused_key)
    if (buf is None or buf.shape != (B, L, D_fused)
            or buf.dtype != attn_output.dtype or buf.device != attn_output.device):
        buf = torch.empty((B, L, D_fused), device=attn_output.device, dtype=attn_output.dtype)
        self.workspace[fused_key] = buf
    buf[:, :, :D_attn].copy_(attn_output)
    buf[:, :, D_attn:].copy_(mlp_hidden_states)
    hidden_states = buf
    
    gate = gate.unsqueeze(1)
    hidden_states = gate * self.proj_out(hidden_states)
    hidden_states = residual + hidden_states
    if hidden_states.dtype == torch.float16:
        hidden_states = hidden_states.clip(-65504, 65504)

    encoder_hidden_states, hidden_states = hidden_states[:, :text_seq_len], hidden_states[:, text_seq_len:]
    return encoder_hidden_states, hidden_states

class CacheFluxAttnProcessor2_0:
    _attention_backend = None
    _parallel_config = None

    def __init__(self):
        if not hasattr(F, "scaled_dot_product_attention"):
            raise ImportError(f"{self.__class__.__name__} requires PyTorch 2.0. Please upgrade your pytorch version.")

    def __call__(
        self,
        attn: "FluxAttention",
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        image_rotary_emb: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        # print("i'm in CacheFluxAttnProcessor2_0")
        should_reuse = ActivationCacheManager.should_reuse(ActivationCacheManager.current_step)
        has_indices = ActivationCacheManager.key_token_indices is not None

        # ---------- 1 双流模式 ----------
        if ActivationCacheManager.stream_type == "double":
            # 文本部分 Q/K/V（全量计算）
            encoder_query = attn.add_q_proj(encoder_hidden_states)
            encoder_key = attn.add_k_proj(encoder_hidden_states)
            encoder_value = attn.add_v_proj(encoder_hidden_states)

            encoder_query = encoder_query.unflatten(-1, (attn.heads, -1))
            encoder_key = encoder_key.unflatten(-1, (attn.heads, -1))
            encoder_value = encoder_value.unflatten(-1, (attn.heads, -1))

            encoder_query = attn.norm_added_q(encoder_query)
            encoder_key = attn.norm_added_k(encoder_key)

            # 图像部分：Q 只计算关键 token，K/V 全量计算
            img_len = hidden_states.shape[1]
            if should_reuse and has_indices:
                key_token_num = min(ActivationCacheManager.key_token_indices.shape[0], img_len)
                # 【优化】Q 只计算关键 token，而不是先全量计算再裁剪
                hidden_states_q = hidden_states[:, :key_token_num, :]
                query = attn.to_q(hidden_states_q)
                # K/V 必须全量计算
                key = attn.to_k(hidden_states)
                value = attn.to_v(hidden_states)
                # print(f"Using cache with {key_token_num} key tokens for Q computation.")
                # print(f"hidden_states.shape: {hidden_states.shape},hidden_states_q.shape: {hidden_states_q.shape}, query.shape: {query.shape}, key.shape: {key.shape}, value.shape: {value.shape}")
            else:
                query = attn.to_q(hidden_states)
                key = attn.to_k(hidden_states)
                value = attn.to_v(hidden_states)

            query = query.unflatten(-1, (attn.heads, -1))
            key = key.unflatten(-1, (attn.heads, -1))
            value = value.unflatten(-1, (attn.heads, -1))

            query = attn.norm_q(query)
            key = attn.norm_k(key)

            # 拼接 Q/K/V
            full_query = torch.cat([encoder_query, query], dim=1)
            full_key = torch.cat([encoder_key, key], dim=1)
            full_value = torch.cat([encoder_value, value], dim=1)

            # RoPE：Q 只用 Q 长度的前缀位置编码，K/V 用全长
            if image_rotary_emb is not None:
                cos, sin = image_rotary_emb
                # 截断 RoPE for Query
                cos_q = cos[:full_query.shape[1], :]
                sin_q = sin[:full_query.shape[1], :]
                full_query = apply_rotary_emb(full_query, (cos_q, sin_q), sequence_dim=1)
                full_key = apply_rotary_emb(full_key, image_rotary_emb, sequence_dim=1)

            attn_output = dispatch_attention_fn(
                full_query, full_key, full_value,
                attn_mask=attention_mask,
                backend=self._attention_backend,
                parallel_config=self._parallel_config,
            )
            attn_output = attn_output.flatten(2, 3).to(query.dtype)

            txt_len = encoder_hidden_states.shape[1]

            # Split
            txt_out = attn_output[:, :txt_len, :]
            img_out = attn_output[:, txt_len:, :]

            # Proj Output
            img_out = attn.to_out[0](img_out)
            img_out = attn.to_out[1](img_out)

            txt_out = attn.to_add_out(txt_out)

            return img_out, txt_out

        # ---------- 2 单流模式 ----------
        else:
            if should_reuse and has_indices:
                key_token_num = ActivationCacheManager.key_token_indices.shape[0]
                txt_len = 512
                calc_len = txt_len + key_token_num

                # 【优化】Q 只计算关键 token，而不是先全量计算再裁剪
                input_q = hidden_states[:, :calc_len, :]
                query = attn.to_q(input_q)
                # K/V 全量计算
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
                query, key, value,
                attn_mask=attention_mask,
                backend=self._attention_backend,
                parallel_config=self._parallel_config,
            )
            attn_output = attn_output.flatten(2, 3).to(query.dtype)

            return attn_output
