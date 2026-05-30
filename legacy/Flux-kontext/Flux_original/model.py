from dataclasses import dataclass

import torch
from torch import Tensor, nn

from modules.layers import (
    DoubleStreamBlock,
    EmbedND,
    LastLayer,
    MLPEmbedder,
    SingleStreamBlock,
    timestep_embedding,
)
from modules.lora import LinearLora, replace_linear_with_lora

from mytools import replace_tensor_with_indices, read_indices_from_csv, get_key_token_indices, filter_diff_points_pure_torch
import os, sys


@dataclass
class FluxParams:
    in_channels: int
    out_channels: int
    vec_in_dim: int
    context_in_dim: int
    hidden_size: int
    mlp_ratio: float
    num_heads: int
    depth: int
    depth_single_blocks: int
    axes_dim: list[int]
    theta: int
    qkv_bias: bool
    guidance_embed: bool


class Flux(nn.Module):
    """
    Transformer model for flow matching on sequences.
    """

    def __init__(self, params: FluxParams):
        super().__init__()

        self.params = params
        self.in_channels = params.in_channels
        self.out_channels = params.out_channels
        if params.hidden_size % params.num_heads != 0:
            raise ValueError(
                f"Hidden size {params.hidden_size} must be divisible by num_heads {params.num_heads}"
            )
        pe_dim = params.hidden_size // params.num_heads
        if sum(params.axes_dim) != pe_dim:
            raise ValueError(f"Got {params.axes_dim} but expected positional dim {pe_dim}")
        self.hidden_size = params.hidden_size
        self.num_heads = params.num_heads
        self.pe_embedder = EmbedND(dim=pe_dim, theta=params.theta, axes_dim=params.axes_dim)
        self.img_in = nn.Linear(self.in_channels, self.hidden_size, bias=True)
        self.time_in = MLPEmbedder(in_dim=256, hidden_dim=self.hidden_size)
        self.vector_in = MLPEmbedder(params.vec_in_dim, self.hidden_size)
        self.guidance_in = (
            MLPEmbedder(in_dim=256, hidden_dim=self.hidden_size) if params.guidance_embed else nn.Identity()
        )
        self.txt_in = nn.Linear(params.context_in_dim, self.hidden_size)

        self.double_blocks = nn.ModuleList(
            [
                DoubleStreamBlock(
                    self.hidden_size,
                    self.num_heads,
                    mlp_ratio=params.mlp_ratio,
                    qkv_bias=params.qkv_bias,
                )
                for _ in range(params.depth)
            ]
        )

        self.single_blocks = nn.ModuleList(
            [
                SingleStreamBlock(self.hidden_size, self.num_heads, mlp_ratio=params.mlp_ratio)
                for _ in range(params.depth_single_blocks)
            ]
        )

        # activation caches kept on device between rounds
        self.prev_cache_double: dict[tuple[int, int], Tensor] = {}
        self.prev_cache_single: dict[tuple[int, int], Tensor] = {}
        self.new_cache_double: dict[tuple[int, int], Tensor] = {}
        self.new_cache_single: dict[tuple[int, int], Tensor] = {}

        self.key_token_indices: Tensor | None = None

        self.use_activation_cache: bool = True
        self.is_round0: bool = True
        self.round_num: int = -1

        # cache on another GPU
        self.cache_device: torch.device = "cuda:1"

        # None = cache all steps; otherwise only cache steps in this set
        self.cache_steps: set[int] | None = {0, 5, 10, 15, 20, 25}

        self.final_layer = LastLayer(self.hidden_size, 1, self.out_channels)

        # ======= for sorted cache & restoration =======
        self.cache_is_sorted: bool = False
        self.perm_cpu: Tensor | None = None
        self.inv_perm_cpu: Tensor | None = None
        self.bg_indices_cpu: Tensor | None = None

        # merge buffer only for single stream (double uses in-place update on prev_local)
        self.merge_buffer: dict[str, Tensor] = {}
        for blk in self.double_blocks:
            blk.workspace = self.merge_buffer
        for blk in self.single_blocks:
            blk.workspace = self.merge_buffer

        # keep original print
        print("params", params)

    # ---------------------------
    # helpers: scheduling/cache
    # ---------------------------
    def map_to_group_min(self, n: int, interval: int):
        group = n // interval
        return group * interval

    def should_cache(self, cur_step: int) -> bool:
        if not self.use_activation_cache:
            return False
        if self.cache_steps is None:
            return True
        return cur_step in self.cache_steps

    # ---------------------------
    # helpers: perm/inv_perm + reorder
    # ---------------------------
    def _compute_bg_indices(self, full_len: int, device: torch.device) -> Tensor:
        assert self.key_token_indices is not None
        key = self.key_token_indices.to(device=device, non_blocking=True)
        mask = torch.ones(full_len, dtype=torch.bool, device=device)
        mask[key] = False
        return torch.nonzero(mask, as_tuple=False).flatten()

    def _build_perm_inv(self, full_len: int, device: torch.device):
        key = self.key_token_indices.to(device=device, non_blocking=True)
        bg = self._compute_bg_indices(full_len, device=device)
        perm = torch.cat([key, bg], dim=0)  # sorted_pos -> orig_pos
        inv = torch.empty_like(perm)
        inv[perm] = torch.arange(full_len, device=device, dtype=perm.dtype)  # orig_pos -> sorted_pos

        self.perm_cpu = perm.detach().to("cpu")
        self.inv_perm_cpu = inv.detach().to("cpu")
        self.bg_indices_cpu = bg.detach().to("cpu")
        return perm, inv, bg

    def _restore_original_order(self, x_sorted: Tensor) -> Tensor:
        assert self.inv_perm_cpu is not None
        inv = self.inv_perm_cpu.to(device=x_sorted.device, non_blocking=True)
        return x_sorted.index_select(1, inv)

    def _unsort_all_caches_to_original(self):
        if not self.cache_is_sorted:
            return
        if self.inv_perm_cpu is None:
            self.cache_is_sorted = False
            return

        inv_cache = self.inv_perm_cpu.to(self.cache_device, non_blocking=True)

        for k, v in list(self.prev_cache_double.items()):
            self.prev_cache_double[k] = v.index_select(1, inv_cache).contiguous()

        for k, v in list(self.prev_cache_single.items()):
            self.prev_cache_single[k] = v.index_select(1, inv_cache).contiguous()

        self.cache_is_sorted = False
        self.perm_cpu = None
        self.inv_perm_cpu = None
        self.bg_indices_cpu = None
        self.merge_buffer.clear()

    def _sort_all_caches_to_key_bg(self, full_len: int, device_for_build: torch.device):
        if self.cache_is_sorted:
            return
        self._build_perm_inv(full_len=full_len, device=device_for_build)
        perm_cache = self.perm_cpu.to(self.cache_device, non_blocking=True)

        for k, v in list(self.prev_cache_double.items()):
            self.prev_cache_double[k] = v.index_select(1, perm_cache).contiguous()
        for k, v in list(self.prev_cache_single.items()):
            self.prev_cache_single[k] = v.index_select(1, perm_cache).contiguous()

        self.cache_is_sorted = True

    # ---------------------------
    # helpers: rearrange current tokens
    # ---------------------------
    def rearrange_tensor_with_key_token_indices(self, img: Tensor, pe_img: Tensor, key_token_indices: Tensor):
        img_key = torch.index_select(img, 1, key_token_indices)
        pe_key = torch.index_select(pe_img, 2, key_token_indices)

        mask = torch.ones(img.size(1), dtype=torch.bool, device=img.device)
        mask[key_token_indices] = False

        img_bg = img[:, mask, :]
        pe_bg = pe_img[:, :, mask, ...]

        img_sorted = torch.cat((img_key, img_bg), dim=1)
        pe_sorted = torch.cat((pe_key, pe_bg), dim=2)
        return img_sorted, pe_sorted

    # ---------------------------
    # forward
    # ---------------------------
    def forward(
        self,
        step: int,
        img: Tensor,
        img_ids: Tensor,
        txt: Tensor,
        txt_ids: Tensor,
        timesteps: Tensor,
        y: Tensor,
        guidance: Tensor | None = None,
    ) -> Tensor:
        if img.ndim != 3 or txt.ndim != 3:
            raise ValueError("Input img and txt tensors must have 3 dimensions.")

        # keep original prints (same location as your old code)
        print("self.prev_cache_double_size: ", len(self.prev_cache_double))
        print("self.prev_cache_single_size: ", len(self.prev_cache_single))

        # ---- round bookkeeping ----
        if step == 0:
            self.round_num += 1
            print("round_num: ", self.round_num)

            # new round: if previous round sorted cache, unsort first to avoid key mismatch
            if self.round_num > 0:
                self._unsort_all_caches_to_original()
                self.is_round0 = False

        is_round0 = self.is_round0

        # ---- embedding / vec ----
        img = self.img_in(img)
        vec = self.time_in(timestep_embedding(timesteps, 256))
        if self.params.guidance_embed:
            if guidance is None:
                raise ValueError("Didn't get guidance strength for guidance distilled model.")
            vec = vec + self.guidance_in(timestep_embedding(guidance, 256))
        vec = vec + self.vector_in(y)

        txt = self.txt_in(txt)

        ids = torch.cat((txt_ids, img_ids), dim=1)
        pe = self.pe_embedder(ids)

        # ---- reuse decision ----
        should_reuse = (not is_round0) and (not self.should_cache(step))

        key_token_num = img.size(1)

        # ---- if reuse: rearrange current img and sort caches once ----
        if should_reuse:
            assert self.key_token_indices is not None, "key_token_indices must be set before reuse"

            full_img_len = img_ids.shape[1]

            pe_img = pe[:, :, txt.shape[1]:, ...]
            img, pe_img_sorted = self.rearrange_tensor_with_key_token_indices(
                img, pe_img, self.key_token_indices.to(img.device, non_blocking=True)
            )
            pe = torch.cat((pe[:, :, :txt.shape[1], ...], pe_img_sorted), dim=2)

            key_token_num = int(self.key_token_indices.numel())

            # sort caches to [Key|BG] contiguous on cache_device
            self._sort_all_caches_to_key_bg(full_len=full_img_len, device_for_build=img.device)

        # ==========================
        # Double blocks
        # ==========================
        for double_layer, block in enumerate(self.double_blocks):
            img, txt = block(
                step=step,
                double_layer=double_layer,
                img=img,
                txt=txt,
                vec=vec,
                pe=pe,
                key_token_num=key_token_num,
                should_reuse=should_reuse,
            )

            if self.use_activation_cache:
                if should_reuse:
                    step_to_load = self.map_to_group_min(step, 5)
                    prev_index = (step_to_load, double_layer)

                    prev = self.prev_cache_double.get(prev_index)
                    if prev is not None:
                        # prev on cache_device, sorted [Key|BG]
                        if prev.device != img.device:
                            prev_local = prev.to(img.device, non_blocking=True)
                        else:
                            prev_local = prev

                        # in-place update: overwrite Key prefix with new Key
                        klen = img.shape[1]
                        prev_local[:, :klen, :].copy_(img)
                        img = prev_local

                if self.should_cache(step):
                    new_index = (step, double_layer)
                    img_to_save = img

                    if self.cache_is_sorted and (self.perm_cpu is not None) and (not should_reuse):
                        perm = self.perm_cpu.to(img.device, non_blocking=True)
                        img_to_save = img.index_select(1, perm).contiguous()

                    self.new_cache_double[new_index] = img_to_save.to(self.cache_device, non_blocking=True)

        # ==========================
        # Single blocks
        # ==========================
        img = torch.cat((txt, img), dim=1)

        for single_layer, block in enumerate(self.single_blocks):
            img = block(
                img,
                vec=vec,
                pe=pe,
                step=step,
                single_layer=single_layer,
                key_token_num=key_token_num,
                should_reuse=should_reuse,
            )

            if self.use_activation_cache:
                if should_reuse:
                    step_to_load = self.map_to_group_min(step, 5)
                    prev_index = (step_to_load, single_layer)

                    prev_img = self.prev_cache_single.get(prev_index)
                    if prev_img is not None:
                        if prev_img.device != img.device:
                            prev_img_local = prev_img.to(img.device, non_blocking=True)
                        else:
                            prev_img_local = prev_img

                        txt_len = txt.shape[1]
                        cur_len = img.shape[1]      # txt + key_img
                        key_len = key_token_num
                        total_len = cur_len + (prev_img_local.shape[1] - key_len)

                        buf = self.merge_buffer.get("single")
                        if (
                            buf is None
                            or buf.shape != (img.shape[0], total_len, img.shape[2])
                            or buf.dtype != img.dtype
                            or buf.device != img.device
                        ):
                            buf = torch.empty(
                                (img.shape[0], total_len, img.shape[2]),
                                dtype=img.dtype, device=img.device
                            )
                            self.merge_buffer["single"] = buf

                        buf[:, :cur_len, :].copy_(img)  # txt + new key
                        buf[:, cur_len:, :].copy_(prev_img_local[:, key_len:, :])  # old BG
                        img = buf

                if self.should_cache(step):
                    new_index = (step, single_layer)
                    img_to_save = img[:, txt.shape[1]:, ...]

                    if self.cache_is_sorted and (self.perm_cpu is not None) and (not should_reuse):
                        perm = self.perm_cpu.to(img_to_save.device, non_blocking=True)
                        img_to_save = img_to_save.index_select(1, perm).contiguous()

                    self.new_cache_single[new_index] = img_to_save.to(self.cache_device, non_blocking=True)

        img = img[:, txt.shape[1]:, ...]

        # ==========================
        # update key_token_indices (same print as your old code)
        # ==========================
        if self.use_activation_cache and (not is_round0) and (step == 0):
            prev_img = self.prev_cache_single.get((step, 37))
            if prev_img is not None:
                if prev_img.device != img.device:
                    prev_img_local = prev_img.to(img.device, non_blocking=True)
                else:
                    prev_img_local = prev_img

                key_token_indices = get_key_token_indices(img[0], prev_img_local[0], 0.975).to(img.device)
                self.key_token_indices = key_token_indices
                print("key_token_indices.shape: ", key_token_indices.shape)

                # key changed => force caches to be treated as unsorted next time
                self.cache_is_sorted = False
                self.perm_cpu = None
                self.inv_perm_cpu = None
                self.bg_indices_cpu = None
                self.merge_buffer.clear()

        # ==========================
        # update cache dicts (same as your original logic)
        # ==========================
        if self.use_activation_cache:
            if is_round0:
                if self.should_cache(step):
                    self.prev_cache_double.update(self.new_cache_double)
                    self.prev_cache_single.update(self.new_cache_single)
                    self.new_cache_double = {}
                    self.new_cache_single = {}
            else:
                if self.should_cache(step + 1) and step < 29:
                    self.prev_cache_double.update(self.new_cache_double)
                    self.prev_cache_single.update(self.new_cache_single)
                    self.new_cache_double = {}
                    self.new_cache_single = {}

        # ==========================
        # restore original order (no clone; keep behavior)
        # ==========================
        if should_reuse:
            img = self._restore_original_order(img)

        img = self.final_layer(img, vec)
        return img


class FluxLoraWrapper(Flux):
    def __init__(
        self,
        lora_rank: int = 128,
        lora_scale: float = 1.0,
        *args,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)

        self.lora_rank = lora_rank

        replace_linear_with_lora(
            self,
            max_rank=lora_rank,
            scale=lora_scale,
        )

    def set_lora_scale(self, scale: float) -> None:
        for module in self.modules():
            if isinstance(module, LinearLora):
                module.set_scale(scale=scale)
