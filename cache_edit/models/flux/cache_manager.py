"""Flux activation cache manager implementation."""

from bisect import bisect_right
from typing import Dict, List, Optional, Set, Tuple

import torch
from torch import Tensor

from cache_edit.core.cache_manager import BaseCacheManager, StreamType


class FluxCacheManager(BaseCacheManager):
    """
    Flux 模型的激活缓存管理器。

    相比 Qwen 版本，Flux 版本特有的功能：
    - 多轮编辑管理（current_round、is_round0、should_reuse）
    - 稀疏 cache_steps + map_to_group_min 就近复用
    - key_token 选择（余弦相似度 < threshold）
    - img + RoPE PE 同步重排
    - token 顺序恢复
    - 多 GPU 显存动态分配
    - stream_type 切换（double / single）
    - 每 step 末按策略 flush_new_cache_after_step
    - 与原始代码 1:1 行为对齐的 reset()

    Attributes:
        stream_type: 当前 stream 模式（"double" 或 "single"），由 Transformer forward 在循环前设置
        num_gpus: GPU 数量，用于显存动态分配
    """

    def __init__(
        self,
        use_activation_cache: bool = True,
        cache_steps: Optional[Set[int]] = None,
        cache_device: torch.device = torch.device("cuda:0"),
        num_gpus: int = 1,
        total_step_num: int = 30,
        threshold: float = 0.97,
        cache_interval: int = 5,
        gpu_memory_limit_gb: Optional[float] = None,
        gpu_memory_buffer_gb: float = 1.0,
    ):
        """
        初始化 Flux 缓存管理器。

        Args:
            use_activation_cache: 总开关
            cache_steps: 显式 cache_steps 集合，None 时根据 cache_interval 自动生成
            cache_device: 起始缓存设备
            num_gpus: 可用 GPU 数量
            total_step_num: 总推理步数
            threshold: key_token 选择的余弦相似度阈值
            cache_interval: 缓存间隔（步数）；值越小缓存越密集，显存占用越大
            gpu_memory_limit_gb: 每张 GPU 的显存上限（GB），None 时自动查询
            gpu_memory_buffer_gb: 显存预留 buffer（GB），防止 OOM
        """
        super().__init__(
            use_activation_cache=use_activation_cache,
            cache_steps=cache_steps,
            cache_device=cache_device,
            total_step_num=total_step_num,
            threshold=threshold,
            cache_interval=cache_interval,
        )

        self.num_gpus = num_gpus
        self.gpu_memory_limit_gb = gpu_memory_limit_gb
        self.gpu_memory_buffer_gb = gpu_memory_buffer_gb
        self.stream_type: StreamType = "single"

        # 自动查询 GPU 显存上限
        if self.gpu_memory_limit_gb is None and torch.cuda.is_available():
            self._auto_detect_gpu_limits()

        # Flux 的缓存以 (stream, step, layer_idx) 为键，单层结构（无 cond/uncond 双模式）
        self.prev_cache: Dict[Tuple[StreamType, int, int], Tensor] = {}
        self.new_cache: Dict[Tuple[StreamType, int, int], Tensor] = {}

        # key_token 索引（Flux 单层结构）
        self.key_token_indices: Optional[Tensor] = None

        # PE 缓存与 mask 缓存（按 round 重置）
        self._rearranged_pe_cache: Optional[Tuple[Tensor, Tensor]] = None
        self._rearranged_pe_cache_version: int = -1
        self._restore_masks: Optional[Tuple[Tensor, Tensor]] = None
        self._prev_mask_cache: Optional[Tensor] = None
        self._key_indices_version: int = 0

        # 排序后的 cache_steps（供 bisect 使用）
        self._refresh_cache_steps_sorted()
        print(f"[FluxCacheManager] Initialized cache_steps: {self.cache_steps}")

    def _auto_detect_gpu_limits(self) -> None:
        """自动检测各 GPU 的显存总量，设置为 limit（留 buffer）。"""
        if self.num_gpus <= 0:
            return
        try:
            # 查询第一张卡的总显存作为默认值
            props = torch.cuda.get_device_properties(0)
            total_gb = props.total_memory / (1024**3)
            self.gpu_memory_limit_gb = total_gb
            print(
                f"[FluxCacheManager] Auto-detected GPU memory: {total_gb:.1f} GB "
                f"(buffer: {self.gpu_memory_buffer_gb:.1f} GB)"
            )
        except Exception as e:
            print(f"[FluxCacheManager] Failed to auto-detect GPU memory: {e}")
            self.gpu_memory_limit_gb = 77.0  # fallback

    # ---------- cache_steps 管理 ----------

    def _refresh_cache_steps_sorted(self) -> None:
        if self.cache_steps is None:
            self._cache_steps_sorted: Optional[List[int]] = None
        else:
            self._cache_steps_sorted = sorted(self.cache_steps)

    def _build_cache_steps_from_interval(self) -> None:
        super()._build_cache_steps_from_interval()
        self._refresh_cache_steps_sorted()

    def set_parameters(
        self,
        num_inference_steps: Optional[int] = None,
        threshold: Optional[float] = None,
        cache_device: Optional[torch.device] = None,
        cache_interval: Optional[int] = None,
        num_gpus: Optional[int] = None,
        gpu_memory_limit_gb: Optional[float] = None,
        gpu_memory_buffer_gb: Optional[float] = None,
    ) -> None:
        """
        动态更新参数（兼容关键字调用与 argparse 风格调用）。

        Args:
            num_inference_steps: 推理步数
            threshold: 相似度阈值
            cache_device: 缓存设备
            cache_interval: 缓存间隔（值越小缓存越密集）
            num_gpus: GPU 数量
            gpu_memory_limit_gb: GPU 显存上限（GB）
            gpu_memory_buffer_gb: 显存预留 buffer（GB）
        """
        super().set_parameters(
            num_inference_steps=num_inference_steps,
            threshold=threshold,
            cache_device=cache_device,
            cache_interval=cache_interval,
        )
        if num_gpus is not None:
            self.num_gpus = num_gpus
        if gpu_memory_limit_gb is not None:
            self.gpu_memory_limit_gb = gpu_memory_limit_gb
        if gpu_memory_buffer_gb is not None:
            self.gpu_memory_buffer_gb = gpu_memory_buffer_gb
        self._refresh_cache_steps_sorted()
        print(f"[FluxCacheManager] Updated cache_steps: {self.cache_steps}")

    def set_parameters_from_args(self, args) -> None:
        """从 argparse Namespace 设置参数（与原始 set_parameters(args) 一致）。"""
        self.set_parameters(
            num_inference_steps=getattr(args, "num_inference_steps", None),
            threshold=getattr(args, "threshold", None),
            cache_interval=getattr(args, "cache_interval", None),
            num_gpus=getattr(args, "num_gpus", None),
        )

    # ---------- 轮次 / step ----------

    def on_step_start(self, step: int) -> None:
        """每 step 起始钩子；step==0 时进入新一轮。"""
        super().on_step_start(step)
        if step == 0:
            self._rearranged_pe_cache = None
            self._rearranged_pe_cache_version = -1
            self._restore_masks = None
            self._prev_mask_cache = None

    def map_to_group_min(self, step: int) -> Optional[int]:
        """
        将任意 step 映射到 cache_steps 中 <= step 的最大值。

        Returns:
            该步骤所属的 cache 分组起点；若所有 cache_step 都 > step，则返回 None
        """
        if self.cache_steps is None:
            return step
        if not self._cache_steps_sorted:
            return None
        idx = bisect_right(self._cache_steps_sorted, step) - 1
        if idx < 0:
            return None
        return self._cache_steps_sorted[idx]

    def should_reuse(self, step: int) -> bool:
        """
        Flux 复用策略：非第一轮 且 当前 step 不需要写 cache → 复用。
        """
        return (not self.is_round0) and (not self.should_cache(step))

    # ---------- 多 GPU 显存管理 ----------

    def gpu_has_space(
        self, device: torch.device, extra_bytes: int, limit_gb: float
    ) -> bool:
        """检查指定 GPU 是否有足够空间（预留 buffer）。"""
        # Use memory_reserved (includes PyTorch's memory pool) instead of
        # memory_allocated (only active tensors) for more accurate check
        used = torch.cuda.memory_reserved(device)
        limit_bytes = int(limit_gb * 1024**3)
        buffer_bytes = int(self.gpu_memory_buffer_gb * 1024**3)
        available = limit_bytes - used - buffer_bytes
        return available >= extra_bytes

    def _select_device(self, extra_bytes: int) -> torch.device:
        """
        从 cache_device 起始逐块尝试有空间的 GPU；都满了则 fallback 到起始设备。
        """
        start_dev = self.cache_device
        if start_dev.type != "cuda":
            return start_dev

        start_idx = start_dev.index if start_dev.index is not None else 0
        limit_gb = self.gpu_memory_limit_gb or 77.0

        for dev_idx in range(start_idx, self.num_gpus):
            dev = torch.device(f"cuda:{dev_idx}")
            if self.gpu_has_space(dev, extra_bytes, limit_gb):
                return dev

        for dev_idx in range(0, start_idx):
            dev = torch.device(f"cuda:{dev_idx}")
            if self.gpu_has_space(dev, extra_bytes, limit_gb):
                return dev

        print(
            f"⚠ All GPUs 0..{self.num_gpus - 1} are almost full, "
            f"falling back to {start_dev} (may OOM)"
        )
        return start_dev

    # ---------- 激活读写 ----------

    def store_activation(
        self,
        stream: StreamType,
        layer_idx: int,
        tensor: Tensor,
    ) -> None:
        """
        简单版存储（直接用 cache_device，无多卡逻辑）。
        """
        if not (self.use_activation_cache and self.should_cache(self.current_step)):
            return
        key = (stream, self.current_step, layer_idx)
        self.new_cache[key] = tensor.to(self.cache_device)

    def maby_store_activation(
        self,
        stream: StreamType,
        layer_idx: int,
        tensor: Tensor,
    ) -> None:
        """
        多卡智能存储：从 cache_device 起始依次尝试，无空间则 fallback。
        """
        if not (self.use_activation_cache and self.should_cache(self.current_step)):
            return
        key = (stream, self.current_step, layer_idx)
        extra_bytes = tensor.numel() * tensor.element_size()
        target_device = self._select_device(extra_bytes)

        # Debug: log cross-GPU placement
        if target_device != self.cache_device:
            print(
                f"[Cache] step={self.current_step} layer={layer_idx} "
                f"→ {target_device} (overflow from {self.cache_device})"
            )

        self.new_cache[key] = tensor.to(target_device)

    def get_activation(
        self,
        stream: StreamType,
        layer_idx: int,
        step: Optional[int] = None,
    ) -> Optional[Tensor]:
        """从 prev_cache 读取（不做设备转换）。"""
        if not self.use_activation_cache:
            return None
        step = step if step is not None else self.current_step
        step_to_load = self.map_to_group_min(step)
        key = (stream, step_to_load, layer_idx)
        return self.prev_cache.get(key, None)

    def load_activation(
        self,
        stream: StreamType,
        layer_idx: int,
        device: torch.device,
        step: Optional[int] = None,
    ) -> Optional[Tensor]:
        """从 prev_cache 读取并搬到指定设备（用 map_to_group_min 映射）。"""
        if not self.use_activation_cache:
            return None
        step = step if step is not None else self.current_step
        step_to_load = self.map_to_group_min(step)
        key = (stream, step_to_load, layer_idx)
        prev = self.prev_cache.get(key, None)
        if prev is None:
            return None
        return prev.to(device)

    def load_key_token_ref(
        self,
        stream: StreamType,
        layer_idx: int,
        device: torch.device,
        step: Optional[int] = None,
    ) -> Optional[Tensor]:
        """读取参考帧激活（不做 group_min 映射，直接按 step 查）。"""
        if not self.use_activation_cache:
            return None
        step = step if step is not None else self.current_step
        key = (stream, step, layer_idx)
        prev = self.prev_cache.get(key, None)
        if prev is None:
            return None
        return prev.to(device)

    def load_key_token_cur(
        self,
        stream: StreamType,
        layer_idx: int,
        device: torch.device,
        step: Optional[int] = None,
    ) -> Optional[Tensor]:
        """读取当前帧激活（从 new_cache）。"""
        if not self.use_activation_cache:
            return None
        step = step if step is not None else self.current_step
        key = (stream, step, layer_idx)
        cur = self.new_cache.get(key, None)
        if cur is None:
            return None
        return cur.to(device)

    def flush_new_cache_after_step(self) -> None:
        """
        每 step 末按策略合并 new_cache 到 prev_cache：
          - round0：若 should_cache(current_step) → 立即合并
          - round>0：若 should_cache(current_step + 1) 且未到最后一步 → 合并
        """
        last_step = self.total_step_num - 1
        if not self.use_activation_cache:
            return

        if self.is_round0:
            if self.should_cache(self.current_step):
                self.prev_cache.update(self.new_cache)
                self.new_cache.clear()
        else:
            if self.current_step >= last_step:
                return
            if self.should_cache(self.current_step + 1):
                self.prev_cache.update(self.new_cache)
                self.new_cache.clear()

    def flush_new_to_prev(self) -> None:
        """覆盖式 flush（直接替换）。"""
        self.prev_cache.update(self.new_cache)
        self.new_cache = {}

    # ---------- key_token 计算与重排 ----------

    def compute_key_indices_fn(
        self,
        tensor1: torch.Tensor,
        tensor2: torch.Tensor,
    ) -> torch.Tensor:
        """
        按行计算余弦相似度，归一化到 [0,1]，返回相似度 < threshold 的索引。

        Args:
            tensor1: (n, d)
            tensor2: (n, d)

        Returns:
            indices: 1D LongTensor
        """
        device = tensor2.device
        tensor1 = tensor1.to(device)
        assert tensor1.shape == tensor2.shape, "两个 tensor 的 shape 必须相同"

        dot_product = (tensor1 * tensor2).sum(dim=1)
        norm1 = tensor1.norm(p=2, dim=1)
        norm2 = tensor2.norm(p=2, dim=1)
        eps = 1e-8
        similarities = dot_product / (norm1 * norm2 + eps)
        normalized_similarities = (similarities + 1) / 2

        mask = normalized_similarities < self.threshold
        indices = mask.nonzero(as_tuple=True)[0]
        return indices

    def update_key_token_indices(
        self,
        cur_img: Tensor,
        ref_img: Optional[Tensor],
    ) -> None:
        """
        在第二轮及之后、当前 step 在 cache_steps 中、且有参考帧时更新 key_token_indices。
        """
        if not self.use_activation_cache:
            return
        if self.is_round0:
            return
        if self.cache_steps is None:
            return
        if self.current_step not in self.cache_steps:
            return
        if ref_img is None or cur_img is None:
            return
        indices = self.compute_key_indices_fn(cur_img[0], ref_img[0])
        self.key_token_indices = indices.to(cur_img.device)
        self._key_indices_version += 1

    @staticmethod
    def rearrange_tensor_with_key_token_indices(
        img: Tensor,
        cos_img: Tensor,
        sin_img: Tensor,
        key_token_indices: Tensor,
    ) -> Tuple[Tensor, Tensor, Tensor]:
        """
        把 key token 调到前面（img + cos/sin 同步重排），其余顺序保持。

        Args:
            img: (B, L_img, C)
            cos_img / sin_img: (L_img, D_pe)
            key_token_indices: 1D LongTensor

        Returns:
            (img_new, cos_img_new, sin_img_new)
        """
        img_key = torch.index_select(img, 1, key_token_indices)
        key_token_indices_pe = key_token_indices.to(cos_img.device)
        cos_img_key = torch.index_select(cos_img, 0, key_token_indices_pe)
        sin_img_key = torch.index_select(sin_img, 0, key_token_indices_pe)

        mask = torch.ones(img.size(1), dtype=torch.bool, device=img.device)
        mask[key_token_indices] = False
        img_not_key = img[:, mask, :]

        mask_pe = mask.to(cos_img.device)
        cos_img_not_key = cos_img[mask_pe, ...]
        sin_img_not_key = sin_img[mask_pe, ...]

        img_new = torch.cat((img_key, img_not_key), dim=1)
        cos_new = torch.cat((cos_img_key, cos_img_not_key), dim=0)
        sin_new = torch.cat((sin_img_key, sin_img_not_key), dim=0)
        return img_new, cos_new, sin_new

    @staticmethod
    def restore_original_token_order(
        x: Tensor,
        key_token_indices: Tensor,
    ) -> Tensor:
        """把前 K 个 token 放回到 key_token_indices 指定的位置。"""
        K = key_token_indices.size(0)
        x_key = x[:, :K, :]
        x_not_key = x[:, K:, :]
        L = x.size(1)

        mask_not_key = torch.ones(L, dtype=torch.bool, device=x.device)
        mask_not_key[key_token_indices] = False
        mask_key = ~mask_not_key

        out = x.clone()
        out[:, mask_not_key, :] = x_not_key
        out[:, mask_key, :] = x_key
        return out

    def precompute_masks(self, total_img_len: int, device: torch.device) -> None:
        """key_token_indices 更新后，预计算 restore mask 与 prev mask。"""
        if self.key_token_indices is None:
            return
        key_token_indices = self.key_token_indices.to(device)

        mask_not_key = torch.ones(total_img_len, dtype=torch.bool, device=device)
        mask_not_key[key_token_indices] = False
        self._restore_masks = (mask_not_key, ~mask_not_key)
        self._prev_mask_cache = mask_not_key.clone()

    def clear_pe_cache(self) -> None:
        """每轮起始或 key_token_indices 变更后清空 PE 重排缓存。"""
        self._rearranged_pe_cache = None
        self._rearranged_pe_cache_version = -1

    def _rearrange_img_only(self, img: Tensor) -> Tensor:
        """只重排 img（PE 已缓存场景下使用）。"""
        key_token_indices = self.key_token_indices
        if key_token_indices is None:
            return img
        if key_token_indices.device != img.device:
            key_token_indices = key_token_indices.to(img.device)
        img_key = torch.index_select(img, 1, key_token_indices)
        mask = torch.ones(img.size(1), dtype=torch.bool, device=img.device)
        mask[key_token_indices] = False
        img_not_key = img[:, mask, :]
        return torch.cat((img_key, img_not_key), dim=1)

    def maybe_rearrange_img_and_pe(
        self,
        img: Tensor,
        pe: Tuple[Tensor, Tensor],
        txt_len: int,
    ) -> Tuple[Tensor, Tuple[Tensor, Tensor], int, bool]:
        """
        若处于复用轮次且已有 key_token_indices，则对 img + PE 同步重排。

        Args:
            img: hidden states，包含 image tokens（不含文本，因为 transformer forward
                调用前 hidden_states 是 image-only）
            pe: (cos, sin)，每个 shape 为 (L_txt + L_img, D_pe)
            txt_len: 文本 token 数

        Returns:
            (img, pe, key_token_num, should_reuse)
        """
        should_reuse = self.should_reuse(self.current_step)

        cos, sin = pe

        if should_reuse and self.key_token_indices is not None:
            cos_txt, cos_img = cos[:txt_len, :], cos[txt_len:, :]
            sin_txt, sin_img = sin[:txt_len, :], sin[txt_len:, :]

            img, cos_img, sin_img = self.rearrange_tensor_with_key_token_indices(
                img, cos_img, sin_img, self.key_token_indices
            )

            cos = torch.cat((cos_txt, cos_img), dim=0)
            sin = torch.cat((sin_txt, sin_img), dim=0)

            pe = (cos, sin)
            key_token_num = self.key_token_indices.size(0)
        else:
            key_token_num = img.size(1)

        return img, pe, key_token_num, should_reuse

    def maybe_restore_img_order(self, img: Tensor) -> Tensor:
        """复用轮次结束时把 token 顺序还原。"""
        if (
            self.should_reuse(self.current_step)
            and self.key_token_indices is not None
        ):
            img = self.restore_original_token_order(img, self.key_token_indices)
        return img

    # ---------- 清理 ----------

    def clear_cache(self) -> None:
        """清空 prev/new cache 与 key_token_indices。"""
        self.prev_cache.clear()
        self.new_cache.clear()
        self.key_token_indices = None

    def reset(self) -> None:
        """
        恢复到初始状态：清空所有缓存、重置 round/step 计数、清空 key_token_indices。
        多图评估时每张图结束后调用。
        """
        self.prev_cache.clear()
        self.new_cache.clear()

        self.current_round = -1
        self.current_step = -1

        self.key_token_indices = None
        self._rearranged_pe_cache = None
        self._rearranged_pe_cache_version = -1
        self._restore_masks = None
        self._prev_mask_cache = None
        self._key_indices_version = 0

        print("[FluxCacheManager] reset to initial state.")

    # ---------- 统计 ----------

    def get_stats(self) -> Dict[str, object]:
        stats = super().get_stats()
        stats.update(
            {
                "num_gpus": self.num_gpus,
                "stream_type": self.stream_type,
                "prev_cache_keys": len(self.prev_cache),
                "new_cache_keys": len(self.new_cache),
                "has_key_token_indices": self.key_token_indices is not None,
            }
        )
        return stats
