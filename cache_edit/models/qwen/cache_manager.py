"""Qwen cache manager implementation."""

from typing import Optional, Dict, Tuple, Set
import torch
from torch import Tensor

from cache_edit.core.cache_manager import BaseCacheManager, StreamType


class QwenCacheManager(BaseCacheManager):
    """
    Qwen 模型的缓存管理器。

    相比基类，增加了以下特性：
    - 支持 cond/uncond 双模式缓存
    - 多 GPU 显存管理
    - 关键 token 索引管理

    Attributes:
        current_mode: 当前模式（"cond" 或 "uncond"）
        cache_device2: 第二块缓存设备
        num_gpus: GPU 数量
    """

    def __init__(
        self,
        use_activation_cache: bool = True,
        cache_steps: Optional[Set[int]] = None,
        cache_device: torch.device = torch.device("cuda"),
        cache_device2: Optional[torch.device] = None,
        total_step_num: int = 30,
        num_gpus: int = 2,
        threshold: float = 0.99,
        cache_interval: int = 5,
    ):
        """
        初始化 Qwen 缓存管理器。

        Args:
            use_activation_cache: 是否启用缓存
            cache_steps: 缓存步骤集合
            cache_device: 主缓存设备
            cache_device2: 第二缓存设备（可选）
            total_step_num: 总步数
            num_gpus: GPU 数量
            threshold: 相似度阈值
            cache_interval: 缓存间隔
        """
        super().__init__(
            use_activation_cache=use_activation_cache,
            cache_steps=cache_steps,
            cache_device=cache_device,
            total_step_num=total_step_num,
            threshold=threshold,
            cache_interval=cache_interval,
        )

        # Qwen 特有属性
        self.cache_device2 = cache_device2
        self.num_gpus = num_gpus
        self.current_mode: str = "cond"  # "cond" or "uncond"
        self.get_key_token_indices_step = 0

        # 按 mode 分两套缓存（用于 KV 缓存模式）
        self.prev_cache: Dict[str, Dict[Tuple[StreamType, int, int], Tensor]] = {
            "cond": {},
            "uncond": {},
        }
        self.new_cache: Dict[str, Dict[Tuple[StreamType, int, int], Tensor]] = {
            "cond": {},
            "uncond": {},
        }

        # 关键 token 索引（激活缓存模式使用单一索引）
        self.key_token_indices: Optional[Tensor] = None

        # 按 mode 分的关键 token（KV 缓存模式使用）
        self.key_token_indices_by_mode: Dict[str, Optional[Tensor]] = {
            "cond": None,
            "uncond": None,
        }

        # 激活缓存（用于 Flux 风格的整块缓存）
        # Key: (mode, stream, layer_idx, step) -> Tensor（按 cond/uncond 分流）
        self.activation_cache: Dict[Tuple[str, str, int, int], Tensor] = {}
        self.prev_activation_cache: Dict[Tuple[str, str, int, int], Tensor] = {}

        # CFG 调度：每步 forward 调用次数（1 = 无 CFG, 2 = cond+uncond）
        self.calls_per_step: int = 1
        self._calls_in_step: int = 0

    def set_mode(self, mode: str):
        """
        设置当前模式。

        Args:
            mode: "cond" 或 "uncond"
        """
        assert mode in ("cond", "uncond"), f"Invalid mode: {mode}"
        self.current_mode = mode

    def on_round_start(self):
        """轮次开始：重置 step 与 CFG 调用计数。"""
        self.current_step = 0
        self.current_round += 1
        self._calls_in_step = 0
        self.current_mode = "cond"

    def begin_forward_call(self):
        """每次 transformer forward 开始时调用。设置 current_mode。"""
        if self.calls_per_step <= 1:
            self.current_mode = "cond"
        else:
            self.current_mode = "cond" if self._calls_in_step == 0 else "uncond"

    def end_forward_call(self):
        """每次 transformer forward 结束时调用。完成一对 (cond, uncond) 后递增 step。"""
        self._calls_in_step += 1
        if self._calls_in_step >= self.calls_per_step:
            self._calls_in_step = 0
            self.current_step += 1

    def gpu_has_space(
        self, device: torch.device, extra_bytes: int, limit_gb: float
    ) -> bool:
        """
        检查 GPU 是否有足够空间。使用 cuda runtime 真实剩余显存（包含驱动/外部进程）。

        Args:
            device: GPU 设备
            extra_bytes: 需要的额外字节数
            limit_gb: GPU 总占用上限（GB），相对于设备总容量

        Returns:
            bool: True 表示有足够空间
        """
        try:
            free, total = torch.cuda.mem_get_info(device)
        except Exception:
            # fallback to allocated-based estimate
            used = torch.cuda.memory_allocated(device)
            total = torch.cuda.get_device_properties(device).total_memory
            free = total - used
        used = total - free
        limit_bytes = int(limit_gb * 1024**3)
        buffer_bytes = int(2 * 1024**3)  # 2GB headroom for fragmentation / workspace
        # Two constraints: (a) total used <= limit; (b) extra_bytes fits in free
        if used + extra_bytes + buffer_bytes > limit_bytes:
            return False
        if extra_bytes + buffer_bytes > free:
            return False
        return True

    def store_activation(
        self,
        stream: StreamType,
        layer_idx: int,
        tensor: Tensor,
    ) -> None:
        """
        存储激活到缓存（支持多 GPU）。

        Args:
            stream: 流类型
            layer_idx: 层索引
            tensor: 要缓存的张量
        """
        if not (self.use_activation_cache and self.should_cache(self.current_step)):
            return

        mode = self.current_mode
        key = (stream, self.current_step, layer_idx)

        if self.new_cache[mode] is None:
            self.new_cache[mode] = {}

        # 计算显存需求
        bytes_per_element = tensor.element_size()
        extra_bytes = tensor.numel() * bytes_per_element

        # 选择目标设备
        target_device = self._select_device(extra_bytes)
        self.new_cache[mode][key] = tensor.to(target_device)

    def _select_device(self, extra_bytes: int) -> torch.device:
        """
        根据显存使用情况选择目标设备。

        Args:
            extra_bytes: 需要的字节数

        Returns:
            torch.device: 选择的设备
        """
        start_dev = self.cache_device
        if start_dev.type != "cuda":
            return start_dev

        start_idx = start_dev.index if start_dev.index is not None else 0

        # cuda:0 carries weights + VAE decode + transformer working memory; do
        # NOT place cache there. Push everything to other GPUs.
        def get_limit_gb(device_idx: int) -> float:
            return 0.0 if device_idx == start_idx else 65.0

        # 从 start_idx 开始尝试
        for dev_idx in range(start_idx, self.num_gpus):
            dev = torch.device(f"cuda:{dev_idx}")
            if self.gpu_has_space(dev, extra_bytes, get_limit_gb(dev_idx)):
                return dev

        # 尝试 0 到 start_idx-1
        for dev_idx in range(0, start_idx):
            dev = torch.device(f"cuda:{dev_idx}")
            if self.gpu_has_space(dev, extra_bytes, get_limit_gb(dev_idx)):
                return dev

        # 所有 GPU 都满了，fallback 到起始设备
        print(
            f"⚠ All GPUs 0..{self.num_gpus-1} are almost full, "
            f"falling back to {start_dev} (may OOM)"
        )
        return start_dev

    def get_activation(
        self,
        stream: StreamType,
        layer_idx: int,
        step: Optional[int] = None,
    ) -> Optional[Tensor]:
        """
        从缓存获取激活。

        Args:
            stream: 流类型
            layer_idx: 层索引
            step: 步骤索引，None 表示使用 current_step

        Returns:
            Optional[Tensor]: 缓存的张量，不存在返回 None
        """
        if step is None:
            step = self.current_step

        mode = self.current_mode
        key = (stream, step, layer_idx)

        return self.prev_cache.get(mode, {}).get(key, None)

    def map_to_group_min(self, step: int) -> Optional[int]:
        """
        将 step 映射为 cache_steps 中 <= step 的最大值。

        Args:
            step: 步骤索引

        Returns:
            Optional[int]: 映射的步骤，不存在返回 None
        """
        if self.cache_steps is None:
            return step

        prev = None
        for s in sorted(self.cache_steps):
            if s <= step:
                prev = s
            else:
                break
        return prev

    def flush_new_to_prev(self) -> None:
        """将 new_cache 刷新到 prev_cache（按 mode）。"""
        for mode in ["cond", "uncond"]:
            self.prev_cache[mode] = self.new_cache[mode]
            self.new_cache[mode] = {}

    def clear_cache(self) -> None:
        """清空所有缓存。"""
        for mode in ["cond", "uncond"]:
            self.prev_cache[mode].clear()
            self.new_cache[mode].clear()
            self.key_token_indices_by_mode[mode] = None

        # 清空激活缓存
        self.key_token_indices = None
        self.activation_cache.clear()
        self.prev_activation_cache.clear()

    def get_stats(self) -> Dict[str, any]:
        """获取缓存统计信息。"""
        stats = super().get_stats()
        stats.update({
            "current_mode": self.current_mode,
            "num_gpus": self.num_gpus,
            "cond_cache_size": len(self.prev_cache.get("cond", {})),
            "uncond_cache_size": len(self.prev_cache.get("uncond", {})),
        })
        return stats

    # Processor 接口方法
    def should_compute_kv(self) -> bool:
        """
        判断是否需要计算 K/V。

        Qwen 策略：
        - Round 0：总是计算（需要建立缓存）
        - Round 1+：
          - 缓存步：不计算（复用缓存）
          - 非缓存步：计算（没有缓存可用）

        Returns:
            bool: True 表示需要计算新的 K/V
        """
        if self.is_round0:
            return True
        # Round 1+：缓存步不计算，非缓存步计算
        return not self.should_cache(self.current_step)

    def should_store_kv(self) -> bool:
        """
        判断是否需要存储 K/V 到缓存。

        Qwen 策略：
        - Round 0 的缓存步：存储
        - 其他情况：不存储

        Returns:
            bool: True 表示需要存储
        """
        return self.is_round0 and self.should_cache(self.current_step)

    def should_reuse_kv(self) -> bool:
        """
        判断是否应该复用缓存的 K/V。

        Qwen 策略：
        - Round 1+ 的缓存步：复用
        - 其他情况：不复用

        Returns:
            bool: True 表示应该复用缓存
        """
        return (not self.is_round0) and self.should_cache(self.current_step)

    def get_selection_ids(self) -> Optional[Tensor]:
        """
        获取需要部分更新的 token 索引（关键 token）。

        Returns:
            Optional[Tensor]: 关键 token 索引，None 表示全部复用
        """
        # 激活缓存模式：返回单一索引
        if self.key_token_indices is not None:
            return self.key_token_indices

        # KV 缓存模式：返回当前 mode 的索引
        return self.key_token_indices_by_mode.get(self.current_mode, None)

    # Flux-style activation caching methods
    def store_activation(
        self,
        stream: str,
        layer_idx: int,
        tensor: Tensor,
    ) -> None:
        """
        存储激活到缓存（Flux 风格，按 current_mode 分流）。

        Args:
            stream: "image" / "text"
            layer_idx: 层索引
            tensor: 要缓存的张量
        """
        if not self.use_activation_cache:
            return

        if not self.should_cache(self.current_step):
            return

        key = (self.current_mode, stream, layer_idx, self.current_step)

        bytes_per_element = tensor.element_size()
        extra_bytes = tensor.numel() * bytes_per_element
        target_device = self._select_device(extra_bytes)

        self.activation_cache[key] = tensor.to(target_device)

    def load_activation(
        self,
        stream: str,
        layer_idx: int,
        device: torch.device,
        step: Optional[int] = None,
    ) -> Optional[Tensor]:
        """
        从缓存加载激活（按 current_mode 分流，自动映射 ≤step 的最近缓存步）。
        """
        if not self.use_activation_cache:
            return None

        if step is None:
            step = self.current_step

        mapped = self.map_to_group_min(step)
        if mapped is None:
            return None
        key = (self.current_mode, stream, layer_idx, mapped)

        tensor = self.prev_activation_cache.get(key)
        if tensor is not None:
            if tensor.device != device:
                tensor = tensor.to(device)
            return tensor

        return None

    def flush_activation_cache(self) -> None:
        """将当前激活缓存刷新到 prev（用于轮次切换）。"""
        self.prev_activation_cache = self.activation_cache
        self.activation_cache = {}

    # ---------- Flux-style key token mechanism ----------

    def compute_key_indices_fn(
        self,
        tensor1: Tensor,
        tensor2: Tensor,
    ) -> Tensor:
        """
        按行计算余弦相似度（归一化到 [0,1]），返回相似度 < threshold 的 token 索引。

        Args:
            tensor1: (n, d)
            tensor2: (n, d)

        Returns:
            indices: 1D LongTensor，"变化区域" 的 token 索引
        """
        device = tensor2.device
        tensor1 = tensor1.to(device)
        assert tensor1.shape == tensor2.shape, (
            f"shape mismatch: {tensor1.shape} vs {tensor2.shape}"
        )
        # 转 fp32 计算更稳
        t1 = tensor1.float()
        t2 = tensor2.float()
        dot = (t1 * t2).sum(dim=1)
        n1 = t1.norm(p=2, dim=1)
        n2 = t2.norm(p=2, dim=1)
        eps = 1e-8
        sims = dot / (n1 * n2 + eps)
        normalized = (sims + 1.0) / 2.0
        mask = normalized < self.threshold
        return mask.nonzero(as_tuple=True)[0]

    def update_key_token_indices(
        self,
        cur_img: Tensor,
        ref_img: Optional[Tensor],
    ) -> None:
        """
        在 Round 1+ 的缓存步更新 key_token_indices，识别"变化区域"。

        Args:
            cur_img: 当前 step 全量计算后的 image hidden_states，shape (B, L, C)
            ref_img: 上一轮在同 step / 同 layer 的缓存激活
        """
        if not self.use_activation_cache:
            return
        if self.is_round0:
            return
        if not self.should_cache(self.current_step):
            return
        if cur_img is None or ref_img is None:
            return
        indices = self.compute_key_indices_fn(cur_img[0], ref_img[0])
        self.key_token_indices = indices.to(cur_img.device)

    @staticmethod
    def _build_mask_not_key(L: int, key_indices: Tensor, device: torch.device) -> Tensor:
        """构造 non-key 位置的 bool mask（长度 L）。"""
        ki = key_indices.to(device)
        mask = torch.ones(L, dtype=torch.bool, device=device)
        mask[ki] = False
        return mask

    def rearrange_image_and_freqs(
        self,
        img: Tensor,
        img_freqs: Tensor,
        key_indices: Tensor,
    ) -> Tuple[Tensor, Tensor, Tensor]:
        """
        把 key tokens 移到序列前面（img + img_freqs 同步重排）。

        Args:
            img: (B, L, C) — image hidden states
            img_freqs: (L, D_rope) — image rotary embeddings (complex tensor for Qwen)
            key_indices: 1D LongTensor

        Returns:
            (img_rearranged, img_freqs_rearranged, mask_not_key_on_img_device)
        """
        # image
        L = img.shape[1]
        ki_img = key_indices.to(img.device)
        mask_img = torch.ones(L, dtype=torch.bool, device=img.device)
        mask_img[ki_img] = False
        img_key = torch.index_select(img, 1, ki_img)
        img_not_key = img[:, mask_img, :]
        img_new = torch.cat([img_key, img_not_key], dim=1)

        # freqs (may be on different device)
        ki_freq = key_indices.to(img_freqs.device)
        mask_freq = torch.ones(L, dtype=torch.bool, device=img_freqs.device)
        mask_freq[ki_freq] = False
        freqs_key = torch.index_select(img_freqs, 0, ki_freq)
        freqs_not_key = img_freqs[mask_freq, ...]
        freqs_new = torch.cat([freqs_key, freqs_not_key], dim=0)

        return img_new, freqs_new, mask_img

    def restore_image_order(self, img: Tensor, key_indices: Tensor) -> Tensor:
        """
        把前 K 个 token 放回到 key_indices 指定位置，其余按 non-key 顺序填充。

        Args:
            img: (B, L, C) — 重排顺序的 hidden states
            key_indices: 1D LongTensor

        Returns:
            (B, L, C) — 还原到原始位置后的张量
        """
        L = img.shape[1]
        K = key_indices.shape[0]
        ki = key_indices.to(img.device)
        mask_not_key = torch.ones(L, dtype=torch.bool, device=img.device)
        mask_not_key[ki] = False

        out = torch.empty_like(img)
        out[:, ki, :] = img[:, :K, :]
        out[:, mask_not_key, :] = img[:, K:, :]
        return out
