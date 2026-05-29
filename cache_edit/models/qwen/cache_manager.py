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

        # 按 mode 分两套缓存
        self.prev_cache: Dict[str, Dict[Tuple[StreamType, int, int], Tensor]] = {
            "cond": {},
            "uncond": {},
        }
        self.new_cache: Dict[str, Dict[Tuple[StreamType, int, int], Tensor]] = {
            "cond": {},
            "uncond": {},
        }

        # 关键 token 按 mode 分两套
        self.key_token_indices: Dict[str, Optional[Tensor]] = {
            "cond": None,
            "uncond": None,
        }

    def set_mode(self, mode: str):
        """
        设置当前模式。

        Args:
            mode: "cond" 或 "uncond"
        """
        assert mode in ("cond", "uncond"), f"Invalid mode: {mode}"
        self.current_mode = mode

    def gpu_has_space(
        self, device: torch.device, extra_bytes: int, limit_gb: float
    ) -> bool:
        """
        检查 GPU 是否有足够空间。

        Args:
            device: GPU 设备
            extra_bytes: 需要的额外字节数
            limit_gb: GPU 限制（GB）

        Returns:
            bool: True 表示有足够空间
        """
        used = torch.cuda.memory_allocated(device)
        limit_bytes = int(limit_gb * 1024**3)
        buffer_bytes = int(1 * 1024**3)  # 1GB buffer
        return used + extra_bytes + buffer_bytes <= limit_bytes

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

        # 默认每个 GPU 限制 77GB
        def get_limit_gb(device_idx: int) -> float:
            return 77.0

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
            self.key_token_indices[mode] = None

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
        # 返回当前 mode 的关键 token 索引
        return self.key_token_indices.get(self.current_mode, None)
