"""Base cache manager for activation caching in diffusion models."""

from abc import ABC, abstractmethod
from typing import Optional, Dict, Any, Tuple, Set, Literal
import torch
from torch import Tensor

StreamType = Literal["double", "single"]


class BaseCacheManager(ABC):
    """
    缓存管理器抽象基类。

    该类定义了多轮图像编辑中激活缓存的标准接口，包括：
    - 轮次和步骤管理
    - 缓存的存储和复用
    - 关键 token 索引管理

    子类需要实现具体的缓存策略和存储逻辑。

    Attributes:
        use_activation_cache: 是否启用激活缓存
        cache_steps: 需要缓存的步骤集合，None 表示所有步骤都缓存
        cache_device: 缓存存储的设备
        total_step_num: 总推理步数
        threshold: 相似度阈值，用于判断是否需要重新计算
        cache_interval: 缓存间隔，用于自动生成 cache_steps
        current_round: 当前编辑轮次（从 0 开始）
        current_step: 当前推理步骤
    """

    def __init__(
        self,
        use_activation_cache: bool = True,
        cache_steps: Optional[Set[int]] = None,
        cache_device: torch.device = torch.device("cuda"),
        total_step_num: int = 30,
        threshold: float = 0.95,
        cache_interval: int = 5,
    ):
        """
        初始化缓存管理器。

        Args:
            use_activation_cache: 是否启用激活缓存
            cache_steps: 需要缓存的步骤集合，None 表示所有步骤
            cache_device: 缓存存储设备
            total_step_num: 总推理步数
            threshold: 相似度阈值
            cache_interval: 缓存间隔（步数）
        """
        self.use_activation_cache = use_activation_cache
        self.cache_steps = cache_steps
        self.cache_device = cache_device
        self.total_step_num = total_step_num
        self.threshold = threshold
        self.cache_interval = cache_interval

        # 轮次和步骤信息
        self.current_round: int = -1
        self.current_step: int = -1

        # 缓存存储
        self.prev_cache: Dict[Any, Tensor] = {}
        self.new_cache: Dict[Any, Tensor] = {}

        # 关键 token 索引
        self.key_token_indices: Optional[Tensor] = None

        # 自动生成 cache_steps
        if self.cache_steps is None:
            self._build_cache_steps_from_interval()

    def _build_cache_steps_from_interval(self):
        """
        根据 total_step_num 和 cache_interval 自动生成 cache_steps。

        例如：total_step_num=40, cache_interval=5 -> {0, 5, 10, 15, 20, 25, 30, 35}
        """
        if self.cache_interval <= 0:
            self.cache_steps = {0}
            return

        steps = list(range(0, self.total_step_num, self.cache_interval))
        self.cache_steps = set(steps)

    @property
    def is_round0(self) -> bool:
        """判断是否为第一轮编辑。"""
        return self.current_round == 0

    def on_step_start(self, step: int):
        """
        在每个推理步骤开始时调用。

        当 step == 0 时，认为开始新一轮编辑。

        Args:
            step: 当前步骤索引
        """
        self.current_step = step
        if step == 0:
            self.current_round += 1

    def should_cache(self, step: int) -> bool:
        """
        判断当前步骤是否需要缓存激活。

        Args:
            step: 步骤索引

        Returns:
            bool: True 表示需要缓存
        """
        if not self.use_activation_cache:
            return False
        if self.cache_steps is None:
            return True
        return step in self.cache_steps

    def should_reuse(self, step: int) -> bool:
        """
        判断当前步骤是否应该复用上一轮的缓存。

        Args:
            step: 步骤索引

        Returns:
            bool: True 表示应该复用缓存
        """
        return (not self.is_round0) and (not self.should_cache(step))

    @abstractmethod
    def store_activation(
        self,
        stream: StreamType,
        layer_idx: int,
        tensor: Tensor,
    ) -> None:
        """
        存储激活到缓存。

        Args:
            stream: 流类型（"double" 或 "single"）
            layer_idx: 层索引
            tensor: 要缓存的张量
        """
        pass

    @abstractmethod
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
            Optional[Tensor]: 缓存的张量，如果不存在返回 None
        """
        pass

    def clear_cache(self) -> None:
        """清空所有缓存。"""
        self.prev_cache.clear()
        self.new_cache.clear()
        self.key_token_indices = None

    def flush_new_to_prev(self) -> None:
        """
        将 new_cache 的内容刷新到 prev_cache。

        通常在一轮编辑结束时调用。
        """
        self.prev_cache = self.new_cache
        self.new_cache = {}

    def set_parameters(
        self,
        num_inference_steps: Optional[int] = None,
        threshold: Optional[float] = None,
        cache_device: Optional[torch.device] = None,
        cache_interval: Optional[int] = None,
    ) -> None:
        """
        动态更新缓存管理器参数。

        Args:
            num_inference_steps: 推理步数
            threshold: 相似度阈值
            cache_device: 缓存设备
            cache_interval: 缓存间隔
        """
        if num_inference_steps is not None:
            self.total_step_num = num_inference_steps
        if threshold is not None:
            self.threshold = threshold
        if cache_device is not None:
            self.cache_device = cache_device
        if cache_interval is not None:
            self.cache_interval = cache_interval
            self._build_cache_steps_from_interval()

    def get_stats(self) -> Dict[str, Any]:
        """
        获取缓存统计信息。

        Returns:
            Dict[str, Any]: 包含缓存大小、命中率等统计信息
        """
        return {
            "current_round": self.current_round,
            "current_step": self.current_step,
            "prev_cache_size": len(self.prev_cache),
            "new_cache_size": len(self.new_cache),
            "cache_steps": self.cache_steps,
            "threshold": self.threshold,
        }

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}("
            f"round={self.current_round}, "
            f"step={self.current_step}, "
            f"cache_size={len(self.prev_cache)}, "
            f"threshold={self.threshold})"
        )
