"""Qwen pipeline factory and initialization."""

from typing import Optional, Union

import torch

try:
    from diffusers import QwenImageEditPipeline
except ImportError:
    QwenImageEditPipeline = None

from cache_edit.models.qwen.cache_manager import QwenCacheManager
from cache_edit.models.qwen.transformer_forward import cache_qwen_transformer_2d_forward


def init_qwen_pipeline(
    model_path: str,
    device: Union[str, torch.device] = "cuda",
    dtype: torch.dtype = torch.bfloat16,
    cache_manager: Optional[QwenCacheManager] = None,
    use_activation_cache: bool = True,
):
    """
    创建并初始化 Qwen 图像编辑 Pipeline（带激活缓存优化）。

    该工厂函数负责：
    1. 加载预训练 Pipeline
    2. 替换 transformer forward 为带激活缓存的版本（类似 Flux）
    3. 附加缓存管理器

    Args:
        model_path: 预训练模型路径或 HuggingFace ID
        device: 目标设备
        dtype: 数据类型
        cache_manager: 缓存管理器实例，None 表示不使用缓存
        use_activation_cache: 是否使用激活缓存（Flux 风格）

    Returns:
        QwenImageEditPipeline: 配置好的 pipeline

    Raises:
        ImportError: 如果 diffusers 未安装

    Examples:
        >>> from cache_edit.models.qwen import QwenCacheManager, init_qwen_pipeline
        >>> manager = QwenCacheManager(cache_device=torch.device("cuda:0"))
        >>> pipeline = init_qwen_pipeline(
        ...     "Qwen/Qwen-VL-Edit",
        ...     device="cuda:0",
        ...     cache_manager=manager,
        ... )
    """
    if QwenImageEditPipeline is None:
        raise ImportError(
            "diffusers is required to use init_qwen_pipeline. "
            "Install it with: pip install diffusers"
        )

    # 1. 加载基础 pipeline
    pipeline = QwenImageEditPipeline.from_pretrained(model_path, torch_dtype=dtype)

    # 2. 如果启用缓存，替换 transformer forward
    if cache_manager is not None and use_activation_cache:
        # 附加缓存管理器到 transformer
        pipeline.transformer.cache_manager = cache_manager

        # 替换 transformer forward 为带激活缓存的版本
        pipeline.transformer.forward = cache_qwen_transformer_2d_forward.__get__(
            pipeline.transformer, pipeline.transformer.__class__
        )

    # 3. 移到目标设备
    pipeline = pipeline.to(device)

    return pipeline


def create_default_cache_manager(
    num_inference_steps: int = 30,
    threshold: float = 0.99,
    cache_interval: int = 5,
    cache_device: Optional[torch.device] = None,
    num_gpus: int = 1,
) -> QwenCacheManager:
    """
    创建默认配置的 Qwen 缓存管理器。

    Args:
        num_inference_steps: 推理步数
        threshold: 相似度阈值
        cache_interval: 缓存间隔
        cache_device: 缓存设备，None 时使用 cuda:0
        num_gpus: GPU 数量

    Returns:
        QwenCacheManager: 配置好的缓存管理器
    """
    if cache_device is None:
        cache_device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    return QwenCacheManager(
        use_activation_cache=True,
        cache_device=cache_device,
        total_step_num=num_inference_steps,
        threshold=threshold,
        cache_interval=cache_interval,
        num_gpus=num_gpus,
    )
