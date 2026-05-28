"""Scheduler utilities for diffusion models."""

from typing import Optional, Union, List
import inspect
from dataclasses import dataclass

try:
    import torch
    from diffusers.utils import BaseOutput
except ImportError:
    torch = None
    BaseOutput = None


def calculate_shift(
    image_seq_len: int,
    base_seq_len: int = 256,
    max_seq_len: int = 4096,
    base_shift: float = 0.5,
    max_shift: float = 1.15,
) -> float:
    """
    计算时间步偏移量，用于调整扩散过程的时间步分布。

    该函数根据图像序列长度线性插值计算偏移量，使得不同分辨率的图像
    可以使用适当的时间步分布。

    Args:
        image_seq_len: 图像序列长度（token 数量）
        base_seq_len: 基准序列长度，默认 256
        max_seq_len: 最大序列长度，默认 4096
        base_shift: 基准偏移量，默认 0.5
        max_shift: 最大偏移量，默认 1.15

    Returns:
        float: 计算得到的偏移量 mu

    Examples:
        >>> calculate_shift(256)  # 基准长度
        0.5
        >>> calculate_shift(4096)  # 最大长度
        1.15
        >>> calculate_shift(2176)  # 中间值
        0.825
    """
    m = (max_shift - base_shift) / (max_seq_len - base_seq_len)
    b = base_shift - m * base_seq_len
    mu = image_seq_len * m + b
    return mu


def retrieve_timesteps(
    scheduler,
    num_inference_steps: Optional[int] = None,
    device: Optional[Union[str, torch.device]] = None,
    timesteps: Optional[List[int]] = None,
    sigmas: Optional[List[float]] = None,
    **kwargs,
):
    """
    从调度器获取时间步序列。

    该函数调用调度器的 `set_timesteps` 方法并返回时间步。支持自定义时间步
    和 sigma 值，提供灵活的时间步控制。

    Args:
        scheduler: 扩散调度器实例（如 FlowMatchEulerDiscreteScheduler）
        num_inference_steps: 推理步数。如果使用，timesteps 必须为 None
        device: 时间步应移动到的设备。如果为 None，则不移动
        timesteps: 自定义时间步列表，用于覆盖调度器的默认时间步策略。
                  如果传入，num_inference_steps 和 sigmas 必须为 None
        sigmas: 自定义 sigma 值列表，用于支持 sigma 参数的调度器。
               如果传入，num_inference_steps 和 timesteps 必须为 None
        **kwargs: 传递给 scheduler.set_timesteps 的额外参数

    Returns:
        tuple: (timesteps, num_inference_steps)
            - timesteps: torch.Tensor，时间步序列
            - num_inference_steps: int，实际推理步数

    Raises:
        ValueError: 当参数组合不合法时（如同时指定 timesteps 和 sigmas）

    Examples:
        >>> from diffusers import FlowMatchEulerDiscreteScheduler
        >>> scheduler = FlowMatchEulerDiscreteScheduler()
        >>> timesteps, steps = retrieve_timesteps(scheduler, num_inference_steps=50)
        >>> len(timesteps)
        50
    """
    # 参数验证
    if timesteps is not None and sigmas is not None:
        raise ValueError("Only one of `timesteps` or `sigmas` can be passed.")

    if timesteps is not None:
        # 使用自定义时间步
        accepts_timesteps = "timesteps" in set(
            inspect.signature(scheduler.set_timesteps).parameters.keys()
        )
        if not accepts_timesteps:
            raise ValueError(
                f"The current scheduler class {scheduler.__class__}'s `set_timesteps` does not support "
                f"custom timestep schedules. Please check whether you are using the correct scheduler."
            )
        scheduler.set_timesteps(timesteps=timesteps, device=device, **kwargs)
        timesteps = scheduler.timesteps
        num_inference_steps = len(timesteps)

    elif sigmas is not None:
        # 使用自定义 sigma 值
        accepts_sigmas = "sigmas" in set(
            inspect.signature(scheduler.set_timesteps).parameters.keys()
        )
        if not accepts_sigmas:
            raise ValueError(
                f"The current scheduler class {scheduler.__class__}'s `set_timesteps` does not support "
                f"custom sigmas schedules. Please check whether you are using the correct scheduler."
            )
        scheduler.set_timesteps(sigmas=sigmas, device=device, **kwargs)
        timesteps = scheduler.timesteps
        num_inference_steps = len(timesteps)

    else:
        # 使用标准推理步数
        scheduler.set_timesteps(num_inference_steps, device=device, **kwargs)
        timesteps = scheduler.timesteps

    return timesteps, num_inference_steps


@dataclass
class FlowMatchEulerDiscreteSchedulerOutput(BaseOutput):
    """
    FlowMatchEulerDiscreteScheduler 的输出类。

    该类封装了调度器单步执行的输出结果，包括前一个样本和预测的原始样本。

    Attributes:
        prev_sample: torch.Tensor，形状为 (batch_size, num_channels, height, width)
            计算得到的前一时间步的样本。这是去噪过程的主要输出。
        pred_original_sample: torch.Tensor，可选，形状同 prev_sample
            基于当前时间步模型输出预测的原始样本（完全去噪的图像）。
            某些调度器会计算此值用于指导或可视化。
    """

    prev_sample: torch.Tensor
    pred_original_sample: Optional[torch.Tensor] = None
