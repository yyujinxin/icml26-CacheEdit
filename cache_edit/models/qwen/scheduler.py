"""Qwen scheduler implementation with region-aware step support."""

from typing import Optional, Union, Tuple

import torch
from diffusers.schedulers import FlowMatchEulerDiscreteScheduler

from cache_edit.utils.scheduler_utils import FlowMatchEulerDiscreteSchedulerOutput


class QwenRegionAwareScheduler(FlowMatchEulerDiscreteScheduler):
    """
    Qwen 区域感知的 FlowMatch Euler 离散调度器。

    扩展了基础调度器，支持：
    - 标准 Euler 步进
    - 区域感知步进（通过 cache_manager 注入）
    - per-token 时间步

    与原始 `RegionEFlowMatchEulerDiscreteScheduler` 的区别：
    - 解耦全局 MANAGER，改为通过参数或依赖注入
    - 区域感知逻辑由外部 cache_manager 提供（通过钩子）
    - 接口更清晰，易于测试

    Attributes:
        cache_manager: 可选的缓存管理器，提供区域感知逻辑
        region_step_fn: 可选的区域感知步进函数
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.cache_manager = None
        self.region_step_fn = None

    def attach_cache_manager(self, cache_manager) -> None:
        """
        附加缓存管理器以启用区域感知步进。

        Args:
            cache_manager: 实现了 region-aware 接口的缓存管理器
        """
        self.cache_manager = cache_manager

    def attach_region_step_fn(self, fn) -> None:
        """
        附加区域感知步进函数（用于替代默认行为）。

        Args:
            fn: 签名为 fn(model_output, sample, dt, current_sigma, next_sigma) -> Tensor
        """
        self.region_step_fn = fn

    def step(
        self,
        model_output: torch.FloatTensor,
        timestep: Union[float, torch.FloatTensor],
        sample: torch.FloatTensor,
        s_churn: float = 0.0,
        s_tmin: float = 0.0,
        s_tmax: float = float("inf"),
        s_noise: float = 1.0,
        generator: Optional[torch.Generator] = None,
        per_token_timesteps: Optional[torch.Tensor] = None,
        return_dict: bool = True,
    ) -> Union[FlowMatchEulerDiscreteSchedulerOutput, Tuple]:
        """
        执行单个去噪步骤。

        Args:
            model_output: 模型输出（预测的噪声/速度）
            timestep: 当前时间步
            sample: 当前样本
            s_churn, s_tmin, s_tmax, s_noise: 噪声调度参数
            generator: 随机数生成器
            per_token_timesteps: 每个 token 的时间步（用于区域感知）
            return_dict: 是否返回字典

        Returns:
            调度器输出（包含 prev_sample）
        """
        if isinstance(timestep, (int, torch.IntTensor, torch.LongTensor)):
            raise ValueError(
                "Passing integer indices as timesteps is not supported. "
                "Pass one of `scheduler.timesteps` instead."
            )

        if self.step_index is None:
            self._init_step_index(timestep)

        # 提升精度以避免数值问题
        sample = sample.to(torch.float32)

        # 计算 sigma 和 dt
        current_sigma, next_sigma, dt = self._compute_sigmas(
            sample, per_token_timesteps
        )

        # 执行步进
        if self.config.stochastic_sampling:
            prev_sample = self._stochastic_step(
                sample, model_output, current_sigma, next_sigma
            )
        elif self.region_step_fn is not None:
            # 使用注入的区域感知步进
            prev_sample = self.region_step_fn(
                model_output=model_output,
                sample=sample,
                dt=dt,
                current_sigma=current_sigma,
                next_sigma=next_sigma,
                scheduler=self,
            )
        else:
            # 标准 Euler 步进
            prev_sample = sample + dt * model_output

        # 更新步数索引
        self._step_index += 1

        # 恢复 dtype
        if per_token_timesteps is None:
            prev_sample = prev_sample.to(model_output.dtype)

        if not return_dict:
            return (prev_sample,)

        return FlowMatchEulerDiscreteSchedulerOutput(prev_sample=prev_sample)

    def _compute_sigmas(
        self,
        sample: torch.Tensor,
        per_token_timesteps: Optional[torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        计算当前和下一个 sigma 及时间差 dt。

        Args:
            sample: 当前样本
            per_token_timesteps: 可选的 per-token 时间步

        Returns:
            tuple: (current_sigma, next_sigma, dt)
        """
        if per_token_timesteps is not None:
            per_token_sigmas = per_token_timesteps / self.config.num_train_timesteps

            sigmas = self.sigmas[:, None, None]
            lower_mask = sigmas < per_token_sigmas[None] - 1e-6
            lower_sigmas = lower_mask * sigmas
            lower_sigmas, _ = lower_sigmas.max(dim=0)

            current_sigma = per_token_sigmas[..., None]
            next_sigma = lower_sigmas[..., None]
            dt = current_sigma - next_sigma
        else:
            sigma_idx = self.step_index
            current_sigma = self.sigmas[sigma_idx]
            next_sigma = self.sigmas[sigma_idx + 1]
            dt = next_sigma - current_sigma

        return current_sigma, next_sigma, dt

    def _stochastic_step(
        self,
        sample: torch.Tensor,
        model_output: torch.Tensor,
        current_sigma: torch.Tensor,
        next_sigma: torch.Tensor,
    ) -> torch.Tensor:
        """
        执行随机采样步进。

        Args:
            sample: 当前样本
            model_output: 模型输出
            current_sigma: 当前 sigma
            next_sigma: 下一个 sigma

        Returns:
            torch.Tensor: 前一个样本
        """
        x0 = sample - current_sigma * model_output
        noise = torch.randn_like(sample)
        prev_sample = (1.0 - next_sigma) * x0 + next_sigma * noise
        return prev_sample
