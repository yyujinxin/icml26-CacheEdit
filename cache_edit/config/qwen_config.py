"""Qwen model configuration."""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from cache_edit.config.base import BaseConfig


@dataclass
class QwenModelConfig(BaseConfig):
    """Qwen 模型配置。"""

    model_path: str = "Qwen/Qwen2-VL-7B-Instruct"
    device: str = "cuda"
    dtype: str = "bfloat16"
    device_map: Optional[str] = None


@dataclass
class QwenCacheConfig(BaseConfig):
    """Qwen 缓存配置。"""

    threshold: float = 0.1
    cache_interval: int = 5
    enable_stats: bool = True
    use_activation_cache: bool = True


@dataclass
class QwenPipelineConfig(BaseConfig):
    """Qwen Pipeline 配置。"""

    num_inference_steps: int = 50
    guidance_scale: float = 4.0
    height: Optional[int] = 1024
    width: Optional[int] = 1024
    max_sequence_length: int = 512


@dataclass
class QwenConfig(BaseConfig):
    """Qwen 总配置。"""

    model: QwenModelConfig = field(default_factory=QwenModelConfig)
    cache: QwenCacheConfig = field(default_factory=QwenCacheConfig)
    pipeline: QwenPipelineConfig = field(default_factory=QwenPipelineConfig)
    output_dir: str = "./outputs/qwen"

    def validate(self) -> None:
        """验证配置有效性。"""
        if self.cache.threshold < 0 or self.cache.threshold > 1:
            raise ValueError(
                f"cache.threshold must be in [0, 1], got {self.cache.threshold}"
            )
        if self.cache.cache_interval <= 0:
            raise ValueError(
                f"cache.cache_interval must be > 0, got {self.cache.cache_interval}"
            )
        if self.pipeline.num_inference_steps <= 0:
            raise ValueError(
                f"pipeline.num_inference_steps must be > 0, "
                f"got {self.pipeline.num_inference_steps}"
            )
        if self.model.dtype not in ("float32", "float16", "bfloat16"):
            raise ValueError(
                f"model.dtype must be one of (float32, float16, bfloat16), "
                f"got {self.model.dtype}"
            )
        output_path = Path(self.output_dir)
        if output_path.exists() and not output_path.is_dir():
            raise ValueError(
                f"output_dir exists but is not a directory: {self.output_dir}"
            )
