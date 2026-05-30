"""Flux Kontext model configuration."""

from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

from cache_edit.config.base import BaseConfig


@dataclass
class FluxModelConfig(BaseConfig):
    """Flux 模型配置。"""

    model_path: str = "black-forest-labs/FLUX.1-dev"
    device: str = "cuda"
    dtype: str = "bfloat16"
    device_map: Optional[str] = None


@dataclass
class FluxCacheConfig(BaseConfig):
    """Flux 缓存配置。"""

    threshold: float = 0.97
    cache_interval: int = 5
    enable_stats: bool = True
    use_activation_cache: bool = True
    num_gpus: int = 1


@dataclass
class FluxPipelineConfig(BaseConfig):
    """Flux Pipeline 配置。"""

    num_inference_steps: int = 28
    guidance_scale: float = 3.5
    height: Optional[int] = 1024
    width: Optional[int] = 1024
    max_sequence_length: int = 512
    true_cfg: bool = True


@dataclass
class FluxVizConfig(BaseConfig):
    """Flux key-token 可视化配置。"""

    enable: bool = False
    gen_dir: Optional[str] = None
    viz_out_dir: Optional[str] = None
    csv_out_path: Optional[str] = None
    edit_ratio_summary_candidates: List[str] = field(default_factory=list)
    rounds_per_image: int = 7
    ref_layer_idx: int = 37
    ref_stream: str = "single"


@dataclass
class FluxConfig(BaseConfig):
    """Flux 总配置。"""

    model: FluxModelConfig = field(default_factory=FluxModelConfig)
    cache: FluxCacheConfig = field(default_factory=FluxCacheConfig)
    pipeline: FluxPipelineConfig = field(default_factory=FluxPipelineConfig)
    viz: FluxVizConfig = field(default_factory=FluxVizConfig)
    output_dir: str = "./outputs/flux"

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
        if self.cache.num_gpus <= 0:
            raise ValueError(
                f"cache.num_gpus must be > 0, got {self.cache.num_gpus}"
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
        if self.viz.enable:
            if self.viz.ref_stream not in ("single", "double"):
                raise ValueError(
                    f"viz.ref_stream must be 'single' or 'double', "
                    f"got {self.viz.ref_stream}"
                )
            if self.viz.ref_layer_idx < 0:
                raise ValueError(
                    f"viz.ref_layer_idx must be >= 0, got {self.viz.ref_layer_idx}"
                )
