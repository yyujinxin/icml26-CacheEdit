"""Shared CLI helpers."""

from pathlib import Path
from typing import Union

import torch

from cache_edit.config import FluxConfig, QwenConfig


DTYPE_MAP = {
    "float32": torch.float32,
    "fp32": torch.float32,
    "float16": torch.float16,
    "fp16": torch.float16,
    "bfloat16": torch.bfloat16,
    "bf16": torch.bfloat16,
}


def resolve_dtype(name: str) -> torch.dtype:
    """字符串 → torch.dtype。"""
    key = name.lower()
    if key not in DTYPE_MAP:
        raise ValueError(
            f"Unsupported dtype '{name}'. Expected one of: "
            f"{', '.join(sorted(set(DTYPE_MAP)))}"
        )
    return DTYPE_MAP[key]


def load_config(
    model: str,
    config_path: Union[str, Path, None] = None,
    apply_env: bool = True,
):
    """
    根据 model 名加载相应的配置实例。

    Args:
        model: "qwen" 或 "flux"
        config_path: 可选的自定义配置文件路径；None 时使用 configs/{model}_default.yaml
        apply_env: 是否应用 CACHEEDIT_* 环境变量覆盖

    Returns:
        QwenConfig 或 FluxConfig 实例
    """
    cls_map = {"qwen": QwenConfig, "flux": FluxConfig}
    if model not in cls_map:
        raise ValueError(
            f"Unknown model '{model}'. Expected one of: {list(cls_map)}"
        )
    cfg_cls = cls_map[model]

    if config_path is None:
        default_path = Path("configs") / f"{model}_default.yaml"
        if default_path.exists():
            cfg = cfg_cls.from_file(default_path)
        else:
            cfg = cfg_cls()
    else:
        cfg = cfg_cls.from_file(config_path)

    if apply_env:
        cfg = cfg.apply_env_overrides(prefix="CACHEEDIT")

    cfg.validate()
    return cfg
