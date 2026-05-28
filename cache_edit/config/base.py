"""Base configuration classes for CacheEdit."""

import os
from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import Any, Dict, Optional, Type, TypeVar, Union

try:
    import yaml

    _HAS_YAML = True
except ImportError:
    _HAS_YAML = False

try:
    import json

    _HAS_JSON = True
except ImportError:
    _HAS_JSON = False


T = TypeVar("T", bound="BaseConfig")


@dataclass
class BaseConfig:
    """
    配置基类，提供通用的加载、合并、环境变量覆盖功能。

    子类应定义具体的配置字段，并可选实现 validate() 方法进行自定义验证。
    """

    @classmethod
    def from_dict(cls: Type[T], data: Dict[str, Any]) -> T:
        """
        从字典创建配置实例。

        Args:
            data: 配置字典

        Returns:
            配置实例
        """
        # 只保留 dataclass 定义的字段
        field_names = {f.name for f in fields(cls)}
        filtered = {k: v for k, v in data.items() if k in field_names}

        # 递归处理嵌套的 dataclass
        for f in fields(cls):
            if f.name in filtered and hasattr(f.type, "__dataclass_fields__"):
                filtered[f.name] = f.type.from_dict(filtered[f.name])

        return cls(**filtered)

    @classmethod
    def from_yaml(cls: Type[T], path: Union[str, Path]) -> T:
        """
        从 YAML 文件加载配置。

        Args:
            path: YAML 文件路径

        Returns:
            配置实例
        """
        if not _HAS_YAML:
            raise ImportError(
                "PyYAML is required to load YAML configs. "
                "Install with: pip install pyyaml"
            )
        with open(path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
        return cls.from_dict(data or {})

    @classmethod
    def from_json(cls: Type[T], path: Union[str, Path]) -> T:
        """
        从 JSON 文件加载配置。

        Args:
            path: JSON 文件路径

        Returns:
            配置实例
        """
        if not _HAS_JSON:
            raise ImportError("json module is required to load JSON configs.")
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return cls.from_dict(data or {})

    @classmethod
    def from_file(cls: Type[T], path: Union[str, Path]) -> T:
        """
        根据文件扩展名自动选择加载器。

        Args:
            path: 配置文件路径（.yaml / .yml / .json）

        Returns:
            配置实例
        """
        path = Path(path)
        suffix = path.suffix.lower()
        if suffix in (".yaml", ".yml"):
            return cls.from_yaml(path)
        elif suffix == ".json":
            return cls.from_json(path)
        else:
            raise ValueError(
                f"Unsupported config file format: {suffix}. "
                "Supported: .yaml, .yml, .json"
            )

    def merge(self: T, other: Union[T, Dict[str, Any]]) -> T:
        """
        合并另一个配置，other 的非 None 字段会覆盖当前配置。

        Args:
            other: 另一个配置实例或字典

        Returns:
            合并后的新配置实例
        """
        if isinstance(other, dict):
            other = self.__class__.from_dict(other)

        merged_data = {}
        for f in fields(self):
            self_val = getattr(self, f.name)
            other_val = getattr(other, f.name)

            # 递归合并嵌套 dataclass
            if (
                hasattr(f.type, "__dataclass_fields__")
                and self_val is not None
                and other_val is not None
            ):
                merged_data[f.name] = self_val.merge(other_val)
            else:
                # other 的非 None 值优先
                merged_data[f.name] = (
                    other_val if other_val is not None else self_val
                )

        return self.__class__(**merged_data)

    def apply_env_overrides(self: T, prefix: str = "CACHEEDIT") -> T:
        """
        从环境变量覆盖配置字段。

        环境变量命名规则：{prefix}_{FIELD_NAME}，例如 CACHEEDIT_CACHE_THRESHOLD。
        嵌套字段用双下划线分隔，例如 CACHEEDIT_MODEL__DEVICE。

        Args:
            prefix: 环境变量前缀

        Returns:
            应用环境变量后的新配置实例
        """
        overrides = {}
        for f in fields(self):
            env_key = f"{prefix}_{f.name.upper()}"
            env_val = os.environ.get(env_key)

            if env_val is not None:
                # 尝试类型转换
                try:
                    if f.type == bool:
                        overrides[f.name] = env_val.lower() in (
                            "true",
                            "1",
                            "yes",
                        )
                    elif f.type == int:
                        overrides[f.name] = int(env_val)
                    elif f.type == float:
                        overrides[f.name] = float(env_val)
                    else:
                        overrides[f.name] = env_val
                except (ValueError, TypeError):
                    # 类型转换失败，保持原值
                    pass

            # 递归处理嵌套 dataclass
            if hasattr(f.type, "__dataclass_fields__"):
                nested_val = getattr(self, f.name)
                if nested_val is not None:
                    nested_prefix = f"{prefix}_{f.name.upper()}"
                    nested_overridden = nested_val.apply_env_overrides(
                        nested_prefix
                    )
                    # 只有当嵌套对象真的被覆盖时才加入 overrides
                    if nested_overridden is not nested_val:
                        overrides[f.name] = nested_overridden.to_dict()

        return self.merge(overrides)

    def validate(self) -> None:
        """
        验证配置的有效性。子类可覆盖此方法实现自定义验证。

        Raises:
            ValueError: 配置无效时抛出
        """
        pass

    def to_dict(self) -> Dict[str, Any]:
        """
        转换为字典。

        Returns:
            配置字典
        """
        result = {}
        for f in fields(self):
            val = getattr(self, f.name)
            if hasattr(val, "to_dict"):
                result[f.name] = val.to_dict()
            else:
                result[f.name] = val
        return result
