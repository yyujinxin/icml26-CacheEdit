"""Image processing utilities."""

import math
from typing import Tuple


def calculate_dimensions(target_area: float, ratio: float) -> Tuple[int, int, int]:
    """
    根据目标面积和宽高比计算图像尺寸。

    该函数计算满足给定面积和宽高比的图像尺寸，并将结果对齐到 32 的倍数
    （VAE 编码器的要求）。

    Args:
        target_area: 目标图像面积（像素数），例如 1024*1024 = 1048576
        ratio: 宽高比（width/height），例如 16:9 = 1.778

    Returns:
        tuple: (width, height, actual_area)
            - width: int，计算得到的宽度（32 的倍数）
            - height: int，计算得到的高度（32 的倍数）
            - actual_area: int，实际面积（可能与目标面积略有差异）

    Examples:
        >>> calculate_dimensions(1024 * 1024, 1.0)  # 正方形
        (1024, 1024, 1048576)
        >>> calculate_dimensions(1024 * 1024, 16/9)  # 16:9
        (1344, 768, 1032192)
        >>> calculate_dimensions(1024 * 1024, 4/3)   # 4:3
        (1184, 896, 1060864)

    Note:
        - 宽度和高度会被四舍五入到最接近的 32 的倍数
        - 实际面积可能与目标面积略有差异（通常在 5% 以内）
        - 该函数确保生成的尺寸与 VAE 编码器兼容
    """
    # 根据面积和宽高比计算原始尺寸
    width = math.sqrt(target_area * ratio)
    height = width / ratio

    # 对齐到 32 的倍数（VAE 要求）
    width = round(width / 32) * 32
    height = round(height / 32) * 32

    # 计算实际面积
    actual_area = width * height

    return width, height, actual_area
