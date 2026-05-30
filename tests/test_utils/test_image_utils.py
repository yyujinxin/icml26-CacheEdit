"""Tests for cache_edit.utils.image_utils."""

import pytest

from cache_edit.utils.image_utils import calculate_dimensions


class TestCalculateDimensions:
    def test_square_1024(self):
        w, h, area = calculate_dimensions(1024 * 1024, 1.0)
        assert w == 1024
        assert h == 1024
        assert area == 1024 * 1024

    def test_aligned_to_32(self):
        for ratio in [1.0, 16 / 9, 4 / 3, 21 / 9, 1 / 2]:
            w, h, _ = calculate_dimensions(1024 * 1024, ratio)
            assert w % 32 == 0, f"width {w} not aligned for ratio {ratio}"
            assert h % 32 == 0, f"height {h} not aligned for ratio {ratio}"

    def test_actual_area_close_to_target(self):
        target = 1024 * 1024
        for ratio in [1.0, 16 / 9, 4 / 3, 0.5]:
            _, _, area = calculate_dimensions(target, ratio)
            # 32-alignment can shift area by up to ~5%
            assert abs(area - target) / target < 0.1

    def test_wide_ratio_gives_wider_image(self):
        w_wide, h_wide, _ = calculate_dimensions(1024 * 1024, 16 / 9)
        w_sq, h_sq, _ = calculate_dimensions(1024 * 1024, 1.0)
        assert w_wide > w_sq
        assert h_wide < h_sq

    def test_returns_ints(self):
        w, h, area = calculate_dimensions(1024 * 1024, 1.0)
        assert isinstance(w, int)
        assert isinstance(h, int)
        assert isinstance(area, int)
