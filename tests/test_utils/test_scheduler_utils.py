"""Tests for cache_edit.utils.scheduler_utils."""

import pytest

from cache_edit.utils.scheduler_utils import calculate_shift


class TestCalculateShift:
    def test_base_seq_len_returns_base_shift(self):
        assert calculate_shift(256) == pytest.approx(0.5, abs=1e-6)

    def test_max_seq_len_returns_max_shift(self):
        assert calculate_shift(4096) == pytest.approx(1.15, abs=1e-6)

    def test_mid_value_in_range(self):
        result = calculate_shift(2176)
        assert 0.5 < result < 1.15

    def test_linear_interpolation(self):
        # Halfway between base and max should be halfway between 0.5 and 1.15
        midpoint_seq = (256 + 4096) / 2
        expected = (0.5 + 1.15) / 2
        assert calculate_shift(midpoint_seq) == pytest.approx(expected, abs=1e-6)

    def test_custom_params(self):
        result = calculate_shift(
            512, base_seq_len=512, max_seq_len=2048,
            base_shift=0.0, max_shift=1.0,
        )
        assert result == pytest.approx(0.0, abs=1e-6)

    def test_extrapolation_below_base(self):
        # 函数是线性的，输入小于 base_seq_len 时仍会插值（非夹紧）
        result = calculate_shift(128)
        assert result < 0.5
