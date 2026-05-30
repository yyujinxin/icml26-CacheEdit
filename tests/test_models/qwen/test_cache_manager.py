"""Tests for QwenCacheManager (CPU-only, mocked GPU paths)."""

import pytest
import torch

from cache_edit.models.qwen.cache_manager import QwenCacheManager


@pytest.fixture
def cpu_mgr():
    return QwenCacheManager(
        use_activation_cache=True,
        cache_steps={0, 5},
        cache_device=torch.device("cpu"),
        total_step_num=10,
        num_gpus=1,
        threshold=0.9,
    )


class TestModeSwitch:
    def test_default_mode_is_cond(self, cpu_mgr):
        assert cpu_mgr.current_mode == "cond"

    def test_set_mode(self, cpu_mgr):
        cpu_mgr.set_mode("uncond")
        assert cpu_mgr.current_mode == "uncond"

    def test_invalid_mode_raises(self, cpu_mgr):
        with pytest.raises(AssertionError):
            cpu_mgr.set_mode("garbage")


class TestStoreLoadPerMode:
    def test_cond_uncond_isolated(self, cpu_mgr):
        cpu_mgr.on_step_start(0)

        cpu_mgr.set_mode("cond")
        t_cond = torch.tensor([1.0, 2.0])
        cpu_mgr.store_activation("double", 0, t_cond)

        cpu_mgr.set_mode("uncond")
        t_uncond = torch.tensor([3.0, 4.0])
        cpu_mgr.store_activation("double", 0, t_uncond)

        assert ("double", 0, 0) in cpu_mgr.new_cache["cond"]
        assert ("double", 0, 0) in cpu_mgr.new_cache["uncond"]
        assert not torch.equal(
            cpu_mgr.new_cache["cond"][("double", 0, 0)],
            cpu_mgr.new_cache["uncond"][("double", 0, 0)],
        )

    def test_store_skipped_for_noncache_step(self, cpu_mgr):
        cpu_mgr.on_step_start(2)  # 2 ∉ {0,5}
        cpu_mgr.store_activation("double", 0, torch.zeros(2))
        assert len(cpu_mgr.new_cache["cond"]) == 0


class TestGetStats:
    def test_includes_qwen_keys(self, cpu_mgr):
        stats = cpu_mgr.get_stats()
        # base keys plus any Qwen extensions
        assert "current_round" in stats
        assert "threshold" in stats


class TestDeviceSelection:
    def test_cpu_device_returns_self(self, cpu_mgr):
        chosen = cpu_mgr._select_device(extra_bytes=1024)
        assert chosen == torch.device("cpu")
