"""Tests for FluxCacheManager (CPU-only)."""

import pytest
import torch

from cache_edit.models.flux import FluxCacheManager


@pytest.fixture
def cpu_mgr():
    return FluxCacheManager(
        use_activation_cache=True,
        total_step_num=28,
        threshold=0.97,
        cache_interval=5,
        cache_device=torch.device("cpu"),
        num_gpus=1,
    )


class TestCacheStepsAndMapping:
    def test_cache_steps_default(self, cpu_mgr):
        assert cpu_mgr.cache_steps == {0, 5, 10, 15, 20, 25}

    def test_map_to_group_min(self, cpu_mgr):
        assert cpu_mgr.map_to_group_min(0) == 0
        assert cpu_mgr.map_to_group_min(4) == 0
        assert cpu_mgr.map_to_group_min(5) == 5
        assert cpu_mgr.map_to_group_min(9) == 5
        assert cpu_mgr.map_to_group_min(25) == 25


class TestShouldReuse:
    def test_round0_never_reuses(self, cpu_mgr):
        cpu_mgr.on_step_start(0)
        assert cpu_mgr.is_round0
        assert cpu_mgr.should_reuse(2) is False

    def test_round1_reuses_when_not_cache_step(self, cpu_mgr):
        cpu_mgr.on_step_start(0)
        cpu_mgr.on_step_start(0)  # → round 1
        assert not cpu_mgr.is_round0
        assert cpu_mgr.should_reuse(2)
        assert not cpu_mgr.should_reuse(5)  # 5 is cache step


class TestStoreFlushLoad:
    def test_round0_store_then_flush_then_load(self, cpu_mgr):
        cpu_mgr.on_step_start(0)
        cpu_mgr.stream_type = "double"
        t = torch.randn(1, 64, 16)
        cpu_mgr.store_activation("double", 0, t)
        assert len(cpu_mgr.new_cache) == 1

        cpu_mgr.flush_new_cache_after_step()
        assert len(cpu_mgr.prev_cache) == 1
        assert len(cpu_mgr.new_cache) == 0

        loaded = cpu_mgr.load_activation("double", 0, torch.device("cpu"))
        assert loaded.shape == t.shape

    def test_load_uses_group_min(self, cpu_mgr):
        # Round 0 step 0: cache and flush
        cpu_mgr.on_step_start(0)
        t = torch.randn(1, 32, 8)
        cpu_mgr.store_activation("double", 0, t)
        cpu_mgr.flush_new_cache_after_step()

        # Round 1 step 2 → should map to step 0
        cpu_mgr.on_step_start(0)
        cpu_mgr.on_step_start(2)
        loaded = cpu_mgr.load_activation("double", 0, torch.device("cpu"))
        assert loaded is not None
        assert torch.equal(loaded, t)


class TestKeyTokenComputation:
    def test_similar_returns_few(self, cpu_mgr):
        a = torch.randn(20, 16)
        b = a + 0.001 * torch.randn(20, 16)
        idx = cpu_mgr.compute_key_indices_fn(a, b)
        assert idx.numel() <= 5  # most rows highly similar

    def test_random_returns_many(self, cpu_mgr):
        a = torch.randn(20, 16)
        b = torch.randn(20, 16)
        idx = cpu_mgr.compute_key_indices_fn(a, b)
        assert idx.numel() >= 10

    def test_update_skipped_in_round0(self, cpu_mgr):
        cpu_mgr.on_step_start(0)
        cur = torch.randn(1, 10, 4)
        ref = torch.randn(1, 10, 4)
        cpu_mgr.update_key_token_indices(cur, ref)
        assert cpu_mgr.key_token_indices is None


class TestRearrangeRestore:
    def test_rearrange_then_restore_inverse(self, cpu_mgr):
        torch.manual_seed(0)
        img = torch.randn(1, 16, 4)
        cos = torch.randn(16, 4)
        sin = torch.randn(16, 4)
        kti = torch.tensor([1, 5, 11])

        img_r, cos_r, sin_r = cpu_mgr.rearrange_tensor_with_key_token_indices(
            img, cos, sin, kti
        )
        # First K tokens of img_r should be the key tokens
        assert torch.equal(img_r[:, :3, :], img[:, kti, :])

        img_back = cpu_mgr.restore_original_token_order(img_r, kti)
        assert torch.allclose(img, img_back)


class TestReset:
    def test_reset_clears_state(self, cpu_mgr):
        cpu_mgr.on_step_start(0)
        cpu_mgr.store_activation("double", 0, torch.zeros(1, 8, 4))
        cpu_mgr.flush_new_cache_after_step()
        cpu_mgr.key_token_indices = torch.tensor([1, 2])

        cpu_mgr.reset()
        assert len(cpu_mgr.prev_cache) == 0
        assert len(cpu_mgr.new_cache) == 0
        assert cpu_mgr.key_token_indices is None
        assert cpu_mgr.current_round == -1
        assert cpu_mgr.current_step == -1
