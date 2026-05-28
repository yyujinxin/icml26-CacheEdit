"""Tests for cache_edit.core.cache_manager.BaseCacheManager."""

from typing import Optional

import pytest
import torch
from torch import Tensor

from cache_edit.core.cache_manager import BaseCacheManager


class _DummyCacheManager(BaseCacheManager):
    """Concrete BaseCacheManager for exercising the abstract API."""

    def store_activation(self, stream, layer_idx, tensor: Tensor) -> None:
        if not (self.use_activation_cache and self.should_cache(self.current_step)):
            return
        key = (stream, self.current_step, layer_idx)
        self.new_cache[key] = tensor.detach().clone()

    def get_activation(self, stream, layer_idx, step: Optional[int] = None):
        if not self.use_activation_cache:
            return None
        s = self.current_step if step is None else step
        return self.prev_cache.get((stream, s, layer_idx))


class TestCacheStepsAutoGeneration:
    def test_default_interval(self):
        mgr = _DummyCacheManager(total_step_num=30, cache_interval=5)
        assert mgr.cache_steps == {0, 5, 10, 15, 20, 25}

    def test_interval_one(self):
        mgr = _DummyCacheManager(total_step_num=5, cache_interval=1)
        assert mgr.cache_steps == {0, 1, 2, 3, 4}

    def test_zero_interval_only_step0(self):
        mgr = _DummyCacheManager(total_step_num=10, cache_interval=0)
        assert mgr.cache_steps == {0}

    def test_explicit_cache_steps_overrides(self):
        mgr = _DummyCacheManager(cache_steps={2, 7})
        assert mgr.cache_steps == {2, 7}


class TestRoundLifecycle:
    def test_round_starts_at_minus_one(self):
        mgr = _DummyCacheManager()
        assert mgr.current_round == -1
        assert mgr.current_step == -1

    def test_step_zero_increments_round(self):
        mgr = _DummyCacheManager()
        mgr.on_step_start(0)
        assert mgr.current_round == 0
        assert mgr.is_round0

        mgr.on_step_start(0)
        assert mgr.current_round == 1
        assert not mgr.is_round0

    def test_step_nonzero_keeps_round(self):
        mgr = _DummyCacheManager()
        mgr.on_step_start(0)
        mgr.on_step_start(3)
        assert mgr.current_step == 3
        assert mgr.current_round == 0


class TestShouldCacheReuse:
    def test_should_cache_disabled(self):
        mgr = _DummyCacheManager(use_activation_cache=False)
        assert mgr.should_cache(0) is False

    def test_should_cache_in_steps(self):
        mgr = _DummyCacheManager(cache_steps={0, 5})
        assert mgr.should_cache(0)
        assert mgr.should_cache(5)
        assert not mgr.should_cache(1)

    def test_should_cache_none_means_all(self):
        mgr = _DummyCacheManager(cache_steps={0})
        # explicit set => only step 0
        assert not mgr.should_cache(1)

        mgr2 = _DummyCacheManager()
        mgr2.cache_steps = None
        assert mgr2.should_cache(99)  # None = every step

    def test_should_reuse_round0_false(self):
        mgr = _DummyCacheManager(cache_steps={0, 5})
        mgr.on_step_start(0)
        assert mgr.is_round0
        assert mgr.should_reuse(1) is False  # round 0 never reuses

    def test_should_reuse_round1(self):
        mgr = _DummyCacheManager(cache_steps={0, 5})
        mgr.on_step_start(0)
        mgr.on_step_start(0)  # → round 1
        assert not mgr.is_round0
        assert not mgr.should_reuse(0)  # 0 ∈ cache_steps → should_cache true
        assert mgr.should_reuse(1)  # not a cache step → reuse


class TestStoreFlushLoad:
    def test_round_trip(self):
        mgr = _DummyCacheManager(cache_steps={0})
        mgr.on_step_start(0)

        t = torch.randn(2, 4)
        mgr.store_activation("double", 0, t)
        assert ("double", 0, 0) in mgr.new_cache

        mgr.flush_new_to_prev()
        assert ("double", 0, 0) in mgr.prev_cache
        assert ("double", 0, 0) not in mgr.new_cache

        loaded = mgr.get_activation("double", 0)
        assert loaded is not None
        assert torch.equal(loaded, t)

    def test_store_skipped_when_not_cache_step(self):
        mgr = _DummyCacheManager(cache_steps={0})
        mgr.on_step_start(3)  # not in cache_steps
        mgr.store_activation("double", 0, torch.zeros(1, 1))
        assert len(mgr.new_cache) == 0

    def test_clear_cache(self):
        mgr = _DummyCacheManager(cache_steps={0})
        mgr.on_step_start(0)
        mgr.store_activation("double", 0, torch.zeros(1))
        mgr.flush_new_to_prev()
        mgr.key_token_indices = torch.tensor([1, 2])

        mgr.clear_cache()
        assert len(mgr.prev_cache) == 0
        assert len(mgr.new_cache) == 0
        assert mgr.key_token_indices is None


class TestSetParameters:
    def test_update_threshold_and_steps(self):
        mgr = _DummyCacheManager(total_step_num=20, cache_interval=5)
        mgr.set_parameters(
            num_inference_steps=40,
            threshold=0.9,
            cache_interval=10,
        )
        assert mgr.total_step_num == 40
        assert mgr.threshold == 0.9
        assert mgr.cache_interval == 10
        assert mgr.cache_steps == {0, 10, 20, 30}

    def test_none_values_skipped(self):
        mgr = _DummyCacheManager(total_step_num=20, threshold=0.5)
        mgr.set_parameters()
        assert mgr.total_step_num == 20
        assert mgr.threshold == 0.5


class TestStatsAndRepr:
    def test_get_stats_keys(self):
        mgr = _DummyCacheManager()
        stats = mgr.get_stats()
        for k in ("current_round", "current_step", "prev_cache_size",
                  "new_cache_size", "cache_steps", "threshold"):
            assert k in stats

    def test_repr_does_not_error(self):
        mgr = _DummyCacheManager()
        s = repr(mgr)
        assert "_DummyCacheManager" in s
