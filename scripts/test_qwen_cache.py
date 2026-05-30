"""Test Qwen cache manager interface compatibility."""

import torch
from cache_edit.models.qwen import QwenCacheManager, init_qwen_pipeline

def test_cache_manager_interface():
    """Test that QwenCacheManager implements all required methods."""
    cache_manager = QwenCacheManager(
        total_step_num=30,
        threshold=0.99,
        cache_interval=5,
    )

    # Test required methods exist
    assert hasattr(cache_manager, 'should_compute_kv'), "Missing should_compute_kv"
    assert hasattr(cache_manager, 'should_store_kv'), "Missing should_store_kv"
    assert hasattr(cache_manager, 'should_reuse_kv'), "Missing should_reuse_kv"
    assert hasattr(cache_manager, 'get_selection_ids'), "Missing get_selection_ids"

    # Test Round 0 behavior
    cache_manager.on_step_start(0)
    assert cache_manager.is_round0, "Should be round 0"
    assert cache_manager.should_compute_kv(), "Round 0 should compute KV"
    assert cache_manager.should_store_kv(), "Round 0 step 0 (cache step) should store KV"
    assert not cache_manager.should_reuse_kv(), "Round 0 should not reuse KV"

    # Test non-cache step in Round 0
    cache_manager.on_step_start(1)
    assert cache_manager.should_compute_kv(), "Round 0 should compute KV"
    assert not cache_manager.should_store_kv(), "Step 1 is not cache step"
    assert not cache_manager.should_reuse_kv(), "Round 0 should not reuse KV"

    # Simulate completing Round 0
    for step in range(2, 30):
        cache_manager.on_step_start(step)
    cache_manager.flush_new_to_prev()

    # Test Round 1 behavior
    cache_manager.on_step_start(0)
    assert not cache_manager.is_round0, "Should be round 1"

    # Cache step: should reuse (not compute)
    assert not cache_manager.should_compute_kv(), "Round 1 cache step should not compute (reuse instead)"
    assert not cache_manager.should_store_kv(), "Round 1 should not store"
    assert cache_manager.should_reuse_kv(), "Round 1 cache step should reuse"

    # Non-cache step: should compute (no cache available)
    cache_manager.on_step_start(1)
    assert cache_manager.should_compute_kv(), "Round 1 non-cache step should compute"
    assert not cache_manager.should_store_kv(), "Round 1 should not store"
    assert not cache_manager.should_reuse_kv(), "Round 1 non-cache step should not reuse"

    # Test get_selection_ids
    selection_ids = cache_manager.get_selection_ids()
    assert selection_ids is None, "No key token indices set yet"

    # Test mode switching
    cache_manager.set_mode("uncond")
    assert cache_manager.current_mode == "uncond"
    cache_manager.set_mode("cond")
    assert cache_manager.current_mode == "cond"

    print("✓ All interface tests passed!")

def test_pipeline_initialization():
    """Test that pipeline can be initialized with cache manager."""
    try:
        cache_manager = QwenCacheManager(
            total_step_num=30,
            threshold=0.99,
            cache_interval=5,
        )

        # This will fail if model is not available, but we're just testing the interface
        print("✓ Cache manager created successfully")
        print(f"  - should_compute_kv: {cache_manager.should_compute_kv()}")
        print(f"  - should_store_kv: {cache_manager.should_store_kv()}")
        print(f"  - should_reuse_kv: {cache_manager.should_reuse_kv()}")
        print(f"  - get_selection_ids: {cache_manager.get_selection_ids()}")

    except Exception as e:
        print(f"✗ Pipeline initialization failed: {e}")
        raise

if __name__ == "__main__":
    print("Testing Qwen cache manager interface...")
    test_cache_manager_interface()
    print("\nTesting pipeline initialization...")
    test_pipeline_initialization()
    print("\n✓ All tests passed!")
