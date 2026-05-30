"""Simple end-to-end test for Qwen cache pipeline."""

import torch
from cache_edit.models.qwen import create_default_cache_manager, init_qwen_pipeline

def test_qwen_pipeline_with_cache():
    """Test that Qwen pipeline can be initialized with cache manager."""
    print("Creating cache manager...")
    cache_manager = create_default_cache_manager(
        num_inference_steps=30,
        threshold=0.99,
        cache_interval=5,
    )

    print("Cache manager created successfully")
    print(f"  - use_activation_cache: {cache_manager.use_activation_cache}")
    print(f"  - cache_steps: {sorted(cache_manager.cache_steps) if cache_manager.cache_steps else 'None'}")
    print(f"  - threshold: {cache_manager.threshold}")
    print(f"  - cache_interval: {cache_manager.cache_interval}")

    # Test interface methods
    print("\nTesting interface methods (Round 0, Step 0):")
    cache_manager.on_step_start(0)
    print(f"  - should_compute_kv: {cache_manager.should_compute_kv()}")
    print(f"  - should_store_kv: {cache_manager.should_store_kv()}")
    print(f"  - should_reuse_kv: {cache_manager.should_reuse_kv()}")
    print(f"  - get_selection_ids: {cache_manager.get_selection_ids()}")

    print("\n✓ Qwen cache manager is ready for use!")
    print("\nTo use with pipeline:")
    print("  pipeline = init_qwen_pipeline(")
    print("      model_path='Qwen/Qwen2-VL-7B-Instruct',")
    print("      device='cuda',")
    print("      dtype=torch.bfloat16,")
    print("      cache_manager=cache_manager,")
    print("  )")

if __name__ == "__main__":
    test_qwen_pipeline_with_cache()
