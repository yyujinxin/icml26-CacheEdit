"""Multi-GPU memory manager for Flux pipeline.

Handles:
1. Encoder offloading to CPU after encoding
2. Dynamic activation placement across GPUs to avoid OOM
3. Smart weight/activation transfer between devices
"""

import torch
from typing import Optional, List, Dict, Any
import logging

logger = logging.getLogger(__name__)


class MultiGPUMemoryManager:
    """Manages memory across multiple GPUs for Flux inference."""

    def __init__(
        self,
        num_gpus: int = 1,
        primary_device: str = "cuda:0",
        memory_limit_per_gpu_gb: float = 20.0,
        memory_buffer_gb: float = 2.0,
        enable_encoder_offload: bool = True,
    ):
        """
        Args:
            num_gpus: Number of GPUs to use
            primary_device: Primary device for main computation
            memory_limit_per_gpu_gb: Maximum memory to use per GPU in GB
            memory_buffer_gb: Memory buffer to keep free in GB
            enable_encoder_offload: Whether to offload encoders to CPU after use
        """
        self.num_gpus = num_gpus
        self.primary_device = torch.device(primary_device)
        self.memory_limit = memory_limit_per_gpu_gb * 1024 * 1024 * 1024
        self.memory_buffer = memory_buffer_gb * 1024 * 1024 * 1024
        self.enable_encoder_offload = enable_encoder_offload

        # Available devices
        self.devices = [torch.device(f"cuda:{i}") for i in range(num_gpus)]

        # Track offloaded components
        self.offloaded_components = {}

        # Round-robin counter for activation placement
        self.activation_device_idx = 0

        logger.info(
            f"[MemoryManager] Initialized with {num_gpus} GPUs, "
            f"limit={memory_limit_per_gpu_gb}GB/GPU, "
            f"buffer={memory_buffer_gb}GB, "
            f"encoder_offload={enable_encoder_offload}"
        )

    def get_device_memory_info(self, device_idx: int) -> Dict[str, float]:
        """Get memory usage info for a specific GPU."""
        device = torch.device(f"cuda:{device_idx}")
        torch.cuda.synchronize(device)

        total = torch.cuda.get_device_properties(device_idx).total_memory
        allocated = torch.cuda.memory_allocated(device_idx)
        reserved = torch.cuda.memory_reserved(device_idx)
        free = total - allocated

        return {
            "total_gb": total / (1024**3),
            "allocated_gb": allocated / (1024**3),
            "reserved_gb": reserved / (1024**3),
            "free_gb": free / (1024**3),
        }

    def print_memory_summary(self):
        """Print memory usage summary for all GPUs."""
        print("\n" + "="*60)
        print("GPU Memory Summary:")
        print("="*60)
        for i in range(self.num_gpus):
            info = self.get_device_memory_info(i)
            print(
                f"GPU {i}: {info['allocated_gb']:.2f}GB / {info['total_gb']:.2f}GB "
                f"({info['allocated_gb']/info['total_gb']*100:.1f}% used, "
                f"{info['free_gb']:.2f}GB free)"
            )
        print("="*60 + "\n")

    def find_best_device_for_activation(self, tensor_size_bytes: int) -> torch.device:
        """
        Find the best GPU to place an activation tensor.

        Strategy: Round-robin with memory check
        """
        if self.num_gpus == 1:
            return self.primary_device

        # Try round-robin first
        attempts = 0
        while attempts < self.num_gpus:
            candidate_idx = self.activation_device_idx
            self.activation_device_idx = (self.activation_device_idx + 1) % self.num_gpus

            device_idx = candidate_idx
            allocated = torch.cuda.memory_allocated(device_idx)

            # Check if this device has enough free memory
            if (allocated + tensor_size_bytes + self.memory_buffer) < self.memory_limit:
                logger.debug(f"[MemoryManager] Placing activation on cuda:{device_idx}")
                return self.devices[device_idx]

            attempts += 1

        # Fallback: use device with most free memory
        min_usage = float('inf')
        best_idx = 0
        for i in range(self.num_gpus):
            usage = torch.cuda.memory_allocated(i)
            if usage < min_usage:
                min_usage = usage
                best_idx = i

        logger.warning(
            f"[MemoryManager] All devices near limit, using cuda:{best_idx} "
            f"(usage: {min_usage/(1024**3):.2f}GB)"
        )
        return self.devices[best_idx]

    def offload_encoder_to_cpu(self, encoder, name: str):
        """Offload an encoder module to CPU."""
        if not self.enable_encoder_offload:
            return

        if encoder is None:
            return

        original_device = next(encoder.parameters()).device
        if original_device.type == 'cpu':
            return

        logger.info(f"[MemoryManager] Offloading {name} to CPU...")
        encoder.to('cpu')
        torch.cuda.empty_cache()

        self.offloaded_components[name] = {
            'module': encoder,
            'original_device': original_device
        }

    def load_encoder_to_gpu(self, name: str, target_device: Optional[torch.device] = None):
        """Temporarily load an offloaded encoder back to GPU."""
        if name not in self.offloaded_components:
            return None

        component = self.offloaded_components[name]
        encoder = component['module']
        device = target_device or component['original_device']

        logger.info(f"[MemoryManager] Loading {name} to {device}...")
        encoder.to(device)

        return encoder

    def offload_encoder_back_to_cpu(self, name: str):
        """Offload encoder back to CPU after temporary use."""
        if name not in self.offloaded_components:
            return

        encoder = self.offloaded_components[name]['module']
        logger.info(f"[MemoryManager] Re-offloading {name} to CPU...")
        encoder.to('cpu')
        torch.cuda.empty_cache()

    def move_activation_if_needed(self, tensor: torch.Tensor, target_device: torch.device) -> torch.Tensor:
        """
        Move activation tensor to target device if needed.
        Handles cross-GPU data transfer efficiently.
        """
        if tensor.device == target_device:
            return tensor

        # Use non_blocking for async transfer when possible
        return tensor.to(target_device, non_blocking=True)

    def balance_transformer_blocks(self, transformer, num_gpus: int):
        """
        Balance transformer blocks across multiple GPUs.

        Args:
            transformer: The transformer model
            num_gpus: Number of GPUs to distribute across
        """
        if num_gpus <= 1:
            return

        # Get total number of blocks
        if hasattr(transformer, 'transformer_blocks'):
            double_blocks = transformer.transformer_blocks
            single_blocks = transformer.single_transformer_blocks

            total_blocks = len(double_blocks) + len(single_blocks)
            blocks_per_gpu = total_blocks // num_gpus

            logger.info(
                f"[MemoryManager] Distributing {total_blocks} blocks across {num_gpus} GPUs "
                f"({blocks_per_gpu} blocks/GPU)"
            )

            # Distribute double blocks
            for idx, block in enumerate(double_blocks):
                gpu_idx = min(idx // blocks_per_gpu, num_gpus - 1)
                device = f"cuda:{gpu_idx}"
                # Note: actual block movement should be handled by device_map
                logger.debug(f"  Double block {idx} -> {device}")

            # Distribute single blocks
            double_count = len(double_blocks)
            for idx, block in enumerate(single_blocks):
                global_idx = double_count + idx
                gpu_idx = min(global_idx // blocks_per_gpu, num_gpus - 1)
                device = f"cuda:{gpu_idx}"
                logger.debug(f"  Single block {idx} -> {device}")

    def optimize_pipeline_memory(self, pipeline):
        """
        Optimize pipeline memory usage.

        1. Offload text encoders to CPU
        2. Keep VAE on GPU but can offload between encode/decode
        3. Distribute transformer across GPUs
        """
        logger.info("[MemoryManager] Optimizing pipeline memory layout...")

        # Offload text encoders
        if hasattr(pipeline, 'text_encoder'):
            self.offload_encoder_to_cpu(pipeline.text_encoder, 'text_encoder')

        if hasattr(pipeline, 'text_encoder_2'):
            self.offload_encoder_to_cpu(pipeline.text_encoder_2, 'text_encoder_2')

        # Print memory after offloading
        self.print_memory_summary()

    def clear_cache(self):
        """Clear CUDA cache on all GPUs."""
        for i in range(self.num_gpus):
            with torch.cuda.device(i):
                torch.cuda.empty_cache()
        logger.debug("[MemoryManager] Cleared CUDA cache on all GPUs")
