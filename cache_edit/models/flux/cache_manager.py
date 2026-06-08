"""Flux activation cache manager implementation."""

import threading
import time
from bisect import bisect_right
from concurrent.futures import Future, ThreadPoolExecutor
from typing import Any, Dict, List, Optional, Set, Tuple

import torch
from torch import Tensor

from cache_edit.core.cache_manager import BaseCacheManager, StreamType


class FluxCacheManager(BaseCacheManager):
    """
    Flux 模型的激活缓存管理器。

    相比 Qwen 版本，Flux 版本特有的功能：
    - 多轮编辑管理（current_round、is_round0、should_reuse）
    - 稀疏 cache_steps + map_to_group_min 就近复用
    - key_token 选择（余弦相似度 < threshold）
    - img + RoPE PE 同步重排
    - token 顺序恢复
    - 多 GPU 显存动态分配
    - stream_type 切换（double / single）
    - 每 step 末按策略 flush_new_cache_after_step
    - 与原始代码 1:1 行为对齐的 reset()

    Attributes:
        stream_type: 当前 stream 模式（"double" 或 "single"），由 Transformer forward 在循环前设置
        num_gpus: GPU 数量，用于显存动态分配
    """

    def __init__(
        self,
        use_activation_cache: bool = True,
        cache_steps: Optional[Set[int]] = None,
        cache_device: torch.device = torch.device("cuda:0"),
        num_gpus: int = 1,
        total_step_num: int = 30,
        threshold: float = 0.97,
        cache_interval: int = 5,
        gpu_memory_limit_gb: Optional[float] = None,
        gpu_memory_buffer_gb: float = 1.0,
        use_compression: bool = False,
        compression_bitrate: float = 5.0,
        compression_codec: str = "hevc",
        compression_gop_length: int = 1,
        compression_frame_interval_p: int = 1,
    ):
        """
        初始化 Flux 缓存管理器。

        Args:
            use_activation_cache: 总开关
            cache_steps: 显式 cache_steps 集合，None 时根据 cache_interval 自动生成
            cache_device: 起始缓存设备
            num_gpus: 可用 GPU 数量
            total_step_num: 总推理步数
            threshold: key_token 选择的余弦相似度阈值
            cache_interval: 缓存间隔（步数）；值越小缓存越密集，显存占用越大
            gpu_memory_limit_gb: 每张 GPU 的显存上限（GB），None 时自动查询
            gpu_memory_buffer_gb: 显存预留 buffer（GB），防止 OOM
            use_compression: 是否使用 LLM.265 NVENC 压缩
            compression_bitrate: 压缩码率（Mbps），1-10 典型值
            compression_codec: 视频编解码器，'hevc' 或 'h264'
            compression_gop_length: 跨连续 layer 的 GOP 长度；<=1 表示全 I 帧
            compression_frame_interval_p: P 帧间隔；1 表示 IPPP
        """
        super().__init__(
            use_activation_cache=use_activation_cache,
            cache_steps=cache_steps,
            cache_device=cache_device,
            total_step_num=total_step_num,
            threshold=threshold,
            cache_interval=cache_interval,
        )

        self.num_gpus = num_gpus
        self.gpu_memory_limit_gb = gpu_memory_limit_gb
        self.gpu_memory_buffer_gb = gpu_memory_buffer_gb
        self.stream_type: StreamType = "single"

        # 压缩配置
        self.use_compression = use_compression
        self.compression_bitrate = compression_bitrate
        self.compression_codec = compression_codec
        self.compression_gop_length = int(compression_gop_length or 1)
        self.compression_frame_interval_p = int(compression_frame_interval_p or 1)

        # 初始化压缩器（仅在需要时）
        self.compressor = None
        self.decompressor = None
        self._pending_compression_group: Optional[Dict[str, Any]] = None
        self._compression_current_image_key: Optional[str] = None
        self._compression_records: List[Dict[str, Any]] = []
        self._decompression_records: List[Dict[str, Any]] = []
        self._decoded_gop_cache: Dict[Tuple[int, str], List[Tensor]] = {}
        self._decoded_gop_access_order: List[Tuple[int, str]] = []
        self._decoded_gop_max_entries = 2
        self._gop_prefetch_window = 2
        self._gop_prefetch_lock = threading.RLock()
        self._gop_decode_lock = threading.Lock()
        self._gop_prefetch_executor: Optional[ThreadPoolExecutor] = None
        self._gop_prefetch_plan: List[Dict[str, Any]] = []
        self._gop_prefetch_next_index = 0
        self._gop_prefetch_target_step: Optional[int] = None
        self._gop_prefetch_futures: Dict[Tuple[int, str], Future] = {}
        self._gop_prefetch_records: List[Dict[str, Any]] = []
        if self.use_compression:
            try:
                from cache_edit.compression.activation_compressor import (
                    ActivationCompressor,
                    ActivationDecompressor,
                )
                self.compressor = ActivationCompressor(
                    bitrate=compression_bitrate,
                    codec=compression_codec,
                )
                self.decompressor = ActivationDecompressor()
                gop_msg = (
                    f", GOP={self.compression_gop_length}, "
                    f"frame_interval_p={self.compression_frame_interval_p}"
                    if self.compression_gop_length > 1
                    else ", all-I"
                )
                print(
                    f"[FluxCacheManager] Compression enabled: "
                    f"{compression_codec} @ {compression_bitrate}Mbps{gop_msg}"
                )
            except Exception as e:
                print(f"[FluxCacheManager] Failed to initialize compression: {e}")
                print(f"[FluxCacheManager] Falling back to uncompressed cache")
                self.use_compression = False

        # 自动查询 GPU 显存上限
        if self.gpu_memory_limit_gb is None and torch.cuda.is_available():
            self._auto_detect_gpu_limits()

        # Flux 的缓存以 (stream, step, layer_idx) 为键，单层结构（无 cond/uncond 双模式）
        self.prev_cache: Dict[Tuple[StreamType, int, int], Tensor] = {}
        self.new_cache: Dict[Tuple[StreamType, int, int], Tensor] = {}

        # key_token 索引（Flux 单层结构）
        self.key_token_indices: Optional[Tensor] = None

        # PE 缓存与 mask 缓存（按 round 重置）
        self._rearranged_pe_cache: Optional[Tuple[Tensor, Tensor]] = None
        self._rearranged_pe_cache_version: int = -1
        self._restore_masks: Optional[Tuple[Tensor, Tensor]] = None
        self._prev_mask_cache: Optional[Tensor] = None
        self._key_indices_version: int = 0

        # 排序后的 cache_steps（供 bisect 使用）
        self._refresh_cache_steps_sorted()
        print(f"[FluxCacheManager] Initialized cache_steps: {self.cache_steps}")

    def _auto_detect_gpu_limits(self) -> None:
        """自动检测各 GPU 的显存总量，设置为 limit（留 buffer）。"""
        if self.num_gpus <= 0:
            return
        try:
            # 查询第一张卡的总显存作为默认值
            props = torch.cuda.get_device_properties(0)
            total_gb = props.total_memory / (1024**3)
            self.gpu_memory_limit_gb = total_gb
            print(
                f"[FluxCacheManager] Auto-detected GPU memory: {total_gb:.1f} GB "
                f"(buffer: {self.gpu_memory_buffer_gb:.1f} GB)"
            )
        except Exception as e:
            print(f"[FluxCacheManager] Failed to auto-detect GPU memory: {e}")
            self.gpu_memory_limit_gb = 77.0  # fallback

    # ---------- cache_steps 管理 ----------

    def _refresh_cache_steps_sorted(self) -> None:
        if self.cache_steps is None:
            self._cache_steps_sorted: Optional[List[int]] = None
        else:
            self._cache_steps_sorted = sorted(self.cache_steps)

    def _build_cache_steps_from_interval(self) -> None:
        super()._build_cache_steps_from_interval()
        self._refresh_cache_steps_sorted()

    def set_parameters(
        self,
        num_inference_steps: Optional[int] = None,
        threshold: Optional[float] = None,
        cache_device: Optional[torch.device] = None,
        cache_interval: Optional[int] = None,
        num_gpus: Optional[int] = None,
        gpu_memory_limit_gb: Optional[float] = None,
        gpu_memory_buffer_gb: Optional[float] = None,
    ) -> None:
        """
        动态更新参数（兼容关键字调用与 argparse 风格调用）。

        Args:
            num_inference_steps: 推理步数
            threshold: 相似度阈值
            cache_device: 缓存设备
            cache_interval: 缓存间隔（值越小缓存越密集）
            num_gpus: GPU 数量
            gpu_memory_limit_gb: GPU 显存上限（GB）
            gpu_memory_buffer_gb: 显存预留 buffer（GB）
        """
        super().set_parameters(
            num_inference_steps=num_inference_steps,
            threshold=threshold,
            cache_device=cache_device,
            cache_interval=cache_interval,
        )
        if num_gpus is not None:
            self.num_gpus = num_gpus
        if gpu_memory_limit_gb is not None:
            self.gpu_memory_limit_gb = gpu_memory_limit_gb
        if gpu_memory_buffer_gb is not None:
            self.gpu_memory_buffer_gb = gpu_memory_buffer_gb
        self._refresh_cache_steps_sorted()
        print(f"[FluxCacheManager] Updated cache_steps: {self.cache_steps}")

    def set_parameters_from_args(self, args) -> None:
        """从 argparse Namespace 设置参数（与原始 set_parameters(args) 一致）。"""
        self.set_parameters(
            num_inference_steps=getattr(args, "num_inference_steps", None),
            threshold=getattr(args, "threshold", None),
            cache_interval=getattr(args, "cache_interval", None),
            num_gpus=getattr(args, "num_gpus", None),
        )

    # ---------- 轮次 / step ----------

    def on_step_start(self, step: int) -> None:
        """每 step 起始钩子；step==0 时进入新一轮。"""
        self._flush_pending_compression_group()
        keep_prefetch = self._gop_prefetch_target_step == int(step)
        if not keep_prefetch:
            self._reset_gop_prefetch_state(wait=True)
            self._clear_decoded_gop_cache()
        super().on_step_start(step)
        if step == 0:
            self._rearranged_pe_cache = None
            self._rearranged_pe_cache_version = -1
            self._restore_masks = None
            self._prev_mask_cache = None
        if self._gop_prefetch_target_step == self.current_step:
            self._schedule_more_gop_prefetch()
        elif self.should_reuse(self.current_step):
            self._start_gop_prefetch_for_step(self.current_step)
        else:
            key_ref_plan = self._build_key_ref_prefetch_plan(self.current_step)
            next_step = self.current_step + 1
            if next_step < self.total_step_num and self.should_reuse(next_step):
                self._start_gop_prefetch_for_step(
                    next_step,
                    extra_plan=key_ref_plan,
                )
            elif key_ref_plan:
                self._start_gop_prefetch_plan(
                    key_ref_plan,
                    preserve_until_step=self.current_step,
                )

    def map_to_group_min(self, step: int) -> Optional[int]:
        """
        将任意 step 映射到 cache_steps 中 <= step 的最大值。

        Returns:
            该步骤所属的 cache 分组起点；若所有 cache_step 都 > step，则返回 None
        """
        if self.cache_steps is None:
            return step
        if not self._cache_steps_sorted:
            return None
        idx = bisect_right(self._cache_steps_sorted, step) - 1
        if idx < 0:
            return None
        return self._cache_steps_sorted[idx]

    def should_reuse(self, step: int) -> bool:
        """
        Flux 复用策略：非第一轮 且 当前 step 不需要写 cache → 复用。
        """
        return (not self.is_round0) and (not self.should_cache(step))

    # ---------- 多 GPU 显存管理 ----------

    def gpu_has_space(
        self, device: torch.device, extra_bytes: int, limit_gb: float
    ) -> bool:
        """检查指定 GPU 是否有足够空间（预留 buffer）。"""
        # Use memory_reserved (includes PyTorch's memory pool) instead of
        # memory_allocated (only active tensors) for more accurate check
        used = torch.cuda.memory_reserved(device)
        limit_bytes = int(limit_gb * 1024**3)
        buffer_bytes = int(self.gpu_memory_buffer_gb * 1024**3)
        available = limit_bytes - used - buffer_bytes
        return available >= extra_bytes

    def _select_device(self, extra_bytes: int) -> torch.device:
        """
        从 cache_device 起始逐块尝试有空间的 GPU；都满了则 fallback 到起始设备。
        """
        start_dev = self.cache_device
        if start_dev.type != "cuda":
            return start_dev

        start_idx = start_dev.index if start_dev.index is not None else 0
        limit_gb = self.gpu_memory_limit_gb or 77.0

        for dev_idx in range(start_idx, self.num_gpus):
            dev = torch.device(f"cuda:{dev_idx}")
            if self.gpu_has_space(dev, extra_bytes, limit_gb):
                return dev

        for dev_idx in range(0, start_idx):
            dev = torch.device(f"cuda:{dev_idx}")
            if self.gpu_has_space(dev, extra_bytes, limit_gb):
                return dev

        print(
            f"⚠ All GPUs 0..{self.num_gpus - 1} are almost full, "
            f"falling back to {start_dev} (may OOM)"
        )
        return start_dev

    # ---------- 激活读写 ----------

    def set_compression_image_key(self, image_key: Optional[str]) -> None:
        self._reset_gop_prefetch_state(wait=True)
        self._clear_decoded_gop_cache()
        self._compression_current_image_key = (
            None if image_key is None else str(image_key)
        )

    @staticmethod
    def _tensor_nbytes(tensor: Tensor) -> int:
        return int(tensor.numel() * tensor.element_size())

    @classmethod
    def _compressed_auxiliary_bytes(cls, value) -> int:
        """
        Count non-bitstream tensor bytes kept beside the encoded payload.

        `code_size` already tracks encoded bitstream bytes. This helper counts
        quantization scales/offsets and NVENC packet-size metadata so the report
        can show both payload-only and total cached sizes.
        """
        if isinstance(value, torch.Tensor):
            return cls._tensor_nbytes(value)
        if isinstance(value, dict):
            total = 0
            for key, item in value.items():
                if key == "bitstream":
                    continue
                total += cls._compressed_auxiliary_bytes(item)
            return total
        if isinstance(value, list):
            return sum(cls._compressed_auxiliary_bytes(item) for item in value)
        if isinstance(value, tuple):
            return sum(cls._compressed_auxiliary_bytes(item) for item in value)
        if hasattr(value, "bitstream") and hasattr(value, "packet_sizes"):
            return cls._tensor_nbytes(value.packet_sizes)
        return 0

    @staticmethod
    def _safe_ratio(numerator: int, denominator: int) -> Optional[float]:
        if denominator <= 0:
            return None
        return float(numerator) / float(denominator)

    def _record_compression_success(
        self,
        *,
        stream: StreamType,
        layer_idx: int,
        tensor: Tensor,
        compressed: Dict[str, Any],
        cache_device: torch.device,
        elapsed_s: float,
    ) -> None:
        original_bytes = self._tensor_nbytes(tensor)
        payload_bytes = int(compressed.get("code_size", 0) or 0)
        auxiliary_bytes = self._compressed_auxiliary_bytes(compressed)
        total_bytes = payload_bytes + auxiliary_bytes

        self._compression_records.append(
            {
                "status": "ok",
                "image_key": self._compression_current_image_key,
                "round": int(self.current_round),
                "step": int(self.current_step),
                "layer": int(layer_idx),
                "stream": str(stream),
                "original_shape": [int(x) for x in tensor.shape],
                "original_dtype": str(tensor.dtype),
                "source_device": str(tensor.device),
                "cache_device": str(cache_device),
                "codec": str(self.compression_codec),
                "bitrate_mbps": float(self.compression_bitrate),
                "compression_mode": compressed.get(
                    "compression_mode", "intra_layer"
                ),
                "original_bytes": original_bytes,
                "compressed_payload_bytes": payload_bytes,
                "compressed_auxiliary_bytes": auxiliary_bytes,
                "compressed_total_bytes": total_bytes,
                "payload_compression_ratio": self._safe_ratio(
                    original_bytes, payload_bytes
                ),
                "total_compression_ratio": self._safe_ratio(
                    original_bytes, total_bytes
                ),
                "compression_time_s": float(elapsed_s),
            }
        )

    def _record_compression_group_success(
        self,
        *,
        stream: StreamType,
        layer_indices: List[int],
        tensors: List[Tensor],
        compressed: Dict[str, Any],
        cache_device: torch.device,
        elapsed_s: float,
    ) -> None:
        original_bytes = int(sum(self._tensor_nbytes(t) for t in tensors))
        payload_bytes = int(compressed.get("code_size", 0) or 0)
        auxiliary_bytes = self._compressed_auxiliary_bytes(compressed)
        total_bytes = payload_bytes + auxiliary_bytes

        self._compression_records.append(
            {
                "status": "ok",
                "image_key": self._compression_current_image_key,
                "round": int(self.current_round),
                "step": int(self.current_step),
                "layer": int(layer_indices[0]),
                "layers": [int(x) for x in layer_indices],
                "stream": str(stream),
                "original_shape": [int(x) for x in tensors[0].shape],
                "original_dtype": str(tensors[0].dtype),
                "source_device": str(tensors[0].device),
                "cache_device": str(cache_device),
                "codec": str(self.compression_codec),
                "bitrate_mbps": float(self.compression_bitrate),
                "compression_mode": compressed.get(
                    "compression_mode", "inter_layer_gop"
                ),
                "gop_length": compressed.get("gop_length"),
                "frame_interval_p": compressed.get("frame_interval_p"),
                "frame_count": len(tensors),
                "original_bytes": original_bytes,
                "compressed_payload_bytes": payload_bytes,
                "compressed_auxiliary_bytes": auxiliary_bytes,
                "compressed_total_bytes": total_bytes,
                "payload_compression_ratio": self._safe_ratio(
                    original_bytes, payload_bytes
                ),
                "total_compression_ratio": self._safe_ratio(
                    original_bytes, total_bytes
                ),
                "compression_time_s": float(elapsed_s),
            }
        )

    def _record_compression_failure(
        self,
        *,
        stream: StreamType,
        layer_idx: int,
        tensor: Tensor,
        cache_device: torch.device,
        elapsed_s: float,
        error: Exception,
    ) -> None:
        original_bytes = self._tensor_nbytes(tensor)
        self._compression_records.append(
            {
                "status": "failed_fallback_uncompressed",
                "image_key": self._compression_current_image_key,
                "round": int(self.current_round),
                "step": int(self.current_step),
                "layer": int(layer_idx),
                "stream": str(stream),
                "original_shape": [int(x) for x in tensor.shape],
                "original_dtype": str(tensor.dtype),
                "source_device": str(tensor.device),
                "cache_device": str(cache_device),
                "codec": str(self.compression_codec),
                "bitrate_mbps": float(self.compression_bitrate),
                "original_bytes": original_bytes,
                "stored_uncompressed_bytes": original_bytes,
                "compression_time_s": float(elapsed_s),
                "error": str(error),
            }
        )

    def _record_compression_group_failure(
        self,
        *,
        stream: StreamType,
        layer_indices: List[int],
        tensors: List[Tensor],
        cache_device: torch.device,
        elapsed_s: float,
        error: Exception,
    ) -> None:
        original_bytes = int(sum(self._tensor_nbytes(t) for t in tensors))
        self._compression_records.append(
            {
                "status": "failed_fallback_uncompressed",
                "image_key": self._compression_current_image_key,
                "round": int(self.current_round),
                "step": int(self.current_step),
                "layer": int(layer_indices[0]),
                "layers": [int(x) for x in layer_indices],
                "stream": str(stream),
                "original_shape": [int(x) for x in tensors[0].shape],
                "original_dtype": str(tensors[0].dtype),
                "source_device": str(tensors[0].device),
                "cache_device": str(cache_device),
                "codec": str(self.compression_codec),
                "bitrate_mbps": float(self.compression_bitrate),
                "compression_mode": "inter_layer_gop",
                "gop_length": int(self.compression_gop_length),
                "frame_interval_p": int(self.compression_frame_interval_p),
                "frame_count": len(tensors),
                "original_bytes": original_bytes,
                "stored_uncompressed_bytes": original_bytes,
                "compression_time_s": float(elapsed_s),
                "error": str(error),
            }
        )

    def _decompress_cached_activation(
        self,
        compressed_data: Dict[str, Any],
        *,
        stream: StreamType,
        layer_idx: int,
        cache_step: Optional[int],
        target_device: Optional[torch.device],
        load_kind: str,
        frame_index: Optional[int] = None,
    ) -> Tensor:
        if self.decompressor is None:
            raise RuntimeError("No decompressor available for compressed cache")

        t0 = time.time()
        status = "ok"
        error = None
        gop_decode_cache_hit = None
        gop_prefetch_wait_s = 0.0
        gop_decode_source = None
        try:
            if compressed_data.get("compression_mode") == "inter_layer_gop":
                if frame_index is None:
                    raise ValueError("GOP compressed cache requires frame_index")
                (
                    decompressed,
                    gop_decode_cache_hit,
                    gop_prefetch_wait_s,
                    gop_decode_source,
                ) = self._get_decoded_gop_frame(
                    compressed_data,
                    frame_index=int(frame_index),
                    target_device=target_device,
                )
            else:
                decompressed = self.decompressor.decompress(
                    compressed_data,
                    target_device=target_device,
                )
            return decompressed
        except Exception as exc:
            status = "failed"
            error = str(exc)
            raise
        finally:
            elapsed_s = time.time() - t0
            payload_bytes = int(compressed_data.get("code_size", 0) or 0)
            auxiliary_bytes = self._compressed_auxiliary_bytes(compressed_data)
            self._decompression_records.append(
                {
                    "status": status,
                    "image_key": self._compression_current_image_key,
                    "round": int(self.current_round),
                    "request_step": int(self.current_step),
                    "cache_step": (
                        int(cache_step) if cache_step is not None else None
                    ),
                    "layer": int(layer_idx),
                    "stream": str(stream),
                    "load_kind": str(load_kind),
                    "compression_mode": compressed_data.get(
                        "compression_mode", "intra_layer"
                    ),
                    "gop_length": compressed_data.get("gop_length"),
                    "frame_index": (
                        int(frame_index) if frame_index is not None else None
                    ),
                    "target_device": (
                        str(target_device) if target_device is not None else None
                    ),
                    "compressed_payload_bytes": payload_bytes,
                    "compressed_auxiliary_bytes": auxiliary_bytes,
                    "compressed_total_bytes": payload_bytes + auxiliary_bytes,
                    "decompression_time_s": float(elapsed_s),
                    "gop_decode_cache_hit": gop_decode_cache_hit,
                    "gop_prefetch_wait_s": float(gop_prefetch_wait_s),
                    "gop_decode_source": gop_decode_source,
                    "decoded_gop_cache_entries": len(self._decoded_gop_cache),
                    "error": error,
                }
            )

    @staticmethod
    def _move_compressed_value(value, device: torch.device):
        if isinstance(value, torch.Tensor):
            return value.to(device)
        if isinstance(value, dict):
            return {
                k: FluxCacheManager._move_compressed_value(v, device)
                for k, v in value.items()
            }
        if isinstance(value, list):
            return [
                FluxCacheManager._move_compressed_value(v, device)
                for v in value
            ]
        if isinstance(value, tuple):
            return tuple(
                FluxCacheManager._move_compressed_value(v, device)
                for v in value
            )
        if hasattr(value, "bitstream") and hasattr(value, "packet_sizes"):
            return {
                "bitstream": value.bitstream.to(device),
                "packet_sizes": value.packet_sizes.to(device),
            }
        return value

    @staticmethod
    def _decoded_gop_device_key(target_device: Optional[torch.device]) -> str:
        if target_device is None:
            return "__original_devices__"
        return str(torch.device(target_device))

    def _clear_decoded_gop_cache(self) -> None:
        with self._gop_prefetch_lock:
            self._decoded_gop_cache.clear()
            self._decoded_gop_access_order.clear()

    def _ensure_gop_prefetch_executor(self) -> ThreadPoolExecutor:
        if self._gop_prefetch_executor is None:
            self._gop_prefetch_executor = ThreadPoolExecutor(
                max_workers=1,
                thread_name_prefix="cacheedit-gop-prefetch",
            )
        return self._gop_prefetch_executor

    def _decode_gop_sequence(
        self,
        compressed_data: Dict[str, Any],
        target_device: Optional[torch.device],
    ) -> List[Tensor]:
        if self.decompressor is None:
            raise RuntimeError("No decompressor available for compressed cache")
        with self._gop_decode_lock:
            return self.decompressor.decompress_sequence(
                compressed_data,
                target_device=target_device,
            )

    def _install_decoded_gop_frames(
        self,
        cache_key: Tuple[int, str],
        frames: List[Tensor],
    ) -> None:
        with self._gop_prefetch_lock:
            if cache_key in self._decoded_gop_cache:
                if cache_key in self._decoded_gop_access_order:
                    self._decoded_gop_access_order.remove(cache_key)
                self._decoded_gop_access_order.append(cache_key)
                return

            while len(self._decoded_gop_cache) >= self._decoded_gop_max_entries:
                lru_key = self._decoded_gop_access_order.pop(0)
                self._decoded_gop_cache.pop(lru_key, None)

            self._decoded_gop_cache[cache_key] = frames
            self._decoded_gop_access_order.append(cache_key)

    def _take_decoded_gop_cache(
        self,
        cache_key: Tuple[int, str],
    ) -> Optional[List[Tensor]]:
        with self._gop_prefetch_lock:
            frames = self._decoded_gop_cache.get(cache_key)
            if frames is None:
                return None
            if cache_key in self._decoded_gop_access_order:
                self._decoded_gop_access_order.remove(cache_key)
            self._decoded_gop_access_order.append(cache_key)
            return frames

    def _run_gop_prefetch(
        self,
        compressed_data: Dict[str, Any],
        plan_item: Dict[str, Any],
    ) -> Dict[str, Any]:
        t0 = time.time()
        try:
            frames = self._decode_gop_sequence(
                compressed_data,
                target_device=None,
            )
            return {
                "status": "ok",
                "frames": frames,
                "elapsed_s": time.time() - t0,
                "plan_item": plan_item,
            }
        except Exception as exc:
            return {
                "status": "failed",
                "frames": None,
                "elapsed_s": time.time() - t0,
                "error": str(exc),
                "plan_item": plan_item,
            }

    def _record_gop_prefetch_result(
        self,
        result: Dict[str, Any],
        *,
        used: bool,
    ) -> None:
        plan_item = result.get("plan_item") or {}
        self._gop_prefetch_records.append(
            {
                "status": str(result.get("status", "unknown")),
                "used": bool(used),
                "image_key": self._compression_current_image_key,
                "round": int(self.current_round),
                "request_step": int(self.current_step),
                "target_step": plan_item.get("target_step"),
                "cache_step": plan_item.get("cache_step"),
                "stream": plan_item.get("stream"),
                "layers": plan_item.get("layers"),
                "purpose": plan_item.get("purpose", "reuse"),
                "compression_mode": "inter_layer_gop",
                "prefetch_time_s": float(result.get("elapsed_s", 0.0) or 0.0),
                "error": result.get("error"),
            }
        )

    def _schedule_gop_prefetch_locked(self) -> None:
        if self.decompressor is None:
            return
        executor = self._ensure_gop_prefetch_executor()
        while (
            len(self._gop_prefetch_futures) < self._gop_prefetch_window
            and self._gop_prefetch_next_index < len(self._gop_prefetch_plan)
        ):
            plan_item = self._gop_prefetch_plan[self._gop_prefetch_next_index]
            self._gop_prefetch_next_index += 1
            compressed_data = plan_item["compressed_data"]
            cache_key = (
                id(compressed_data),
                self._decoded_gop_device_key(None),
            )
            if (
                cache_key in self._decoded_gop_cache
                or cache_key in self._gop_prefetch_futures
            ):
                continue
            future = executor.submit(
                self._run_gop_prefetch,
                compressed_data,
                plan_item,
            )
            self._gop_prefetch_futures[cache_key] = future

    def _schedule_more_gop_prefetch(self) -> None:
        with self._gop_prefetch_lock:
            self._schedule_gop_prefetch_locked()

    def _reset_gop_prefetch_state(self, *, wait: bool) -> None:
        with self._gop_prefetch_lock:
            futures = list(self._gop_prefetch_futures.items())
            self._gop_prefetch_futures.clear()
            self._gop_prefetch_plan = []
            self._gop_prefetch_next_index = 0
            self._gop_prefetch_target_step = None

        for _cache_key, future in futures:
            if not future.done():
                future.cancel()
            if wait and not future.cancelled():
                try:
                    result = future.result()
                    self._record_gop_prefetch_result(result, used=False)
                except Exception as exc:
                    self._gop_prefetch_records.append(
                        {
                            "status": "failed",
                            "used": False,
                            "image_key": self._compression_current_image_key,
                            "round": int(self.current_round),
                            "request_step": int(self.current_step),
                            "target_step": None,
                            "cache_step": None,
                            "stream": None,
                            "layers": None,
                            "compression_mode": "inter_layer_gop",
                            "prefetch_time_s": 0.0,
                            "error": str(exc),
                        }
                    )

    def _shutdown_gop_prefetch_executor(self) -> None:
        executor = self._gop_prefetch_executor
        self._gop_prefetch_executor = None
        if executor is not None:
            executor.shutdown(wait=True, cancel_futures=True)

    def _build_gop_prefetch_plan(self, step: int) -> List[Dict[str, Any]]:
        if not self.should_reuse(step):
            return []
        step_to_load = self.map_to_group_min(step)
        if step_to_load is None:
            return []

        plan: List[Dict[str, Any]] = []
        seen: Set[int] = set()
        stream_order: Tuple[StreamType, StreamType] = ("double", "single")
        for stream in stream_order:
            stream_items: List[Tuple[int, Dict[str, Any]]] = []
            for (
                cached_stream,
                cached_step,
                layer_idx,
            ), cached in self.prev_cache.items():
                if cached_stream != stream or cached_step != step_to_load:
                    continue
                if not (isinstance(cached, dict) and cached.get("compressed")):
                    continue
                compressed_data = cached.get("data")
                if not (
                    isinstance(compressed_data, dict)
                    and compressed_data.get("compression_mode") == "inter_layer_gop"
                ):
                    continue
                group_id = id(compressed_data)
                if group_id in seen:
                    continue
                seen.add(group_id)
                layers = cached.get("group_layers") or [int(layer_idx)]
                layers = [int(x) for x in layers]
                stream_items.append(
                    (
                        min(layers),
                        {
                            "compressed_data": compressed_data,
                            "target_step": int(step),
                            "cache_step": int(step_to_load),
                            "stream": str(stream),
                            "layers": layers,
                        },
                    )
                )
            stream_items.sort(key=lambda item: item[0])
            plan.extend(item for _first_layer, item in stream_items)
        return plan

    def _build_key_ref_prefetch_plan(self, step: int) -> List[Dict[str, Any]]:
        if (
            self.is_round0
            or self.cache_steps is None
            or step not in self.cache_steps
        ):
            return []

        # Key-token update uses single stream layer 37 by default. Prefetch the
        # containing GOP group at step start so that late-step reference loading
        # does not synchronously decode the whole group.
        key = ("single", int(step), 37)
        cached = self.prev_cache.get(key)
        if not (isinstance(cached, dict) and cached.get("compressed")):
            return []
        compressed_data = cached.get("data")
        if not (
            isinstance(compressed_data, dict)
            and compressed_data.get("compression_mode") == "inter_layer_gop"
        ):
            return []
        layers = cached.get("group_layers") or [37]
        return [
            {
                "compressed_data": compressed_data,
                "target_step": int(step),
                "cache_step": int(step),
                "stream": "single",
                "layers": [int(x) for x in layers],
                "purpose": "key_ref",
            }
        ]

    @staticmethod
    def _merge_prefetch_plans(
        *plans: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        merged: List[Dict[str, Any]] = []
        for plan in plans:
            for item in plan:
                merged.append(item)
        return merged

    def _start_gop_prefetch_plan(
        self,
        plan: List[Dict[str, Any]],
        *,
        preserve_until_step: int,
    ) -> None:
        if not (
            self.use_compression
            and self.decompressor is not None
            and self.compression_gop_length > 1
        ):
            return
        if not plan:
            return
        with self._gop_prefetch_lock:
            self._gop_prefetch_plan = plan
            self._gop_prefetch_next_index = 0
            self._gop_prefetch_target_step = int(preserve_until_step)
            self._schedule_gop_prefetch_locked()

    def _start_gop_prefetch_for_step(
        self,
        step: int,
        *,
        extra_plan: Optional[List[Dict[str, Any]]] = None,
    ) -> None:
        plan = self._merge_prefetch_plans(
            extra_plan or [],
            self._build_gop_prefetch_plan(step),
        )
        self._start_gop_prefetch_plan(
            plan,
            preserve_until_step=step,
        )

    def _get_decoded_gop_frame(
        self,
        compressed_data: Dict[str, Any],
        *,
        frame_index: int,
        target_device: Optional[torch.device],
    ) -> Tuple[Tensor, bool, float, Optional[str]]:
        """
        Decode a GOP group once per target device and reuse frames by index.

        `decompress_sequence_frame()` decodes the whole video sequence internally.
        Calling it once per layer repeats that expensive NVDEC work. This cache
        stores restored tensors for the current step so consecutive layers in
        the same GOP can reuse one decode.
        """
        if self.decompressor is None:
            raise RuntimeError("No decompressor available for compressed cache")

        target_device_obj = (
            torch.device(target_device) if target_device is not None else None
        )
        cache_key = (
            id(compressed_data),
            self._decoded_gop_device_key(target_device_obj),
        )
        canonical_key = (
            id(compressed_data),
            self._decoded_gop_device_key(None),
        )

        cached_frames = self._take_decoded_gop_cache(cache_key)
        if cached_frames is not None:
            self._schedule_more_gop_prefetch()
            return cached_frames[int(frame_index)], True, 0.0, "decoded_cache"

        canonical_frames = self._take_decoded_gop_cache(canonical_key)
        if canonical_frames is not None:
            frame = canonical_frames[int(frame_index)]
            if target_device_obj is not None and frame.device != target_device_obj:
                frame = frame.to(target_device_obj)
            self._schedule_more_gop_prefetch()
            return frame, True, 0.0, "decoded_cache_original_devices"

        future = None
        future_key = None
        with self._gop_prefetch_lock:
            future = self._gop_prefetch_futures.get(cache_key)
            future_key = cache_key if future is not None else None
            if future is None:
                future = self._gop_prefetch_futures.get(canonical_key)
                future_key = canonical_key if future is not None else None

        if future is not None and future_key is not None:
            wait_t0 = time.time()
            result = future.result()
            wait_s = time.time() - wait_t0
            with self._gop_prefetch_lock:
                self._gop_prefetch_futures.pop(future_key, None)
            self._record_gop_prefetch_result(result, used=True)
            if result.get("status") == "ok" and result.get("frames") is not None:
                frames = result["frames"]
                self._install_decoded_gop_frames(canonical_key, frames)
                frame = frames[int(frame_index)]
                if target_device_obj is not None and frame.device != target_device_obj:
                    frame = frame.to(target_device_obj)
                self._schedule_more_gop_prefetch()
                return frame, True, wait_s, "prefetch_future"

        # Synchronous fallback decodes to each frame's original device. This keeps
        # later layers reusable even when the prefetch window did not reach them.
        frames = self._decode_gop_sequence(
            compressed_data,
            target_device=None,
        )
        self._install_decoded_gop_frames(canonical_key, frames)
        frame = frames[int(frame_index)]
        if target_device_obj is not None and frame.device != target_device_obj:
            frame = frame.to(target_device_obj)
        self._schedule_more_gop_prefetch()
        return frame, False, 0.0, "sync_decode"

    def _gop_compression_enabled(self) -> bool:
        return (
            self.use_compression
            and self.compressor is not None
            and self.compression_gop_length > 1
        )

    def _store_uncompressed_activation(
        self,
        key: Tuple[StreamType, int, int],
        tensor: Tensor,
        *,
        smart_device: bool,
    ) -> None:
        if smart_device:
            extra_bytes = tensor.numel() * tensor.element_size()
            target_device = self._select_device(extra_bytes)
            if target_device != self.cache_device:
                print(
                    f"[Cache] step={key[1]} layer={key[2]} "
                    f"→ {target_device} (overflow from {self.cache_device})"
                )
            tensor = tensor.clone().detach()
        else:
            target_device = self.cache_device
        self.new_cache[key] = tensor.to(target_device)

    def _compress_and_store_single(
        self,
        stream: StreamType,
        layer_idx: int,
        tensor: Tensor,
        *,
        smart_device: bool,
    ) -> None:
        key = (stream, self.current_step, layer_idx)
        t0 = time.time()
        try:
            compressed = self.compressor.compress(
                tensor,
                name=f"step{self.current_step}_layer{layer_idx}",
            )
            elapsed_s = time.time() - t0
            extra_bytes = int(compressed.get("code_size", 0) or 0)
            target_device = (
                self._select_device(extra_bytes)
                if smart_device
                else self.cache_device
            )

            if smart_device and target_device != self.cache_device:
                print(
                    f"[Cache] step={self.current_step} layer={layer_idx} "
                    f"→ {target_device} (overflow from {self.cache_device})"
                )

            compressed_on_device = self._move_compressed_value(
                compressed,
                target_device,
            )
            self._record_compression_success(
                stream=stream,
                layer_idx=layer_idx,
                tensor=tensor,
                compressed=compressed,
                cache_device=target_device,
                elapsed_s=elapsed_s,
            )
            self.new_cache[key] = {
                "compressed": True,
                "data": compressed_on_device,
                "frame_index": None,
            }
        except Exception as e:
            elapsed_s = time.time() - t0
            fallback_device = (
                self._select_device(tensor.numel() * tensor.element_size())
                if smart_device
                else self.cache_device
            )
            self._record_compression_failure(
                stream=stream,
                layer_idx=layer_idx,
                tensor=tensor,
                cache_device=fallback_device,
                elapsed_s=elapsed_s,
                error=e,
            )
            print(
                f"[Cache] Compression failed for step={self.current_step} "
                f"layer={layer_idx}: {e}"
            )
            self._store_uncompressed_activation(
                key,
                tensor,
                smart_device=smart_device,
            )

    def _pending_group_matches(
        self,
        stream: StreamType,
        layer_idx: int,
        tensor: Tensor,
        smart_device: bool,
    ) -> bool:
        pending = self._pending_compression_group
        if pending is None:
            return False
        last_entry = pending["entries"][-1]
        return (
            pending["stream"] == stream
            and pending["step"] == self.current_step
            and pending["smart_device"] == smart_device
            and tuple(pending["shape"]) == tuple(tensor.shape)
            and pending["dtype"] == tensor.dtype
            and int(last_entry["layer_idx"]) + 1 == int(layer_idx)
        )

    def _queue_gop_activation(
        self,
        stream: StreamType,
        layer_idx: int,
        tensor: Tensor,
        *,
        smart_device: bool,
    ) -> None:
        if not self._pending_group_matches(
            stream,
            layer_idx,
            tensor,
            smart_device,
        ):
            self._flush_pending_compression_group()
            self._pending_compression_group = {
                "stream": stream,
                "step": self.current_step,
                "smart_device": smart_device,
                "shape": tuple(tensor.shape),
                "dtype": tensor.dtype,
                "entries": [],
            }

        self._pending_compression_group["entries"].append(
            {
                "layer_idx": int(layer_idx),
                "tensor": tensor.detach(),
            }
        )
        if (
            len(self._pending_compression_group["entries"])
            >= self.compression_gop_length
        ):
            self._flush_pending_compression_group()

    def _flush_pending_compression_group(self) -> None:
        pending = self._pending_compression_group
        if not pending:
            return
        self._pending_compression_group = None

        entries = pending["entries"]
        if not entries:
            return

        stream = pending["stream"]
        smart_device = bool(pending["smart_device"])
        layer_indices = [int(entry["layer_idx"]) for entry in entries]
        tensors = [entry["tensor"] for entry in entries]

        if len(entries) == 1:
            self._compress_and_store_single(
                stream,
                layer_indices[0],
                tensors[0],
                smart_device=smart_device,
            )
            return

        t0 = time.time()
        try:
            compressed = self.compressor.compress_sequence(
                tensors,
                name=(
                    f"step{self.current_step}_{stream}_layers"
                    f"{layer_indices[0]}-{layer_indices[-1]}"
                ),
                gop_length=min(self.compression_gop_length, len(tensors)),
                frame_interval_p=self.compression_frame_interval_p,
            )
            elapsed_s = time.time() - t0
            extra_bytes = int(compressed.get("code_size", 0) or 0)
            target_device = (
                self._select_device(extra_bytes)
                if smart_device
                else self.cache_device
            )

            if smart_device and target_device != self.cache_device:
                print(
                    f"[Cache] step={self.current_step} layers="
                    f"{layer_indices[0]}-{layer_indices[-1]} "
                    f"→ {target_device} (overflow from {self.cache_device})"
                )

            compressed_on_device = self._move_compressed_value(
                compressed,
                target_device,
            )
            self._record_compression_group_success(
                stream=stream,
                layer_indices=layer_indices,
                tensors=tensors,
                compressed=compressed,
                cache_device=target_device,
                elapsed_s=elapsed_s,
            )

            for frame_index, layer_idx in enumerate(layer_indices):
                self.new_cache[(stream, self.current_step, layer_idx)] = {
                    "compressed": True,
                    "data": compressed_on_device,
                    "frame_index": int(frame_index),
                    "group_layers": list(layer_indices),
                }
        except Exception as e:
            elapsed_s = time.time() - t0
            fallback_device = (
                self._select_device(sum(self._tensor_nbytes(t) for t in tensors))
                if smart_device
                else self.cache_device
            )
            self._record_compression_group_failure(
                stream=stream,
                layer_indices=layer_indices,
                tensors=tensors,
                cache_device=fallback_device,
                elapsed_s=elapsed_s,
                error=e,
            )
            print(
                f"[Cache] GOP compression failed for step={self.current_step} "
                f"layers={layer_indices[0]}-{layer_indices[-1]}: {e}"
            )
            for layer_idx, tensor in zip(layer_indices, tensors):
                self._store_uncompressed_activation(
                    (stream, self.current_step, layer_idx),
                    tensor,
                    smart_device=smart_device,
                )

    def _store_activation_impl(
        self,
        stream: StreamType,
        layer_idx: int,
        tensor: Tensor,
        *,
        smart_device: bool,
    ) -> None:
        if not (self.use_activation_cache and self.should_cache(self.current_step)):
            return
        if self.use_compression and self.compressor is not None:
            if self._gop_compression_enabled():
                self._queue_gop_activation(
                    stream,
                    layer_idx,
                    tensor,
                    smart_device=smart_device,
                )
            else:
                self._compress_and_store_single(
                    stream,
                    layer_idx,
                    tensor,
                    smart_device=smart_device,
                )
        else:
            self._store_uncompressed_activation(
                (stream, self.current_step, layer_idx),
                tensor,
                smart_device=smart_device,
            )

    def store_activation(
        self,
        stream: StreamType,
        layer_idx: int,
        tensor: Tensor,
    ) -> None:
        """
        简单版存储（直接用 cache_device，无多卡逻辑）。
        """
        self._store_activation_impl(
            stream,
            layer_idx,
            tensor,
            smart_device=False,
        )

    def maby_store_activation(
        self,
        stream: StreamType,
        layer_idx: int,
        tensor: Tensor,
    ) -> None:
        """
        多卡智能存储：从 cache_device 起始依次尝试，无空间则 fallback。
        """
        self._store_activation_impl(
            stream,
            layer_idx,
            tensor,
            smart_device=True,
        )

    def get_activation(
        self,
        stream: StreamType,
        layer_idx: int,
        step: Optional[int] = None,
    ) -> Optional[Tensor]:
        """从 prev_cache 读取（不做设备转换）。"""
        if not self.use_activation_cache:
            return None
        step = step if step is not None else self.current_step
        step_to_load = self.map_to_group_min(step)
        key = (stream, step_to_load, layer_idx)
        cached = self.prev_cache.get(key, None)

        if cached is None:
            return None

        # 检查是否是压缩数据
        if isinstance(cached, dict) and cached.get('compressed', False):
            # 解压缩
            if self.decompressor is not None:
                try:
                    decompressed = self._decompress_cached_activation(
                        cached['data'],
                        stream=stream,
                        layer_idx=layer_idx,
                        cache_step=step_to_load,
                        target_device=None,
                        load_kind="get_activation",
                        frame_index=cached.get("frame_index"),
                    )
                    # 立即清理 CUDA 缓存以防止碎片
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    return decompressed
                except Exception as e:
                    print(f"[Cache] Decompression failed for step={step_to_load} layer={layer_idx}: {e}")
                    return None
            else:
                print(f"[Cache] No decompressor available for compressed cache")
                return None
        else:
            return cached

    def load_activation(
        self,
        stream: StreamType,
        layer_idx: int,
        device: torch.device,
        step: Optional[int] = None,
    ) -> Optional[Tensor]:
        """从 prev_cache 读取并搬到指定设备（用 map_to_group_min 映射）。"""
        if not self.use_activation_cache:
            return None
        step = step if step is not None else self.current_step
        step_to_load = self.map_to_group_min(step)
        key = (stream, step_to_load, layer_idx)
        prev = self.prev_cache.get(key, None)
        if prev is None:
            return None

        # 检查是否是压缩数据
        if isinstance(prev, dict) and prev.get('compressed', False):
            # 解压缩到目标设备
            if self.decompressor is not None:
                try:
                    decompressed = self._decompress_cached_activation(
                        prev['data'],
                        stream=stream,
                        layer_idx=layer_idx,
                        cache_step=step_to_load,
                        target_device=device,
                        load_kind="load_activation",
                        frame_index=prev.get("frame_index"),
                    )
                    return decompressed
                except Exception as e:
                    print(f"[Cache] Decompression failed for step={step_to_load} layer={layer_idx}: {e}")
                    return None
            else:
                print(f"[Cache] No decompressor available for compressed cache")
                return None
        else:
            return prev.to(device)

    def load_key_token_ref(
        self,
        stream: StreamType,
        layer_idx: int,
        device: torch.device,
        step: Optional[int] = None,
    ) -> Optional[Tensor]:
        """读取参考帧激活（不做 group_min 映射，直接按 step 查）。"""
        if not self.use_activation_cache:
            return None
        step = step if step is not None else self.current_step
        key = (stream, step, layer_idx)
        prev = self.prev_cache.get(key, None)
        if prev is None:
            return None

        # 检查是否是压缩数据
        if isinstance(prev, dict) and prev.get('compressed', False):
            # 解压缩到目标设备
            if self.decompressor is not None:
                try:
                    decompressed = self._decompress_cached_activation(
                        prev['data'],
                        stream=stream,
                        layer_idx=layer_idx,
                        cache_step=step,
                        target_device=device,
                        load_kind="load_key_token_ref",
                        frame_index=prev.get("frame_index"),
                    )
                    return decompressed
                except Exception as e:
                    print(f"[Cache] Decompression failed for step={step} layer={layer_idx}: {e}")
                    return None
            else:
                print(f"[Cache] No decompressor available for compressed cache")
                return None
        else:
            return prev.to(device)

    def load_key_token_cur(
        self,
        stream: StreamType,
        layer_idx: int,
        device: torch.device,
        step: Optional[int] = None,
    ) -> Optional[Tensor]:
        """读取当前帧激活（从 new_cache）。"""
        if not self.use_activation_cache:
            return None
        step = step if step is not None else self.current_step
        key = (stream, step, layer_idx)
        cur = self.new_cache.get(key, None)
        if cur is None:
            return None
        if isinstance(cur, dict) and cur.get('compressed', False):
            if self.decompressor is not None:
                try:
                    return self._decompress_cached_activation(
                        cur['data'],
                        stream=stream,
                        layer_idx=layer_idx,
                        cache_step=step,
                        target_device=device,
                        load_kind="load_key_token_cur",
                        frame_index=cur.get("frame_index"),
                    )
                except Exception as e:
                    print(f"[Cache] Decompression failed for step={step} layer={layer_idx}: {e}")
                    return None
            print(f"[Cache] No decompressor available for compressed cache")
            return None
        return cur.to(device)

    def flush_new_cache_after_step(self) -> None:
        """
        每 step 末按策略合并 new_cache 到 prev_cache：
          - round0：若 should_cache(current_step) → 立即合并
          - round>0：若 should_cache(current_step + 1) 且未到最后一步 → 合并
        """
        last_step = self.total_step_num - 1
        if not self.use_activation_cache:
            return
        self._flush_pending_compression_group()
        if not (
            self._gop_prefetch_target_step is not None
            and self._gop_prefetch_target_step > self.current_step
        ):
            self._reset_gop_prefetch_state(wait=True)
            self._clear_decoded_gop_cache()

        if self.is_round0:
            if self.should_cache(self.current_step):
                self.prev_cache.update(self.new_cache)
                self.new_cache.clear()
        else:
            if self.current_step >= last_step:
                return
            if self.should_cache(self.current_step + 1):
                self.prev_cache.update(self.new_cache)
                self.new_cache.clear()

    def flush_new_to_prev(self) -> None:
        """覆盖式 flush（直接替换）。"""
        self._flush_pending_compression_group()
        if not (
            self._gop_prefetch_target_step is not None
            and self._gop_prefetch_target_step > self.current_step
        ):
            self._reset_gop_prefetch_state(wait=True)
            self._clear_decoded_gop_cache()
        self.prev_cache.update(self.new_cache)
        self.new_cache = {}

    # ---------- key_token 计算与重排 ----------

    def compute_key_indices_fn(
        self,
        tensor1: torch.Tensor,
        tensor2: torch.Tensor,
    ) -> torch.Tensor:
        """
        按行计算余弦相似度，归一化到 [0,1]，返回相似度 < threshold 的索引。

        Args:
            tensor1: (n, d)
            tensor2: (n, d)

        Returns:
            indices: 1D LongTensor
        """
        device = tensor2.device
        tensor1 = tensor1.to(device)
        assert tensor1.shape == tensor2.shape, "两个 tensor 的 shape 必须相同"

        dot_product = (tensor1 * tensor2).sum(dim=1)
        norm1 = tensor1.norm(p=2, dim=1)
        norm2 = tensor2.norm(p=2, dim=1)
        eps = 1e-8
        similarities = dot_product / (norm1 * norm2 + eps)
        normalized_similarities = (similarities + 1) / 2

        mask = normalized_similarities < self.threshold
        indices = mask.nonzero(as_tuple=True)[0]
        return indices

    def update_key_token_indices(
        self,
        cur_img: Tensor,
        ref_img: Optional[Tensor],
    ) -> None:
        """
        在第二轮及之后、当前 step 在 cache_steps 中、且有参考帧时更新 key_token_indices。
        """
        if not self.use_activation_cache:
            return
        if self.is_round0:
            return
        if self.cache_steps is None:
            return
        if self.current_step not in self.cache_steps:
            return
        if ref_img is None or cur_img is None:
            return
        indices = self.compute_key_indices_fn(cur_img[0], ref_img[0])
        self.key_token_indices = indices.to(cur_img.device)
        self._key_indices_version += 1

    @staticmethod
    def rearrange_tensor_with_key_token_indices(
        img: Tensor,
        cos_img: Tensor,
        sin_img: Tensor,
        key_token_indices: Tensor,
    ) -> Tuple[Tensor, Tensor, Tensor]:
        """
        把 key token 调到前面（img + cos/sin 同步重排），其余顺序保持。

        Args:
            img: (B, L_img, C)
            cos_img / sin_img: (L_img, D_pe)
            key_token_indices: 1D LongTensor

        Returns:
            (img_new, cos_img_new, sin_img_new)
        """
        img_key = torch.index_select(img, 1, key_token_indices.to(img.device))
        key_token_indices_pe = key_token_indices.to(cos_img.device)
        cos_img_key = torch.index_select(cos_img, 0, key_token_indices_pe)
        sin_img_key = torch.index_select(sin_img, 0, key_token_indices_pe)

        key_token_indices_img = key_token_indices.to(img.device)
        mask = torch.ones(img.size(1), dtype=torch.bool, device=img.device)
        mask[key_token_indices_img] = False
        img_not_key = img[:, mask, :]

        mask_pe = mask.to(cos_img.device)
        cos_img_not_key = cos_img[mask_pe, ...]
        sin_img_not_key = sin_img[mask_pe, ...]

        img_new = torch.cat((img_key, img_not_key), dim=1)
        cos_new = torch.cat((cos_img_key, cos_img_not_key), dim=0)
        sin_new = torch.cat((sin_img_key, sin_img_not_key), dim=0)
        return img_new, cos_new, sin_new

    @staticmethod
    def restore_original_token_order(
        x: Tensor,
        key_token_indices: Tensor,
    ) -> Tensor:
        """把前 K 个 token 放回到 key_token_indices 指定的位置。"""
        K = key_token_indices.size(0)
        x_key = x[:, :K, :]
        x_not_key = x[:, K:, :]
        L = x.size(1)

        mask_not_key = torch.ones(L, dtype=torch.bool, device=x.device)
        mask_not_key[key_token_indices] = False
        mask_key = ~mask_not_key

        out = x.clone()
        out[:, mask_not_key, :] = x_not_key
        out[:, mask_key, :] = x_key
        return out

    def precompute_masks(self, total_img_len: int, device: torch.device) -> None:
        """key_token_indices 更新后，预计算 restore mask 与 prev mask。"""
        if self.key_token_indices is None:
            return
        key_token_indices = self.key_token_indices.to(device)

        mask_not_key = torch.ones(total_img_len, dtype=torch.bool, device=device)
        mask_not_key[key_token_indices] = False
        self._restore_masks = (mask_not_key, ~mask_not_key)
        self._prev_mask_cache = mask_not_key.clone()

    def clear_pe_cache(self) -> None:
        """每轮起始或 key_token_indices 变更后清空 PE 重排缓存。"""
        self._rearranged_pe_cache = None
        self._rearranged_pe_cache_version = -1

    def _rearrange_img_only(self, img: Tensor) -> Tensor:
        """只重排 img（PE 已缓存场景下使用）。"""
        key_token_indices = self.key_token_indices
        if key_token_indices is None:
            return img
        if key_token_indices.device != img.device:
            key_token_indices = key_token_indices.to(img.device)
        img_key = torch.index_select(img, 1, key_token_indices)
        mask = torch.ones(img.size(1), dtype=torch.bool, device=img.device)
        mask[key_token_indices] = False
        img_not_key = img[:, mask, :]
        return torch.cat((img_key, img_not_key), dim=1)

    def maybe_rearrange_img_and_pe(
        self,
        img: Tensor,
        pe: Tuple[Tensor, Tensor],
        txt_len: int,
    ) -> Tuple[Tensor, Tuple[Tensor, Tensor], int, bool]:
        """
        若处于复用轮次且已有 key_token_indices，则对 img + PE 同步重排。

        Args:
            img: hidden states，包含 image tokens（不含文本，因为 transformer forward
                调用前 hidden_states 是 image-only）
            pe: (cos, sin)，每个 shape 为 (L_txt + L_img, D_pe)
            txt_len: 文本 token 数

        Returns:
            (img, pe, key_token_num, should_reuse)
        """
        should_reuse = self.should_reuse(self.current_step)

        cos, sin = pe

        if should_reuse and self.key_token_indices is not None:
            cos_txt, cos_img = cos[:txt_len, :], cos[txt_len:, :]
            sin_txt, sin_img = sin[:txt_len, :], sin[txt_len:, :]

            img, cos_img, sin_img = self.rearrange_tensor_with_key_token_indices(
                img, cos_img, sin_img, self.key_token_indices
            )

            cos = torch.cat((cos_txt, cos_img), dim=0)
            sin = torch.cat((sin_txt, sin_img), dim=0)

            pe = (cos, sin)
            key_token_num = self.key_token_indices.size(0)
        else:
            key_token_num = img.size(1)

        return img, pe, key_token_num, should_reuse

    def maybe_restore_img_order(self, img: Tensor) -> Tensor:
        """复用轮次结束时把 token 顺序还原。"""
        if (
            self.should_reuse(self.current_step)
            and self.key_token_indices is not None
        ):
            img = self.restore_original_token_order(img, self.key_token_indices)
        return img

    # ---------- 清理 ----------

    def clear_cache(self) -> None:
        """清空 prev/new cache 与 key_token_indices。"""
        self._pending_compression_group = None
        self._reset_gop_prefetch_state(wait=True)
        self._clear_decoded_gop_cache()
        self.prev_cache.clear()
        self.new_cache.clear()
        self.key_token_indices = None

    def reset(self) -> None:
        """
        恢复到初始状态：清空所有缓存、重置 round/step 计数、清空 key_token_indices。
        多图评估时每张图结束后调用。
        """
        self._flush_pending_compression_group()
        self._reset_gop_prefetch_state(wait=True)
        self._clear_decoded_gop_cache()
        self.prev_cache.clear()
        self.new_cache.clear()

        self.current_round = -1
        self.current_step = -1

        self.key_token_indices = None
        self._rearranged_pe_cache = None
        self._rearranged_pe_cache_version = -1
        self._restore_masks = None
        self._prev_mask_cache = None
        self._key_indices_version = 0
        self._pending_compression_group = None
        self._clear_decoded_gop_cache()
        self._shutdown_gop_prefetch_executor()
        self._compression_current_image_key = None

        print("[FluxCacheManager] reset to initial state.")

    # ---------- 统计 ----------

    def reset_compression_stats(self) -> None:
        """清空压缩/解压统计记录，不影响缓存内容。"""
        self._compression_records.clear()
        self._decompression_records.clear()
        self._gop_prefetch_records.clear()

    @staticmethod
    def _bytes_to_mib(num_bytes: int) -> float:
        return float(num_bytes) / 1024.0 / 1024.0

    def get_compression_report(
        self,
        include_records: bool = True,
    ) -> Dict[str, Any]:
        """
        Return JSON-serializable compression effectiveness statistics.

        The report distinguishes encoded payload bytes from total cached bytes.
        Total cached bytes include payload plus side metadata such as
        quantization scale/offset tensors and NVENC packet-size tensors.
        """
        success_records = [
            r for r in self._compression_records if r.get("status") == "ok"
        ]
        failure_records = [
            r for r in self._compression_records if r.get("status") != "ok"
        ]
        decompress_success = [
            r for r in self._decompression_records if r.get("status") == "ok"
        ]
        decompress_failure = [
            r for r in self._decompression_records if r.get("status") != "ok"
        ]
        gop_decode_cache_hits = sum(
            1
            for r in self._decompression_records
            if r.get("gop_decode_cache_hit") is True
        )
        gop_decode_cache_misses = sum(
            1
            for r in self._decompression_records
            if r.get("gop_decode_cache_hit") is False
        )
        gop_prefetch_success = [
            r for r in self._gop_prefetch_records if r.get("status") == "ok"
        ]
        gop_prefetch_failure = [
            r for r in self._gop_prefetch_records if r.get("status") != "ok"
        ]
        gop_prefetch_time_s = float(
            sum(
                float(r.get("prefetch_time_s", 0.0) or 0.0)
                for r in self._gop_prefetch_records
            )
        )
        gop_prefetch_wait_s = float(
            sum(
                float(r.get("gop_prefetch_wait_s", 0.0) or 0.0)
                for r in self._decompression_records
            )
        )

        original_bytes = int(
            sum(int(r.get("original_bytes", 0) or 0) for r in success_records)
        )
        payload_bytes = int(
            sum(
                int(r.get("compressed_payload_bytes", 0) or 0)
                for r in success_records
            )
        )
        auxiliary_bytes = int(
            sum(
                int(r.get("compressed_auxiliary_bytes", 0) or 0)
                for r in success_records
            )
        )
        total_bytes = int(
            sum(
                int(r.get("compressed_total_bytes", 0) or 0)
                for r in success_records
            )
        )
        uncompressed_fallback_bytes = int(
            sum(
                int(r.get("stored_uncompressed_bytes", 0) or 0)
                for r in failure_records
            )
        )
        compression_time_s = float(
            sum(
                float(r.get("compression_time_s", 0.0) or 0.0)
                for r in self._compression_records
            )
        )
        decompression_time_s = float(
            sum(
                float(r.get("decompression_time_s", 0.0) or 0.0)
                for r in self._decompression_records
            )
        )
        by_mode: Dict[str, int] = {}
        for record in success_records:
            mode = str(record.get("compression_mode", "intra_layer"))
            by_mode[mode] = by_mode.get(mode, 0) + 1

        summary = {
            "enabled": bool(self.use_compression),
            "codec": str(self.compression_codec),
            "bitrate_mbps": float(self.compression_bitrate),
            "configured_gop_length": int(self.compression_gop_length),
            "configured_frame_interval_p": int(self.compression_frame_interval_p),
            "success_count_by_mode": by_mode,
            "attempt_count": len(self._compression_records),
            "success_count": len(success_records),
            "failure_count": len(failure_records),
            "decompression_count": len(self._decompression_records),
            "decompression_success_count": len(decompress_success),
            "decompression_failure_count": len(decompress_failure),
            "gop_decode_cache_hit_count": int(gop_decode_cache_hits),
            "gop_decode_cache_miss_count": int(gop_decode_cache_misses),
            "gop_prefetch_count": len(self._gop_prefetch_records),
            "gop_prefetch_success_count": len(gop_prefetch_success),
            "gop_prefetch_failure_count": len(gop_prefetch_failure),
            "total_gop_prefetch_time_s": gop_prefetch_time_s,
            "avg_gop_prefetch_time_s": self._safe_ratio(
                gop_prefetch_time_s, len(self._gop_prefetch_records)
            ),
            "total_gop_prefetch_wait_s": gop_prefetch_wait_s,
            "original_bytes": original_bytes,
            "compressed_payload_bytes": payload_bytes,
            "compressed_auxiliary_bytes": auxiliary_bytes,
            "compressed_total_bytes": total_bytes,
            "uncompressed_fallback_bytes": uncompressed_fallback_bytes,
            "original_mib": self._bytes_to_mib(original_bytes),
            "compressed_payload_mib": self._bytes_to_mib(payload_bytes),
            "compressed_auxiliary_mib": self._bytes_to_mib(auxiliary_bytes),
            "compressed_total_mib": self._bytes_to_mib(total_bytes),
            "uncompressed_fallback_mib": self._bytes_to_mib(
                uncompressed_fallback_bytes
            ),
            "payload_compression_ratio": self._safe_ratio(
                original_bytes, payload_bytes
            ),
            "total_compression_ratio": self._safe_ratio(
                original_bytes, total_bytes
            ),
            "auxiliary_over_payload_ratio": self._safe_ratio(
                auxiliary_bytes, payload_bytes
            ),
            "total_compression_time_s": compression_time_s,
            "avg_compression_time_s": self._safe_ratio(
                compression_time_s, len(self._compression_records)
            ),
            "total_decompression_time_s": decompression_time_s,
            "avg_decompression_time_s": self._safe_ratio(
                decompression_time_s, len(self._decompression_records)
            ),
        }

        report: Dict[str, Any] = {"summary": summary}
        if include_records:
            report["compression_records"] = list(self._compression_records)
            report["decompression_records"] = list(self._decompression_records)
            report["gop_prefetch_records"] = list(self._gop_prefetch_records)
        return report

    def get_stats(self) -> Dict[str, object]:
        stats = super().get_stats()
        stats.update(
            {
                "num_gpus": self.num_gpus,
                "stream_type": self.stream_type,
                "prev_cache_keys": len(self.prev_cache),
                "new_cache_keys": len(self.new_cache),
                "has_key_token_indices": self.key_token_indices is not None,
                "compression": self.get_compression_report(
                    include_records=False
                ),
            }
        )
        return stats
