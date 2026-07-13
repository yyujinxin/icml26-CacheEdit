"""
Activation compression using LLM.265 NVENC hardware acceleration.

Compresses Flux activation tensors with NVENC/NVDEC codecs to reduce cache memory.

`hevc` and `h264` use lossy video rate control. `lossless` still uses the
HEVC/NVENC codec, but switches the codec to lossless mode after the activation
has been quantized into uint8 frames.
"""

import torch
from typing import Dict, Any, List, Optional
import logging
import gc
from contextlib import nullcontext

logger = logging.getLogger(__name__)


def _move_compressed_value(value, device: torch.device):
    if isinstance(value, torch.Tensor):
        return value.to(device)
    if isinstance(value, dict):
        return {
            k: _move_compressed_value(v, device)
            for k, v in value.items()
        }
    if isinstance(value, list):
        return [
            _move_compressed_value(v, device)
            for v in value
        ]
    if isinstance(value, tuple):
        return tuple(
            _move_compressed_value(v, device)
            for v in value
        )
    if hasattr(value, "bitstream") and hasattr(value, "packet_sizes"):
        return {
            "bitstream": value.bitstream.to(device),
            "packet_sizes": value.packet_sizes.to(device),
        }
    return value


def _slice_outliers_for_frame(
    data_dict: Dict[str, Any],
    frame_index: int,
    rows_per_frame: int,
    group_size: int,
) -> Dict[str, Any]:
    indices = data_dict.get("outlier_indices")
    residuals = data_dict.get("outlier_residuals")
    if not (
        isinstance(indices, torch.Tensor)
        and isinstance(residuals, torch.Tensor)
        and indices.numel() > 0
        and residuals.numel() > 0
    ):
        return data_dict

    flat_start = int(frame_index) * int(rows_per_frame) * int(group_size)
    flat_end = flat_start + int(rows_per_frame) * int(group_size)
    mask = (indices >= flat_start) & (indices < flat_end)
    data_dict["outlier_indices"] = (indices[mask] - flat_start).to(torch.int32)
    data_dict["outlier_residuals"] = residuals[mask]
    return data_dict


try:
    from .ops import (
        CodecType, RateControlMode, PresetType, TuningInfo, InputFormat,
        TensorEncodeConfig, TensorEncoder, TensorDecoder, EncodeQp
    )
    from .pipeline import (
        Pipeline,
        CWQuantization,
        GWQuantization,
        GWOutlierQuantization,
        FixedTiling,
        MonoNVEncode,
        MonoNVEncodeSequence,
    )
    from .pipeline.definitions import Step
    NVENC_AVAILABLE = True
except ImportError as e:
    logger.warning(f"NVENC ops not available - compression will be disabled: {e}")
    NVENC_AVAILABLE = False
    # Define dummy classes for type hints
    Pipeline = object
    CWQuantization = object
    GWQuantization = object
    GWOutlierQuantization = object
    FixedTiling = object
    MonoNVEncode = object
    MonoNVEncodeSequence = object


DEFAULT_LOSSLESS_QUANT_GROUP_SIZE = 256


def _normalize_quant_group_size(group_size: Optional[int]) -> Optional[int]:
    if group_size is None:
        return None
    group_size = int(group_size)
    if group_size <= 0:
        return None
    return group_size


def _normalize_quant_outlier_ratio(outlier_ratio: Optional[float]) -> float:
    if outlier_ratio is None:
        return 0.0
    return max(0.0, float(outlier_ratio))


def _make_quantization_step(
    codec: str,
    width: int,
    quant_group_size: Optional[int] = DEFAULT_LOSSLESS_QUANT_GROUP_SIZE,
    quant_outlier_ratio: float = 0.0,
):
    """Choose the quantizer paired with the codec path."""
    group_size = _normalize_quant_group_size(quant_group_size)
    if group_size and width % group_size == 0:
        outlier_ratio = _normalize_quant_outlier_ratio(quant_outlier_ratio)
        if outlier_ratio > 0.0:
            return GWOutlierQuantization(
                groupsize=group_size,
                outlier_ratio=outlier_ratio,
            )
        return GWQuantization(groupsize=group_size)
    return CWQuantization()


def _quantization_name(
    codec: str,
    width: int,
    quant_group_size: Optional[int] = DEFAULT_LOSSLESS_QUANT_GROUP_SIZE,
    quant_outlier_ratio: float = 0.0,
) -> str:
    group_size = _normalize_quant_group_size(quant_group_size)
    if group_size and width % group_size == 0:
        if _normalize_quant_outlier_ratio(quant_outlier_ratio) > 0.0:
            return f"gwo{group_size}"
        return f"gw{group_size}"
    return "cw"


def _quantization_variant(quantization: str) -> str:
    if str(quantization).startswith("gwo"):
        return "group_round_zero_point_outlier"
    if str(quantization).startswith("gw"):
        return "group_round_zero_point"
    return "channel_min_offset"


def _quantization_rows_per_frame(
    quantization: Optional[str],
    height: int,
    width: int,
) -> int:
    quantization = str(quantization or "")
    if quantization.startswith("gwo"):
        prefix_len = 3
    elif quantization.startswith("gw"):
        prefix_len = 2
    else:
        prefix_len = 0
    if prefix_len:
        try:
            group_size = int(quantization[prefix_len:])
        except ValueError:
            group_size = DEFAULT_LOSSLESS_QUANT_GROUP_SIZE
        return int(height * width // group_size)
    return int(height)


class ActivationCompressor:
    """
    Compresses Flux activation tensors using NVENC hardware acceleration.

    Supports variable activation shapes by adapting tiling strategy.
    """

    def __init__(
        self,
        bitrate: float = 5.0,
        codec: str = "hevc",
        tile_height: int = 2048,
        tile_width: int = 2048,
        bitrate_max_multiplier: float = 10.0,
        max_cached_pipelines: int = 2,
        quant_group_size: Optional[int] = DEFAULT_LOSSLESS_QUANT_GROUP_SIZE,
        quant_outlier_ratio: float = 0.0,
        rc_mode: str = "vbr",
        const_qp: Optional[int] = None,
        const_qp_intra: Optional[int] = None,
        const_qp_inter_p: Optional[int] = None,
        const_qp_inter_b: Optional[int] = None,
        codec_preset: str = "p7",
        codec_tuning: str = "high_quality",
        codec_spatial_aq: Optional[int] = None,
        codec_temporal_aq: Optional[bool] = None,
        codec_target_quality: Optional[int] = None,
    ):
        """
        Args:
            bitrate: Average bitrate in Mbps (1-10 typical)
            codec: 'lossless', 'hevc', or 'h264'. lossless means HEVC/NVENC
                lossless coding of the quantized frames, not raw tensor storage.
            tile_height: Height of video tiles
            tile_width: Width of video tiles
            bitrate_max_multiplier: Max bitrate as multiple of average
            max_cached_pipelines: Maximum number of pipelines to cache (to limit NVENC resources)
            quant_group_size: Group size for lossless codec uint8 activation
                quantization. Use <=0 to force channel-wise quantization.
            quant_outlier_ratio: Optional fraction of the worst quantization
                residuals to store exactly beside the codec payload.
            rc_mode: Rate-control mode for hevc/h264: vbr, cbr, or constqp.
            const_qp: Constant QP value used when rc_mode=constqp.
            const_qp_intra: Optional I-frame QP override for constqp mode.
            const_qp_inter_p: Optional P-frame QP override for constqp mode.
            const_qp_inter_b: Optional B-frame QP override for constqp mode.
            codec_preset: NVENC preset p1..p7. p7 is highest quality/slowest.
            codec_tuning: NVENC tuning: high_quality, low_latency,
                ultra_low_latency.
            codec_spatial_aq: Optional NVENC spatial AQ strength.
            codec_temporal_aq: Optional NVENC temporal AQ toggle.
            codec_target_quality: Optional NVENC VBR target quality value.
        """
        codec = str(codec).lower()
        if not NVENC_AVAILABLE:
            raise RuntimeError("NVENC not available - cannot create ActivationCompressor")

        self.bitrate = bitrate
        self.codec = codec
        self.tile_height = tile_height
        self.tile_width = tile_width
        self.bitrate_max_multiplier = bitrate_max_multiplier
        self.max_cached_pipelines = max_cached_pipelines
        self.quant_group_size = _normalize_quant_group_size(quant_group_size)
        self.quant_outlier_ratio = _normalize_quant_outlier_ratio(
            quant_outlier_ratio
        )
        self.rc_mode = str(rc_mode or "vbr").lower()
        if self.rc_mode == "constqp":
            self.const_qp = 28 if const_qp is None else int(const_qp)
        else:
            self.const_qp = None if const_qp is None else int(const_qp)
        self.const_qp_intra = (
            None if const_qp_intra is None else int(const_qp_intra)
        )
        self.const_qp_inter_p = (
            None if const_qp_inter_p is None else int(const_qp_inter_p)
        )
        self.const_qp_inter_b = (
            None if const_qp_inter_b is None else int(const_qp_inter_b)
        )
        self.codec_preset = str(codec_preset or "p7").lower()
        self.codec_tuning = str(codec_tuning or "high_quality").lower()
        self.codec_spatial_aq = (
            None if codec_spatial_aq is None else int(codec_spatial_aq)
        )
        self.codec_temporal_aq = codec_temporal_aq
        self.codec_target_quality = (
            None if codec_target_quality is None else int(codec_target_quality)
        )

        # Cache pipelines per GPU: {gpu_id: {shape_key: pipeline}}
        self._pipeline_cache_per_gpu = {}
        self._pipeline_access_order_per_gpu = {}
        self._sequence_pipeline_cache_per_gpu = {}
        self._sequence_pipeline_access_order_per_gpu = {}

        logger.info(
            f"[ActivationCompressor] Initialized: bitrate={bitrate}Mbps, "
            f"codec={codec}, quant_group_size={self.quant_group_size}, "
            f"quant_outlier_ratio={self.quant_outlier_ratio}, "
            f"rc_mode={self.rc_mode}, const_qp={self.const_qp}, "
            f"const_qp_intra={self.const_qp_intra}, "
            f"const_qp_inter_p={self.const_qp_inter_p}, "
            f"const_qp_inter_b={self.const_qp_inter_b}, "
            f"preset={self.codec_preset}, tuning={self.codec_tuning}, "
            f"spatial_aq={self.codec_spatial_aq}, "
            f"temporal_aq={self.codec_temporal_aq}, "
            f"target_quality={self.codec_target_quality}, "
            f"max_pipelines={max_cached_pipelines} per GPU"
        )

    def clear_pipeline_cache(self) -> None:
        """Drop cached pipeline objects so native NVENC resources can be freed."""
        self._pipeline_cache_per_gpu.clear()
        self._pipeline_access_order_per_gpu.clear()
        self._sequence_pipeline_cache_per_gpu.clear()
        self._sequence_pipeline_access_order_per_gpu.clear()
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def _create_config(
        self,
        gop_length: Optional[int] = None,
        frame_interval_p: Optional[int] = None,
    ) -> TensorEncodeConfig:
        config = TensorEncodeConfig()
        config.input_format = InputFormat.NV12
        config.codec_type = CodecType.H264 if self.codec == "h264" else CodecType.HEVC
        if self.codec == "lossless":
            qp = EncodeQp()
            qp.qpInterP = 0
            qp.qpInterB = 0
            qp.qpIntra = 0
            config.rc_mode = RateControlMode.ConstQP
            config.const_qp = qp
            config.preset = PresetType.P1
            config.tuning_info = TuningInfo.Lossless
        else:
            if self.rc_mode == "constqp":
                qp = EncodeQp()
                base_qp = int(self.const_qp if self.const_qp is not None else 28)
                qp.qpInterP = int(
                    self.const_qp_inter_p
                    if self.const_qp_inter_p is not None
                    else base_qp
                )
                qp.qpInterB = int(
                    self.const_qp_inter_b
                    if self.const_qp_inter_b is not None
                    else base_qp
                )
                qp.qpIntra = int(
                    self.const_qp_intra
                    if self.const_qp_intra is not None
                    else base_qp
                )
                config.rc_mode = RateControlMode.ConstQP
                config.const_qp = qp
            elif self.rc_mode == "cbr":
                config.average_bit_rate = int(self.bitrate * 1000000)
                config.max_bit_rate = int(
                    self.bitrate * 1000000 * self.bitrate_max_multiplier
                )
                config.rc_mode = RateControlMode.CBR
            else:
                config.average_bit_rate = int(self.bitrate * 1000000)
                config.max_bit_rate = int(
                    self.bitrate * 1000000 * self.bitrate_max_multiplier
                )
                if self.codec_target_quality is not None:
                    config.target_quality = int(self.codec_target_quality)
                config.rc_mode = RateControlMode.VBR
            config.preset = self._preset_type(self.codec_preset)
            config.tuning_info = self._tuning_info(self.codec_tuning)
            if self.codec_spatial_aq is not None:
                config.spatial_aq = int(self.codec_spatial_aq)
            if self.codec_temporal_aq is not None:
                config.temporal_aq = bool(self.codec_temporal_aq)
        config.monochrome = True

        if gop_length is not None and gop_length > 1:
            config.gop_length = int(gop_length)
            config.frame_interval_p = int(frame_interval_p or 1)
        else:
            config.gop_length = None
            config.frame_interval_p = None
        return config

    @staticmethod
    def _preset_type(value: str):
        table = {
            "p1": PresetType.P1,
            "p2": PresetType.P2,
            "p3": PresetType.P3,
            "p4": PresetType.P4,
            "p5": PresetType.P5,
            "p6": PresetType.P6,
            "p7": PresetType.P7,
        }
        key = str(value or "p7").lower()
        if key not in table:
            raise ValueError(f"Unsupported NVENC preset: {value}")
        return table[key]

    @staticmethod
    def _tuning_info(value: str):
        table = {
            "high_quality": TuningInfo.HighQuality,
            "hq": TuningInfo.HighQuality,
            "low_latency": TuningInfo.LowLatency,
            "ll": TuningInfo.LowLatency,
            "ultra_low_latency": TuningInfo.UltraLowLatency,
            "ull": TuningInfo.UltraLowLatency,
        }
        key = str(value or "high_quality").lower()
        if key not in table:
            raise ValueError(f"Unsupported NVENC tuning: {value}")
        return table[key]

    def _create_pipeline(self, height: int, width: int) -> Pipeline:
        """Create all-I compression pipeline for one activation tensor."""
        config = self._create_config()

        # Calculate padded dimensions that are divisible by tile size
        padded_height = ((height + self.tile_height - 1) // self.tile_height) * self.tile_height
        padded_width = ((width + self.tile_width - 1) // self.tile_width) * self.tile_width

        # Note: We need to add a step that injects 'shape' key before FixedTiling
        class AddShape(Step):
            def __init__(self):
                super().__init__("AddShape", required_keys=["data"], yield_keys=["shape"])

            def forward(self, data_dict):
                data_dict["shape"] = data_dict["data"].shape
                return data_dict

            def backward(self, data_dict):
                return data_dict

        pipeline = Pipeline([
            AddShape(),  # Add shape key for FixedTiling
            _make_quantization_step(
                self.codec,
                width,
                self.quant_group_size,
                self.quant_outlier_ratio,
            ),  # FP16 -> uint8 quantization
            FixedTiling(
                pad_to_shape=[padded_height, padded_width],
                resize_to_shape=[1, 1, padded_height, padded_width],
                tile_shape=[1, 1, self.tile_height, self.tile_width]
            ),
            MonoNVEncode(config, self.tile_height, self.tile_width),
        ])

        return pipeline

    def _create_sequence_pipeline(
        self,
        height: int,
        width: int,
        frame_count: int,
        gop_length: Optional[int],
        frame_interval_p: Optional[int],
    ) -> Pipeline:
        """Create inter-layer GOP pipeline for a sequence of activations."""
        effective_gop = min(int(gop_length or frame_count), frame_count)
        config = self._create_config(
            gop_length=effective_gop,
            frame_interval_p=frame_interval_p or 1,
        )

        padded_height = ((height + self.tile_height - 1) // self.tile_height) * self.tile_height
        padded_width = ((width + self.tile_width - 1) // self.tile_width) * self.tile_width

        class AddShape(Step):
            def __init__(self):
                super().__init__("AddShape", required_keys=["data"], yield_keys=["shape"])

            def forward(self, data_dict):
                data_dict["shape"] = data_dict["data"].shape
                return data_dict

            def backward(self, data_dict):
                return data_dict

        return Pipeline([
            AddShape(),
            _make_quantization_step(
                self.codec,
                width,
                self.quant_group_size,
                self.quant_outlier_ratio,
            ),
            FixedTiling(
                pad_to_shape=[frame_count, padded_height, padded_width],
                resize_to_shape=[frame_count, 1, padded_height, padded_width],
                tile_shape=[1, 1, self.tile_height, self.tile_width],
            ),
            MonoNVEncodeSequence(config, self.tile_height, self.tile_width),
        ])

    def _get_pipeline(self, height: int, width: int, device: torch.device) -> Pipeline:
        """Get or create cached pipeline for shape and device with LRU eviction."""
        # Get GPU ID from device
        gpu_id = device.index if device.type == 'cuda' and device.index is not None else 0
        shape_key = (height, width, self.quant_outlier_ratio)

        # Initialize cache for this GPU if needed
        if gpu_id not in self._pipeline_cache_per_gpu:
            self._pipeline_cache_per_gpu[gpu_id] = {}
            self._pipeline_access_order_per_gpu[gpu_id] = []

        pipeline_cache = self._pipeline_cache_per_gpu[gpu_id]
        access_order = self._pipeline_access_order_per_gpu[gpu_id]

        # Check if already cached for this GPU
        if shape_key in pipeline_cache:
            # Move to end (most recently used)
            access_order.remove(shape_key)
            access_order.append(shape_key)
            return pipeline_cache[shape_key]

        # Need to create new pipeline
        # First check if we need to evict old pipelines on this GPU
        while len(pipeline_cache) >= self.max_cached_pipelines:
            # Evict least recently used
            lru_key = access_order.pop(0)
            evicted = pipeline_cache.pop(lru_key)
            logger.info(f"[ActivationCompressor] Evicted pipeline for GPU {gpu_id}, shape {lru_key} (LRU)")
            del evicted  # Explicitly delete to free NVENC resources

        # Create new pipeline on the correct GPU
        logger.info(f"[ActivationCompressor] Creating pipeline for GPU {gpu_id}, shape ({height}, {width})")
        pipeline = self._create_pipeline(height, width)
        pipeline_cache[shape_key] = pipeline
        access_order.append(shape_key)

        return pipeline

    def _get_sequence_pipeline(
        self,
        height: int,
        width: int,
        frame_count: int,
        device: torch.device,
        gop_length: Optional[int],
        frame_interval_p: Optional[int],
    ) -> Pipeline:
        gpu_id = device.index if device.type == 'cuda' and device.index is not None else 0
        effective_gop = min(int(gop_length or frame_count), frame_count)
        shape_key = (
            height,
            width,
            frame_count,
            effective_gop,
            int(frame_interval_p or 1),
            self.quant_outlier_ratio,
        )

        if gpu_id not in self._sequence_pipeline_cache_per_gpu:
            self._sequence_pipeline_cache_per_gpu[gpu_id] = {}
            self._sequence_pipeline_access_order_per_gpu[gpu_id] = []

        pipeline_cache = self._sequence_pipeline_cache_per_gpu[gpu_id]
        access_order = self._sequence_pipeline_access_order_per_gpu[gpu_id]

        if shape_key in pipeline_cache:
            access_order.remove(shape_key)
            access_order.append(shape_key)
            return pipeline_cache[shape_key]

        while len(pipeline_cache) >= self.max_cached_pipelines:
            lru_key = access_order.pop(0)
            evicted = pipeline_cache.pop(lru_key)
            logger.info(
                f"[ActivationCompressor] Evicted GOP pipeline for GPU {gpu_id}, "
                f"shape {lru_key} (LRU)"
            )
            del evicted

        logger.info(
            f"[ActivationCompressor] Creating GOP pipeline for GPU {gpu_id}, "
            f"shape ({height}, {width}), frames={frame_count}, gop={effective_gop}"
        )
        pipeline = self._create_sequence_pipeline(
            height,
            width,
            frame_count,
            effective_gop,
            frame_interval_p,
        )
        pipeline_cache[shape_key] = pipeline
        access_order.append(shape_key)
        return pipeline

    def compress(
        self,
        activation: torch.Tensor,
        name: str = "activation"
    ) -> Dict[str, Any]:
        """
        Compress activation tensor.

        Args:
            activation: Activation tensor, typically [batch, seq_len, hidden_dim]
            name: Name for debugging

        Returns:
            Dictionary containing:
                - 'compressed': Encoded bitstream
                - 'shape': Original shape
                - 'dtype': Original dtype
                - 'device': Original device
                - 'code_size': Size of compressed data
                - 'scale': Quantization scale
                - 'offset': Quantization offset
        """
        if not NVENC_AVAILABLE:
            raise RuntimeError("NVENC not available")

        original_shape = activation.shape
        original_dtype = activation.dtype
        original_device = activation.device

        # Reshape to 2D for compression
        # [batch, seq_len, hidden_dim] -> [seq_len*batch, hidden_dim]
        if len(activation.shape) == 3:
            batch, seq_len, hidden_dim = activation.shape
            activation_2d = activation.reshape(-1, hidden_dim)
        elif len(activation.shape) == 2:
            activation_2d = activation
        else:
            raise ValueError(f"Unsupported activation shape: {activation.shape}")

        height, width = activation_2d.shape

        # Ensure activation is on GPU. Do not force bf16/fp32 activations to
        # fp16 here: Flux runs in bf16 and some activations exceed fp16 range.
        # The quantization step converts to fp32 before min/max, then emits
        # uint8 frames for the codec.
        original_device = activation_2d.device
        if activation_2d.device.type != 'cuda':
            activation_2d = activation_2d.cuda()

        # Compress - pipeline.forward() expects a tensor directly
        try:
            device_ctx = (
                torch.cuda.device(activation_2d.device)
                if activation_2d.device.type == "cuda"
                else nullcontext()
            )
            with device_ctx:
                # Get appropriate pipeline for this GPU while that GPU is the
                # active CUDA context. NVENC uses the current CUDA context.
                pipeline = self._get_pipeline(
                    height, width, activation_2d.device
                )
                compressed_dict = pipeline.forward(activation_2d, name=name)

            # Add metadata
            compressed_dict['original_shape'] = original_shape
            compressed_dict['original_dtype'] = original_dtype
            compressed_dict['original_device'] = original_device
            compressed_dict['compression_mode'] = 'intra_layer'
            compressed_dict['codec'] = self.codec
            compressed_dict['rc_mode'] = self.rc_mode
            compressed_dict['const_qp'] = self.const_qp
            compressed_dict['const_qp_intra'] = self.const_qp_intra
            compressed_dict['const_qp_inter_p'] = self.const_qp_inter_p
            compressed_dict['const_qp_inter_b'] = self.const_qp_inter_b
            compressed_dict['bitrate_mbps'] = float(self.bitrate)
            compressed_dict['bitrate_max_multiplier'] = float(
                self.bitrate_max_multiplier
            )
            compressed_dict['codec_preset'] = self.codec_preset
            compressed_dict['codec_tuning'] = self.codec_tuning
            compressed_dict['codec_spatial_aq'] = self.codec_spatial_aq
            compressed_dict['codec_temporal_aq'] = self.codec_temporal_aq
            compressed_dict['codec_target_quality'] = self.codec_target_quality
            compressed_dict['quantization'] = _quantization_name(
                self.codec,
                width,
                self.quant_group_size,
                self.quant_outlier_ratio,
            )
            compressed_dict['quantization_variant'] = _quantization_variant(
                compressed_dict['quantization']
            )
            compressed_dict['quant_group_size'] = self.quant_group_size
            compressed_dict['quant_outlier_ratio'] = self.quant_outlier_ratio

            logger.debug(
                f"[Compress] {name}: {original_shape} -> {compressed_dict['code_size']/1024/1024:.2f}MB "
                f"({original_shape[0]*original_shape[1]*original_shape[2]*2/compressed_dict['code_size']:.1f}x)"
            )

            return compressed_dict

        except Exception as e:
            logger.error(f"[Compress] Failed for {name}: {e}")
            raise

    def compress_sequence(
        self,
        activations: List[torch.Tensor],
        name: str = "activation_sequence",
        gop_length: Optional[int] = None,
        frame_interval_p: Optional[int] = 1,
        target_device: Optional[torch.device] = None,
        original_devices_override: Optional[List[torch.device]] = None,
    ) -> Dict[str, Any]:
        """
        Compress consecutive layer activations as inter-frame video frames.
        """
        if not NVENC_AVAILABLE:
            raise RuntimeError("NVENC not available")
        if not activations:
            raise ValueError("compress_sequence requires at least one activation")

        original_shapes = [a.shape for a in activations]
        original_dtypes = [a.dtype for a in activations]
        original_devices = (
            list(original_devices_override)
            if original_devices_override is not None
            else [a.device for a in activations]
        )
        first_shape = original_shapes[0]
        if any(shape != first_shape for shape in original_shapes):
            raise ValueError(f"All GOP activation shapes must match: {original_shapes}")

        if len(first_shape) == 3:
            batch, seq_len, hidden_dim = first_shape
            height = batch * seq_len
            width = hidden_dim
            tensors_2d = [a.reshape(-1, hidden_dim) for a in activations]
        elif len(first_shape) == 2:
            height, width = first_shape
            tensors_2d = list(activations)
        else:
            raise ValueError(f"Unsupported activation shape: {first_shape}")

        if target_device is None:
            for dev in original_devices:
                if dev.type == "cuda":
                    target_device = dev
                    break
        if target_device is None:
            target_device = torch.device("cuda")

        sequence = torch.stack(
            [
                tensor.to(device=target_device)
                for tensor in tensors_2d
            ],
            dim=0,
        )

        try:
            device_ctx = (
                torch.cuda.device(sequence.device)
                if sequence.device.type == "cuda"
                else nullcontext()
            )
            with device_ctx:
                pipeline = self._get_sequence_pipeline(
                    height,
                    width,
                    len(activations),
                    sequence.device,
                    gop_length=gop_length,
                    frame_interval_p=frame_interval_p,
                )
                compressed_dict = pipeline.forward(sequence, name=name)

            effective_gop = min(int(gop_length or len(activations)), len(activations))
            compressed_dict["compression_mode"] = "inter_layer_gop"
            compressed_dict["codec"] = self.codec
            compressed_dict["rc_mode"] = self.rc_mode
            compressed_dict["const_qp"] = self.const_qp
            compressed_dict["const_qp_intra"] = self.const_qp_intra
            compressed_dict["const_qp_inter_p"] = self.const_qp_inter_p
            compressed_dict["const_qp_inter_b"] = self.const_qp_inter_b
            compressed_dict["bitrate_mbps"] = float(self.bitrate)
            compressed_dict["bitrate_max_multiplier"] = float(
                self.bitrate_max_multiplier
            )
            compressed_dict["codec_preset"] = self.codec_preset
            compressed_dict["codec_tuning"] = self.codec_tuning
            compressed_dict["codec_spatial_aq"] = self.codec_spatial_aq
            compressed_dict["codec_temporal_aq"] = self.codec_temporal_aq
            compressed_dict["codec_target_quality"] = self.codec_target_quality
            compressed_dict["quantization"] = _quantization_name(
                self.codec,
                width,
                self.quant_group_size,
                self.quant_outlier_ratio,
            )
            compressed_dict["quantization_variant"] = _quantization_variant(
                compressed_dict["quantization"]
            )
            compressed_dict["quant_group_size"] = self.quant_group_size
            compressed_dict["quant_outlier_ratio"] = self.quant_outlier_ratio
            compressed_dict["original_shapes"] = original_shapes
            compressed_dict["original_dtypes"] = original_dtypes
            compressed_dict["original_devices"] = original_devices
            compressed_dict["sequence_shape"] = sequence.shape
            compressed_dict["frame_count"] = len(activations)
            compressed_dict["gop_length"] = effective_gop
            compressed_dict["frame_interval_p"] = int(frame_interval_p or 1)
            compressed_dict["original_shape"] = first_shape
            compressed_dict["original_dtype"] = original_dtypes[0]
            compressed_dict["original_device"] = original_devices[0]
            return compressed_dict
        except Exception as e:
            logger.error(f"[Compress] Failed for {name}: {e}")
            raise


class ActivationDecompressor:
    """
    Decompresses activation tensors compressed by ActivationCompressor.
    """

    def __init__(
        self,
        tile_height: int = 2048,
        tile_width: int = 2048,
        max_cached_pipelines: int = 2,
    ):
        """
        Args:
            tile_height: Height of video tiles (must match compressor)
            tile_width: Width of video tiles (must match compressor)
            max_cached_pipelines: Maximum number of pipelines to cache
        """
        if not NVENC_AVAILABLE:
            raise RuntimeError("NVENC not available - cannot create ActivationDecompressor")

        self.tile_height = tile_height
        self.tile_width = tile_width
        self.max_cached_pipelines = max_cached_pipelines

        # Cache pipelines per GPU: {gpu_id: {shape_key: pipeline}}
        self._pipeline_cache_per_gpu = {}
        self._pipeline_access_order_per_gpu = {}
        self._sequence_pipeline_cache_per_gpu = {}
        self._sequence_pipeline_access_order_per_gpu = {}

        logger.info(f"[ActivationDecompressor] Initialized with max_pipelines={max_cached_pipelines} per GPU")

    @staticmethod
    def _apply_codec_residual(
        compressed_dict: Dict[str, Any],
        recovered: torch.Tensor,
        frame_index: Optional[int] = None,
    ) -> torch.Tensor:
        indices_list = compressed_dict.get("codec_residual_indices")
        values_list = compressed_dict.get("codec_residual_values")
        if indices_list is None or values_list is None:
            return recovered

        if frame_index is None:
            indices = indices_list
            values = values_list
        else:
            if frame_index >= len(indices_list) or frame_index >= len(values_list):
                return recovered
            indices = indices_list[frame_index]
            values = values_list[frame_index]

        if not isinstance(indices, torch.Tensor) or not isinstance(values, torch.Tensor):
            return recovered
        if indices.numel() == 0 or values.numel() == 0:
            return recovered

        flat = recovered.reshape(-1)
        indices = indices.to(device=flat.device, dtype=torch.long)
        values = values.to(device=flat.device, dtype=torch.float32)
        updated = flat.index_add(0, indices, values.to(dtype=flat.dtype))
        return updated.reshape_as(recovered)

    def clear_pipeline_cache(self) -> None:
        """Drop cached pipeline objects so native NVDEC resources can be freed."""
        self._pipeline_cache_per_gpu.clear()
        self._pipeline_access_order_per_gpu.clear()
        self._sequence_pipeline_cache_per_gpu.clear()
        self._sequence_pipeline_access_order_per_gpu.clear()
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    @staticmethod
    def _codec_type(codec: str):
        return CodecType.H264 if str(codec).lower() == "h264" else CodecType.HEVC

    def _create_pipeline(
        self,
        height: int,
        width: int,
        codec: str = "hevc",
        quant_group_size: Optional[int] = DEFAULT_LOSSLESS_QUANT_GROUP_SIZE,
        quant_outlier_ratio: float = 0.0,
    ) -> Pipeline:
        """Create decompression pipeline (reuses compression pipeline)."""
        # Calculate padded dimensions that are divisible by tile size
        padded_height = ((height + self.tile_height - 1) // self.tile_height) * self.tile_height
        padded_width = ((width + self.tile_width - 1) // self.tile_width) * self.tile_width

        # Use dummy config for backward pass
        config = TensorEncodeConfig()
        config.input_format = InputFormat.NV12
        config.average_bit_rate = 1000000  # Dummy value
        config.max_bit_rate = 10000000
        config.codec_type = self._codec_type(codec)
        config.rc_mode = RateControlMode.VBR
        config.preset = PresetType.P7
        config.tuning_info = TuningInfo.HighQuality
        config.monochrome = True

        class AddShape(Step):
            def __init__(self):
                super().__init__("AddShape", required_keys=["data"], yield_keys=["shape"])

            def forward(self, data_dict):
                data_dict["shape"] = data_dict["data"].shape
                return data_dict

            def backward(self, data_dict):
                return data_dict

        pipeline = Pipeline([
            AddShape(),
            _make_quantization_step(
                codec,
                width,
                quant_group_size,
                quant_outlier_ratio,
            ),
            FixedTiling(
                pad_to_shape=[padded_height, padded_width],
                resize_to_shape=[1, 1, padded_height, padded_width],
                tile_shape=[1, 1, self.tile_height, self.tile_width]
            ),
            MonoNVEncode(config, self.tile_height, self.tile_width),
        ])

        return pipeline

    def _create_sequence_pipeline(
        self,
        height: int,
        width: int,
        frame_count: int,
        codec: str = "hevc",
        quant_group_size: Optional[int] = DEFAULT_LOSSLESS_QUANT_GROUP_SIZE,
        quant_outlier_ratio: float = 0.0,
    ) -> Pipeline:
        padded_height = ((height + self.tile_height - 1) // self.tile_height) * self.tile_height
        padded_width = ((width + self.tile_width - 1) // self.tile_width) * self.tile_width

        config = TensorEncodeConfig()
        config.input_format = InputFormat.NV12
        config.average_bit_rate = 1000000
        config.max_bit_rate = 10000000
        config.codec_type = self._codec_type(codec)
        config.rc_mode = RateControlMode.VBR
        config.preset = PresetType.P7
        config.tuning_info = TuningInfo.HighQuality
        config.monochrome = True

        class AddShape(Step):
            def __init__(self):
                super().__init__("AddShape", required_keys=["data"], yield_keys=["shape"])

            def forward(self, data_dict):
                data_dict["shape"] = data_dict["data"].shape
                return data_dict

            def backward(self, data_dict):
                return data_dict

        return Pipeline([
            AddShape(),
            _make_quantization_step(
                codec,
                width,
                quant_group_size,
                quant_outlier_ratio,
            ),
            FixedTiling(
                pad_to_shape=[frame_count, padded_height, padded_width],
                resize_to_shape=[frame_count, 1, padded_height, padded_width],
                tile_shape=[1, 1, self.tile_height, self.tile_width],
            ),
            MonoNVEncodeSequence(config, self.tile_height, self.tile_width),
        ])

    def _get_pipeline(
        self,
        height: int,
        width: int,
        device: torch.device,
        codec: str = "hevc",
        quant_group_size: Optional[int] = DEFAULT_LOSSLESS_QUANT_GROUP_SIZE,
        quant_outlier_ratio: float = 0.0,
    ) -> Pipeline:
        """Get or create cached pipeline for shape and device with LRU eviction."""
        # Get GPU ID from device
        gpu_id = device.index if device.type == 'cuda' and device.index is not None else 0
        shape_key = (
            height,
            width,
            str(codec).lower(),
            _quantization_name(
                codec,
                width,
                quant_group_size,
                quant_outlier_ratio,
            ),
        )

        # Initialize cache for this GPU if needed
        if gpu_id not in self._pipeline_cache_per_gpu:
            self._pipeline_cache_per_gpu[gpu_id] = {}
            self._pipeline_access_order_per_gpu[gpu_id] = []

        pipeline_cache = self._pipeline_cache_per_gpu[gpu_id]
        access_order = self._pipeline_access_order_per_gpu[gpu_id]

        # Check if already cached for this GPU
        if shape_key in pipeline_cache:
            # Move to end (most recently used)
            access_order.remove(shape_key)
            access_order.append(shape_key)
            return pipeline_cache[shape_key]

        # Need to create new pipeline
        # First check if we need to evict old pipelines on this GPU
        while len(pipeline_cache) >= self.max_cached_pipelines:
            # Evict least recently used
            lru_key = access_order.pop(0)
            evicted = pipeline_cache.pop(lru_key)
            logger.info(f"[ActivationDecompressor] Evicted pipeline for GPU {gpu_id}, shape {lru_key} (LRU)")
            del evicted  # Explicitly delete to free NVDEC resources

        # Create new pipeline on the correct GPU
        logger.info(f"[ActivationDecompressor] Creating pipeline for GPU {gpu_id}, shape ({height}, {width})")
        pipeline = self._create_pipeline(
            height,
            width,
            codec=codec,
            quant_group_size=quant_group_size,
            quant_outlier_ratio=quant_outlier_ratio,
        )
        pipeline_cache[shape_key] = pipeline
        access_order.append(shape_key)

        return pipeline

    def _get_sequence_pipeline(
        self,
        height: int,
        width: int,
        frame_count: int,
        device: torch.device,
        codec: str = "hevc",
        quant_group_size: Optional[int] = DEFAULT_LOSSLESS_QUANT_GROUP_SIZE,
        quant_outlier_ratio: float = 0.0,
    ) -> Pipeline:
        gpu_id = device.index if device.type == 'cuda' and device.index is not None else 0
        shape_key = (
            height,
            width,
            frame_count,
            str(codec).lower(),
            _quantization_name(
                codec,
                width,
                quant_group_size,
                quant_outlier_ratio,
            ),
        )

        if gpu_id not in self._sequence_pipeline_cache_per_gpu:
            self._sequence_pipeline_cache_per_gpu[gpu_id] = {}
            self._sequence_pipeline_access_order_per_gpu[gpu_id] = []

        pipeline_cache = self._sequence_pipeline_cache_per_gpu[gpu_id]
        access_order = self._sequence_pipeline_access_order_per_gpu[gpu_id]

        if shape_key in pipeline_cache:
            access_order.remove(shape_key)
            access_order.append(shape_key)
            return pipeline_cache[shape_key]

        while len(pipeline_cache) >= self.max_cached_pipelines:
            lru_key = access_order.pop(0)
            evicted = pipeline_cache.pop(lru_key)
            logger.info(
                f"[ActivationDecompressor] Evicted GOP pipeline for GPU {gpu_id}, "
                f"shape {lru_key} (LRU)"
            )
            del evicted

        logger.info(
            f"[ActivationDecompressor] Creating GOP pipeline for GPU {gpu_id}, "
            f"shape ({height}, {width}), frames={frame_count}"
        )
        pipeline = self._create_sequence_pipeline(
            height,
            width,
            frame_count,
            codec=codec,
            quant_group_size=quant_group_size,
            quant_outlier_ratio=quant_outlier_ratio,
        )
        pipeline_cache[shape_key] = pipeline
        access_order.append(shape_key)
        return pipeline

    def decompress(
        self,
        compressed_dict: Dict[str, Any],
        target_device: Optional[torch.device] = None
    ) -> torch.Tensor:
        """
        Decompress activation tensor.

        Args:
            compressed_dict: Dictionary from compress() containing compressed data
            target_device: Device to place decompressed tensor (None = original device)

        Returns:
            Decompressed activation tensor with original shape
        """
        original_shape = compressed_dict['original_shape']
        original_dtype = compressed_dict['original_dtype']
        original_device = compressed_dict['original_device']

        if target_device is None:
            target_device = original_device

        if not NVENC_AVAILABLE:
            raise RuntimeError("NVENC not available")

        # Get 2D shape
        if len(original_shape) == 3:
            batch, seq_len, hidden_dim = original_shape
            height = batch * seq_len
            width = hidden_dim
        elif len(original_shape) == 2:
            height, width = original_shape
        else:
            raise ValueError(f"Unsupported shape: {original_shape}")

        # Calculate padded dimensions
        padded_height = ((height + self.tile_height - 1) // self.tile_height) * self.tile_height
        padded_width = ((width + self.tile_width - 1) // self.tile_width) * self.tile_width

        # Decompress
        try:
            device_ctx = (
                torch.cuda.device(target_device)
                if target_device.type == "cuda"
                else nullcontext()
            )
            # For backward pass, we need to handle padding correctly
            # The issue is: FixedTiling.backward outputs padded tensor,
            # but quantization backward expects original size.
            with device_ctx:
                # Get pipeline while the target GPU is the active CUDA context.
                pipeline = self._get_pipeline(
                    height,
                    width,
                    target_device,
                    codec=compressed_dict.get("codec", "hevc"),
                    quant_group_size=compressed_dict.get(
                        "quant_group_size", DEFAULT_LOSSLESS_QUANT_GROUP_SIZE
                    ),
                    quant_outlier_ratio=compressed_dict.get(
                        "quant_outlier_ratio", 0.0
                    ),
                )

                # We'll manually handle the backward pass step by step
                data_dict = compressed_dict.copy()

                # Compressed cache may live on CPU; NVDEC and dequantization need
                # nested bitstreams and quantization tensors on target_device.
                data_dict = _move_compressed_value(data_dict, target_device)

                # Step 1: MonoNVEncode backward (decode video)
                data_dict = pipeline.steps[-1].backward(data_dict)

                # Step 2: FixedTiling backward (untile to padded shape)
                data_dict['shape'] = torch.Size([padded_height, padded_width])
                data_dict = pipeline.steps[-2].backward(data_dict)

                # Step 3: Crop padding before dequantization
                quantized_data = data_dict['data']
                quantized_data = quantized_data[:height, :width]
                data_dict['data'] = quantized_data

                # Step 4: Quantization backward (dequantize with original size)
                data_dict['shape'] = torch.Size([height, width])
                data_dict = pipeline.steps[-3].backward(data_dict)

                # Step 5: AddShape backward (no-op)
                data_dict = pipeline.steps[-4].backward(data_dict)

                recovered_2d = data_dict['data']

            # Reshape back to original shape
            if len(original_shape) == 3:
                recovered = recovered_2d.reshape(original_shape)
            else:
                recovered = recovered_2d

            # Restore dtype and device
            if recovered.dtype != original_dtype:
                recovered = recovered.to(original_dtype)
            if recovered.device != target_device:
                recovered = recovered.to(target_device)
            recovered = self._apply_codec_residual(compressed_dict, recovered)

            logger.debug(f"[Decompress] Restored shape: {recovered.shape}")

            return recovered

        except Exception as e:
            logger.error(f"[Decompress] Failed: {e}")
            raise

    def decompress_sequence(
        self,
        compressed_dict: Dict[str, Any],
        target_device: Optional[torch.device] = None,
    ) -> List[torch.Tensor]:
        """Decode and restore every frame/layer from an inter-layer GOP cache."""
        if not NVENC_AVAILABLE:
            raise RuntimeError("NVENC not available")

        original_shapes = compressed_dict["original_shapes"]
        original_dtypes = compressed_dict["original_dtypes"]
        original_devices = compressed_dict["original_devices"]
        frame_count = int(compressed_dict["frame_count"])
        if len(original_shapes) != frame_count:
            raise ValueError(
                f"original_shapes has {len(original_shapes)} entries, "
                f"expected frame_count={frame_count}"
            )

        first_shape = original_shapes[0]
        if len(first_shape) == 3:
            batch, seq_len, hidden_dim = first_shape
            height = batch * seq_len
            width = hidden_dim
        elif len(first_shape) == 2:
            height, width = first_shape
        else:
            raise ValueError(f"Unsupported shape: {first_shape}")

        padded_height = ((height + self.tile_height - 1) // self.tile_height) * self.tile_height
        padded_width = ((width + self.tile_width - 1) // self.tile_width) * self.tile_width

        decode_device = target_device
        if decode_device is None:
            decode_device = next(
                (
                    device
                    for device in original_devices
                    if getattr(device, "type", None) == "cuda"
                ),
                original_devices[0],
            )
        if not isinstance(decode_device, torch.device):
            decode_device = torch.device(decode_device)

        try:
            device_ctx = (
                torch.cuda.device(decode_device)
                if decode_device.type == "cuda"
                else nullcontext()
            )
            with device_ctx:
                sequence_pipeline = self._get_sequence_pipeline(
                    height,
                    width,
                    frame_count,
                    decode_device,
                    codec=compressed_dict.get("codec", "hevc"),
                    quant_group_size=compressed_dict.get(
                        "quant_group_size", DEFAULT_LOSSLESS_QUANT_GROUP_SIZE
                    ),
                    quant_outlier_ratio=compressed_dict.get(
                        "quant_outlier_ratio", 0.0
                    ),
                )
                single_pipeline = self._get_pipeline(
                    height,
                    width,
                    decode_device,
                    codec=compressed_dict.get("codec", "hevc"),
                    quant_group_size=compressed_dict.get(
                        "quant_group_size", DEFAULT_LOSSLESS_QUANT_GROUP_SIZE
                    ),
                    quant_outlier_ratio=compressed_dict.get(
                        "quant_outlier_ratio", 0.0
                    ),
                )

                data_dict = compressed_dict.copy()
                data_dict = _move_compressed_value(data_dict, decode_device)

                # This is the expensive NVDEC step. Do it once for the whole GOP.
                decoded_dict = sequence_pipeline.steps[-1].backward(data_dict)
                decoded_tiles = decoded_dict["data"]
                tiles_shape = decoded_dict["tiles_shape"]
                scale = decoded_dict["scale"]
                offset = decoded_dict["offset"]

                recovered_frames: List[torch.Tensor] = []
                for frame_index in range(frame_count):
                    original_shape = original_shapes[frame_index]
                    original_dtype = original_dtypes[frame_index]
                    frame_target_device = (
                        target_device
                        if target_device is not None
                        else original_devices[frame_index]
                    )
                    if not isinstance(frame_target_device, torch.device):
                        frame_target_device = torch.device(frame_target_device)

                    frame_dict = decoded_dict.copy()
                    frame_dict["data"] = decoded_tiles[frame_index].contiguous()
                    frame_dict["tiles_shape"] = [
                        1,
                        tiles_shape[1],
                        tiles_shape[2],
                        tiles_shape[3],
                    ]
                    frame_dict["shape"] = torch.Size([padded_height, padded_width])
                    frame_dict = single_pipeline.steps[-2].backward(frame_dict)

                    quantized_data = frame_dict["data"][:height, :width]
                    frame_dict["data"] = quantized_data

                    rows_per_frame = _quantization_rows_per_frame(
                        compressed_dict.get("quantization"),
                        height,
                        width,
                    )
                    row_start = frame_index * rows_per_frame
                    row_end = row_start + rows_per_frame
                    frame_dict["scale"] = scale[row_start:row_end]
                    frame_dict["offset"] = offset[row_start:row_end]
                    group_size = int(
                        compressed_dict.get(
                            "quant_group_size",
                            DEFAULT_LOSSLESS_QUANT_GROUP_SIZE,
                        )
                        or DEFAULT_LOSSLESS_QUANT_GROUP_SIZE
                    )
                    frame_dict = _slice_outliers_for_frame(
                        frame_dict,
                        frame_index,
                        rows_per_frame,
                        group_size,
                    )

                    frame_dict["shape"] = torch.Size([height, width])
                    frame_dict = single_pipeline.steps[-3].backward(frame_dict)
                    frame_dict = single_pipeline.steps[-4].backward(frame_dict)
                    recovered_2d = frame_dict["data"]

                    if len(original_shape) == 3:
                        recovered = recovered_2d.reshape(original_shape)
                    else:
                        recovered = recovered_2d

                    if recovered.dtype != original_dtype:
                        recovered = recovered.to(original_dtype)
                    if recovered.device != frame_target_device:
                        recovered = recovered.to(frame_target_device)
                    recovered = self._apply_codec_residual(
                        compressed_dict,
                        recovered,
                        frame_index=frame_index,
                    )
                    recovered_frames.append(recovered)

            return recovered_frames
        except Exception as e:
            logger.error(f"[Decompress] Failed for GOP sequence: {e}")
            raise

    def validate_sequence_finite(
        self,
        compressed_dict: Dict[str, Any],
        target_device: Optional[torch.device] = None,
    ) -> None:
        """Decode a GOP once and check each restored frame without retaining it."""
        if not NVENC_AVAILABLE:
            raise RuntimeError("NVENC not available")

        original_shapes = compressed_dict["original_shapes"]
        original_devices = compressed_dict["original_devices"]
        frame_count = int(compressed_dict["frame_count"])
        if len(original_shapes) != frame_count:
            raise ValueError(
                f"original_shapes has {len(original_shapes)} entries, "
                f"expected frame_count={frame_count}"
            )

        first_shape = original_shapes[0]
        if len(first_shape) == 3:
            batch, seq_len, hidden_dim = first_shape
            height = batch * seq_len
            width = hidden_dim
        elif len(first_shape) == 2:
            height, width = first_shape
        else:
            raise ValueError(f"Unsupported shape: {first_shape}")

        padded_height = ((height + self.tile_height - 1) // self.tile_height) * self.tile_height
        padded_width = ((width + self.tile_width - 1) // self.tile_width) * self.tile_width

        decode_device = target_device
        if decode_device is None:
            decode_device = next(
                (
                    device
                    for device in original_devices
                    if getattr(device, "type", None) == "cuda"
                ),
                original_devices[0],
            )
        if not isinstance(decode_device, torch.device):
            decode_device = torch.device(decode_device)

        try:
            device_ctx = (
                torch.cuda.device(decode_device)
                if decode_device.type == "cuda"
                else nullcontext()
            )
            with device_ctx:
                sequence_pipeline = self._get_sequence_pipeline(
                    height,
                    width,
                    frame_count,
                    decode_device,
                    codec=compressed_dict.get("codec", "hevc"),
                    quant_group_size=compressed_dict.get(
                        "quant_group_size", DEFAULT_LOSSLESS_QUANT_GROUP_SIZE
                    ),
                    quant_outlier_ratio=compressed_dict.get(
                        "quant_outlier_ratio", 0.0
                    ),
                )
                single_pipeline = self._get_pipeline(
                    height,
                    width,
                    decode_device,
                    codec=compressed_dict.get("codec", "hevc"),
                    quant_group_size=compressed_dict.get(
                        "quant_group_size", DEFAULT_LOSSLESS_QUANT_GROUP_SIZE
                    ),
                    quant_outlier_ratio=compressed_dict.get(
                        "quant_outlier_ratio", 0.0
                    ),
                )

                data_dict = compressed_dict.copy()
                data_dict = _move_compressed_value(data_dict, decode_device)
                decoded_dict = sequence_pipeline.steps[-1].backward(data_dict)
                decoded_tiles = decoded_dict["data"]
                tiles_shape = decoded_dict["tiles_shape"]
                scale = decoded_dict["scale"]
                offset = decoded_dict["offset"]

                rows_per_frame = _quantization_rows_per_frame(
                    compressed_dict.get("quantization"),
                    height,
                    width,
                )
                group_size = int(
                    compressed_dict.get(
                        "quant_group_size",
                        DEFAULT_LOSSLESS_QUANT_GROUP_SIZE,
                    )
                    or DEFAULT_LOSSLESS_QUANT_GROUP_SIZE
                )

                for frame_index in range(frame_count):
                    frame_dict = decoded_dict.copy()
                    frame_dict["data"] = decoded_tiles[frame_index].contiguous()
                    frame_dict["tiles_shape"] = [
                        1,
                        tiles_shape[1],
                        tiles_shape[2],
                        tiles_shape[3],
                    ]
                    frame_dict["shape"] = torch.Size([padded_height, padded_width])
                    frame_dict = single_pipeline.steps[-2].backward(frame_dict)

                    frame_dict["data"] = frame_dict["data"][:height, :width]
                    row_start = frame_index * rows_per_frame
                    row_end = row_start + rows_per_frame
                    frame_dict["scale"] = scale[row_start:row_end]
                    frame_dict["offset"] = offset[row_start:row_end]
                    frame_dict = _slice_outliers_for_frame(
                        frame_dict,
                        frame_index,
                        rows_per_frame,
                        group_size,
                    )

                    frame_dict["shape"] = torch.Size([height, width])
                    frame_dict = single_pipeline.steps[-3].backward(frame_dict)
                    frame_dict = single_pipeline.steps[-4].backward(frame_dict)
                    restored = frame_dict["data"]
                    if not torch.isfinite(restored).all().item():
                        raise RuntimeError(
                            "validation decoded non-finite activation "
                            f"frame={frame_index}"
                        )
                    del frame_dict, restored

                del decoded_dict, decoded_tiles
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
        except Exception:
            raise

    def decompress_sequence_frame(
        self,
        compressed_dict: Dict[str, Any],
        frame_index: int,
        target_device: Optional[torch.device] = None,
    ) -> torch.Tensor:
        """Decode one frame/layer from an inter-layer GOP cache entry."""
        if not NVENC_AVAILABLE:
            raise RuntimeError("NVENC not available")

        original_shapes = compressed_dict["original_shapes"]
        frame_count = int(compressed_dict["frame_count"])
        if frame_index < 0 or frame_index >= frame_count:
            raise IndexError(
                f"frame_index={frame_index} outside GOP frame_count={frame_count}"
            )
        if frame_index >= len(original_shapes):
            raise IndexError(
                f"frame_index={frame_index} outside original_shapes length "
                f"{len(original_shapes)}"
            )

        try:
            return self.decompress_sequence(
                compressed_dict,
                target_device=target_device,
            )[frame_index]
        except Exception as e:
            logger.error(f"[Decompress] Failed for GOP frame {frame_index}: {e}")
            raise


def test_compression():
    """Test activation compression/decompression."""
    print("\n" + "="*60)
    print("Testing Activation Compression")
    print("="*60)

    if not NVENC_AVAILABLE:
        print("NVENC not available - skipping test")
        return

    # Create test activation (simulate Flux activation)
    # Typical shape: [batch=1, seq_len=~12M, hidden_dim=3072]
    # For testing, use smaller size
    test_activation = torch.randn(1, 10000, 3072, dtype=torch.float16, device='cuda')
    original_size = test_activation.numel() * 2  # bytes

    print(f"\nTest activation shape: {test_activation.shape}")
    print(f"Original size: {original_size / 1024 / 1024:.2f} MB")

    # Compress
    compressor = ActivationCompressor(bitrate=5.0, codec="lossless")
    import time
    start = time.time()
    compressed = compressor.compress(test_activation, name="test")
    compress_time = time.time() - start

    compressed_size = compressed['code_size']
    compression_ratio = original_size / compressed_size

    print(f"\nCompressed size: {compressed_size / 1024 / 1024:.2f} MB")
    print(f"Compression ratio: {compression_ratio:.2f}x")
    print(f"Compression time: {compress_time*1000:.1f} ms")

    # Decompress
    decompressor = ActivationDecompressor()
    start = time.time()
    recovered = decompressor.decompress(compressed)
    decompress_time = time.time() - start

    print(f"Decompression time: {decompress_time*1000:.1f} ms")

    # Check quality
    mse = torch.mean((test_activation - recovered) ** 2).item()
    max_error = torch.max(torch.abs(test_activation - recovered)).item()

    print(f"\nQuality metrics:")
    print(f"  MSE: {mse:.6f}")
    print(f"  Max error: {max_error:.6f}")
    print(f"  Shape match: {test_activation.shape == recovered.shape}")
    print(f"  Dtype match: {test_activation.dtype == recovered.dtype}")

    print("\n" + "="*60)
    print("✅ Test completed successfully!")
    print("="*60 + "\n")


if __name__ == "__main__":
    test_compression()
