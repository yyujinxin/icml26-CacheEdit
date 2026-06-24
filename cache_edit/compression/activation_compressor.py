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


try:
    from .ops import (
        CodecType, RateControlMode, PresetType, TuningInfo, InputFormat,
        TensorEncodeConfig, TensorEncoder, TensorDecoder, EncodeQp
    )
    from .pipeline import (
        Pipeline,
        CWQuantization,
        GWQuantization,
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
    FixedTiling = object
    MonoNVEncode = object
    MonoNVEncodeSequence = object


LOSSLESS_QUANT_GROUP_SIZE = 64


def _make_quantization_step(codec: str, width: int):
    """Choose the quantizer paired with the codec path."""
    if str(codec).lower() == "lossless" and width % LOSSLESS_QUANT_GROUP_SIZE == 0:
        return GWQuantization(groupsize=LOSSLESS_QUANT_GROUP_SIZE)
    return CWQuantization()


def _quantization_name(codec: str, width: int) -> str:
    if str(codec).lower() == "lossless" and width % LOSSLESS_QUANT_GROUP_SIZE == 0:
        return f"gw{LOSSLESS_QUANT_GROUP_SIZE}"
    return "cw"


def _quantization_rows_per_frame(
    quantization: Optional[str],
    height: int,
    width: int,
) -> int:
    if str(quantization or "").startswith("gw"):
        return int(height * width // LOSSLESS_QUANT_GROUP_SIZE)
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

        # Cache pipelines per GPU: {gpu_id: {shape_key: pipeline}}
        self._pipeline_cache_per_gpu = {}
        self._pipeline_access_order_per_gpu = {}
        self._sequence_pipeline_cache_per_gpu = {}
        self._sequence_pipeline_access_order_per_gpu = {}

        logger.info(f"[ActivationCompressor] Initialized: bitrate={bitrate}Mbps, codec={codec}, max_pipelines={max_cached_pipelines} per GPU")

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
            config.average_bit_rate = int(self.bitrate * 1000000)
            config.max_bit_rate = int(self.bitrate * 1000000 * self.bitrate_max_multiplier)
            config.rc_mode = RateControlMode.VBR
            config.preset = PresetType.P7
            config.tuning_info = TuningInfo.HighQuality
        config.monochrome = True

        if gop_length is not None and gop_length > 1:
            config.gop_length = int(gop_length)
            config.frame_interval_p = int(frame_interval_p or 1)
        else:
            config.gop_length = None
            config.frame_interval_p = None
        return config

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
            _make_quantization_step(self.codec, width),  # FP16 -> uint8 quantization
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
            _make_quantization_step(self.codec, width),
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
        shape_key = (height, width)

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
        shape_key = (height, width, frame_count, effective_gop, int(frame_interval_p or 1))

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

        # Ensure activation is on GPU and FP16
        original_device = activation_2d.device
        if activation_2d.device.type != 'cuda':
            activation_2d = activation_2d.cuda()
        if activation_2d.dtype != torch.float16:
            activation_2d = activation_2d.to(torch.float16)

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
            compressed_dict['quantization'] = _quantization_name(self.codec, width)

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
                tensor.to(device=target_device, dtype=torch.float16)
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
            compressed_dict["quantization"] = _quantization_name(self.codec, width)
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

    def _create_pipeline(self, height: int, width: int, codec: str = "hevc") -> Pipeline:
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
            _make_quantization_step(codec, width),
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
            _make_quantization_step(codec, width),
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
    ) -> Pipeline:
        """Get or create cached pipeline for shape and device with LRU eviction."""
        # Get GPU ID from device
        gpu_id = device.index if device.type == 'cuda' and device.index is not None else 0
        shape_key = (height, width, str(codec).lower())

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
        pipeline = self._create_pipeline(height, width, codec=codec)
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
    ) -> Pipeline:
        gpu_id = device.index if device.type == 'cuda' and device.index is not None else 0
        shape_key = (height, width, frame_count, str(codec).lower())

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
                )
                single_pipeline = self._get_pipeline(
                    height,
                    width,
                    decode_device,
                    codec=compressed_dict.get("codec", "hevc"),
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
                    recovered_frames.append(recovered)

            return recovered_frames
        except Exception as e:
            logger.error(f"[Decompress] Failed for GOP sequence: {e}")
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
