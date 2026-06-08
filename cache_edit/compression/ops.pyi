import torch
from enum import Enum

class InputFormat(Enum):
    YUV444 = 0
    ARGB = 1
    NV12 = 2

class CodecType(Enum):
    H264 = 0
    HEVC = 1

class RateControlMode(Enum):
    ConstQP = 0
    CBR = 1
    VBR = 2

class PresetType(Enum):
    P1 = 0
    P2 = 1
    P3 = 2
    P4 = 3
    P5 = 4
    P6 = 5
    P7 = 6


class TuningInfo(Enum):
    HighQuality = 0
    LowLatency = 1
    UltraLowLatency = 2
    Lossless = 3


class EncodeQp:
    qpInterP: int
    qpInterB: int
    qpIntra: int


class TensorEncodeOutput:
    bitstream: torch.Tensor
    packet_sizes: torch.Tensor


class TensorEncodeConfig:
    input_format: InputFormat

    codec_type: CodecType
    rc_mode: RateControlMode
    preset: PresetType
    tuning_info: TuningInfo

    average_bit_rate: int | None
    max_bit_rate: int | None
    target_quality: int | None
    const_qp: EncodeQp | None
    spatial_aq: int | None
    temporal_aq: bool | None
    monochrome: bool | None


class TensorEncoder:
    def __init__(self, config: TensorEncodeConfig, width: int, height: int): ...
    def encode(self, tensor: torch.Tensor) -> TensorEncodeOutput: ...


class TensorDecoder:
    def __init__(self, codec_type: CodecType, max_width: int, max_height: int): ...
    def decode(self, bitstream: torch.Tensor, packet_sizes: torch.Tensor) -> torch.Tensor: ...