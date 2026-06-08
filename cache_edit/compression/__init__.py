import torch
try:
    from .ops import *
except ImportError:
    print("Warning: NVENC ops not available - need to build CUDA extensions")
    # Define fallback enums when CUDA extensions are not available
    from enum import IntEnum

    class CodecType(IntEnum):
        H264 = 0
        HEVC = 1

    class RateControlMode(IntEnum):
        CBR = 0
        VBR = 1

    class PresetType(IntEnum):
        P1 = 0
        P7 = 6

    class TuningInfo(IntEnum):
        HighQuality = 0

    class InputFormat(IntEnum):
        NV12 = 0

    class TensorEncodeConfig:
        def __init__(self):
            self.input_format = None
            self.average_bit_rate = None
            self.max_bit_rate = None
            self.codec_type = None
            self.rc_mode = None
            self.preset = None
            self.tuning_info = None
            self.monochrome = None

    class TensorEncoder:
        def __init__(self, config, width, height):
            self.config = config
            self.width = width
            self.height = height

        def encode(self, tensor):
            raise NotImplementedError("CUDA extensions required for encoding")

    class TensorDecoder:
        def __init__(self, width, height):
            self.width = width
            self.height = height

        def decode(self, encoded_data):
            raise NotImplementedError("CUDA extensions required for decoding")


def rgb2yuv(input: torch.Tensor) -> torch.Tensor:
    """
    Convert RGB tensor to YUV tensor

    Args:
        tensor (torch.Tensor): RGB tensor with shape (N, 3, H, W)

    Returns:
        torch.Tensor: YUV tensor with shape (N, 3, H, W)
    """
    output = torch.empty_like(input)
    red = input[:, 0, :, :]
    green = input[:, 1, :, :]
    blue = input[:, 2, :, :]
    output[:, 0, :, :] = 0.2126 * red + 0.7152 * green + 0.0722 * blue
    output[:, 1, :, :] = -0.09991 * red - 0.33609 * green + 0.436 * blue
    output[:, 2, :, :] = 0.615 * red - 0.55861 * green - 0.05639 * blue
    return output


def yuv2rgb(input: torch.Tensor) -> torch.Tensor:
    """
    Convert YUV tensor to RGB tensor

    Args:
        tensor (torch.Tensor): YUV tensor with shape (N, 3, H, W)

    Returns:
        torch.Tensor: RGB tensor with shape (N, 3, H, W)
    """
    output = torch.empty_like(input)
    luma = input[:, 0, :, :]
    u = input[:, 1, :, :]
    v = input[:, 2, :, :]
    output[:, 0, :, :] = luma + 1.28033 * v
    output[:, 1, :, :] = luma - 0.21482 * u - 0.38059 * v
    output[:, 2, :, :] = luma + 2.12798 * u
    return output
