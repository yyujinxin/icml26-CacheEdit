from typing import Dict, Any, List
from contextlib import contextmanager
from .definitions import Step
import torch
from .. import CodecType, RateControlMode, PresetType, TuningInfo
from .. import TensorEncodeConfig, TensorEncoder
from .. import TensorDecoder
# from torchvision.io import write_video
# import matplotlib.image  # Commented out - not needed for activation compression
import os
import sys
import gc
import ctypes
import threading


_libc = ctypes.CDLL(None)
_native_stdout_lock = threading.RLock()


def _flush_native_stdio():
    try:
        _libc.fflush(None)
    except Exception:
        pass


def _encoded_tensors(encoded):
    if isinstance(encoded, dict):
        return encoded["bitstream"], encoded["packet_sizes"]
    return encoded.bitstream, encoded.packet_sizes


@contextmanager
def _suppress_native_stdout():
    # fd 1/2 are process-global. Async compression can call into this helper
    # from multiple threads, so serialize dup2 restore pairs to avoid leaking
    # stdout/stderr to /dev/null or to a stale saved fd.
    with _native_stdout_lock:
        sys.stdout.flush()
        sys.stderr.flush()
        _flush_native_stdio()
        saved_stdout = os.dup(1)
        saved_stderr = os.dup(2)
        try:
            with open(os.devnull, "w") as devnull:
                os.dup2(devnull.fileno(), 1)
                os.dup2(devnull.fileno(), 2)
                yield
                sys.stdout.flush()
                sys.stderr.flush()
                _flush_native_stdio()
        finally:
            sys.stdout.flush()
            sys.stderr.flush()
            _flush_native_stdio()
            gc.collect()
            os.dup2(saved_stdout, 1)
            os.dup2(saved_stderr, 2)
            os.close(saved_stdout)
            os.close(saved_stderr)


class FixedTiling(Step):
    def __init__(self, pad_to_shape: List[int], resize_to_shape: List[int], tile_shape: List[int]):
        super(FixedTiling, self).__init__("FixedTiling",
                                          required_keys=["data"],
                                          yield_keys=["shape"]
                                          )
        self.pad_to_shape = pad_to_shape
        self.resize_to_shape = resize_to_shape
        self.tile_shape = tile_shape

    def forward(self, data_dict: Dict[str, Any]) -> Dict[str, Any]:
        orig_shape = data_dict["shape"]
        pad_to_shape = self.pad_to_shape
        resize_to_shape = self.resize_to_shape
        tile_shape = self.tile_shape
        x = data_dict["data"]
        # pad first
        pad_size = []
        for i in range(len(orig_shape) - 1, -1, -1):
            pad_size.append(0)
            pad_size.append(pad_to_shape[i] - orig_shape[i])
        x = torch.nn.functional.pad(x, pad_size)
        x = x.view(resize_to_shape)

        # Tile Shape
        Nt, Ct, Ht, Wt = tile_shape
        N, C, H, W = x.shape
        N_tiles = N // Nt
        C_tiles = C // Ct
        H_tiles = H // Ht
        W_tiles = W // Wt

        # Ensure divisibility
        assert N % Nt == 0 and C % Ct == 0 and H % Ht == 0 and W % Wt == 0, \
            "The tensor's dimensions must be divisible by the tile shape" + \
            f"({N}, {C}, {H}, {W})\n" + \
            f"({Nt}, {Ct}, {Ht}, {Wt})\n" + \
            f"Remainders: ({N % Nt}, {C % Ct}, {H % Ht}, {W % Wt})"

        # Reshape to separate tiles in each dimension
        x = x.view(N_tiles, Nt, C_tiles, Ct, H_tiles, Ht, W_tiles, Wt)

        # Permute to bring tiles next to each other
        x = x.permute(0, 2, 4, 6, 1, 3, 5, 7).contiguous()
        x = x.view(N_tiles * C_tiles * H_tiles * W_tiles, Nt, Ct, Ht, Wt)
        # save x to disk
        # torch.save(x, "data/tiled_tensor_fwd.pt")
        data_dict["data"] = x
        data_dict["tiles_shape"] = [N_tiles, C_tiles, H_tiles, W_tiles]

        return data_dict

    def backward(self, data_dict: Dict[str, Any]) -> Dict[str, Any]:
        x = data_dict['data']
        orig_shape = data_dict['shape']
        Nt, Ct, Ht, Wt = self.tile_shape
        N_tiles, C_tiles, H_tiles, W_tiles = data_dict['tiles_shape']
        padded_shape = self.pad_to_shape
        # torch.save(x, "data/tiled_tensor_rcv.pt")

        # reshape to [N_tiles, C_tiles, H_tiles, W_tiles, Nt, Ct, Ht, Wt]
        x = x.view(N_tiles, C_tiles, H_tiles, W_tiles, Nt, Ct, Ht, Wt)
        # permute to [N_tiles, Nt, C_tiles, Ct, H_tiles, Ht, W_tiles, Wt]
        x = x.permute(0, 4, 1, 5, 2, 6, 3, 7).contiguous()
        # reshape to padded shape
        x = x.view(padded_shape)
        # remove padding, x can be any shape
        # for i in range(len(orig_shape)):
        #     x = x.narrow(i, 0, orig_shape[i])
        data_dict["data"] = x
        return data_dict


class SplitIntoBatch(Step):
    """
    This step splits the input tensor into batches of size batch_size, such that each batch can be decoded separately.
    """
    def __init__(self, batch_size: int, last_dim_size: int, debug=True):
        super(SplitIntoBatch, self).__init__("SplitIntoBatch",
                                             required_keys=["data"],
                                             yield_keys=["data"]
                                             )
        self.batch_size = batch_size
        self.last_dim_size = last_dim_size
        self.each_batch_dim = last_dim_size // batch_size
        self.pad_size = self.each_batch_dim * batch_size - last_dim_size
        self.debug = debug

    def forward(self, data_dict: Dict[str, Any]) -> List[Dict[str, Any]]:
        # assert input is a dict
        x = data_dict["data"]
        if self.debug:
            assert isinstance(data_dict, dict), "Input must be a dictionary"
            # assert last dim size is correct
            assert x.shape[-1] == self.last_dim_size, "Last dimension size is not correct"

        # pad to make it divisible by batch_size
        x = torch.nn.functional.pad(x, (0, self.pad_size))
        # split
        x = x.view(*x.shape[:-1], self.each_batch_dim, self.batch_size)
        ret = []
        for i in range(self.batch_size):
            ret.append({"data": x[..., i]})
        return ret


    def backward(self, data_dict: List[Dict[str, Any]]) -> Dict[str, Any]:
        if self.debug:
            assert isinstance(data_dict, list), "Input must be a list"
            assert len(data_dict) > 0, "Input list must not be empty"
            assert isinstance(data_dict[0], dict), "Each element in the list must be a dictionary"
            for item in data_dict:
                assert 'data' in item, "Each dictionary must contain 'data' key"

        # Concatenate the split tensors back together along a new last dimension
        x = torch.stack([item['data'] for item in data_dict], dim=-1)

        # view back
        x = x.view(*x.shape[:-2], -1)

        # Remove any padding added during the forward pass
        if self.pad_size > 0:
            x = x[..., :-self.pad_size]

        return {'data': x}


class NVEncode(Step):
    def __init__(self, config, width, height):
        super(NVEncode, self).__init__("NVEncode",
                                       required_keys=["data", "tiles_shape"],
                                       yield_keys=["data"],
                                       preserve_keys=["code_size"]
                                       )
        self.config = config
        self.width = width
        self.height = height

    def forward(self, data_dict: Dict[str, Any]) -> Dict[str, Any]:
        x = data_dict["data"]
        num_tiles = x.shape[0]
        ret = []
        total_size = 0

        # Create encoder once and reuse for all tiles
        with _suppress_native_stdout():
            encoder = TensorEncoder(self.config, self.width, self.height)
            try:
                for i in range(num_tiles):
                    this_tile = torch.clone(x[i].contiguous())
                    encoded = encoder.encode(this_tile)
                    total_size += encoded.packet_sizes.sum().item()
                    ret.append(encoded)
            finally:
                if hasattr(encoder, "close"):
                    encoder.close()
                del encoder

        # print("Total size: ", total_size)
        data_dict["data"] = ret
        data_dict["code_size"] = total_size
        return data_dict

    def backward(self, data_dict: Dict[str, Any]) -> Dict[str, Any]:
        bitstreams = data_dict["data"]
        ret = []

        # Create decoder once and reuse for all tiles
        with _suppress_native_stdout():
            decoder = TensorDecoder(self.config.codec_type, self.width, self.height)
            try:
                for i in range(len(bitstreams)):
                    bitstream, packet_sizes = _encoded_tensors(bitstreams[i])
                    decoded = decoder.decode(bitstream, packet_sizes)
                    assert decoded.numel() != 0, "Decoded tensor is empty"
                    ret.append(decoded)
            finally:
                if hasattr(decoder, "close"):
                    decoder.close()
                del decoder

        data_dict["data"] = torch.stack(ret)
        return data_dict


class PadUVChannel(Step):
    def __init__(self):
        super(PadUVChannel, self).__init__("PadUVChannel", required_keys=["data"], yield_keys=["data"])

    def forward(self, data_dict: Dict[str, Any]) -> Dict[str, Any]:
        x = data_dict["data"]
        B, N, C, H, W = x.shape
        # pad C from 1 to 3 with 0
        assert C == 1, "Expect 1 channel, got {}".format(C)
        # pad the third dim to 3
        x = torch.nn.functional.pad(x, (0, 0, 0, 0, 0, 2, 0, 0))
        data_dict["data"] = x
        return data_dict

    def backward(self, data_dict: Dict[str, Any]) -> Dict[str, Any]:
        # unpad
        x = data_dict["data"]
        B, N, C, H, W = x.shape
        assert C == 3, "Expect 3 channel, got {}".format(C)
        x = x[:, :, 0:1, :, :]
        data_dict["data"] = x
        return data_dict


class MonoNVEncode(Step):
    def __init__(self, config, width, height):
        super(MonoNVEncode, self).__init__("MonoNVEncode",
                                       required_keys=["data", "tiles_shape"],
                                       yield_keys=["data"],
                                       preserve_keys=["code_size"]
                                       )
        self.config = config
        self.width = width
        self.height = height

    def convert_tensor_to_monochrome(self, tensor):
        assert tensor.ndim == 3
        luma_height = tensor.size(-2)
        total_height = luma_height + (luma_height + 1) // 2
        monochrome = torch.zeros(tensor.size(0), total_height, tensor.size(-1), dtype=torch.uint8, device=tensor.device)
        monochrome[:, :luma_height, :] = tensor
        return monochrome

    def convert_monochrome_to_tensor(self, monochrome):
        # Calculate the luma height from the monochrome height
        total_height = monochrome.size(-2)
        luma_height = (2 * total_height) // 3
        out_tensor = monochrome[:, :luma_height, :]
        return out_tensor

    def forward(self, data_dict: Dict[str, Any]) -> Dict[str, Any]:
        x = data_dict["data"]
        B, N, C, H, W = x.shape
        # assert C is 1
        assert C == 1, "Expect 1 channel, got {}".format(C)
        num_tiles = x.shape[0]
        ret = []
        total_size = 0

        # Create encoder once and reuse for all tiles
        with _suppress_native_stdout():
            encoder = TensorEncoder(self.config, self.width, self.height)
            try:
                for i in range(num_tiles):
                    this_tile = torch.clone(x[i].squeeze(1).contiguous())
                    # print(this_tile.shape)
                    filled = self.convert_tensor_to_monochrome(this_tile)
                    encoded = encoder.encode(filled)
                    total_size += encoded.packet_sizes.sum().item()
                    ret.append(encoded)
            finally:
                if hasattr(encoder, "close"):
                    encoder.close()
                del encoder

        # print("Total size: ", total_size)
        data_dict["data"] = ret
        data_dict["code_size"] = total_size
        return data_dict

    def backward(self, data_dict: Dict[str, Any]) -> Dict[str, Any]:
        bitstreams = data_dict["data"]
        ret = []

        # Create decoder once and reuse for all tiles
        with _suppress_native_stdout():
            decoder = TensorDecoder(self.config.codec_type, self.width, self.height)
            try:
                for i in range(len(bitstreams)):
                    bitstream, packet_sizes = _encoded_tensors(bitstreams[i])
                    decoded = decoder.decode(bitstream, packet_sizes)
                    # unsqueeze the channel dim
                    tensor = self.convert_monochrome_to_tensor(decoded)
                    assert decoded.numel() != 0, "Decoded tensor is empty"
                    ret.append(tensor)
            finally:
                if hasattr(decoder, "close"):
                    decoder.close()
                del decoder

        data_dict["data"] = torch.stack(ret)
        return data_dict


class MonoNVEncodeSequence(Step):
    """
    Encode the same tile position across consecutive layers as a frame sequence.

    Input shape is [frames, num_tiles, 1, 1, tile_h, tile_w]. Each tile index is
    encoded as its own video stream so inter-frame prediction happens across
    layers, not across unrelated tile positions.
    """

    def __init__(self, config, width, height):
        super(MonoNVEncodeSequence, self).__init__(
            "MonoNVEncodeSequence",
            required_keys=["data", "tiles_shape"],
            yield_keys=["data"],
            preserve_keys=["code_size"],
        )
        self.config = config
        self.width = width
        self.height = height

    def convert_tensor_to_monochrome(self, tensor):
        assert tensor.ndim == 3
        luma_height = tensor.size(-2)
        total_height = luma_height + (luma_height + 1) // 2
        monochrome = torch.zeros(
            tensor.size(0),
            total_height,
            tensor.size(-1),
            dtype=torch.uint8,
            device=tensor.device,
        )
        monochrome[:, :luma_height, :] = tensor
        return monochrome

    def convert_monochrome_to_tensor(self, monochrome):
        total_height = monochrome.size(-2)
        luma_height = (2 * total_height) // 3
        return monochrome[:, :luma_height, :]

    def forward(self, data_dict: Dict[str, Any]) -> Dict[str, Any]:
        x = data_dict["data"]
        if x.ndim == 5:
            frame_count = int(data_dict["tiles_shape"][0])
            num_tiles = x.shape[0] // frame_count
            x = x.view(frame_count, num_tiles, *x.shape[1:])
        assert x.ndim == 6, f"Expect [frames, tiles, 1, 1, H, W], got {x.shape}"
        frame_count, num_tiles, nt, ct, _, _ = x.shape
        assert nt == 1 and ct == 1, f"Expect singleton N/C tile dims, got {x.shape}"

        ret = []
        total_size = 0
        with _suppress_native_stdout():
            encoder = None
            try:
                encoder = TensorEncoder(self.config, self.width, self.height)
                for tile_idx in range(num_tiles):
                    tile_sequence = x[:, tile_idx].squeeze(1).squeeze(1).contiguous()
                    assert tile_sequence.shape[0] == frame_count
                    filled = self.convert_tensor_to_monochrome(tile_sequence)
                    encoded = encoder.encode(filled)
                    total_size += encoded.packet_sizes.sum().item()
                    ret.append(encoded)
            finally:
                if encoder is not None and hasattr(encoder, "close"):
                    encoder.close()
                if encoder is not None:
                    del encoder

        data_dict["data"] = ret
        data_dict["code_size"] = total_size
        data_dict["frame_count"] = frame_count
        return data_dict

    def backward(self, data_dict: Dict[str, Any]) -> Dict[str, Any]:
        bitstreams = data_dict["data"]
        ret = []

        with _suppress_native_stdout():
            decoder = None
            try:
                decoder = TensorDecoder(self.config.codec_type, self.width, self.height)
                for tile_idx in range(len(bitstreams)):
                    bitstream, packet_sizes = _encoded_tensors(bitstreams[tile_idx])
                    decoded = decoder.decode(bitstream, packet_sizes)
                    assert decoded.numel() != 0, "Decoded tensor is empty"
                    ret.append(self.convert_monochrome_to_tensor(decoded))
            finally:
                if decoder is not None and hasattr(decoder, "close"):
                    decoder.close()
                if decoder is not None:
                    del decoder

        # [tiles, frames, H, W] -> [frames, tiles, 1, 1, H, W]
        data_dict["data"] = torch.stack(ret, dim=1).unsqueeze(2).unsqueeze(2)
        return data_dict
