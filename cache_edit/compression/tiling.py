import torch
from typing import Dict, Any, List


def adaptive_tile_anyshape_tensor_to_video(x: torch.Tensor, max_tile_size=[4, 3, 1024, 1024]) -> Dict[str, Any]:
    # x: a tensor of 2D, 3D or 4D
    # max_tile_size: the maximum size of each tile, must be 4D, [Nt, Ct, Ht, Wt], representing a video
    # Ct must be exactly the same as the channel size of x
    # represented as a list. e.g. [10, 3, 256, 256]

    # Step 1: Convert any shape tensor to [-1, Ct, -1,  -1]

    # if x is 3D: [C, H, W], make it [-1, Ct, H, W] tile it and return it. The return value should be in shape [T, Nt, Ct, Ht, Wt]
    ...


def fixed_tile(x: torch.Tensor, pad_to_shape: List[int], resize_to_shape: List[int], tile_shape: List[int]):
    # Ensure the input tensor is a float tensor for `pad` compatibility
    # View and Pad
    orig_shape = x.shape
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
    assert N % Nt == 0 and C % Ct == 0 and H % Ht == 0 and W % Wt == 0, "The tensor's dimensions must be divisible by the tile shape"

    # Reshape to separate tiles in each dimension
    x = x.view(N_tiles, Nt, C_tiles, Ct, H_tiles, Ht, W_tiles, Wt)

    # Permute to bring tiles next to each other
    x = x.permute(0, 2, 4, 6, 1, 3, 5, 7).contiguous()
    x = x.view(N_tiles * C_tiles * H_tiles * W_tiles, Nt, Ct, Ht, Wt)
    ret_dict = {
        "data": x,
        "orig_shape": orig_shape,
        "padded_shape": pad_to_shape,
        "tile_shape": tile_shape,
        "tiles_shape": [N_tiles, C_tiles, H_tiles, W_tiles],
    }
    return ret_dict


def fixed_untile(data_dict):
    # Extract necessary details from data_dict
    x = data_dict['data']
    orig_shape = data_dict['orig_shape']
    Nt, Ct, Ht, Wt = data_dict['tile_shape']
    N_tiles, C_tiles, H_tiles, W_tiles = data_dict['tiles_shape']
    padded_shape = data_dict['padded_shape']

    # reshape to [N_tiles, C_tiles, H_tiles, W_tiles, Nt, Ct, Ht, Wt]
    x = x.view(N_tiles, C_tiles, H_tiles, W_tiles, Nt, Ct, Ht, Wt)
    # permute to [N_tiles, Nt, C_tiles, Ct, H_tiles, Ht, W_tiles, Wt]
    x = x.permute(0, 4, 1, 5, 2, 6, 3, 7).contiguous()
    # reshape to padded shape
    x = x.view(padded_shape)
    # remove padding, x can be any shape
    for i in range(len(orig_shape)):
        x = x.narrow(i, 0, orig_shape[i])
    return x


if __name__ == "__main__":
    x = torch.randn(4096, 4096)
    Nt, Ct, Ht, Wt = 1, 3, 683, 1024
    tiled = fixed_tile(x, [4098, 4096 * 2], [1, 3, 1366, 4096 * 2], [Nt, Ct, Ht, Wt])
    untiled = fixed_untile(tiled)
    print(x.shape)
    print(untiled.shape)
    diff = torch.abs(x - untiled)
    print(diff.max())
