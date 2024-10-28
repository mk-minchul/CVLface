import torch
import math

@torch.no_grad()
def make_rel_keypoints(keypoints, query):
    seq_length = query.shape[1]
    side = int(math.sqrt(seq_length))
    assert side == math.sqrt(seq_length)

    # make a grid of points from 0 to 1
    coord = torch.linspace(0, 1, side+1, device=query.device, dtype=query.dtype)
    coord = (coord[:-1] + coord[1:]) / 2  # get center of patches

    x, y = torch.meshgrid(coord, coord, indexing='ij')
    grid = torch.stack([y, x], dim=-1).reshape(-1, 2).unsqueeze(0).unsqueeze(-2)  # BxNx1x2
    _keypoints = keypoints.unsqueeze(-3)  # Bx1x5x2
    diff = (grid - _keypoints)  # BxNx5x2
    diff = diff.flatten(2)  # BxNx10
    return diff


def make_grid_0_1(side, device, dtype):
    if isinstance(side, tuple):
        one_side = side[0]
        assert side[0] == side[1]
    else:
        one_side = side
    # make a grid of points from 0 to 1
    coord = torch.linspace(0, 1, one_side+1, device=device, dtype=dtype)
    coord = (coord[:-1] + coord[1:]) / 2  # get center of patches
    x, y = torch.meshgrid(coord, coord, indexing='ij')
    grid = torch.stack([y, x], dim=-1).reshape(-1, 2).unsqueeze(0).unsqueeze(-2)  # BxNx1x2
    return grid

def calc_rel_keypoints(keypoints, grid):
    _keypoints = keypoints.unsqueeze(-3)  # Bx1x5x2
    diff = (grid - _keypoints)  # BxNx5x2
    diff = diff.flatten(2)  # BxNx10
    return diff