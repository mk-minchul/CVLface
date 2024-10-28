import torch
import torch.nn as nn

from .RPE.KPRPE.kprpe_shared import get_rpe_config
from .RPE.KPRPE import relative_keypoints
import torch.nn.functional as F


def make_kprpe_shared(rpe_config, net):

    assert rpe_config.rpe_on == 'k'
    num_buckets = get_rpe_config(
        ratio=rpe_config.ratio,
        method=rpe_config.method,
        mode=rpe_config.mode,
        shared_head=rpe_config.shared_head,
        skip=0,
        rpe_on=rpe_config.rpe_on,
    )['rpe_k']['num_buckets']

    if rpe_config.ctx_type == 'rel_keypoint_splithead_unshared':
        swin_config = get_swin_config(net)

        module_list = []
        for side, blocks_cfg in swin_config.items():
            total_heads = sum([block_cfg['num_heads'] for block_cfg in blocks_cfg])
            keypoint_linear = nn.Linear(2 * rpe_config.num_keypoints, num_buckets * total_heads)
            # init zero
            keypoint_linear.weight.data.zero_()
            keypoint_linear.bias.data.zero_()
            module_list.append(keypoint_linear)
        keypoint_linear = nn.ModuleList(module_list)
    else:
        raise ValueError(f'Not support ctx_type: {rpe_config.ctx_type}')

    return keypoint_linear, num_buckets, swin_config



def make_kprpe_bias(keypoints, x, keypoint_linear, rpe_config, swin_config, num_buckets, kprpes):
    B = x.shape[0]
    ctx_type = rpe_config.get('ctx_type', '')
    num_kp = rpe_config.num_keypoints
    if ctx_type == 'rel_keypoint_splithead_unshared':

        extra_ctx = []
        for feature_size, linear, kprpe in zip(swin_config.keys(), keypoint_linear, kprpes):
            grid = relative_keypoints.make_grid_0_1(feature_size, x.device, x.dtype)
            rel_keypoints = relative_keypoints.calc_rel_keypoints(keypoints, grid)[:, :, :2 * num_kp]
            rel_keypoints = linear(rel_keypoints)  # B H N D
            blocks_cfg = swin_config[feature_size]
            heads = [block_cfg['num_heads'] for block_cfg in blocks_cfg]
            rel_keypoints = rel_keypoints.view(B, -1, sum(heads), num_buckets).transpose(1, 2)
            rel_keypoints = torch.split(rel_keypoints, heads, dim=1)
            for rel_kp, block_cfg in zip(rel_keypoints, blocks_cfg):
                # make table
                windowed_rel_kp = split_window(rel_kp, feature_size[0], feature_size[1],
                                               block_cfg['window_size'], block_cfg['shift_size'])

                per_window_rel_kp = torch.split(windowed_rel_kp, 1, dim=2)

                window_rpe_biases = []
                for window_kp in per_window_rel_kp:
                    per_window_rpe_biases = kprpe(window_kp.squeeze(2))
                    window_rpe_biases.append(per_window_rpe_biases)
                window_rpe_biases = torch.stack(window_rpe_biases, dim=1)
                B, n_window, nhead, attn_sz, attn_sz2 = window_rpe_biases.shape
                window_rpe_biases = window_rpe_biases.view(B * n_window, nhead, attn_sz, attn_sz2).contiguous()

                row = {'num_heads': block_cfg['num_heads'],
                       'shift_size': block_cfg['shift_size'],
                       'feature_size': block_cfg['feature_size'],
                       'window_size': block_cfg['window_size'],
                       'rel_keypoints': window_rpe_biases}
                extra_ctx.append(row)
    else:
        raise ValueError(f'Not support ctx_type: {ctx_type}')

    return extra_ctx




def split_window(ctx, x_H, x_W, window_size, shift_size):
    # ctx: # B H N D
    B = ctx.shape[0]
    num_head = ctx.shape[1]
    D = ctx.shape[3]
    ctx = ctx.reshape(B, num_head, x_H, x_W, D)
    ctx = ctx.reshape(B * num_head, x_H, x_W, D)

    BH, H, W, C = ctx.shape
    # pad feature maps to multiples of window size
    pad_r = (window_size[1] - W % window_size[1]) % window_size[1]
    pad_b = (window_size[0] - H % window_size[0]) % window_size[0]
    ctx = F.pad(ctx, (0, 0, 0, pad_r, 0, pad_b))
    _, pad_H, pad_W, _ = ctx.shape

    shift_size = shift_size.copy()
    # If window size is larger than feature size, there is no need to shift window
    if window_size[0] >= pad_H:
        shift_size[0] = 0
    if window_size[1] >= pad_W:
        shift_size[1] = 0

    # cyclic shift
    if sum(shift_size) > 0:
        ctx = torch.roll(ctx, shifts=(-shift_size[0], -shift_size[1]), dims=(1, 2))

    # partition windows
    grid_size = ((pad_H // window_size[0]), (pad_W // window_size[1]))
    num_windows = grid_size[0] * grid_size[1]
    ctx = ctx.view(BH, pad_H // window_size[0], window_size[0], pad_W // window_size[1], window_size[1], C)
    ctx = ctx.permute(0, 1, 3, 2, 4, 5)
    ctx = ctx.reshape(B, num_head, grid_size[0], grid_size[1], window_size[0], window_size[1], C)
    ctx = ctx.reshape(B, num_head, num_windows, window_size[0] * window_size[1], C)  # B, H, num_grid, num_el_window, C
    # ex : [32, 4, 64, 25, 49]
    return ctx.contiguous()


def get_swin_config(net):
    assert hasattr(net, 'features')
    features = net.features

    input = torch.rand(1, 3, 112, 112)
    config = {}
    for module in features:
        if hasattr(module, 'attn'):
            attn = module.attn
            num_heads = attn.num_heads
            shift_size = attn.shift_size
            feature_size = tuple(input.shape[1:3])
            window_size = attn.window_size
            if feature_size not in config:
                config[feature_size] = []
            config[feature_size].append({'num_heads': num_heads, 'shift_size': shift_size,
                                         'feature_size': feature_size, 'window_size': window_size,
                                         })
        input = module(input)
    # {(37, 37): [{'num_heads': 4, 'shift_size': [0, 0], 'feature_size': (37, 37)},
    #   {'num_heads': 4, 'shift_size': [2, 2], 'feature_size': (37, 37)}],
    #  (19, 19): [{'num_heads': 8, 'shift_size': [0, 0], 'feature_size': (19, 19)},
    #   {'num_heads': 8, 'shift_size': [2, 2], 'feature_size': (19, 19)}],
    #  (10, 10): [{'num_heads': 16, 'shift_size': [0, 0], 'feature_size': (10, 10)},
    #   {'num_heads': 16, 'shift_size': [2, 2], 'feature_size': (10, 10)},
    #   {'num_heads': 16, 'shift_size': [0, 0], 'feature_size': (10, 10)},
    # ...
    #   {'num_heads': 16, 'shift_size': [0, 0], 'feature_size': (10, 10)},
    #   {'num_heads': 16, 'shift_size': [2, 2], 'feature_size': (10, 10)}],
    #  (5, 5): [{'num_heads': 32, 'shift_size': [0, 0], 'feature_size': (5, 5)},
    #   {'num_heads': 32, 'shift_size': [2, 2], 'feature_size': (5, 5)}]}
    return config