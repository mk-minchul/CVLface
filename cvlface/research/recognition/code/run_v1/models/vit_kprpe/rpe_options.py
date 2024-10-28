import torch
import torch.nn as nn
from .RPE.KPRPE.kprpe_shared import get_rpe_config
from .RPE.KPRPE import relative_keypoints


def make_kprpe_shared(rpe_config, depth, num_heads):

    assert rpe_config.rpe_on == 'k'
    num_buckets = get_rpe_config(
        ratio=rpe_config.ratio,
        method=rpe_config.method,
        mode=rpe_config.mode,
        shared_head=rpe_config.shared_head,
        skip=0,
        rpe_on=rpe_config.rpe_on,
    )['rpe_k']['num_buckets']
    if rpe_config.ctx_type == 'rel_keypoint':
        keypoint_linear = nn.Linear(2 * rpe_config.num_keypoints, num_buckets)
        # init zero
        keypoint_linear.weight.data.zero_()
        keypoint_linear.bias.data.zero_()

    elif rpe_config.ctx_type == 'rel_keypoint_unshared':
        keypoint_linear = nn.Linear(2 * rpe_config.num_keypoints, num_buckets * depth)
        # init zero
        keypoint_linear.weight.data.zero_()
        keypoint_linear.bias.data.zero_()

    elif rpe_config.ctx_type == 'rel_keypoint_unshared_v2':
        keypoint_linear = nn.Sequential(
            nn.Linear(2 * rpe_config.num_keypoints, 256),
            nn.ReLU(inplace=True),
            nn.LayerNorm(256),
            nn.Linear(256, num_buckets * depth),
        )
        # init zero
        keypoint_linear[-1].weight.data.zero_()
        keypoint_linear[-1].bias.data.zero_()

    elif rpe_config.ctx_type == 'rel_keypoint_splithead':
        keypoint_linear = nn.Linear(2 * rpe_config.num_keypoints, num_buckets * num_heads)
        # init zero
        keypoint_linear.weight.data.zero_()
        keypoint_linear.bias.data.zero_()

    elif rpe_config.ctx_type == 'rel_keypoint_splithead_unshared':
        keypoint_linear = nn.Linear(2 * rpe_config.num_keypoints, num_buckets * num_heads * depth)
        # init zero
        keypoint_linear.weight.data.zero_()
        keypoint_linear.bias.data.zero_()

    elif rpe_config.ctx_type == 'rel_keypoint_v2':
        keypoint_linear = nn.Sequential(
            nn.Linear(2 * rpe_config.num_keypoints, 256),
            nn.ReLU(inplace=True),
            nn.LayerNorm(256),
            nn.Linear(256, num_buckets),
        )
        # init zero
        keypoint_linear[-1].weight.data.zero_()
        keypoint_linear[-1].bias.data.zero_()
    elif rpe_config.ctx_type == 'keypoint':
        keypoint_linear = nn.Linear(2 * rpe_config.num_keypoints, num_buckets)
        # init zero
        keypoint_linear.weight.data.zero_()
        keypoint_linear.bias.data.zero_()
    else:
        raise ValueError(f'Not support ctx_type: {rpe_config.ctx_type}')

    return keypoint_linear, num_buckets



def make_kprpe_input(keypoints, x, keypoint_linear, rpe_config, mask_ratio, depth, num_heads, num_buckets):
    B = x.shape[0]
    ctx_type = rpe_config.get('ctx_type', '')
    num_kp = rpe_config.num_keypoints
    if ctx_type == 'rel_keypoint':
        assert mask_ratio == 0
        rel_keypoints = relative_keypoints.make_rel_keypoints(keypoints, x)[:, :, :2 * num_kp]
        rel_keypoints = keypoint_linear(rel_keypoints).unsqueeze(1)  # B H N D
        extra_ctx = {'rel_keypoints': rel_keypoints}

    elif ctx_type == 'rel_keypoint_unshared':
        assert mask_ratio == 0
        rel_keypoints = relative_keypoints.make_rel_keypoints(keypoints, x)[:, :, :2 * num_kp]
        rel_keypoints = keypoint_linear(rel_keypoints)  # B H N D
        rel_keypoints = rel_keypoints.view(B, -1, depth, num_buckets).transpose(1, 2)
        rel_keypoints = torch.chunk(rel_keypoints, depth, dim=1)
        extra_ctx = [{'rel_keypoints': rel_keypoint} for rel_keypoint in rel_keypoints]

    elif ctx_type == 'rel_keypoint_unshared_v2':
        assert mask_ratio == 0
        rel_keypoints = relative_keypoints.make_rel_keypoints(keypoints, x)[:, :, :2 * num_kp]
        rel_keypoints = keypoint_linear(rel_keypoints)  # B H N D
        rel_keypoints = rel_keypoints.view(B, -1, depth, num_buckets).transpose(1, 2)
        rel_keypoints = torch.chunk(rel_keypoints, depth, dim=1)
        extra_ctx = [{'rel_keypoints': rel_keypoint} for rel_keypoint in rel_keypoints]

    elif ctx_type == 'rel_keypoint_splithead':
        assert mask_ratio == 0
        rel_keypoints = relative_keypoints.make_rel_keypoints(keypoints, x)[:, :, :2 * num_kp]
        rel_keypoints = keypoint_linear(rel_keypoints)  # B H N D
        rel_keypoints = rel_keypoints.view(B, -1, num_heads, num_buckets).transpose(1, 2)
        extra_ctx = {'rel_keypoints': rel_keypoints}
    elif ctx_type == 'rel_keypoint_splithead_unshared':
        assert mask_ratio == 0
        rel_keypoints = relative_keypoints.make_rel_keypoints(keypoints, x)[:, :, :2 * num_kp]
        rel_keypoints = keypoint_linear(rel_keypoints)  # B H N D
        rel_keypoints = rel_keypoints.view(B, -1, num_heads * depth, num_buckets).transpose(1, 2)
        rel_keypoints = torch.chunk(rel_keypoints, depth, dim=1)
        extra_ctx = [{'rel_keypoints': rel_keypoint} for rel_keypoint in rel_keypoints]

    elif ctx_type == 'rel_keypoint_v2':
        assert mask_ratio == 0
        rel_keypoints = relative_keypoints.make_rel_keypoints(keypoints, x)[:, :, :2 * num_kp]
        rel_keypoints = keypoint_linear(rel_keypoints).unsqueeze(1)  # B H N D
        extra_ctx = {'rel_keypoints': rel_keypoints}

    elif ctx_type == 'keypoint':
        keypoints = keypoints.flatten(1).unsqueeze(1)
        keypoints = keypoint_linear(keypoints).unsqueeze(1)
        extra_ctx = {'rel_keypoints': keypoints}
    else:
        raise ValueError(f'Not support ctx_type: {ctx_type}')

    return extra_ctx