from PIL import Image
import numpy as np
import os
import cv2
import torch
import random


def sample_param_debug():
    result = {
        'px': 0.5,
        'py': 0.5,
        'signx': 1,
        'signy': 1,
        'scale': 1,
        'angle': 0,
        'hflip': 0,
        'extra_offset_val': 0,
    }
    return result


def sample_param(scale_min=1.0, scale_max=1.0, rot_prob=0.0, max_rot=0, hflip_prob=0.0, extra_offset=0.0, **kwargs, ):
    px = np.random.uniform(0, 1)
    py = np.random.uniform(0, 1)
    signx = (-1) ** np.random.randint(0, 2)
    signy = (-1) ** np.random.randint(0, 2)
    scale = np.random.uniform(scale_min, scale_max)

    if random.random() < rot_prob:
        angle = np.random.uniform(-max_rot, max_rot)
    else:
        angle = 0

    if random.random() < hflip_prob:
        hflip = 1
    else:
        hflip = 0

    extra_offset_val = np.random.uniform(0, extra_offset)
    result = {
        'px': px,
        'py': py,
        'signx': signx,
        'signy': signy,
        'scale': scale,
        'angle': angle,
        'hflip': hflip,
        'extra_offset_val': extra_offset_val,
    }
    return result



def generate_transform_torch(image_np, output_width, output_height, px, py, signx, signy, scale, angle, hflip, extra_offset_val):
    assert image_np.ndim == 3
    orig_shape = image_np.shape
    orig_width = orig_shape[1]
    orig_height = orig_shape[0]

    # origin
    center_x = 0
    center_y = 0

    # extreme
    extreme_x = 1 - (scale * 1 * orig_width / output_width)
    extreme_y = 1 - (scale * 1 * orig_height / output_height)

    extreme_x = extreme_x + extra_offset_val * (scale * 1 * orig_width / output_width)
    extreme_y = extreme_y + extra_offset_val * (scale * 1 * orig_height / output_height)


    tx = center_x * (1 - px) + extreme_x * px
    ty = center_y * (1 - py) + extreme_y * py
    tx = tx * signx
    ty = ty * signy

    transforms = []

    # horizontal flip
    if hflip:
        transforms.append(np.asarray([[-1, 0, 0],
                                      [0, 1, 0],
                                      [0, 0, 1]], dtype=np.float32))

    translation = np.asarray([[1, 0, +tx],  # -1 to 1
                              [0, 1, +ty],  # -1 to 1
                              [0, 0, 1]], dtype=np.float32)
    transforms.append(translation)

    # scale
    scale_x = output_width / orig_width / scale
    scale_y = output_height / orig_height / scale
    scale_matrix = np.asarray([[scale_x, 0, 0],
                               [0, scale_y, 0],
                               [0, 0, 1]], dtype=np.float32)
    transforms.append(scale_matrix)

    # rotation
    angle = np.deg2rad(angle)
    rotation = np.asarray([[np.cos(angle), -np.sin(angle), 0],
                           [np.sin(angle), np.cos(angle), 0],
                           [0, 0, 1]], dtype=np.float32)
    transforms.append(rotation)

    # aggregate
    final_transform = np.eye(3)
    for t in transforms:
        final_transform = np.matmul(t, final_transform)
    final_transform = torch.from_numpy(final_transform).float()[None, :2, :]
    return final_transform


def augment_torch_deterministic(image_np, transform, output_width, output_height):
    grid = torch.nn.functional.affine_grid(transform,
                                           [1, image_np.shape[2], output_height, output_width], align_corners=True)
    image_torch = torch.from_numpy(image_np).float().permute(2, 0, 1)[None, :, :, :]
    image_t = torch.nn.functional.grid_sample(image_torch, grid, align_corners=True)
    image_t = image_t.permute(0, 2, 3, 1).numpy()[0].astype(np.uint8)
    image_t = Image.fromarray(image_t)
    return image_t
