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


def sample_param(scale_min=1.0, scale_max=1.0, rot_prob=0.0, max_rot=0, hflip_prob=0.0, extra_offset=0.0):
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


def generate_transform_cv2(image_np, output_width, output_height, px, py, signx, signy, scale, angle, hflip, extra_offset_val):
    assert image_np.ndim == 3
    orig_shape = image_np.shape

    transforms = []

    # origin
    center_x = orig_shape[1] // 2
    center_y = orig_shape[0] // 2

    # rotation
    if angle != 0:
        translation = np.asarray([[1, 0, -center_x],
                                  [0, 1, -center_y],
                                  [0, 0, 1]], dtype=np.float32)
        transforms.append(translation)

        angle = np.deg2rad(angle)
        rotation = np.asarray([[np.cos(-angle), -np.sin(-angle), 0],
                               [np.sin(-angle), np.cos(-angle), 0],
                               [0, 0, 1]], dtype=np.float32)
        transforms.append(rotation)
        translation = np.asarray([[1, 0, center_x],
                                  [0, 1, center_y],
                                  [0, 0, 1]], dtype=np.float32)
        transforms.append(translation)
    #########################

    # possible offset without going out of bounds
    padx = (output_width // 2 / scale - orig_shape[1] // 2)
    pady = (output_height // 2 / scale - orig_shape[0] // 2)
    # padx = np.abs(padx)
    # pady = np.abs(pady)
    padx = padx + extra_offset_val * center_x
    pady = pady + extra_offset_val * center_y

    padx = padx * signx
    pady = pady * signy

    # patch center
    tx = center_x * (1-px) + (center_x + padx) * px
    ty = center_y * (1-py) + (center_y + pady) * py

    translation = np.asarray([[1, 0, -tx],
                              [0, 1, -ty],
                              [0, 0, 1]], dtype=np.float32)
    transforms.append(translation)

    # scale
    scale_matrix = np.asarray([[scale, 0, 0],
                               [0, scale, 0],
                               [0, 0, 1]], dtype=np.float32)
    transforms.append(scale_matrix)


    # horizontal flip
    if hflip:
        transforms.append(np.asarray([[-1, 0, 0],
                                      [0, 1, 0],
                                      [0, 0, 1]], dtype=np.float32))


    # move to center
    reverse_translation = np.asarray([[1, 0, output_width / 2],
                                      [0, 1, output_height / 2],
                                      [0, 0, 1]], dtype=np.float32)
    transforms.append(reverse_translation)

    # aggregate
    final_transform = np.eye(3)
    for t in transforms:
        final_transform = np.matmul(t, final_transform)

    return final_transform


def augment_cv2_deterministic(image_np, transform, output_width, output_height):
    image_t = cv2.warpPerspective(image_np,
                                  transform,
                                  (output_width, output_height),
                                  borderValue=0)
    image_t = Image.fromarray(image_t)
    return image_t

