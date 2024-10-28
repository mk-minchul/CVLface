from data_augs.aug_utils import transform_cv2
from data_augs.aug_utils import transform_torch
import os
import numpy as np
from PIL import Image
import random

if __name__ == '__main__':

    # set seed
    np.random.seed(0)
    random.seed(0)

    root = '/data/data/faces/casia_webface/raw/CASIA-WebFace_raw_aligned_mtcnn/3010926'
    images = [os.path.join(root, name) for name in os.listdir(root)]
    sample_images = [np.array(Image.open(image_path)) for image_path in images]

    # original shape
    os.makedirs('/mckim/temp/temp_aug_v12', exist_ok=True)
    os.makedirs('/mckim/temp/temp_aug_v12_torch', exist_ok=True)
    output_width, output_height = 160, 160

    os.makedirs('/mckim/temp/temp_aug_v12_determ', exist_ok=True)
    for i, image in enumerate(sample_images):

        params = transform_cv2.sample_param(scale_min=0.8, scale_max=1.2, rot_prob=1.0, max_rot=45, hflip_prob=0.5, extra_offset=0.5)

        mat_cv = transform_cv2.generate_transform_cv2(image, output_width, output_height, **params)
        output_cv = transform_cv2.augment_cv2_deterministic(image.copy(), mat_cv, output_width, output_height)
        output_cv.save('/mckim/temp/temp_aug_v12_determ/{}_cv.png'.format(i))

        mat_torch = transform_torch.generate_transform_torch(image, output_width, output_height, **params)
        output_torch = transform_torch.augment_torch_deterministic(image.copy(), mat_torch, output_width, output_height)
        output_torch.save('/mckim/temp/temp_aug_v12_determ/{}_torch.png'.format(i))


    # time it (torch is slower)
    import time

    start = time.time()
    for _ in range(1000):
        for i, image in enumerate(sample_images):
            params = transform_cv2.sample_param(scale_min=0.8, scale_max=1.2, rot_prob=1.0, max_rot=45, hflip_prob=0.5, extra_offset=0.5)
            mat_cv = transform_cv2.generate_transform_cv2(image, output_width, output_height, **params)
            output_cv = transform_cv2.augment_cv2_deterministic(image.copy(), mat_cv, output_width, output_height)

    end = time.time()
    print('cv2 time: {}'.format(end - start))

    start = time.time()
    for _ in range(1000):
        for i, image in enumerate(sample_images):
            params = transform_cv2.sample_param(scale_min=0.8, scale_max=1.2, rot_prob=1.0, max_rot=45, hflip_prob=0.5, extra_offset=0.5)
            mat_torch = transform_torch.generate_transform_torch(image, output_width, output_height, **params)
            output_torch = transform_torch.augment_torch_deterministic(image.copy(), mat_torch, output_width, output_height)

    end = time.time()
    print('torch time: {}'.format(end - start))
