import numpy as np
from data_augs.aug_utils import transform_torch
from data_augs.aug_utils import transform_cv2
from PIL import Image
from PIL import ImageDraw
import torch
from typing import Tuple, Dict
from torch import Tensor
from torchvision.transforms import functional as F
import imgaug.augmenters as iaa
import cv2
import albumentations as A
from torchvision import transforms


class GridSampleAugmenter():

    '''
    GridSampleAugmenter:
    This class is used to augment the input image while keeping track of the corresponding theta for grid sampling.
    Output is (image, theta) where theta can be used as

    >>>from torchvision.transforms import ToTensor
    >>>image_tensor = ToTensor()(image_pil).unsqueeze(0)
    >>>align_input_theta = theta.unsqueeze(0)
    >>>b, c, h, w = image_tensor.shape
    >>>sample_grid = torch.nn.functional.affine_grid(align_input_theta, [b, c, h, w], align_corners=True)
    >>>image_tensor_aug = torch.nn.functional.grid_sample(image_tensor, sample_grid, align_corners=True)
    '''

    def __init__(self, aug_params, input_size=112):

        print('GridSampleAugmenter')
        self.aug_params = aug_params
        self.input_size = input_size
        self.photo_aug = PhotometricRandAugment(num_ops=self.aug_params['photometric_num_ops'],
                                                magnitude=self.aug_params['photometric_magnitude'],
                                                magnitude_offset=self.aug_params['photometric_magnitude_offset'],
                                                num_magnitude_bins=self.aug_params['photometric_num_magnitude_bins'])
        self.blur_aug = BlurAugmenter(magnitude=self.aug_params['blur_magnitude'], prob=self.aug_params['blur_prob'])
        self.cutout = CutoutAugment(aug_params['cutout_prob'])

    def augment(self, sample):
        image_np = np.array(sample)

        # augment
        params = transform_torch.sample_param(
            scale_min=self.aug_params['scale_min'],
            scale_max=self.aug_params['scale_max'],
            rot_prob=self.aug_params['rot_prob'],
            max_rot=self.aug_params['max_rot'],
            hflip_prob=self.aug_params['hflip_prob'],
            extra_offset=self.aug_params['extra_offset'],
        )
        mat = transform_cv2.generate_transform_cv2(image_np, self.input_size, self.input_size, **params)
        aug_sample = transform_cv2.augment_cv2_deterministic(image_np, mat, self.input_size, self.input_size)

        # corresponding theta
        align_input_theta = transform_torch.generate_transform_torch(image_np, self.input_size, self.input_size, **params)
        align_input_theta = align_input_theta.squeeze(0)

        # cutout
        aug_sample = self.cutout.augment(aug_sample)

        # blur
        blur_params = self.blur_aug.sample_param()
        aug_sample = self.blur_aug.augment(aug_sample, param=blur_params)

        # photometric
        photo_params = self.photo_aug.sample_param()
        aug_sample = self.photo_aug.augment(aug_sample, param=photo_params)

        return aug_sample, align_input_theta


class CutoutAugment():

    def __init__(self, cutout_prob):
        self.cutout_prob = cutout_prob
        self.dropout = A.CoarseDropout(max_holes=20,  # Maximum number of regions to zero out. (default: 8)
                                       max_height=16,  # Maximum height of the hole. (default: 8)
                                       max_width=16,  # Maximum width of the hole. (default: 8)
                                       min_holes=12, # Maximum number of regions to zero out. (default: None, which equals max_holes)
                                       min_height=None, # Maximum height of the hole. (default: None, which equals max_height)
                                       min_width=None, # Maximum width of the hole. (default: None, which equals max_width)
                                       fill_value=0,  # value for dropped pixels.
                                       mask_fill_value=None,  # fill value for dropped pixels in mask.
                                       always_apply=False,
                                       p=1.0
                                       )
        self.random_resized_crop = transforms.RandomResizedCrop(size=(112, 112),
                                                                scale=(0.2, 1.0),
                                                                ratio=(0.75, 1.3333333333333333))

    def augment(self, sample):
        if np.random.random() < self.cutout_prob:
            if np.random.random() < 0.05:
                # not too natural
                return Image.fromarray(self.dropout(image=np.array(sample))['image'])
            else:
                new = np.zeros_like(np.array(sample))
                i, j, h, w = self.random_resized_crop.get_params(sample,
                                                                 self.random_resized_crop.scale,
                                                                 self.random_resized_crop.ratio)
                cropped = F.crop(sample, i, j, h, w)
                new[i:i+h,j:j+w, :] = np.array(cropped)
                sample = Image.fromarray(new.astype(np.uint8))
                return sample
        else:
            return sample


class PhotometricRandAugment():

    def __init__(self,
                 num_ops: int = 2,
                 magnitude: int = 9,
                 magnitude_offset: int = 4,
                 num_magnitude_bins: int = 31) -> None:
        self.num_ops = num_ops
        self.magnitude = magnitude
        self.magnitude_offset = magnitude_offset
        self.num_magnitude_bins = num_magnitude_bins
        self.op_names = list(self._augmentation_space(self.num_magnitude_bins).keys())
        self.op_meta = self._augmentation_space(self.num_magnitude_bins)

    def _augmentation_space(self, num_bins: int) -> Dict[str, Tuple[Tensor, bool]]:
        return {
            # op_name: (magnitudes, signed)
            "Identity": (torch.tensor(0.0), False),
            "Brightness": (torch.linspace(0.0, 0.9, num_bins), True),
            "Saturate": (torch.linspace(0.0, 0.9, num_bins), True),
            "Contrast": (torch.linspace(0.0, 0.9, num_bins), True),
            "Sharpness": (torch.linspace(0.0, 0.9, num_bins), True),
            "Equalize": (torch.tensor(0.0), False),
            "Grayscale": (torch.tensor(0.0), False),
        }

    def apply_op(self, img: Tensor, op_name: str, magnitude: float):
        if op_name == "Brightness":
            img = F.adjust_brightness(img, 1.0 + magnitude)
        elif op_name == "Saturate":
            img = F.adjust_saturation(img, 1.0 + magnitude)
        elif op_name == "Contrast":
            img = F.adjust_contrast(img, 1.0 + magnitude)
        elif op_name == "Sharpness":
            img = F.adjust_sharpness(img, 1.0 + magnitude)
        elif op_name == "Equalize":
            img = F.equalize(img)
        elif op_name == 'Grayscale':
            img = F.to_grayscale(img, num_output_channels=3)
        elif op_name == "Identity":
            pass
        else:
            raise ValueError("The provided operator {} is not recognized.".format(op_name))
        return img

    def sample_param(self):
        ops = []
        for _ in range(self.num_ops):
            # random sample op
            op_name = np.random.choice(self.op_names)
            # reduce probability of these two ops
            if op_name in ['Equalize', 'Grayscale']:
                op_name = np.random.choice(self.op_names)
                if op_name in ['Equalize', 'Grayscale']:
                    op_name = np.random.choice(self.op_names)

            magnitudes, signed = self.op_meta[op_name]
            # random sample magnitude
            magnitude_idx = np.random.randint(self.magnitude-self.magnitude_offset,
                                              self.magnitude+self.magnitude_offset)
            magnitude_idx = np.clip(magnitude_idx, 0, self.num_magnitude_bins-1)
            if magnitudes.ndim > 0:
                magnitude = float(magnitudes[magnitude_idx].item())
            else:
                magnitude = 0.0
            if signed and torch.randint(2, (1,)):
                magnitude *= -1.0
            ops.append((op_name, magnitude))
        return ops

    def augment(self, img: Tensor, param=None) -> Tensor:
        """
            img (PIL Image or Tensor): Image to be transformed.
        Returns:
            PIL Image or Tensor: Transformed image.
        """
        if param is None:
            param = self.sample_param()
        for op_name, magnitude in param:
            img = self.apply_op(img, op_name, magnitude)

        return img



class BlurAugmenter():

    def __init__(self, magnitude=0.5, prob=0.2):
        self.magnitude = magnitude
        self.prob = prob

    def sample_param(self):
        if np.random.random() < self.prob:
            blur_method = np.random.choice(['avg', 'gaussian',
                                            'resize', 'resize', 'resize', 'resize',
                                            'resize', 'resize', 'resize', 'resize'])  # more resizing aug, no motion
            if blur_method == 'avg':
                k = np.random.randint(1, int(10 * self.magnitude))
                param = [blur_method, k]
            elif blur_method == 'gaussian':
                sigma = np.random.random() * 4 * self.magnitude
                param = [blur_method, sigma]
            elif blur_method == 'motion':
                k = np.random.randint(5, max(int(10 * self.magnitude), 6))
                angle = np.random.randint(-45, 45)
                direction = np.random.random() * 2 - 1
                param = [blur_method, k, angle, direction]
            elif blur_method == 'resize':
                side_ratio = np.random.uniform(1.0 - 0.8 * self.magnitude, 1.0)
                interpolation1 = np.random.choice([cv2.INTER_NEAREST, cv2.INTER_LINEAR, cv2.INTER_AREA,
                                                  cv2.INTER_CUBIC, cv2.INTER_LANCZOS4])
                interpolation2 = np.random.choice([cv2.INTER_NEAREST, cv2.INTER_LINEAR, cv2.INTER_AREA,
                                                  cv2.INTER_CUBIC, cv2.INTER_LANCZOS4])
                param = [blur_method, side_ratio, [interpolation1, interpolation2]]
            else:
                raise ValueError('not a correct blur')
        else:
            param = ['skip']

        return param

    def augment(self, sample, param=None):
        if param is None:
            param = self.sample_param()
        blur_method = param[0]
        if blur_method == 'skip':
            return sample

        if blur_method == 'avg':
            blur_method, k = param
            avg_blur = iaa.AverageBlur(k=k) # max 10
            blurred = avg_blur(image=np.array(sample))
        elif blur_method == 'gaussian':
            blur_method, sigma = param
            gaussian_blur = iaa.GaussianBlur(sigma=sigma) # 4 is max
            blurred = gaussian_blur(image=np.array(sample))
        elif blur_method == 'motion':
            blur_method, k, angle, direction = param
            motion_blur = iaa.MotionBlur(k=k, angle=angle, direction=direction)  # k 20 max angle:-45 45, dir:-1 1
            blurred = motion_blur(image=np.array(sample))
        elif blur_method == 'resize':
            blur_method, side_ratio, interpolation = param
            blurred = self.low_res_augmentation(np.array(sample), side_ratio, interpolation)
        else:
            raise ValueError('not a correct blur')

        sample = Image.fromarray(blurred.astype(np.uint8))

        return sample

    def low_res_augmentation(self, img, side_ratio, interpolation):
        # resize the image to a small size and enlarge it back
        img_shape = img.shape
        small_side = int(side_ratio * img_shape[0])
        small_img = cv2.resize(img, (small_side, small_side), interpolation=interpolation[0])
        aug_img = cv2.resize(small_img, (img_shape[1], img_shape[0]), interpolation=interpolation[1])
        return aug_img


def main():
    image = Image.open('/data/data/faces/ms1mv2_subset_images/84946/5770863.jpg')
    # draw a square box on the image
    image_draw = ImageDraw.Draw(image)
    image_draw.rectangle((10, 10, 110, 110), outline='red')
    image_draw.rectangle((0, 0, 120, 120), outline='blue')

    scale_min = 0.7
    scale_max = 2.0
    rot_prob = 0.2
    max_rot = 30
    hflip_prob = 0.5
    extra_offset = 0.15

    photometric_num_ops = 2
    photometric_magnitude = 14
    photometric_magnitude_offset = 9
    photometric_num_magnitude_bins = 31

    blur_magnitude = 1.0
    blur_prob = 0.3
    cutout_prob = 0.2

    aug_params = {
        'scale_min': scale_min,
        'scale_max': scale_max,
        'rot_prob': rot_prob,
        'max_rot': max_rot,
        'hflip_prob': hflip_prob,
        'extra_offset': extra_offset,
        'photometric_num_ops': photometric_num_ops,
        'photometric_magnitude': photometric_magnitude,
        'photometric_magnitude_offset': photometric_magnitude_offset,
        'photometric_num_magnitude_bins': photometric_num_magnitude_bins,
        'blur_magnitude': blur_magnitude,
        'blur_prob': blur_prob,
        'cutout_prob': cutout_prob
    }
    align_input_size = 112
    augmenter = GridSampleAugmenter(aug_params, align_input_size)
    # make a grid 10x10
    grids = []
    grids_theta = []
    for i in range(10):
        grid = []
        grid_theta = []
        for j in range(10):
            align_input_sample, align_input_theta = augmenter.augment(image)
            grid.append(align_input_sample)
            from torchvision.transforms import ToTensor
            image_tensor = ToTensor()(image).unsqueeze(0)
            align_input_theta = align_input_theta.unsqueeze(0)
            b, c, h, w = image_tensor.shape
            sample_grid = torch.nn.functional.affine_grid(align_input_theta, [b, c, h, w], align_corners=True)
            image_tensor_aug = torch.nn.functional.grid_sample(image_tensor, sample_grid, align_corners=True)
            from general_utils.img_utils import tensor_to_pil
            grid_theta.append(tensor_to_pil(image_tensor_aug)[0])
        grids.append(grid)
        grids_theta.append(grid_theta)
    # save the grid
    grid_image = Image.new('RGB', (1120, 1120))
    for i in range(10):
        for j in range(10):
            grid_image.paste(grids[i][j], (112 * j, 112 * i))
    grid_image.save(f'/mckim/temp/GridSampleAugmenter.jpg')

    grid_theta_image = Image.new('RGB', (1120, 1120))
    for i in range(10):
        for j in range(10):
            grid_theta_image.paste(grids_theta[i][j], (112 * j, 112 * i))
    grid_theta_image.save(f'/mckim/temp/GridSampleAugmenter_by_theta.jpg')


if __name__ == '__main__':
    main()
