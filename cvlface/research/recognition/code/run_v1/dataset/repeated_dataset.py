from .augment_dataset import AugmentMXDataset
from tqdm import tqdm
import mxnet as mx
import numbers
import pandas as pd
import os
import torch

class RepeatedSamplingMXDataset(AugmentMXDataset):
    def __init__(self,
                 root_dir,
                 local_rank,
                 augmentation_version='v1',
                 aug_params=None,
                 repeated_sampling_cfg=None,
                 ):
        super(RepeatedSamplingMXDataset, self).__init__(root_dir, local_rank, augmentation_version, aug_params)
        # self.augmenter
        # self.imgidx
        assert repeated_sampling_cfg is not None
        self.repeated_sampling_cfg = repeated_sampling_cfg


    def __getitem__(self, index, skip_augment=False):
        sample1, target = self.read_sample(index)
        theta1 = torch.tensor([0])
        theta2 = torch.tensor([0])

        if not skip_augment:
            sample1 = self.augmenter.augment(sample1)
            if isinstance(sample1, tuple):
                sample1, theta1 = sample1

        sample1 = self.transform(sample1)

        if self.repeated_sampling_cfg.use_same_image:
            # same image
            extra_index = index
        else:
            # same subject
            extra_index = self.label_info[target.item()].sample(1).index.item()

        sample2, _target = self.read_sample(extra_index)
        assert target == _target
        if not skip_augment and self.repeated_sampling_cfg.second_img_augment:
            sample2 = self.augmenter.augment(sample2)
            if isinstance(sample2, tuple):
                sample2, theta2 = sample2

        if self.transform is not None:
            sample2 = self.transform(sample2)

        # import cv2
        # cv2.imwrite('/mckim/temp/temp.png',255*0.5*sample.transpose(0,1).transpose(1,2).numpy() + 0.5)

        if theta1.ndim != 1:
            return sample1, sample2, target, theta1, theta2
        else:
            # dummy theta
            return sample1, sample2, target



