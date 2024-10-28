from .augment_dataset import AugmentMXDataset
from tqdm import tqdm
import mxnet as mx
import numbers
import pandas as pd
import os
import torch
import cv2
import numpy as np

class RepeatedWithLdmkThetaMXDataset(AugmentMXDataset):
    def __init__(self,
                 root_dir,
                 local_rank,
                 augmentation_version='v1',
                 aug_params=None,
                 repeated_sampling_cfg=None,
                 ):
        super(RepeatedWithLdmkThetaMXDataset, self).__init__(root_dir, local_rank, augmentation_version, aug_params)
        # self.augmenter
        # self.imgidx
        assert repeated_sampling_cfg is not None
        self.repeated_sampling_cfg = repeated_sampling_cfg
        self.disable_repeat = repeated_sampling_cfg.disable_repeat
        self.skip_aug_prob_in_disable_repeat = repeated_sampling_cfg.skip_aug_prob_in_disable_repeat

        self.ldmk_info = pd.read_csv(os.path.join(root_dir, repeated_sampling_cfg.ldmk_path), sep=',', index_col=0)
        self.identity_theta = torch.zeros(2, 3)
        self.identity_theta[0, 0] = 1
        self.identity_theta[1, 1] = 1

        self.do_augment = True
        self.prev_index = None
        self.prev_label = None
        self.repeated = False

    def set_augmentation(self, value):
        print('set augmentation', value)
        self.do_augment = value

    def get_one_sample(self, index, augment=True):
        sample, target = self.read_sample(index)

        theta = None

        if augment:
            sample = self.augmenter.augment(sample)
            if isinstance(sample, tuple):
                sample, theta = sample

        if self.transform is not None:
            sample = self.transform(sample)

        # load landmark
        ldmk = self.ldmk_info.loc[index].values
        if len(ldmk) == 10:
            ldmk = ldmk.reshape(-1, 2)
        else:
            ldmk = ldmk.reshape(-1, 3)[:, :2]

        ldmk = torch.from_numpy(ldmk)
        if theta is not None:
            ldmk = self.transform_ldmk(ldmk, theta)
        else:
            theta = self.identity_theta.clone()

        ldmk = ldmk.float()
        return sample, target, ldmk, theta


    def __getitem__(self, index, skip_augment=False):

        placeholder = 0
        augment = not skip_augment and self.do_augment

        if self.prev_index is not None and augment:
            if self.repeated_sampling_cfg.repeated_augment_prob > 0:
                if np.random.rand() < self.repeated_sampling_cfg.repeated_augment_prob and not self.repeated:
                    self.repeated = True
                    if self.repeated_sampling_cfg.use_same_image:
                        index = self.prev_index
                    else:
                        index = self.label_info[self.prev_label].sample(1).index.item()
                else:
                    self.repeated = False


        if self.disable_repeat:
            if np.random.rand() < self.skip_aug_prob_in_disable_repeat:
                augment = False
        sample1, target, ldmk1, theta1 = self.get_one_sample(index, augment=augment)
        if self.repeated:
            if self.prev_label is not None:
                if augment and self.prev_label != target.item():
                    print('Warning repeated label different {} {}'.format(target.item(), self.prev_label))

        self.prev_index = index
        self.prev_label = target.item()

        # vis1 = visualize_landmark(sample1, ldmk1)
        # cv2.imwrite('/mckim/temp/temp.png', vis1[:, :, ::-1])


        if self.disable_repeat:
            return sample1, target, ldmk1, theta1, placeholder, placeholder, placeholder

        # get extra image index
        if self.repeated_sampling_cfg.use_same_image:
            extra_index = index
        else:
            extra_index = self.label_info[target.item()].sample(1).index.item()

        extra_augment = augment and self.repeated_sampling_cfg.second_img_augment
        sample2, target2, ldmk2, theta2 = self.get_one_sample(extra_index, augment=extra_augment)
        assert target == target2

        # vis1 = visualize_landmark(sample1, ldmk1)
        # vis2 = visualize_landmark(sample2, ldmk2)
        # vis = np.concatenate([vis1, vis2], axis=1)
        # cv2.imwrite('/mckim/temp/temp.png', vis[:, :, ::-1])



        return sample1, target, ldmk1, theta1, sample2, ldmk2, theta2

    def transform_ldmk(self, ldmk, theta):
        inv_theta = inv_matrix(theta.unsqueeze(0)).squeeze(0)
        ldmk = torch.cat([ldmk, torch.ones(ldmk.shape[0], 1)], dim=1).float()
        transformed_ldmk = (((ldmk) * 2 - 1) @ inv_theta.T) / 2 + 0.5
        if inv_theta[0, 0] < 0:
            transformed_ldmk = self.mirror_ldmk(transformed_ldmk)
        return transformed_ldmk

    def mirror_ldmk(self, ldmk):
        if len(ldmk) == 5:
            return self.mirror_ldmk_5(ldmk)
        else:
            return self.mirror_ldmk_34(ldmk)

    def mirror_ldmk_5(self, ldmk):
        # landm
        new_ldmk = ldmk.clone()
        tmp = new_ldmk[1, :].clone()
        new_ldmk[1, :] = new_ldmk[0, :]
        new_ldmk[0, :] = tmp
        tmp1 = new_ldmk[4, :].clone()
        new_ldmk[4, :] = new_ldmk[3, :]
        new_ldmk[3, :] = tmp1
        return new_ldmk

    def mirror_ldmk_34(self, ldmk):
        raise NotImplementedError




def visualize_landmark(img, landmark):
    if isinstance(img, torch.Tensor):
        img = img.clone()
        # make it to numpy array
        img = img.permute(1, 2, 0).numpy()
        img = (img * 0.5 + 0.5) * 255
        img = img.astype(np.uint8)
        img = img.copy()
    else:
        img = img.copy()
    landmark = landmark.clone().reshape(-1, 2) * torch.tensor([[img.shape[1], img.shape[0]]])
    for i in range(landmark.shape[0]):
        cv2.circle(img, (int(landmark[i][0]), int(landmark[i][1])), 2, (0, 0, 255), -1)
        # put index on the landmark
        cv2.putText(img, str(i), (int(landmark[i][0]), int(landmark[i][1])), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
    return img

def inv_matrix(theta):
    # torch batched version
    assert theta.ndim == 3
    a, b, t1 = theta[:, 0,0], theta[:, 0,1], theta[:, 0,2]
    c, d, t2 = theta[:, 1,0], theta[:, 1,1], theta[:, 1,2]
    det = a * d - b * c
    inv_det = 1.0 / det
    inv_mat = torch.stack([
        torch.stack([d * inv_det, -b * inv_det, (b * t2 - d * t1) * inv_det], dim=1),
        torch.stack([-c * inv_det, a * inv_det, (c * t1 - a * t2) * inv_det], dim=1)
    ], dim=1)
    return inv_mat
