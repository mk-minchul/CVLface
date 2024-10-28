from tqdm import tqdm
import mxnet as mx
import numbers
import pandas as pd
import torch
import cv2
import numpy as np
from torch.utils.data import Dataset
from PIL import Image



class GeneralAugmentDataset(Dataset):
    def __init__(self, dataset, augmenter, transform):
        super(GeneralAugmentDataset, self).__init__()
        self.dataset = dataset
        self.augmenter = augmenter
        self.transform = transform

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        batch = self.dataset[index]
        sample, target = batch
        assert isinstance(sample, Image.Image)

        sample = self.augmenter.augment(sample)
        theta = None
        if isinstance(sample, tuple):
            sample, theta = sample

        if self.transform is not None:
            sample = self.transform(sample)

        if theta is not None:
            placeholder = 0
            assert theta.shape == (2, 3)
            return sample, placeholder, target, theta
        else:
            return sample, target


class GeneralAugmentLmdkThetaDataset(Dataset):
    def __init__(self,
                 dataset,
                 augmenter,
                 transform,
                 ):
        super(GeneralAugmentLmdkThetaDataset, self).__init__()
        self.dataset = dataset
        self.augmenter = augmenter
        self.transform = transform

        self.identity_theta = torch.zeros(2, 3)
        self.identity_theta[0, 0] = 1
        self.identity_theta[1, 1] = 1

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        placeholder = 0
        sample, target, ldmk = self.dataset[index]
        assert isinstance(sample, Image.Image)

        theta = None
        sample = self.augmenter.augment(sample)
        if isinstance(sample, tuple):
            sample, theta = sample

        if self.transform is not None:
            sample = self.transform(sample)

        if theta is not None:
            ldmk = self.transform_ldmk(ldmk, theta)
        else:
            theta = self.identity_theta.clone()
        ldmk = ldmk.float()
        return sample, target, ldmk, theta, placeholder, placeholder, placeholder


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


def iterate_record(imgidx, record):
    # make one yourself
    record_info = []
    for idx in tqdm(imgidx, total=len(imgidx)):
        s = record.read_idx(idx)
        header, _ = mx.recordio.unpack(s)
        label = header.label
        if not isinstance(label, numbers.Number):
            label = label[0]
        label = int(label)
        row = {'idx':idx, 'path': f'{label}/{idx}.jpg', 'label': label}
        record_info.append(row)
    record_info = pd.DataFrame(record_info)
    return record_info


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
