import os
from torchvision.datasets import ImageFolder
import numpy as np
from torchvision.utils import make_grid
import torch
import cv2
from PIL import Image
import random

from .base_dataset import SyntheticDataset, MXFaceDataset
from .augment_dataset import AugmentMXDataset
from .repeated_dataset import RepeatedSamplingMXDataset
from .repeated_dataset_with_ldmk_theta import RepeatedWithLdmkThetaMXDataset
from .subset_dataset import SubsetDataset

def get_train_dataset(dataset_cfg, train_transform, aug_cfg, local_rank=0):

    # batch_size = cfg.trainers.batch_size
    # num_workers = cfg.trainers.num_workers
    # local_rank = cfg.trainers.local_rank
    # world_size = cfg.trainers.world_size

    root_dir = os.path.join(dataset_cfg.data_root, dataset_cfg.rec)
    rec = os.path.join(root_dir, 'train.rec')
    idx = os.path.join(root_dir, 'train.idx')

    # Synthetic
    if dataset_cfg.rec == "synthetic":
        train_set = SyntheticDataset(dataset_cfg.num_classes, dataset_cfg.num_image)
        label_mapping = None

    # Mxnet RecordIO
    elif os.path.exists(rec) and os.path.exists(idx):
        if aug_cfg.augmentation_version == 'none':
            assert dataset_cfg.repeated_sampling_cfg is None
            train_set = MXFaceDataset(root_dir=root_dir, local_rank=local_rank)
        else:
            if dataset_cfg.repeated_sampling_cfg is not None:
                if dataset_cfg.repeated_sampling_cfg.ldmk_path:
                    # repeated sampling + augmentation + ldmk
                    train_set = RepeatedWithLdmkThetaMXDataset(root_dir=root_dir, local_rank=local_rank,
                                                               augmentation_version=aug_cfg.augmentation_version,
                                                               aug_params=aug_cfg.aug_params,
                                                               repeated_sampling_cfg=dataset_cfg.repeated_sampling_cfg)
                else:
                    # repeated sampling + augmentation
                    train_set = RepeatedSamplingMXDataset(root_dir=root_dir, local_rank=local_rank,
                                                          augmentation_version=aug_cfg.augmentation_version,
                                                          aug_params=aug_cfg.aug_params,
                                                          repeated_sampling_cfg=dataset_cfg.repeated_sampling_cfg)
            else:
                # augmentation
                train_set = AugmentMXDataset(root_dir=root_dir, local_rank=local_rank,
                                             augmentation_version=aug_cfg.augmentation_version,
                                             aug_params=aug_cfg.aug_params)

        train_set.transform = train_transform

        # resample dataset if needed
        if hasattr(dataset_cfg, 'resample_dataset') and dataset_cfg.resample_dataset:

            if dataset_cfg.resample_dataset == 'one_half':
                removing_index = list(set(range(0, len(train_set))) - set(range(0, len(train_set), 2)))
            elif dataset_cfg.resample_dataset == 'one_fourth':
                removing_index = list(set(range(0, len(train_set))) - set(range(0, len(train_set), 4)))
            else:
                removing_index = np.load(os.path.join(root_dir, dataset_cfg.resample_dataset))
            train_set = SubsetDataset(train_set, removing_index)
            dataset_cfg.num_classes = len(train_set.unique_label)
            label_mapping = train_set.label_mapping
        else:
            label_mapping = None

    elif dataset_cfg.rec == '':
        raise ValueError('No dataset is provided')

    # Image Folder
    else:
        train_set = ImageFolder(root_dir, train_transform)
        label_mapping = None

    train_set.color_space = dataset_cfg.color_space

    return train_set, label_mapping


def set_epoch(dataloader, epoch, cfg):
    if hasattr(dataloader.sampler, 'set_epoch'):
        if cfg.trainers.local_rank == 0:
            print(f'Dataloader set epoch: {epoch}')
        dataloader.sampler.set_epoch(epoch)
    if hasattr(dataloader.dataset, 'set_augmentation'):
        if hasattr(cfg.data_augs, 'disable_aug_during_warmup') and cfg.data_augs.disable_aug_during_warmup:
            if cfg.trainers.local_rank == 0:
                print(f'set augmentation, epoch: {epoch} : {epoch >= cfg.optims.warmup_epoch}')
            dataloader.dataset.set_augmentation(epoch >= cfg.optims.warmup_epoch)


def visualize_dataset(dataloader, save_path):
    batch = [dataloader.dataset[i] for i in range(4)]
    batch_img = torch.stack([b[0] for b in batch], dim=0)
    grid = make_grid(batch_img, nrow=4, padding=2, normalize=False)
    grid = tensor_to_numpy_uin8(grid)
    if dataloader.dataset.color_space == 'RGB':
        Image.fromarray(grid).save(save_path)
    else:
        cv2.imwrite(save_path, grid)

def tensor_to_numpy_uin8(tensor):
    array = ((tensor * 0.5 + 0.5)*256).cpu().numpy().astype(np.uint8)
    return np.transpose(array, (1, 2, 0))


def worker_init_fn(worker_id, num_workers, rank, seed):
    # The seed of each worker equals to
    # num_worker * rank + worker_id + user_seed
    worker_seed = num_workers * rank + worker_id + seed
    np.random.seed(worker_seed)
    random.seed(worker_seed)
    torch.manual_seed(worker_seed)
