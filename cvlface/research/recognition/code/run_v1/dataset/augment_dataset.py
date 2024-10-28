from .base_dataset import MXFaceDataset
from data_augs import make_augmenter

class AugmentMXDataset(MXFaceDataset):
    def __init__(self,
                 root_dir,
                 local_rank,
                 augmentation_version='v1',
                 aug_params=None,
                 ):
        super(AugmentMXDataset, self).__init__(root_dir, local_rank)
        print('augmentation_version', augmentation_version)
        self.augmenter = make_augmenter(augmentation_version, aug_params)

    def __getitem__(self, index, skip_augment=False):
        sample, target = self.read_sample(index)
        theta = None
        if not skip_augment:
            sample = self.augmenter.augment(sample)
            if isinstance(sample, tuple):
                sample, theta = sample

        if self.transform is not None:
            sample = self.transform(sample)

        # import cv2
        # cv2.imwrite('/mckim/temp/temp.png',255*0.5*sample.transpose(0,1).transpose(1,2).numpy() + 0.5)

        if theta is not None:
            placeholder = 0
            assert theta.shape == (2, 3)
            return sample, placeholder, target, theta
        else:
            return sample, target

