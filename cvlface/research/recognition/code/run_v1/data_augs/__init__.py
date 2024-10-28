from .basic_augmenter import BasicAugmenter
from .gridsample_augmenter import GridSampleAugmenter

def make_augmenter(augmentation_version, aug_params):
    if augmentation_version == 'basic':
        augmenter = BasicAugmenter(crop_augmentation_prob=aug_params.crop_augmentation_prob,
                                   photometric_augmentation_prob=aug_params.photometric_augmentation_prob,
                                   low_res_augmentation_prob=aug_params.low_res_augmentation_prob,
                                   )
    elif augmentation_version == 'gridsample':
        augmenter = GridSampleAugmenter(aug_params, input_size=112)
    else:
        raise ValueError('not correct augmentation version')
    return augmenter