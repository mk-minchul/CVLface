from torch.utils.data import Dataset
from .augment_dataset import AugmentMXDataset
from .repeated_dataset import RepeatedSamplingMXDataset
from .repeated_dataset_with_ldmk_theta import RepeatedWithLdmkThetaMXDataset
import numpy as np
import torch

class SubsetDataset(Dataset):
    def __init__(self, dataset, drop_indices):
        """
        Initializes the SubsetDataset.
        Args:
        - dataset (Dataset): The original dataset.
        - drop_indices (set of int): The indices to drop from the dataset.
        """
        self.dataset = dataset
        self.drop_indices = set(drop_indices)

        # Calculate the indices to keep
        self.indices = [i for i in range(len(dataset)) if i not in self.drop_indices]

        print('original sample count :', len(self.dataset))
        print('original label count :', len(self.dataset.info['label'].unique()))
        print(f'removing {len(self.drop_indices)} ({len(self.drop_indices) / len(self.dataset) * 100 :.2f}%) samples ')
        dropped_info = self.dataset.info.copy()
        dropped_info = dropped_info.drop(self.drop_indices)

        unique_label = dropped_info['label'].unique()
        unique_label = np.sort(unique_label)
        self.unique_label = unique_label
        self.label_mapping = {k: i for i, k in enumerate(unique_label)}
        print('new sample count :', len(dropped_info))
        print('new label count :', len(self.label_mapping))

        # adjust self.label_info if there is one, so you don't repeat samples from drop indices
        if hasattr(self.dataset, 'label_info'):
            new_label_info = {k: v for k, v in dropped_info.groupby('label')}
            self.dataset.label_info = new_label_info


    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        # Adjust the index to refer to the index in the original dataset
        result = self.dataset[self.indices[idx]]
        if isinstance(self.dataset, RepeatedWithLdmkThetaMXDataset):
            target_index = 1
        elif isinstance(self.dataset, RepeatedSamplingMXDataset):
            target_index = 2
        elif isinstance(self.dataset, AugmentMXDataset):
            target_index = 2 if len(result) == 4 else 1
        else:
            raise NotImplementedError

        target = result[target_index]
        remapped_target = self.remap_target(target)

        result = list(result)
        result[target_index] = remapped_target
        result = tuple(result)
        return result

    def remap_target(self, target):
        if isinstance(target, int):
            return self.label_mapping[target]
        elif isinstance(target, np.ndarray):
            return np.array([self.label_mapping[t] for t in target])
        elif isinstance(target, float):
            return self.label_mapping[int(target)]
        elif isinstance(target, torch.Tensor):
            return torch.tensor(self.label_mapping[target.item()], dtype=torch.long)
        else:
            raise NotImplementedError
        pass