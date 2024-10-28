import os
from typing import Union
import torch
from torch import device
from .utils import get_parameter_device, get_parameter_dtype, save_state_dict_and_config, load_state_dict_from_path
from general_utils.os_utils import natural_sort

class BaseClassifier(torch.nn.Module):

    def __init__(self, config=None):
        super(BaseClassifier, self).__init__()
        self.config = config

    @classmethod
    def from_config(cls, classifier_cfg, margin_loss_fn, model_cfg, dataset_cfg, rank, world_size) -> "BaseClassifier":
        raise NotImplementedError('from_config must be implemented in subclass')

    def forward(self, local_embeddings, local_labels):
        raise NotImplementedError('from_config must be implemented in subclass')


    @property
    def device(self) -> device:
        return get_parameter_device(self)

    @property
    def dtype(self) -> torch.dtype:
        return get_parameter_dtype(self)

    def num_parameters(self, only_trainable: bool = False) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad or not only_trainable)

    def has_trainable_params(self):
        for param in self.parameters():
            if param.requires_grad:
                return True
        return False

    def save_pretrained(
        self,
        save_dir: Union[str, os.PathLike],
        name: str = 'model.pt',
        rank: int = 0,
    ):
        rank_added_name = os.path.splitext(name)[0] + f'_rank{rank}' + os.path.splitext(name)[1]
        save_path = os.path.join(save_dir, rank_added_name)
        save_state_dict_and_config(self.state_dict(), self.config, save_path)


    def load_state_dict_from_path(self, pretrained_model_path):

        save_dir = os.path.dirname(pretrained_model_path)
        save_name = os.path.basename(pretrained_model_path)
        rank_added_name = os.path.splitext(save_name)[0] + f'_rank{self.rank}' + os.path.splitext(save_name)[1]
        pretrained_model_path = os.path.join(save_dir, rank_added_name)

        all_partitions = [name for name in os.listdir(save_dir) if '_rank' in name and '.pt' in name]
        all_partitions = natural_sort(all_partitions)
        ckpt_worldsize = len(all_partitions)

        if self.world_size != ckpt_worldsize:
            # we need to redistribute the partialfc weights
            part_ckpts = [torch.load(os.path.join(save_dir, name), map_location='cpu') for name in all_partitions]
            total_ckpt_num_subjects = sum([ckpt['partial_fc.weight'].shape[0] for ckpt in part_ckpts])
            assert total_ckpt_num_subjects - self.partial_fc.num_classes < 10, \
                (f"total_ckpt_num_subjects: {total_ckpt_num_subjects}, "
                 f"self.partial_fc.num_classes: {self.partial_fc.num_classes}"
                 f"The number can be slightly different due to the last partition.")

            combined_weight = torch.cat([ckpt['partial_fc.weight'] for ckpt in part_ckpts], dim=0)
            state_dict = part_ckpts[0]

            class_start = self.partial_fc.class_start
            num_sample = self.partial_fc.num_local
            sub_center = combined_weight[class_start:class_start + num_sample, :]
            if sub_center.shape[0] != num_sample:
                # append zero
                extra_center = torch.zeros(num_sample - sub_center.shape[0], sub_center.shape[1],
                                           device=self.device, dtype=self.dtype)
                sub_center = torch.cat([sub_center, extra_center], dim=0)
            state_dict['partial_fc.weight'] = sub_center

        else:
            state_dict = load_state_dict_from_path(pretrained_model_path)

        result = self.load_state_dict(state_dict, strict=False)
        print(result)