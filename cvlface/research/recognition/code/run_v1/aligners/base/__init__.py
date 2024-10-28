import os
from typing import Union
import torch
from torch import device
from .utils import get_parameter_device, get_parameter_dtype, save_state_dict_and_config, load_state_dict_from_path

class BaseAligner(torch.nn.Module):

    def __init__(self, config=None):
        super().__init__()
        self.config = config

    @classmethod
    def from_config(cls, config) -> "BaseAligner":
        raise NotImplementedError('from_config must be implemented in subclass')

    def make_train_transform(self):
        raise NotImplementedError('from_config must be implemented in subclass')

    def make_test_transform(self):
        raise NotImplementedError('from_config must be implemented in subclass')

    def forward(self, x):
        raise NotImplementedError('from_config must be implemented in subclass')

    def save_pretrained(
        self,
        save_dir: Union[str, os.PathLike],
        name: str = 'model.pt',
        rank: int = 0,
    ):
        save_path = os.path.join(save_dir, name)
        if rank == 0:
            save_state_dict_and_config(self.state_dict(), self.config, save_path)

    def load_state_dict_from_path(self, pretrained_model_path):
        state_dict = load_state_dict_from_path(pretrained_model_path)
        result = self.load_state_dict(state_dict)
        print(f"Loaded pretrained aligner from {pretrained_model_path}")


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

    def has_params(self):
        return len(list(self.parameters())) > 0