import itertools
from typing import List, Optional, Tuple, Union
import safetensors
import torch
from torch import Tensor
import os
from pathlib import Path
from omegaconf import DictConfig, OmegaConf


def get_parameter_device(parameter: torch.nn.Module):
    try:
        parameters_and_buffers = itertools.chain(parameter.parameters(), parameter.buffers())
        return next(parameters_and_buffers).device
    except StopIteration:
        # For torch.nn.DataParallel compatibility in PyTorch 1.5
        def find_tensor_attributes(module: torch.nn.Module) -> List[Tuple[str, Tensor]]:
            tuples = [(k, v) for k, v in module.__dict__.items() if torch.is_tensor(v)]
            return tuples
        gen = parameter._named_members(get_members_fn=find_tensor_attributes)
        first_tuple = next(gen)
        return first_tuple[1].device


def get_parameter_dtype(parameter: torch.nn.Module):
    try:
        params = tuple(parameter.parameters())
        if len(params) > 0:
            return params[0].dtype

        buffers = tuple(parameter.buffers())
        if len(buffers) > 0:
            return buffers[0].dtype

    except StopIteration:
        # For torch.nn.DataParallel compatibility in PyTorch 1.5

        def find_tensor_attributes(module: torch.nn.Module) -> List[Tuple[str, Tensor]]:
            tuples = [(k, v) for k, v in module.__dict__.items() if torch.is_tensor(v)]
            return tuples

        gen = parameter._named_members(get_members_fn=find_tensor_attributes)
        first_tuple = next(gen)
        return first_tuple[1].dtype


def get_parent_directory(save_path: Union[str, os.PathLike]) -> Path:
    path_obj = Path(save_path)
    return path_obj.parent

def get_base_name(save_path: Union[str, os.PathLike]) -> str:
    path_obj = Path(save_path)
    return path_obj.name

def load_state_dict_from_path(path: Union[str, os.PathLike]):
    # Load a state dict from a path.
    if 'safetensors' in path:
        state_dict = safetensors.torch.load_file(path)
    else:
        state_dict = torch.load(path, map_location="cpu")
    return state_dict

def replace_extension(path, new_extension):
    if not new_extension.startswith('.'):
        new_extension = '.' + new_extension
    return os.path.splitext(path)[0] + new_extension

def make_config_path(save_path):
    config_path = replace_extension(save_path, '.yaml')
    return config_path

def save_config(config, config_path):
    assert isinstance(config, dict) or isinstance(config, DictConfig)
    os.makedirs(get_parent_directory(config_path), exist_ok=True)
    if isinstance(config, dict):
        config = OmegaConf.create(config)
    OmegaConf.save(config, config_path)


def save_state_dict_and_config(state_dict, config, save_path):
    os.makedirs(get_parent_directory(save_path), exist_ok=True)

    # save config dict
    config_path = make_config_path(save_path)
    save_config(config, config_path)

    # Save the model
    if 'safetensors' in save_path:
        safetensors.torch.save_file(state_dict, save_path, metadata={"format": "pt"})
    else:
        torch.save(state_dict, save_path)
