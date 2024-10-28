from omegaconf import OmegaConf
import sys
import os
import importlib
def load_config(config_path):
    cfg = OmegaConf.load(config_path)
    return cfg

def import_from_config(root, cfg, module_names):
    modules = []
    task = cfg.trainers.task
    path = os.path.join(root, 'tasks', task)
    sys.path.insert(0, path)

    imported_functions = []
    for module_name in module_names:
        module = importlib.import_module(module_name.split('.')[0])
        function = getattr(module, module_name.split('.')[1])
        assert len(module_name.split('.')) == 2
        imported_functions.append(function)

    return imported_functions