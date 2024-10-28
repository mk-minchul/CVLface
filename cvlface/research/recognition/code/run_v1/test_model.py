import pyrootutils
root = pyrootutils.setup_root(
    search_from=__file__,
    indicator=["__root__.txt"],
    pythonpath=True,
    dotenv=True,
)
import os, sys
sys.path.append(os.path.join(root))
import numpy as np

np.bool = np.bool_  # fix bug for mxnet 1.9.1
np.object = np.object_
np.float = np.float_

import torch
from general_utils.config_utils import load_config
from models import get_model

if __name__ == '__main__':

    inputs_shape = (2, 3, 112, 112)
    inputs = torch.randn(inputs_shape)

    # setting 1: input is image
    for config_name in [
        'models/iresnet/configs/v1_ir50.yaml',
        'models/vit/configs/v1_base.yaml',
        'models/swin/configs/v1_base.yaml',
        'models/vit_irpe/configs/v1_base_irpe.yaml',
        'models/part_fvit/configs/v1_base.yaml',
    ]:
        config = load_config(config_name)
        config.yaml_path = config_name
        model = get_model(config, task='run_v1')
        out = model(inputs)
        print(f'{config_name} has input shape {inputs_shape} and output shape {out.shape}')

    # setting 2: input is image + keypoints
    for config_name in [
        'models/vit_kprpe/configs/v1_base_kprpe_splithead_unshared.yaml',
        'models/swin_kprpe/configs/v1_base_kprpe_splithead_unshared.yaml',
    ]:
        config = load_config(config_name)
        config.yaml_path = config_name
        keypoints = torch.randn(2, 49, 2)
        model = get_model(config, task='run_v1')
        out = model(inputs, keypoints)
        print(f'{config_name} has input shape {inputs_shape} and output shape {out.shape}')