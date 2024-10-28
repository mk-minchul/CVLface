from hydra import compose, initialize
import hydra
import omegaconf
from omegaconf import OmegaConf
import os
from time import gmtime, strftime
from dataclasses import dataclass


@dataclass
class Config:
    trainers: omegaconf.dictconfig.DictConfig
    optims: omegaconf.dictconfig.DictConfig
    models: omegaconf.dictconfig.DictConfig
    dataset: omegaconf.dictconfig.DictConfig
    data_augs: omegaconf.dictconfig.DictConfig
    losses: omegaconf.dictconfig.DictConfig
    classifiers: omegaconf.dictconfig.DictConfig
    aligners: omegaconf.dictconfig.DictConfig
    pipelines: omegaconf.dictconfig.DictConfig
    evaluations: omegaconf.dictconfig.DictConfig
    pefts: omegaconf.dictconfig.DictConfig


def init(root):
    initialize(config_path="./", job_name="configs")
    args_parser = hydra._internal.utils.get_args_parser()
    args = args_parser.parse_args()
    cfg = compose(config_name="base", overrides=args.overrides)


    # writing config's path
    OmegaConf.set_struct(cfg, False)
    base_cfg = {}
    for row in OmegaConf.load('base.yaml')['defaults']:
        for key, val in row.items():
            base_cfg[key] = val
    for key in Config.__dataclass_fields__.keys():
        has_override = [
            x.split('=')[1] if '.yaml' in x.split('=')[1] else x.split('=')[1] + '.yaml'
            for x in args.overrides if key + '=' in x
        ]
        if has_override:
            print('has_override', has_override[0])
            getattr(cfg, key).yaml_path = '/'+has_override[0]
        else:
            getattr(cfg, key).yaml_path = '/'+base_cfg[key]
    OmegaConf.set_struct(cfg, True)

    # make output_dir
    output_dir = prepare_output_dir(cfg, root)
    cfg.trainers.output_dir = output_dir
    os.makedirs(cfg.trainers.output_dir, exist_ok=True)

    return cfg


def parse_config_string(config_string):
    lst = config_string.split('.')
    assert len(lst) <= 2
    if len(lst) == 1:
        return 'configs/' + lst[0] + '.yaml'
    else:
        return lst[0] + '/configs/' + lst[1] + '.yaml'

def load_yaml(config_string, directory='models'):
    yaml_path = os.path.join(directory, parse_config_string(config_string))
    assert os.path.exists(yaml_path), yaml_path
    cfg = OmegaConf.load(yaml_path)
    cfg.yaml_path = yaml_path
    return cfg

def is_used_directory(directory):
    return os.path.isdir(directory) and os.path.exists(os.path.join(directory, 'train.py'))

def prepare_output_dir(cfg, root):
    # set working dir
    cur_time = strftime("%m-%d_0", gmtime())
    task = os.path.basename(os.path.dirname(__file__))
    cfg.trainers.task = task
    output_dir = os.path.join(root, 'research/recognition/experiments', cfg.trainers.task, cfg.trainers.prefix + "_" + cur_time)
    if is_used_directory(output_dir):
        while True:
            cur_exp_number = int(output_dir[-2:].replace('_', ""))
            output_dir = output_dir[:-2] + "_{}".format(cur_exp_number+1)
            # replace repeating _ with _
            output_dir = output_dir.replace('__', '_')
            if not is_used_directory(output_dir):
                break

    if cfg.pipelines.resume:
        print('resume ', cfg.pipelines.resume)
        assert os.path.isdir(cfg.pipelines.resume)
        if '/checkpoints/' in cfg.pipelines.resume:
            output_dir = cfg.pipelines.resume.split('/checkpoints/')[0]
        else:
            output_dir = cfg.pipelines.resume

    os.makedirs(output_dir, exist_ok=True)
    return output_dir
