# Copyright 2024 Ant Group.
import pathlib
from omegaconf import OmegaConf


def load_config(config_name: str):
    config_filename = config_name + ".yaml"
    return OmegaConf.load(pathlib.Path(__file__).parent.resolve() / config_filename)

poa_default_configs = {
    'vit': load_config("vit_ssl_default_config"), 
    'swin': load_config("swin_ssl_default_config"),
    'resnet': load_config("resnet_ssl_default_config"),
}

def load_and_merge_config(config_name: str):
    if 'vit' in config_name:
        poa_default_config = poa_default_configs['vit']
    elif 'swin' in config_name:
        poa_default_config = poa_default_configs['swin']
    elif 'resnet' in config_name:
        poa_default_config = poa_default_configs['resnet']
    else:
        raise Exception(f'Unsupported config: {config_name}')
    default_config = OmegaConf.create(poa_default_config)
    loaded_config = load_config(config_name)
    return OmegaConf.merge(default_config, loaded_config)
