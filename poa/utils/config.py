# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the Apache License, Version 2.0
# found in the LICENSE file in the root directory of this source tree.

import math
import logging
import os

from omegaconf import OmegaConf

import poa.distributed as distributed
from poa.logging import setup_logging
from poa.utils import utils
from poa.configs import poa_default_configs


logger = logging.getLogger("poa")


def apply_scaling_rules_to_cfg(cfg):  # to fix
    if cfg.optim.scaling_rule == "sqrt_wrt_1024":
        base_lr = cfg.optim.base_lr
        cfg.optim.lr = base_lr
        cfg.optim.lr *= math.sqrt(cfg.train.batch_size_per_gpu * distributed.get_global_size() / 1024.0)
        logger.info(f"sqrt scaling learning rate; base: {base_lr}, new: {cfg.optim.lr}")
    else:
        raise NotImplementedError
    return cfg


def write_config(cfg, output_dir, name="config.yaml"):
    logger.info(OmegaConf.to_yaml(cfg))
    saved_cfg_path = os.path.join(output_dir, name)
    with open(saved_cfg_path, "w") as f:
        OmegaConf.save(config=cfg, f=f)
    return saved_cfg_path


def get_cfg_from_args(args):
    args.output_dir = os.path.abspath(args.output_dir)
    args.opts += [f"train.output_dir={args.output_dir}"]
    if 'vit' in args.config_file:
        poa_default_config = poa_default_configs['vit']
    elif 'swin' in args.config_file:
        poa_default_config = poa_default_configs['swin']
    elif 'resnet' in args.config_file:
        poa_default_config = poa_default_configs['resnet']
    else:
        raise Exception(f'Unsupported config: {args.config_file}')
    default_cfg = OmegaConf.create(poa_default_config)
    cfg = OmegaConf.load(args.config_file)
    cfg = OmegaConf.merge(default_cfg, cfg, OmegaConf.from_cli(args.opts))
    return cfg


def default_setup(args):
    distributed.enable(overwrite=True)
    seed = getattr(args, "seed", 0)
    rank = distributed.get_global_rank()

    global logger
    setup_logging(output=args.output_dir, level=logging.INFO)
    logger = logging.getLogger("poa")

    utils.fix_random_seeds(seed + rank)
    logger.info("git:\n  {}\n".format(utils.get_sha()))
    logger.info("\n".join("%s: %s" % (k, str(v)) for k, v in sorted(dict(vars(args)).items())))


def setup(args):
    """
    Create configs and perform basic setups.
    """
    cfg = get_cfg_from_args(args)
    os.makedirs(args.output_dir, exist_ok=True)
    default_setup(args)
    apply_scaling_rules_to_cfg(cfg)
    write_config(cfg, args.output_dir)
    return cfg
