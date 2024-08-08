# Copyright 2024 Ant Group.
import logging

from . import vision_transformer as vit
from . import swin_transformer as swin
from . import resnet as resnet


logger = logging.getLogger("poa")


def build_model(args, only_teacher=False, img_size=224):
    args.arch = args.arch.removesuffix("_memeff")
    if "vit" in args.arch:
        vit_kwargs = dict(
            min_embed_dim=args.min_embed_dim,
            max_embed_dim=args.max_embed_dim,
            min_depth=args.min_depth,
            max_depth=args.max_depth,
            head_dim=args.head_dim,
            mode=args.mode,
            img_size=img_size,
            patch_size=args.patch_size,
            init_values=args.layerscale,
            ffn_layer=args.ffn_layer,
            qkv_bias=args.qkv_bias,
            proj_bias=args.proj_bias,
            ffn_bias=args.ffn_bias,
            width_interval=args.width_interval,
            depth_interval=args.depth_interval,
            net_select_scale=args.net_select_scale,
        )
        teacher = vit.__dict__[args.arch](**vit_kwargs)
        if only_teacher:
            return teacher, teacher.max_embed_dim
        student = vit.__dict__[args.arch](
            **vit_kwargs,
            drop_path_rate=args.drop_path_rate,
            drop_path_uniform=args.drop_path_uniform,
        )
        max_embed_dim = student.max_embed_dim
    elif "swin" in args.arch:
        swin_kwargs = dict(
            min_embed_dim=args.min_embed_dim,
            max_embed_dim=args.max_embed_dim,
            min_s3_depth=args.min_s3_depth,
            max_depths=list(args.max_depths),
            head_dim=args.head_dim,
            mode=args.mode,
            window_size=args.window_size,
            img_size=img_size,
            patch_size=args.patch_size,
            ibot_scale_factor=args.ibot_scale_factor, 
            qkv_bias=args.qkv_bias,
        )
        teacher = swin.__dict__[args.arch](**swin_kwargs)
        if only_teacher:
            return teacher, teacher.max_feat_dim
        student = swin.__dict__[args.arch](
            **swin_kwargs,
            drop_path_rate=args.drop_path_rate,
        )
        max_embed_dim = student.max_feat_dim
    elif "resnet" in args.arch:
        resnet_kwargs = dict(
            min_depths=list(args.min_depths),
            max_depths=list(args.max_depths),
            stem_channels=args.stem_channels,
            base_channels=args.base_channels,
            num_gn_groups=args.num_gn_groups,
            mode=args.mode,
            ibot_scale_factor=args.ibot_scale_factor, 
            strides=list(args.strides),
        )
        teacher = resnet.__dict__[args.arch](**resnet_kwargs)
        if only_teacher:
            return teacher, teacher.max_embed_dim
        student = resnet.__dict__[args.arch](**resnet_kwargs)
        max_embed_dim = student.max_embed_dim
    else:
        raise TypeError(f'Unsupported arch: {args.arch}')
    num_networks = student.num_networks
    return student, teacher, max_embed_dim, num_networks


def build_model_from_cfg(cfg, only_teacher=False):
    return build_model(cfg.student, only_teacher=only_teacher, img_size=cfg.crops.global_crops_size)

