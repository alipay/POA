# Copyright 2024 Ant Group.
import logging
from typing import Sequence, Tuple, Union
import torch.utils.checkpoint
from poa.layers import DynamicPatchEmbed, DynamicLayerNorm
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import ModuleList
from poa.layers import DynamicPatchMerging, DynamicSwinBlockSequence
logger = logging.getLogger("poa")


class DynamicSwinTransformer(nn.Module):
    """Swin Transformer backbone.

    This backbone is the implementation of `Swin Transformer:
    Hierarchical Vision Transformer using Shifted
    Windows <https://arxiv.org/abs/2103.14030>`_.
    Inspiration from https://github.com/microsoft/Swin-Transformer.

    Args:
        pretrain_img_size (int | tuple[int]): The size of input image when
            pretrain. Defaults: 224.
        in_channels (int): The num of input channels.
            Defaults: 3.
        embed_dims (int): The feature dimension. Default: 96.
        patch_size (int | tuple[int]): Patch size. Default: 4.
        window_size (int): Window size. Default: 7.
        mlp_ratio (int | float): Ratio of mlp hidden dim to embedding dim.
            Default: 4.
        depths (tuple[int]): Depths of each Swin Transformer stage.
            Default: (2, 2, 6, 2).
        num_heads (tuple[int]): Parallel attention heads of each Swin
            Transformer stage. Default: (3, 6, 12, 24).
        strides (tuple[int]): The patch merging or patch embedding stride of
            each Swin Transformer stage. (In swin, we set kernel size equal to
            stride.) Default: (4, 2, 2, 2).
        out_indices (tuple[int]): Output from which stages.
            Default: (0, 1, 2, 3).
        qkv_bias (bool, optional): If True, add a learnable bias to query, key,
            value. Default: True
        qk_scale (float | None, optional): Override default qk scale of
            head_dim ** -0.5 if set. Default: None.
        patch_norm (bool): If add a norm layer for patch embed and patch
            merging. Default: True.
        drop_rate (float): Dropout rate. Defaults: 0.
        attn_drop_rate (float): Attention dropout rate. Default: 0.
        drop_path_rate (float): Stochastic depth rate. Defaults: 0.1.
        use_abs_pos_embed (bool): If True, add absolute position embedding to
            the patch embedding. Defaults: False.
        act_cfg (dict): Config dict for activation layer.
            Default: dict(type='LN').
        norm_cfg (dict): Config dict for normalization layer at
            output of backone. Defaults: dict(type='LN').
        with_cp (bool, optional): Use checkpoint or not. Using checkpoint
            will save some memory while slowing down the training speed.
            Default: False.
        pretrained (str, optional): model pretrained path. Default: None.
        frozen_stages (int): Stages to be frozen (stop grad and set eval mode).
            -1 means not freezing any parameters.
        init_cfg (dict, optional): The Config for initialization.
            Defaults to None.
    """

    def __init__(self,
                 img_size=224,
                 in_channels=3,
                 min_embed_dim=96,
                 max_embed_dim=192,
                 ibot_scale_factor=2,
                 mode='order',
                 net_select_scale=2,
                 patch_size=4,
                 window_size=7,
                 mlp_ratio=4,
                 min_s3_depth=6,
                 max_depths=(2, 2, 18, 2),
                 width_interval=1,
                 depth_interval=1,
                 head_dim=16,
                 strides=(4, 2, 2, 2),
                 qkv_bias=True,
                 qk_scale=None,
                 drop_rate=0.,
                 attn_drop_rate=0.,
                 drop_path_rate=0.0,
                 with_cp=False,
                 ):
        super(DynamicSwinTransformer, self).__init__()
        num_stages = len(max_depths)
        self.num_stages = num_stages
        self.max_depths = max_depths
        self.width_interval=width_interval
        self.depth_interval=depth_interval
        assert ibot_scale_factor in [1, 2, 4]
        self.ibot_scale_factor = ibot_scale_factor
        self.depths = max_depths
        assert strides[0] == patch_size, 'Use non-overlapping patch embed.'
        self.patch_embed = DynamicPatchEmbed(
            img_size=img_size,
            patch_size=patch_size,
            in_chans=in_channels,
            max_embed_dim=max_embed_dim,
            mode=mode,
            norm_layer=DynamicLayerNorm,
        )
        self.num_patch = img_size // patch_size // 8 * ibot_scale_factor
        self.drop_after_pos = nn.Dropout(p=drop_rate)
        # set stochastic depth decay rule
        max_total_depth = sum(max_depths)
        dpr = [
            x.item() for x in torch.linspace(0, drop_path_rate, max_total_depth)
        ]
        self.stages = ModuleList()
        max_in_channels = max_embed_dim
        self.head_dim = head_dim
        self.min_s3_depth = min_s3_depth
        self.max_s3_depth = max_depths[2]
        self.min_embed_dim = min_embed_dim
        self.max_embed_dim = max_embed_dim
        self.mask_token = nn.Parameter(torch.zeros(1, max_embed_dim))
        self.head = nn.Identity()
        self.net_select_scale = net_select_scale
        print('WINDOW SIZE: {}'.format(window_size))
        for i in range(num_stages):
            if i < num_stages - 1:
                downsample = DynamicPatchMerging(
                    max_in_channels=max_in_channels,
                    max_out_channels=2 * max_in_channels,
                    mode=mode,
                    stride=strides[i + 1],
                )
            else:
                downsample = None

            stage = DynamicSwinBlockSequence(
                max_embed_dim=max_in_channels,
                head_dim=head_dim,
                mode=mode,
                max_feedforward_channels=int(mlp_ratio * max_in_channels),
                max_depth=max_depths[i],
                window_size=window_size,
                qkv_bias=qkv_bias,
                qk_scale=qk_scale,
                drop_rate=drop_rate,
                attn_drop_rate=attn_drop_rate,
                drop_path_rate=dpr[sum(max_depths[:i]):sum(max_depths[:i + 1])],
                downsample=downsample,
                with_cp=with_cp)
            self.stages.append(stage)
            if downsample:
                max_in_channels = downsample.max_out_channels

        self.max_feat_dim = max_in_channels
        self.max_num_features = [int(max_embed_dim * 2**i) for i in range(num_stages)]
        self.patch_idx = None
        self.block_idx = None
        self.s3_depth = None
        # Add a norm layer for each output
        for i in range(num_stages):
            layer = DynamicLayerNorm(self.max_num_features[i])
            layer_name = f'norm{i}'
            self.add_module(layer_name, layer)
        self.create_dynamic_networks()

    def get_eval_arch(self,):
        return ['tiny', 'small', 'base']

    def get_by_type(self, arch='small'):
        width, depth = None, None
        if arch=='tiny':
            width, depth = 96, 6
        elif arch=='small':
            width, depth = 96, 18
        elif arch=='base':
            width, depth = 128, 18
        elif arch=='large':
            width, depth = 192, 18
        else:
            width, depth = arch.split('_')
            width = int(width)
            depth = int(depth)
        self.s3_depth = depth
        idx_in = self.patch_embed.get_index(width)
        self.patch_idx = idx_in 
        for i in range(self.num_stages):
            depth_ = self.max_depths[i] if i != 2 else depth
            self.stages[i].set_depth(depth_)
            idx_in_down, idx_in = self.stages[i].get_index(idx_in, width)
            if i == 2:
                self.block_idx = self.stages[2].block_idx
            width *= 2
            getattr(self, f'norm{i}').set_index(idx_in)
            idx_in = idx_in_down
        self.feat_dim = idx_in 
        return idx_in

    def get_depths(self):
        return [2, 2, self.s3_depth, 2]

    def get_num_layers(self):
        return self.s3_depth + 6

    def no_weight_decay(self):
        return {'pos_embed'}

    def get_index(self, network_id=0):
        width, depth = self.networks[network_id]
        self.s3_depth = depth
        idx_in = self.patch_embed.get_index(width)
        self.patch_idx = idx_in 
        for i in range(self.num_stages):
            depth_ = self.max_depths[i] if i != 2 else depth
            self.stages[i].set_depth(depth_)
            idx_in_down, idx_in = self.stages[i].get_index(idx_in, width)
            if i == 2:
                self.block_idx = self.stages[2].block_idx
            width *= 2
            getattr(self, f'norm{i}').set_index(idx_in)
            idx_in = idx_in_down
        return idx_in

    def prepare_tokens_with_masks(self, x, masks=None):
        x, hw_shape = self.patch_embed(x, return_hw=True)  # shape: B, hw, nc
        if masks is not None:    # shape: B, hw
            # interploate masks
            B = masks.shape[0]
            masks = masks.reshape(B, 1, self.num_patch, self.num_patch).float()
            scale_factor = 8 // self.ibot_scale_factor
            masks = F.interpolate(
                masks, scale_factor=scale_factor, mode='nearest'
            ).reshape(B, -1, 1).bool()
            # masks shape: B, hw, 1    mask_token shape: 1, 1, nc
            mask_token = self.mask_token[..., self.patch_idx]
            x = torch.where(masks, mask_token.to(x.dtype).unsqueeze(0), x)
        # x shape: B, hw, nc
        return x, hw_shape

    def forward_features(self, x, masks=None):
        if isinstance(x, list):
            return self.forward_features_list(x, masks)

        x, hw_shape = self.prepare_tokens_with_masks(x, masks)
        x = self.drop_after_pos(x)

        out = None
        for i, stage in enumerate(self.stages):
            x, hw_shape, out, out_hw_shape = stage(x, hw_shape)

        x_norm = self.norm3(out)
        B, hw, C = x_norm.shape
        x_region = x_norm
        if self.ibot_scale_factor > 1: 
            x_region = x_region.reshape(B, hw_shape[0], hw_shape[1], C).permute(0, 3, 1, 2)
            x_region = F.interpolate(x_region, scale_factor=self.ibot_scale_factor, mode='nearest')
            x_region = x_region.permute(0, 2, 3, 1).flatten(1, 2)
        return {
            "x_norm_clstoken": x_norm.mean(1),
            "x_mean_pooling": x_norm.mean(1),
            "x_norm_patchtokens": x_region,
            "x_prenorm": out,
            "masks": masks,
        }

    def get_intermediate_layers(
        self,
        x: torch.Tensor,
        n: Union[int, Sequence] = 1,  # Layers or n last layers to take
        reshape: bool = False,
        return_class_token: bool = False,
        norm=True,
    ) -> Tuple[Union[torch.Tensor, Tuple[torch.Tensor]]]:
        x, hw_shape = self.prepare_tokens_with_masks(x)
        x = self.drop_after_pos(x)
        # If n is an int, take the n last blocks. If it's a list, take them
        outputs, total_block_len = [], len(self.stages)
        blocks_to_take = range(total_block_len - n, total_block_len) if isinstance(n, int) else n
        norm_idxs = []
        for i, stage in enumerate(self.stages):
            x, hw_shape, out, out_hw_shape = stage(x, hw_shape)
            if i in blocks_to_take:
                norm_idxs.append(i)
                outputs.append(out)
        assert len(outputs) == len(blocks_to_take), f"only {len(outputs)} / {len(blocks_to_take)} " \
                                                    f"blocks found"
        if norm:
            outputs = [getattr(self, f'norm{norm_idxs[i]}')(out) for i, out in enumerate(outputs)]
        class_tokens = [out.mean(1) for out in outputs]
        outputs = [out for out in outputs]
        if reshape:
            B, _, w, h = x.shape
            outputs = [
                out.reshape(B, w // self.patch_size, h // self.patch_size, -1).permute(0, 3, 1, 2).contiguous()
                for out in outputs
            ]
        if return_class_token:
            return tuple(zip(outputs, class_tokens))
        return tuple(outputs)

    def forward_features_list(self, x_list, masks_list):
        output = []
        for x, masks in zip(x_list, masks_list):
            output.append(self.forward_features(x, masks))
        return output

    def forward(self, *args, is_training=False, **kwargs):
        ret = self.forward_features(*args, **kwargs)
        if is_training:
            return ret
        else:
            return self.head(ret["x_norm_clstoken"])

    def create_dynamic_networks(self,):
        assert self.min_embed_dim % self.head_dim == 0
        assert self.max_embed_dim % self.head_dim == 0
        self.widths = [x for x in range(self.min_embed_dim, self.max_embed_dim + 1,
                                        self.head_dim * self.width_interval)]
        self.s3_depths = [x for x in range(self.min_s3_depth, self.max_s3_depth + 1, 
                                        self.depth_interval)]
        logger.info("Elastic Widths: {} -- {}".format(len(self.widths), str(self.widths)))
        logger.info("Elastic Depths: {} -- {}".format(len(self.s3_depths), str(self.s3_depths)))
        self.networks = []
        w_select_scale = np.linspace(self.net_select_scale, 1, len(self.widths))
        d_select_scale = np.linspace(self.net_select_scale, 1, len(self.s3_depths))
        for w, ws in zip(self.widths, w_select_scale):
            for d, ds in zip(self.s3_depths, d_select_scale):
                num = int(np.round(ws * ds))
                self.networks.extend([(w, d)] * num)
        self.num_networks = len(self.networks)


def swin_dynamic(
        patch_size=4,
        min_embed_dim=96,
        max_embed_dim=192,
        min_s3_depth=6,
        max_depths=(2, 2, 18, 2),
        head_dim=16,
        drop_path_rate=0.0,
        mode='order',
        window_size=7,
        ibot_scale_factor=2,
        **kwargs):

    model = DynamicSwinTransformer(
        patch_size=patch_size,
        min_embed_dim=min_embed_dim,
        max_embed_dim=max_embed_dim,
        min_s3_depth=min_s3_depth,
        max_depths=max_depths,
        head_dim=head_dim,
        mode=mode,
        window_size=window_size,
        ibot_scale_factor=ibot_scale_factor, 
        drop_path_rate=drop_path_rate,
        **kwargs,
    )
    return model


