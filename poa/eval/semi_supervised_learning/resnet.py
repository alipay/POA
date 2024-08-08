# Copyright 2024 Ant Group.
import logging
import numpy as np
from typing import Sequence, Tuple, Union
from poa.layers import DynamicBottleneck
import torch
import torch.nn as nn
import torch.nn.functional as F
from poa.eval.semi_supervised_learning.utils_for_semi import trunc_normal_
logger = logging.getLogger("poa")


class DynamicResLayer(nn.Module):
    """ResLayer to build ResNet style backbone.
    """
    def __init__(self,
                 block,
                 min_num_blocks,
                 max_num_blocks,
                 in_channels,
                 out_channels,
                 mode,
                 layer_id,
                 num_groups=32,
                 stride=1,
                 avg_down=False,
        ):
        super(DynamicResLayer, self).__init__()
        self.block = block
        self.min_num_blocks = min_num_blocks
        self.max_num_blocks = max_num_blocks
        assert max_num_blocks >= self.min_num_blocks
        downsample = None
        if stride != 1 or in_channels != out_channels:
            downsample = []
            conv_stride = stride
            if avg_down and stride != 1:
                conv_stride = 1
                downsample.append(
                    nn.AvgPool2d(
                        kernel_size=stride,
                        stride=stride,
                        ceil_mode=True,
                        count_include_pad=False))
            downsample.extend([
                nn.Conv2d(
                    in_channels,
                    out_channels,
                    kernel_size=1,
                    stride=conv_stride,
                    bias=False),
                nn.GroupNorm(num_groups, out_channels)
            ])
            downsample = nn.Sequential(*downsample)
        self.layers = nn.ModuleList()
        self.layers.append(
            block(
                in_channels=in_channels,
                out_channels=out_channels,
                mode=mode,
                layer_id=layer_id,
                num_groups=num_groups,
                stride=stride,
                downsample=downsample,
            )
        )
        in_channels = out_channels
        for i in range(1, max_num_blocks):
            self.layers.append(
                block(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    mode=mode,
                    layer_id=layer_id,
                    num_groups=num_groups,
                    stride=1,
                )
            )
        self.widths = self.layers[0].get_widths()
        self.block_idxs = None

    def get_widths(self):
        return self.widths

    def get_block_idx(self, depth):
        block_idx = np.linspace(0, self.max_num_blocks - 1, depth).astype(np.int64)
        return block_idx.tolist()

    def get_index(self, depth, width):
        assert self.min_num_blocks<= depth <= self.max_num_blocks
        self.block_idxs = self.get_block_idx(depth)
        assert width in self.widths
        idx_out = None
        for i in self.block_idxs:
            idx_out = self.layers[i].get_index(width)
        # remove unused layers
        for i in range(len(self.layers)):
            if i not in self.block_idxs:
                self.layers[i] = nn.Identity()
        return idx_out

    def forward(self, x):
        for i in self.block_idxs:
            x = self.layers[i](x)
        return x


class DynamicResNet(nn.Module):
    """ResNet backbone.
    Please refer to the `paper <https://arxiv.org/abs/1512.03385>`__ for
    details.
    """

    def __init__(self,
                 img_size=224,
                 in_channels=3,
                 ibot_scale_factor=2,
                 mode='order',
                 net_select_scale=2,
                 min_depths=[3, 4, 6, 3],
                 max_depths=[3, 8, 36, 3],
                 stem_channels=64,
                 base_channels=64,
                 num_gn_groups=32,
                 strides=(1, 2, 2, 2),
                 avg_down=False,
                 zero_init_residual=True,
        ):
        super(DynamicResNet, self).__init__()
        assert mode in ['order', 'random']
        self.min_depths = min_depths
        self.max_depths = max_depths
        self.stem_channels = stem_channels
        self.base_channels = base_channels
        self.num_gn_groups = num_gn_groups
        self.num_stages = 4
        assert ibot_scale_factor in [1, 2, 4]
        self.ibot_scale_factor = ibot_scale_factor
        self.strides = strides
        self.avg_down = avg_down
        self.zero_init_residual = zero_init_residual
        self.block = DynamicBottleneck
        self.patch_size = 4
        self.num_patch = img_size // self.patch_size // 8 * ibot_scale_factor
        #self.mask_token = nn.Parameter(torch.zeros(1, stem_channels, 1, 1)) 
        self.net_select_scale = net_select_scale
        

        self.conv1 = nn.Conv2d(
            in_channels, stem_channels, kernel_size=7, stride=2, padding=3, bias=False
        )
        self.norm1 = nn.GroupNorm(num_gn_groups, stem_channels)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.res_layers = []
        _in_channels = stem_channels
        _out_channels = base_channels * 4
        self.min_mid_channels = []
        for i, num_blocks in enumerate(self.max_depths):
            stride = strides[i]
            res_layer = DynamicResLayer(
                block=self.block,
                min_num_blocks=self.min_depths[i],
                max_num_blocks=num_blocks,
                in_channels=_in_channels,
                out_channels=_out_channels,
                mode=mode,
                layer_id=i,
                num_groups=num_gn_groups,
                stride=stride,
                avg_down=self.avg_down,
            )
            _in_channels = _out_channels
            _out_channels *= 2
            layer_name = f'layer{i + 1}'
            self.add_module(layer_name, res_layer)
            self.res_layers.append(layer_name)
        self.max_embed_dim = _out_channels // 2
        self.create_dynamic_networks()
        self.depth_each_stage = None
        self.head = nn.Identity()
        self.block_idx = None
        self.feat_dim = 3584

    def create_dynamic_networks(self,):
        depths = [
            list(range(min_d, max_d + 1)) for min_d, max_d in zip(self.min_depths, self.max_depths)
        ]
        self.depths = [
            (d1, d2, d3, d4) for d1 in depths[0] for d2 in depths[1] for d3 in depths[2] for d4
            in depths[3]
        ]
        self.depths = sorted(self.depths, key=lambda x: sum(x))
        widths = [getattr(self, layer_name).get_widths() for layer_name in self.res_layers]
        assert len(depths) == len(widths)
        #self.widths = [
        #    (w1, w2, w3, w4) for w1 in widths[0] for w2 in widths[1] for w3 in widths[2] for w4
        #    in widths[3]
        #]
        self.widths = list(zip(*widths))
        d_select_scale = np.linspace(self.net_select_scale, 1, len(self.depths))
        #self.networks = [(w, d) for w in self.widths for d in self.depths]
        self.networks = []
        for w in self.widths:
            for d, ds in zip(self.depths, d_select_scale):  
                num = int(np.round(ds))  
                self.networks.extend([(w, d)] * num)   
        self.num_networks = len(self.networks)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def get_depths(self):
        return self.depth_each_stage

    def get_num_layers(self):
        return sum(self.depth_each_stage)

    def get_eval_arch(self,):
        return ['R50', 'R101', 'R152']

    def no_weight_decay(self):
        return {'conv1.weight', 'norm1.weight', 'norm1.bias'}

    def get_by_type(self, arch='R50'):
        width, depth = [64, 128, 256, 512], None
        if arch=='R50':
            depth = [3, 4, 6, 3]
        elif arch=='R101':
            depth = [3, 4, 23, 3]
        elif arch=='R152':
            depth = [3, 8, 36, 3]
        else:
            raise TypeError('Not supported arch {}'.format(arch))
        idx_out = None
        self.block_idx = []
        for i, layer_name in enumerate(self.res_layers):
            idx_out = getattr(self, layer_name).get_index(depth[i], width[i])
            if i in [1,2]: 
                self.block_idx.append(getattr(self, layer_name).get_block_idx(depth[i]))
        self.depth_each_stage = depth
        return idx_out

    def get_index(self, network_id=0):
        width, depth = self.networks[network_id]
        idx_out = None
        self.block_idx = []
        for i, layer_name in enumerate(self.res_layers):
            idx_out = getattr(self, layer_name).get_index(depth[i], width[i])
            if i in [1,2]: 
                self.block_idx.append(getattr(self, layer_name).get_block_idx(depth[i]))
        self.depth_each_stage = depth
        return idx_out

    def prepare_tokens_with_masks(self, x, masks=None):
        x = self.conv1(x)
        x = self.norm1(x)
        x = self.relu(x)
        x = self.maxpool(x) # shape: B, c, h, w
        if masks is not None:    # shape: B, hw
            # interploate masks
            B = masks.shape[0]
            masks = masks.reshape(B, 1, self.num_patch, self.num_patch).float()
            scale_factor = 8 // self.ibot_scale_factor
            masks = F.interpolate(masks, scale_factor=scale_factor, mode='nearest').bool()
            mask_token = self.mask_token
            # masks shape: B, 1, h, w mask_token shape: 1, c, 1, 1
            x = torch.where(masks, mask_token.to(x.dtype), x)
        return x

    def forward_features(self, x, masks=None):
        if isinstance(x, list):
            return self.forward_features_list(x, masks)
        x = self.prepare_tokens_with_masks(x, masks)
        for i, layer_name in enumerate(self.res_layers):
            res_layer = getattr(self, layer_name)
            x = res_layer(x)
        if self.ibot_scale_factor > 1:
            x_region = F.interpolate(x, scale_factor=self.ibot_scale_factor, mode='nearest')
            x_region = x_region.flatten(2).permute(0, 2, 1) 
        x = x.flatten(2).permute(0, 2, 1)
        x_mean_pooling = x.mean(1)
        return {
            "x_norm_clstoken": x_mean_pooling,
            "x_mean_pooling": x_mean_pooling,
            "x_norm_patchtokens": x_region,
            "x_prenorm": x,
            "masks": masks,
        }

    def forward_multistage_features(self, x):
        x = self.prepare_tokens_with_masks(x, None)
        x_mean_pooling = []
        for i, layer_name in enumerate(self.res_layers):
            res_layer = getattr(self, layer_name)
            x = res_layer(x)
            if i > 0:
                x_ = x.flatten(2).permute(0, 2, 1)
                x_mean_pooling.append(x_.mean(1))
        x_mean_pooling = torch.cat(x_mean_pooling, 1)
        return {
            "x_mean_pooling": x_mean_pooling,
        }

    def get_intermediate_layers(
        self,
        x: torch.Tensor,
        n: Union[int, Sequence] = 1,  # Layers or n last layers to take
        reshape: bool = False,
        return_class_token: bool = False,
        norm=True,
    ) -> Tuple[Union[torch.Tensor, Tuple[torch.Tensor]]]:
        x = self.conv1(x)
        x = self.norm1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        # If n is an int, take the n last blocks. If it's a list, take them
        outputs, total_block_len = [], self.num_stages
        blocks_to_take = range(total_block_len - n, total_block_len) if isinstance(n, int) else n
        for i, layer_name in enumerate(self.res_layers):
            res_layer = getattr(self, layer_name)
            x = res_layer(x)
            if i in blocks_to_take:
                if reshape:
                    outputs.append(x)
                else:
                    outputs.append(x.flatten(2).permute(0, 2, 1))
        assert len(outputs) == len(blocks_to_take), f"only {len(outputs)} / {len(blocks_to_take)} " \
                                                    f"blocks found"
        if reshape:
            class_tokens = [out.flatten(2).permute(0, 2, 1).mean(1) for out in outputs]
        else:
            class_tokens = [out.mean(1) for out in outputs]
        outputs = [out for out in outputs]
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
        return ret['x_mean_pooling']

def resnet_dynamic(
        min_depths=[3, 4, 6, 3],
        max_depths=[3, 8, 36, 3],
        ibot_scale_factor=2,
        stem_channels=64,
        base_channels=64,
        num_gn_groups=32,
        mode='order',
        strides=(1, 2, 2, 2)
):
    model = DynamicResNet(
        img_size=224,
        in_channels=3,
        ibot_scale_factor=ibot_scale_factor,
        min_depths=min_depths,
        max_depths=max_depths,
        stem_channels=stem_channels,
        base_channels=base_channels,
        num_gn_groups=num_gn_groups,
        mode=mode,
        strides=strides
    )
    return model


