# Copyright 2024 Ant Group.
from functools import partial
import math
import logging
import numpy as np
from typing import Sequence, Union, Callable

import torch
import torch.nn as nn
import torch.utils.checkpoint
from torch.nn.init import trunc_normal_

from poa.layers import DynamicMlp, DynamicPatchEmbed, DynamicSwiGLUFFNFused, \
    MemEffDynamicAttention, DynamicLayerNorm
from poa.layers import NestedTensorDynamicBlock as DynamicBlock


logger = logging.getLogger("poa")


def named_apply(fn: Callable, module: nn.Module, name="", depth_first=True, include_root=False) -> nn.Module:
    if not depth_first and include_root:
        fn(module=module, name=name)
    for child_name, child_module in module.named_children():
        child_name = ".".join((name, child_name)) if name else child_name
        named_apply(fn=fn, module=child_module, name=child_name, depth_first=depth_first, include_root=True)
    if depth_first and include_root:
        fn(module=module, name=name)
    return module


class DinoVisionTransformer(nn.Module):
    def __init__(
        self,
        img_size=224,
        patch_size=16,
        in_chans=3,
        max_embed_dim=1024,
        min_embed_dim=384,
        max_depth=24,
        width_interval=1,
        min_depth=12,
        depth_interval=1,
        head_dim=64,
        mode='order',
        net_select_scale=2,
        mlp_ratio=4.0,
        qkv_bias=True,
        ffn_bias=True,
        proj_bias=True,
        drop_path_rate=0.0,
        drop_path_uniform=False,
        init_values=None,  # for layerscale: None or 0 => no layerscale
        embed_layer=DynamicPatchEmbed,
        act_layer=nn.GELU,
        block_fn=DynamicBlock,
        ffn_layer="mlp",
    ):
        """
        Args:
            img_size (int, tuple): input image size
            patch_size (int, tuple): patch size
            in_chans (int): number of input channels
            embed_dim (int): embedding dimension
            depth (int): depth of transformer
            num_heads (int): number of attention heads
            mlp_ratio (int): ratio of mlp hidden dim to embedding dim
            qkv_bias (bool): enable bias for qkv if True
            proj_bias (bool): enable bias for proj in attn if True
            ffn_bias (bool): enable bias for ffn if True
            drop_path_rate (float): stochastic depth rate
            drop_path_uniform (bool): apply uniform drop rate across blocks
            weight_init (str): weight init scheme
            init_values (float): layer-scale init values
            embed_layer (nn.Module): patch embedding layer
            act_layer (nn.Module): MLP activation layer
            block_fn (nn.Module): transformer block class
            ffn_layer (str): "mlp", "swiglu", "swiglufused" or "identity"
        """
        super().__init__()
        norm_layer = partial(DynamicLayerNorm, eps=1e-6)

        self.max_num_features = self.max_embed_dim = max_embed_dim  # num_features for consistency
        self.min_embed_dim = min_embed_dim
        self.num_tokens = 1
        self.max_n_blocks = self.max_depth = max_depth
        self.min_depth = min_depth
        self.head_dim = head_dim
        self.patch_size = patch_size
        self.net_select_scale = net_select_scale
        self.width_interval=width_interval
        self.depth_interval=depth_interval

        self.patch_embed = embed_layer(
            img_size=img_size,
            patch_size=patch_size,
            in_chans=in_chans,
            max_embed_dim=max_embed_dim,
            mode=mode,
        )
        num_patches = self.patch_embed.num_patches

        self.cls_token = nn.Parameter(torch.zeros(1, 1, max_embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + self.num_tokens, max_embed_dim))

        if drop_path_uniform is True:
            dpr = [drop_path_rate] * max_depth
        else:
            dpr = [x.item() for x in torch.linspace(0, drop_path_rate, max_depth)]  # stochastic depth decay rule

        if ffn_layer == "mlp":
            logger.info("using MLP layer as FFN")
            ffn_layer = DynamicMlp
        elif ffn_layer == "swiglufused" or ffn_layer == "swiglu":
            logger.info("using SwiGLU layer as FFN")
            ffn_layer = DynamicSwiGLUFFNFused
        elif ffn_layer == "identity":
            logger.info("using Identity layer as FFN")

            def f(*args, **kwargs):
                return nn.Identity()

            ffn_layer = f
        else:
            raise NotImplementedError

        blocks_list = [
            block_fn(
                max_dim=max_embed_dim,
                head_dim=head_dim,
                mode=mode,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                proj_bias=proj_bias,
                ffn_bias=ffn_bias,
                drop_path=dpr[i],
                norm_layer=norm_layer,
                act_layer=act_layer,
                ffn_layer=ffn_layer,
                init_values=init_values,
            )
            for i in range(max_depth)
        ]
        self.blocks = nn.ModuleList(blocks_list)
        self.num_pos = num_patches + self.num_tokens
        self.in_chans = in_chans
        max_embed_dim * max_embed_dim + mlp_ratio * max_embed_dim * (max_embed_dim * 2 + 1)
        self.mlp_ratio = mlp_ratio
        self.norm = norm_layer(max_embed_dim)
        self.head = nn.Identity()
        self.mask_token = nn.Parameter(torch.zeros(1, max_embed_dim))
        self.create_dynamic_networks()
        self.patch_idx = None
        self.block_idx = None
        self.init_weights()
        self.use_mean_pooling = False

    def get_by_type(self, arch='small'):
        width, depth = None, None
        if arch=='small':
            width, depth = 384, 12
        elif arch=='base':
            width, depth = 768, 12
        elif arch=='large':
            width, depth = 1024, 24
        elif arch=='huge':
            width, depth = 1280, 32
        else:
            width, depth = arch.split('_')
            width = int(width)
            depth = int(depth)
        block_idx = self.get_block_idx(depth)
        self.block_idx = block_idx
        patch_idx = self.patch_embed.get_index(width)
        self.patch_idx = patch_idx
        idx_in = patch_idx
        for i in block_idx:
            idx_in, idx_out = self.blocks[i].get_index(idx_in, width)
            idx_in = idx_out[-1]
        self.norm.set_index(idx_in)
        self.feat_dim = width
        return idx_in

    def get_num_layers(self):
        return len(self.block_idx)

    def no_weight_decay(self):
        return {'pos_embed', 'cls_token'}

    def get_eval_arch(self):
        return ['small', 'base', 'large']

    def get_index(self, network_id=0):
        width, depth = self.networks[network_id]
        block_idx = self.get_block_idx(depth)
        self.block_idx = block_idx
        patch_idx = self.patch_embed.get_index(width)
        self.patch_idx = patch_idx
        idx_in = patch_idx
        for i in block_idx:
            idx_in, idx_out = self.blocks[i].get_index(idx_in, width)
            idx_in = idx_out[-1]
        self.norm.set_index(idx_in)
        return idx_in

    def create_dynamic_networks(self,):
        assert self.min_embed_dim % self.head_dim == 0
        assert self.max_embed_dim % self.head_dim == 0
        self.widths = [x for x in range(self.min_embed_dim, self.max_embed_dim + 1,
                                        self.head_dim * self.width_interval)]
        self.depths = [x for x in range(self.min_depth, self.max_depth + 1, self.depth_interval)]
        #self.networks = [(w, d) for w in self.widths for d in self.depths]
        logger.info("Elastic Widths: {} -- {}".format(len(self.widths), str(self.widths)))
        logger.info("Elastic Depths: {} -- {}".format(len(self.depths), str(self.depths)))
        self.networks = []
        w_select_scale = np.linspace(self.net_select_scale, 1, len(self.widths))
        d_select_scale = np.linspace(self.net_select_scale, 1, len(self.depths))
        for w, ws in zip(self.widths, w_select_scale):
            for d, ds in zip(self.depths, d_select_scale):
                num = int(np.round(ws * ds))
                self.networks.extend([(w, d)] * num)
        self.num_networks = len(self.networks)

    def get_block_idx(self, depth):
        block_idx = np.linspace(0, self.max_depth - 1, depth).astype(np.int64)
        return block_idx.tolist()

    def init_weights(self):
        trunc_normal_(self.pos_embed, std=0.02)
        nn.init.normal_(self.cls_token, std=1e-6)
        named_apply(init_weights_vit_timm, self)

    def interpolate_pos_encoding(self, x, w, h, idx):
        previous_dtype = x.dtype
        npatch = x.shape[1] - 1
        N = self.pos_embed.shape[1] - 1
        if npatch == N and w == h:
            return self.pos_embed[..., idx]
        pos_embed = self.pos_embed[..., idx].float()
        class_pos_embed = pos_embed[:, 0]
        patch_pos_embed = pos_embed[:, 1:]
        dim = x.shape[-1]
        w0 = w // self.patch_size
        h0 = h // self.patch_size
        # we add a small number to avoid floating point error in the interpolation
        # see discussion at https://github.com/facebookresearch/dino/issues/8
        w0, h0 = w0 + 0.1, h0 + 0.1

        sqrt_N = math.sqrt(N)
        sx, sy = float(w0) / sqrt_N, float(h0) / sqrt_N
        patch_pos_embed = nn.functional.interpolate(
            patch_pos_embed.reshape(1, int(sqrt_N), int(sqrt_N), dim).permute(0, 3, 1, 2),
            scale_factor=(sx, sy),
            mode="bicubic",
        )

        assert int(w0) == patch_pos_embed.shape[-2]
        assert int(h0) == patch_pos_embed.shape[-1]
        patch_pos_embed = patch_pos_embed.permute(0, 2, 3, 1).view(1, -1, dim)
        return torch.cat((class_pos_embed.unsqueeze(0), patch_pos_embed), dim=1).to(previous_dtype)

    def prepare_tokens_with_masks(self, x, masks=None):
        B, nc, w, h = x.shape
        x = self.patch_embed(x)
        if masks is not None:
            mask_token = self.mask_token[..., self.patch_idx]
            x = torch.where(masks.unsqueeze(-1), mask_token.to(x.dtype).unsqueeze(0), x)

        cls_token = self.cls_token[..., self.patch_idx]
        x = torch.cat((cls_token.expand(x.shape[0], -1, -1), x), dim=1)
        x = x + self.interpolate_pos_encoding(x, w, h, self.patch_idx)

        return x

    def forward_features_list(self, x_list, masks_list):
        x = [self.prepare_tokens_with_masks(x, masks) for x, masks in zip(x_list, masks_list)]
        for i in self.block_idx:
            x = self.blocks[i](x)

        all_x = x
        output = []
        for x, masks in zip(all_x, masks_list):
            x_norm = self.norm(x)
            output.append(
                {
                    "x_norm_clstoken": x_norm[:, 0],      # shape: B, C
                    "x_norm_patchtokens": x_norm[:, 1:],  # shape: B, hw, C
                    "x_prenorm": x,  # shape: B, 1+hw, nc
                    "masks": masks,  # shape: B, hw
                }
            )
        return output

    def forward_features(self, x, masks=None):
        if isinstance(x, list):
            return self.forward_features_list(x, masks)

        x = self.prepare_tokens_with_masks(x, masks)
        for i in self.block_idx:
            x = self.blocks[i](x)
        if self.use_mean_pooling:
            x_mean_pooling = self.norm(x[:, 1:].mean(1))
        else:
            x_mean_pooling = None
        x_norm = self.norm(x)
        return {
            "x_norm_clstoken": x_norm[:, 0],
            "x_mean_pooling": x_mean_pooling,
            "x_norm_patchtokens": x_norm[:, 1:],
            "x_prenorm": x,
            "masks": masks,
        }

    def _get_intermediate_layers(self, x, n=1):
        x = self.prepare_tokens_with_masks(x)
        # If n is an int, take the n last blocks. If it's a list, take them
        output, total_block_len = [], len(self.block_idx)
        blocks_to_take = [int(np.round(total_block_len / 4 * i)) for i in range(4 - n + 1, 4)]
        blocks_to_take.append(total_block_len - 1)
        #blocks_to_take = range(total_block_len - n, total_block_len) if isinstance(n, int) else n
        for i, blk in enumerate(self.block_idx):
            x = self.blocks[blk](x)
            if i in blocks_to_take:
                output.append(x)
        assert len(output) == len(blocks_to_take), f"only {len(output)} / {len(blocks_to_take)} blocks found"
        return output

    def get_intermediate_layers(
        self,
        x: torch.Tensor,
        n: Union[int, Sequence] = 1,  # Layers or n last layers to take
        reshape: bool = False,
        return_class_token: bool = False,
        norm=True,
    ):
        outputs = self._get_intermediate_layers(x, n)
        if norm:
            outputs = [self.norm(out) for out in outputs]
        class_tokens = [out[:, 0] for out in outputs]
        outputs = [out[:, 1:] for out in outputs]
        if reshape:
            B, _, w, h = x.shape
            outputs = [
                out.reshape(B, w // self.patch_size, h // self.patch_size, -1).permute(0, 3, 1, 2).contiguous()
                for out in outputs
            ]
        if return_class_token:
            return tuple(zip(outputs, class_tokens))
        return tuple(outputs)

    def get_intermediate_layers_for_mugs(self, x: torch.Tensor, n: Union[int, Sequence] = 1):
        x = self.prepare_tokens_with_masks(x)
        # If n is an int, take the n last blocks. If it's a list, take them
        outputs, total_block_len = [], len(self.block_idx)
        blocks_to_take = range(total_block_len - n, total_block_len) if isinstance(n, int) else n
        for i, blk in enumerate(self.block_idx):
            x = self.blocks[blk](x)
            if i in blocks_to_take:
                outputs.append(x)
        assert len(outputs) == len(blocks_to_take), f"only {len(output)} / {len(blocks_to_take)} blocks found"
        outputs = [self.norm(out) for out in outputs]
        return outputs

    def forward(self, *args, is_training=False, **kwargs):
        ret = self.forward_features(*args, **kwargs)
        if is_training:
            return ret
        else:
            return self.head(ret["x_norm_clstoken"])


def init_weights_vit_timm(module: nn.Module, name: str = ""):
    """ViT weight initialization, original timm impl (for reproducibility)"""
    if isinstance(module, nn.Linear):
        trunc_normal_(module.weight, std=0.02)
        if module.bias is not None:
            nn.init.zeros_(module.bias)


def vit_dynamic(
        patch_size=16,
        min_embed_dim=384,
        max_embed_dim=1024,
        min_depth=12,
        max_depth=24,
        head_dim=64,
        mode='order',
        **kwargs):
    model = DinoVisionTransformer(
        patch_size=patch_size,
        min_embed_dim=min_embed_dim,
        max_embed_dim=max_embed_dim,
        min_depth=min_depth,
        max_depth=max_depth,
        head_dim=head_dim,
        mode=mode,
        mlp_ratio=4,
        block_fn=partial(DynamicBlock, attn_class=MemEffDynamicAttention),
        **kwargs,
    )
    return model
