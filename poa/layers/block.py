# Copyright 2024 Ant Group.
import logging
import os
from typing import Callable, List, Any, Tuple, Dict
from copy import deepcopy
import numpy as np
import warnings

import torch
from torch import nn, Tensor

from .attention import DynamicAttention, MemEffDynamicAttention
from .attention import DynamicShiftWindowMSA
from .drop_path import DropPath
from .dynamic_ops import DynamicLayerScale, DynamicLayerNorm
from .dynamic_ops import DynamicConv2d, DynamicGroupNorm
from .mlp import DynamicMlp, DynamicFFN
import torch.utils.checkpoint as cp


logger = logging.getLogger("poa")


XFORMERS_ENABLED = os.environ.get("XFORMERS_DISABLED") is None
try:
    if XFORMERS_ENABLED:
        from xformers.ops import fmha, scaled_index_add, index_select_cat

        XFORMERS_AVAILABLE = True
        warnings.warn("xFormers is available (Block)")
    else:
        warnings.warn("xFormers is disabled (Block)")
        raise ImportError
except ImportError:
    XFORMERS_AVAILABLE = False

    warnings.warn("xFormers is not available (Block)")


class DynamicBlock(nn.Module):

    def __init__(
        self,
        max_dim: int,
        head_dim: int = 64,
        mode: str = 'order',
        mlp_ratio: float = 4.0,
        qkv_bias: bool = False,
        proj_bias: bool = True,
        ffn_bias: bool = True,
        drop: float = 0.0,
        attn_drop: float = 0.0,
        init_values=None,
        drop_path: float = 0.0,
        act_layer: Callable[..., nn.Module] = nn.GELU,
        norm_layer: Callable[..., nn.Module] = DynamicLayerNorm,
        attn_class: Callable[..., nn.Module] = DynamicAttention,
        ffn_layer: Callable[..., nn.Module] = DynamicMlp,
    ) -> None:
        super().__init__()
        # print(f"biases: qkv: {qkv_bias}, proj: {proj_bias}, ffn: {ffn_bias}")
        self.norm1 = norm_layer(max_dim)
        self.attn = attn_class(
            max_dim=max_dim,
            head_dim=head_dim,
            mode=mode,
            qkv_bias=qkv_bias,
            proj_bias=proj_bias,
            attn_drop=attn_drop,
            proj_drop=drop,
        )
        self.ls1 = DynamicLayerScale(max_dim, init_values=init_values) if init_values else \
            nn.Identity()
        self.drop_path1 = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

        self.norm2 = norm_layer(max_dim)
        max_mlp_hidden_dim = int(max_dim * mlp_ratio)
        self.mlp = ffn_layer(
            max_in_features=max_dim,
            max_hidden_features=max_mlp_hidden_dim,
            mode=mode,
            act_layer=act_layer,
            drop=drop,
            bias=ffn_bias,
        )
        self.ls2 = DynamicLayerScale(max_dim, init_values=init_values) if init_values else \
            nn.Identity()
        self.drop_path2 = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

        self.sample_drop_ratio = drop_path

    def get_index(self, idx_in, out_features):
        idx_ins, idx_outs = [], []
        self.norm1.set_index(idx_in)
        idx_in, idx_out = self.attn.get_index(idx_in, out_features)
        idx_ins.extend(idx_in)
        idx_outs.extend(idx_out)
        idx_in = idx_out[-1]
        self.ls1.set_index(idx_in)

        self.norm2.set_index(idx_in)
        idx_in, idx_out = self.mlp.get_index(idx_in, out_features)
        idx_ins.extend(idx_in)
        idx_outs.extend(idx_out)
        idx_in = idx_out[-1]
        self.ls2.set_index(idx_in)
        return idx_ins, idx_outs

    def set_index(self, idx_ins, idx_outs):
        assert len(idx_ins) == len(idx_outs)
        assert len(idx_ins) == 4
        self.norm1.set_index(idx_ins[0])
        self.attn.set_index(idx_ins[:2], idx_outs[:2])
        self.ls1.set_index(idx_outs[1])
        self.norm2.set_index(idx_outs[1])
        self.mlp.set_index(idx_ins[2:], idx_outs[2:])
        self.ls2.set_index(idx_outs[-1])

    def forward(self, x: Tensor) -> Tensor:
        def attn_residual_func(x: Tensor) -> Tensor:
            return self.ls1(self.attn(self.norm1(x)))

        def ffn_residual_func(x: Tensor) -> Tensor:
            return self.ls2(self.mlp(self.norm2(x)))

        if self.training and self.sample_drop_ratio > 0.1:
            # the overhead is compensated only for a drop path rate larger than 0.1
            x = drop_add_residual_stochastic_depth(
                x,
                residual_func=attn_residual_func,
                sample_drop_ratio=self.sample_drop_ratio,
            )
            x = drop_add_residual_stochastic_depth(
                x,
                residual_func=ffn_residual_func,
                sample_drop_ratio=self.sample_drop_ratio,
            )
        elif self.training and self.sample_drop_ratio > 0.0:
            x = x + self.drop_path1(attn_residual_func(x))
            x = x + self.drop_path1(ffn_residual_func(x))  # FIXME: drop_path2
        else:
            x = x + attn_residual_func(x)
            x = x + ffn_residual_func(x)
        return x


def drop_add_residual_stochastic_depth(
    x: Tensor,
    residual_func: Callable[[Tensor], Tensor],
    sample_drop_ratio: float = 0.0,
) -> Tensor:
    # 1) extract subset using permutation
    b, n, d = x.shape
    sample_subset_size = max(int(b * (1 - sample_drop_ratio)), 1)
    brange = (torch.randperm(b, device=x.device))[:sample_subset_size]
    x_subset = x[brange]

    # 2) apply residual_func to get residual
    residual = residual_func(x_subset)

    x_flat = x.flatten(1)
    residual = residual.flatten(1)

    residual_scale_factor = b / sample_subset_size

    # 3) add the residual
    x_plus_residual = torch.index_add(x_flat, 0, brange, residual.to(dtype=x.dtype), alpha=residual_scale_factor)
    return x_plus_residual.view_as(x)


def get_branges_scales(x, sample_drop_ratio=0.0):
    b, n, d = x.shape
    sample_subset_size = max(int(b * (1 - sample_drop_ratio)), 1)
    brange = (torch.randperm(b, device=x.device))[:sample_subset_size]
    residual_scale_factor = b / sample_subset_size
    return brange, residual_scale_factor


def add_residual(x, brange, residual, residual_scale_factor, scaling_vector=None):
    if scaling_vector is None:
        x_flat = x.flatten(1)
        residual = residual.flatten(1)
        x_plus_residual = torch.index_add(x_flat, 0, brange, residual.to(dtype=x.dtype), alpha=residual_scale_factor)
    else:
        x_plus_residual = scaled_index_add(
            x, brange, residual.to(dtype=x.dtype), scaling=scaling_vector, alpha=residual_scale_factor
        )
    return x_plus_residual


attn_bias_cache: Dict[Tuple, Any] = {}


def get_attn_bias_and_cat(x_list, branges=None):
    """
    this will perform the index select, cat the tensors, and provide the attn_bias from cache
    """
    batch_sizes = [b.shape[0] for b in branges] if branges is not None else [x.shape[0] for x in x_list]
    all_shapes = tuple((b, x.shape[1]) for b, x in zip(batch_sizes, x_list))
    if all_shapes not in attn_bias_cache.keys():
        seqlens = []
        for b, x in zip(batch_sizes, x_list):
            for _ in range(b):
                seqlens.append(x.shape[1])
        attn_bias = fmha.BlockDiagonalMask.from_seqlens(seqlens)
        attn_bias._batch_sizes = batch_sizes
        attn_bias_cache[all_shapes] = attn_bias

    if branges is not None:
        cat_tensors = index_select_cat([x.flatten(1) for x in x_list], branges).view(1, -1, x_list[0].shape[-1])
    else:
        tensors_bs1 = tuple(x.reshape([1, -1, *x.shape[2:]]) for x in x_list)
        cat_tensors = torch.cat(tensors_bs1, dim=1)

    return attn_bias_cache[all_shapes], cat_tensors


def drop_add_residual_stochastic_depth_list(
    x_list: List[Tensor],
    residual_func: Callable[[Tensor, Any], Tensor],
    sample_drop_ratio: float = 0.0,
    scaling_vector=None,
) -> List[Tensor]:
    # 1) generate random set of indices for dropping samples in the batch
    branges_scales = [get_branges_scales(x, sample_drop_ratio=sample_drop_ratio) for x in x_list]
    branges = [s[0] for s in branges_scales]
    residual_scale_factors = [s[1] for s in branges_scales]

    # 2) get attention bias and index+concat the tensors
    attn_bias, x_cat = get_attn_bias_and_cat(x_list, branges)

    # 3) apply residual_func to get residual, and split the result
    residual_list = attn_bias.split(residual_func(x_cat, attn_bias=attn_bias))  # type: ignore

    outputs = []
    for x, brange, residual, residual_scale_factor in zip(x_list, branges, residual_list, residual_scale_factors):
        outputs.append(add_residual(x, brange, residual, residual_scale_factor, scaling_vector).view_as(x))
    return outputs


class NestedTensorDynamicBlock(DynamicBlock):

    def forward_nested(self, x_list: List[Tensor]) -> List[Tensor]:
        """
        x_list contains a list of tensors to nest together and run
        """
        assert isinstance(self.attn, MemEffDynamicAttention)

        if self.training and self.sample_drop_ratio > 0.0:

            def attn_residual_func(x: Tensor, attn_bias=None) -> Tensor:
                return self.attn(self.norm1(x), attn_bias=attn_bias)

            def ffn_residual_func(x: Tensor, attn_bias=None) -> Tensor:
                return self.mlp(self.norm2(x))

            x_list = drop_add_residual_stochastic_depth_list(
                x_list,
                residual_func=attn_residual_func,
                sample_drop_ratio=self.sample_drop_ratio,
                scaling_vector=self.ls1.get_gamma() if isinstance(self.ls1, DynamicLayerScale) else
                None,
            )
            x_list = drop_add_residual_stochastic_depth_list(
                x_list,
                residual_func=ffn_residual_func,
                sample_drop_ratio=self.sample_drop_ratio,
                scaling_vector=self.ls2.get_gamma() if isinstance(self.ls2, DynamicLayerScale) else
                None,
            )
            return x_list
        else:

            def attn_residual_func(x: Tensor, attn_bias=None) -> Tensor:
                return self.ls1(self.attn(self.norm1(x), attn_bias=attn_bias))

            def ffn_residual_func(x: Tensor, attn_bias=None) -> Tensor:
                return self.ls2(self.mlp(self.norm2(x)))

            attn_bias, x = get_attn_bias_and_cat(x_list)
            x = x + attn_residual_func(x, attn_bias=attn_bias)
            x = x + ffn_residual_func(x)
            return attn_bias.split(x)

    def forward(self, x_or_x_list):
        if isinstance(x_or_x_list, Tensor):
            return super().forward(x_or_x_list)
        elif isinstance(x_or_x_list, list):
            assert XFORMERS_AVAILABLE, "Please install xFormers for nested tensors usage"
            return self.forward_nested(x_or_x_list)
        else:
            raise AssertionError


class DynamicSwinBlock(nn.Module):
    """"
    Args:
        embed_dims (int): The feature dimension.
        num_heads (int): Parallel attention heads.
        feedforward_channels (int): The hidden dimension for FFNs.
        window_size (int, optional): The local window scale. Default: 7.
        shift (bool, optional): whether to shift window or not. Default False.
        qkv_bias (bool, optional): enable bias for qkv if True. Default: True.
        qk_scale (float | None, optional): Override default qk scale of
            head_dim ** -0.5 if set. Default: None.
        drop_rate (float, optional): Dropout rate. Default: 0.
        attn_drop_rate (float, optional): Attention dropout rate. Default: 0.
        drop_path_rate (float, optional): Stochastic depth rate. Default: 0.
        act_cfg (dict, optional): The config dict of activation function.
            Default: dict(type='GELU').
        norm_cfg (dict, optional): The config dict of normalization.
            Default: dict(type='LN').
        with_cp (bool, optional): Use checkpoint or not. Using checkpoint
            will save some memory while slowing down the training speed.
            Default: False.
        init_cfg (dict | list | None, optional): The init config.
            Default: None.
    """

    def __init__(self,
                 max_embed_dim,
                 head_dim,
                 mode,
                 max_feedforward_channels,
                 window_size=7,
                 shift=False,
                 qkv_bias=True,
                 qk_scale=None,
                 drop_rate=0.,
                 attn_drop_rate=0.,
                 drop_path_rate=0.,
                 act_layer=nn.GELU,
                 norm_layer=DynamicLayerNorm,
                 with_cp=False):

        super(DynamicSwinBlock, self).__init__()
        self.with_cp = with_cp
        self.norm1 = norm_layer(max_embed_dim)
        self.attn = DynamicShiftWindowMSA(
            max_embed_dim=max_embed_dim,
            head_dim=head_dim,
            mode=mode,
            window_size=window_size,
            shift_size=window_size // 2 if shift else 0,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            attn_drop_rate=attn_drop_rate,
            proj_drop_rate=drop_rate,
            dropout_layer=DropPath(drop_prob=drop_path_rate)
        )
        self.norm2 = norm_layer(max_embed_dim)
        self.ffn = DynamicFFN(
            max_embed_dim=max_embed_dim,
            max_feedforward_channels=max_feedforward_channels,
            mode=mode,
            ffn_drop=drop_rate,
            dropout_layer=DropPath(drop_prob=drop_path_rate),
            act_func=act_layer,
            add_identity=True,
        )

    def get_index(self, idx_in, out_features):
        self.norm1.set_index(idx_in)
        idx_in = self.attn.get_index(idx_in, out_features)
        self.norm2.set_index(idx_in)
        idx_out = self.ffn.get_index(idx_in, out_features)
        return idx_out

    def forward(self, x, hw_shape):
        def _inner_forward(x):
            identity = x
            x = self.norm1(x)
            x = self.attn(x, hw_shape)
            x = x + identity
            identity = x
            x = self.norm2(x)
            x = self.ffn(x, identity=identity)
            return x

        if self.with_cp and x.requires_grad:
            x = cp.checkpoint(_inner_forward, x)
        else:
            x = _inner_forward(x)

        return x


class DynamicSwinBlockSequence(nn.Module):
    """Implements one stage in Swin Transformer.

    Args:
        embed_dims (int): The feature dimension.
        num_heads (int): Parallel attention heads.
        feedforward_channels (int): The hidden dimension for FFNs.
        depth (int): The number of blocks in this stage.
        window_size (int, optional): The local window scale. Default: 7.
        qkv_bias (bool, optional): enable bias for qkv if True. Default: True.
        qk_scale (float | None, optional): Override default qk scale of
            head_dim ** -0.5 if set. Default: None.
        drop_rate (float, optional): Dropout rate. Default: 0.
        attn_drop_rate (float, optional): Attention dropout rate. Default: 0.
        drop_path_rate (float | list[float], optional): Stochastic depth
            rate. Default: 0.
        downsample (BaseModule | None, optional): The downsample operation
            module. Default: None.
        act_cfg (dict, optional): The config dict of activation function.
            Default: dict(type='GELU').
        norm_cfg (dict, optional): The config dict of normalization.
            Default: dict(type='LN').
        with_cp (bool, optional): Use checkpoint or not. Using checkpoint
            will save some memory while slowing down the training speed.
            Default: False.
        init_cfg (dict | list | None, optional): The init config.
            Default: None.
    """

    def __init__(self,
                 max_embed_dim,
                 head_dim,
                 mode,
                 max_feedforward_channels,
                 max_depth,
                 window_size=7,
                 qkv_bias=True,
                 qk_scale=None,
                 drop_rate=0.,
                 attn_drop_rate=0.,
                 drop_path_rate=0.,
                 downsample=None,
                 with_cp=False):
        super().__init__()

        if isinstance(drop_path_rate, list):
            drop_path_rates = drop_path_rate
            assert len(drop_path_rates) == max_depth
        else:
            drop_path_rates = [deepcopy(drop_path_rate) for _ in range(max_depth)]

        self.blocks = nn.ModuleList()
        self.max_depth = max_depth
        for i in range(max_depth):
            block = DynamicSwinBlock(
                max_embed_dim=max_embed_dim,
                head_dim=head_dim,
                mode=mode,
                max_feedforward_channels=max_feedforward_channels,
                window_size=window_size,
                shift=False if i % 2 == 0 else True,
                qkv_bias=qkv_bias,
                qk_scale=qk_scale,
                drop_rate=drop_rate,
                attn_drop_rate=attn_drop_rate,
                drop_path_rate=drop_path_rates[i],
                with_cp=with_cp,
            )
            self.blocks.append(block)

        self.downsample = downsample

    def get_index(self, idx_in, out_features):
        for i in range(self.max_depth):
            if i in self.block_idx:
                idx_in = self.blocks[i].get_index(idx_in, out_features)
        if self.downsample is not None:
            idx_in_down = self.downsample.get_index(idx_in, out_features * 2)
        else:
            idx_in_down = idx_in
        return idx_in_down, idx_in

    def set_depth(self, depth):
        self.block_idx = np.linspace(0, self.max_depth - 1, depth).astype(np.int64)

    def forward(self, x, hw_shape):
        for i, block in enumerate(self.blocks):
            if i in self.block_idx:
                x = block(x, hw_shape)
        if self.downsample is not None:
            x_down, down_hw_shape = self.downsample(x, hw_shape)
            return x_down, down_hw_shape, x, hw_shape
        else:
            return x, hw_shape, x, hw_shape


class DynamicBottleneck(nn.Module):
    """Bottleneck block for ResNet."""

    def __init__(self,
                 in_channels,
                 out_channels,
                 mode,
                 layer_id,
                 num_groups=32,
                 stride=1,
                 dilation=1,
                 downsample=None,
                 drop_path_rate=0.0,
                 eps=1.0e-5):
        super(DynamicBottleneck, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.expansion = 4
        self.min_expansion = 2
        assert out_channels % self.expansion == 0
        assert out_channels % self.min_expansion == 0
        self.min_mid_channels = out_channels // self.expansion
        self.max_mid_channels = out_channels // self.min_expansion
        assert self.min_mid_channels % num_groups == 0
        assert self.max_mid_channels % num_groups == 0
        self.stride = stride
        self.dilation = dilation
        self.conv1_stride = 1
        self.conv2_stride = stride
        self.widths = list(range(self.min_mid_channels, self.max_mid_channels + 1, num_groups * (2 ** layer_id)))

        self.conv1 = DynamicConv2d(
            in_channels, self.max_mid_channels, mode, 1, stride=self.conv1_stride, bias=False
        )
        self.norm1 = DynamicGroupNorm(num_groups, self.max_mid_channels, mode)
        self.conv2 = DynamicConv2d(
            self.max_mid_channels, self.max_mid_channels, mode, 3, stride=self.conv2_stride,
            padding=dilation, dilation=dilation, bias=False
        )
        self.norm2 = DynamicGroupNorm(num_groups, self.max_mid_channels, mode)
        self.conv3 = DynamicConv2d(
            self.max_mid_channels, out_channels, mode, 1, bias=False
        )
        self.norm3 = DynamicGroupNorm(num_groups, out_channels, mode)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.drop_path = DropPath(drop_prob=drop_path_rate
                                  ) if drop_path_rate > eps else nn.Identity()

    def get_widths(self):
        return self.widths

    def get_index(self, mid_planes):
        idx_in = torch.arange(self.in_channels, device=self.conv1.conv.weight.device)
        idx_in = self.conv1.get_index(idx_in, mid_planes)
        self.norm1.set_index(idx_in)
        idx_in = self.conv2.get_index(idx_in, mid_planes)
        self.norm2.set_index(idx_in)
        idx_out = self.conv3.get_index(idx_in, self.out_channels)
        self.norm3.set_index(idx_out)
        return idx_out

    def forward(self, x):
        identity = x
        out = self.conv1(x)
        out = self.norm1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.norm2(out)
        out = self.relu(out)
        out = self.conv3(out)
        out = self.norm3(out)
        if self.downsample is not None:
            identity = self.downsample(x)
        out = self.drop_path(out)
        out += identity
        out = self.relu(out)
        return out
