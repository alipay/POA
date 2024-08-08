# Copyright 2024 Ant Group.
from typing import Callable, Optional, Tuple, Union, Sequence
import torch
import math
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F
from .dynamic_ops import DynamicLayerNorm, ScaledDynamicLinear


def make_2tuple(x):
    if isinstance(x, tuple):
        assert len(x) == 2
        return x

    assert isinstance(x, int)
    return (x, x)


class DynamicPatchEmbed(nn.Module):
    """
    2D image to patch embedding: (B,C,H,W) -> (B,N,D)

    Args:
        img_size: Image size.
        patch_size: Patch token size.
        in_chans: Number of input image channels.
        embed_dim: Number of linear projection output channels.
        norm_layer: Normalization layer.
    """

    def __init__(
        self,
        img_size: Union[int, Tuple[int, int]] = 224,
        patch_size: Union[int, Tuple[int, int]] = 16,
        in_chans: int = 3,
        max_embed_dim: int = 768,
        mode: str = 'order',
        norm_layer: Optional[Callable] = None,
        flatten_embedding: bool = True,
    ) -> None:
        super().__init__()

        assert mode in ['order', 'sort', 'random']
        self.mode = mode
        image_HW = make_2tuple(img_size)
        patch_HW = make_2tuple(patch_size)
        patch_grid_size = (
            image_HW[0] // patch_HW[0],
            image_HW[1] // patch_HW[1],
        )

        self.img_size = image_HW
        self.patch_size = patch_HW
        self.patches_resolution = patch_grid_size
        self.num_patches = patch_grid_size[0] * patch_grid_size[1]

        self.in_chans = in_chans
        self.max_embed_dim = max_embed_dim

        self.flatten_embedding = flatten_embedding

        self.proj = nn.Conv2d(in_chans, max_embed_dim, kernel_size=patch_HW, stride=patch_HW)
        self.norm = norm_layer(max_embed_dim) if norm_layer is not None else None
        self.idx_out = None

    def get_index(self, out_features):
        w = self.proj.weight
        if self.mode == 'order':
            idx_in = self.max_embed_dim
            idx_out = torch.arange(out_features, device=self.proj.weight.device)
        elif self.mode == 'random':
            idx_in = torch.arange(self.max_embed_dim, device=self.proj.weight.device)
            idx_out = torch.randperm(
                self.max_embed_dim, device=self.proj.weight.device
            )[:out_features]
        else:
            raise TypeError(f'Unsupported mode: {self.mode}')
        self.idx_out = idx_out
        if self.norm is not None:
            self.norm.set_index(idx_in)
        return idx_out

    def set_index(self, idx_out):
        self.idx_out = idx_out

    def get_out_idx(self, out_features):
        if self.mode == 'order':
            return self.get_idx_out_order(out_features)
        if self.mode == 'random':
            return self.get_idx_out_random(out_features)
        if self.mode == 'sort':
            return self.get_idx_out_order(out_features)

    def forward(self, x: Tensor, return_hw: bool=False) -> Tensor:
        _, _, H, W = x.shape
        patch_H, patch_W = self.patch_size

        assert H % patch_H == 0, f"Input image height {H} is not a multiple of patch height {patch_H}"
        assert W % patch_W == 0, f"Input image width {W} is not a multiple of patch width: {patch_W}"

        x = self.proj(x)  # B C H W
        H, W = x.size(2), x.size(3)
        x = x.flatten(2).transpose(1, 2)  # B HW C
        x = self.norm(x) if self.norm is not None else x
        x = x[..., self.idx_out]
        out_features = len(self.idx_out)
        if not self.flatten_embedding:
            x = x.reshape(-1, H, W, out_features)  # B H W C
        if return_hw:
            return x, (H, W)
        return x

    def flops(self) -> float:
        Ho, Wo = self.patches_resolution
        flops = Ho * Wo * self.embed_dim * self.in_chans * (self.patch_size[0] * self.patch_size[1])
        if self.norm is not None:
            flops += Ho * Wo * self.embed_dim
        return flops



class AdaptivePadding(nn.Module):
    """Applies padding to input (if needed) so that input can get fully covered
    by filter you specified. It support two modes "same" and "corner". The
    "same" mode is same with "SAME" padding mode in TensorFlow, pad zero around
    input. The "corner"  mode would pad zero to bottom right.

    Args:
        kernel_size (int | tuple): Size of the kernel:
        stride (int | tuple): Stride of the filter. Default: 1:
        dilation (int | tuple): Spacing between kernel elements.
            Default: 1.
        padding (str): Support "same" and "corner", "corner" mode
            would pad zero to bottom right, and "same" mode would
            pad zero around input. Default: "corner".
    Example:
        >>> kernel_size = 16
        >>> stride = 16
        >>> dilation = 1
        >>> input = torch.rand(1, 1, 15, 17)
        >>> adap_pad = AdaptivePadding(
        >>>     kernel_size=kernel_size,
        >>>     stride=stride,
        >>>     dilation=dilation,
        >>>     padding="corner")
        >>> out = adap_pad(input)
        >>> assert (out.shape[2], out.shape[3]) == (16, 32)
        >>> input = torch.rand(1, 1, 16, 17)
        >>> out = adap_pad(input)
        >>> assert (out.shape[2], out.shape[3]) == (16, 32)
    """

    def __init__(self, kernel_size=1, stride=1, dilation=1, padding='corner'):

        super().__init__()

        assert padding in ('same', 'corner')

        kernel_size = make_2tuple(kernel_size)
        stride = make_2tuple(stride)
        dilation = make_2tuple(dilation)

        self.padding = padding
        self.kernel_size = kernel_size
        self.stride = stride
        self.dilation = dilation

    def get_pad_shape(self, input_shape):
        input_h, input_w = input_shape
        kernel_h, kernel_w = self.kernel_size
        stride_h, stride_w = self.stride
        output_h = math.ceil(input_h / stride_h)
        output_w = math.ceil(input_w / stride_w)
        pad_h = max((output_h - 1) * stride_h +
                    (kernel_h - 1) * self.dilation[0] + 1 - input_h, 0)
        pad_w = max((output_w - 1) * stride_w +
                    (kernel_w - 1) * self.dilation[1] + 1 - input_w, 0)
        return pad_h, pad_w

    def forward(self, x):
        pad_h, pad_w = self.get_pad_shape(x.size()[-2:])
        if pad_h > 0 or pad_w > 0:
            if self.padding == 'corner':
                x = F.pad(x, [0, pad_w, 0, pad_h])
            elif self.padding == 'same':
                x = F.pad(x, [
                    pad_w // 2, pad_w - pad_w // 2, pad_h // 2,
                    pad_h - pad_h // 2
                ])
        return x


class DynamicPatchMerging(nn.Module):
    """Merge patch feature map.

    This layer groups feature map by kernel_size, and applies norm and linear
    layers to the grouped feature map. Our implementation uses `nn.Unfold` to
    merge patch, which is about 25% faster than original implementation.
    Instead, we need to modify pretrained models for compatibility.

    Args:
        in_channels (int): The num of input channels.
        out_channels (int): The num of output channels.
        kernel_size (int | tuple, optional): the kernel size in the unfold
            layer. Defaults to 2.
        stride (int | tuple, optional): the stride of the sliding blocks in the
            unfold layer. Default: None. (Would be set as `kernel_size`)
        padding (int | tuple | string ): The padding length of
            embedding conv. When it is a string, it means the mode
            of adaptive padding, support "same" and "corner" now.
            Default: "corner".
        dilation (int | tuple, optional): dilation parameter in the unfold
            layer. Default: 1.
        bias (bool, optional): Whether to add bias in linear layer or not.
            Defaults: False.
        norm_cfg (dict, optional): Config dict for normalization layer.
            Default: dict(type='LN').
        init_cfg (dict, optional): The extra config for initialization.
            Default: None.
    """

    def __init__(self,
                 max_in_channels,
                 max_out_channels,
                 mode='order',
                 kernel_size=2,
                 stride=None,
                 padding='corner',
                 dilation=1,
                 bias=False,
                 norm_layer=DynamicLayerNorm):
        super().__init__()
        self.max_in_channels = max_in_channels
        self.max_out_channels = max_out_channels
        if stride:
            stride = stride
        else:
            stride = kernel_size

        kernel_size = make_2tuple(kernel_size)
        self.num_scale = kernel_size[0] * kernel_size[1]
        stride = make_2tuple(stride)
        dilation = make_2tuple(dilation)
        self.mode = mode

        if isinstance(padding, str):
            self.adap_padding = AdaptivePadding(
                kernel_size=kernel_size,
                stride=stride,
                dilation=dilation,
                padding=padding)
            # disable the padding of unfold
            padding = 0
        else:
            self.adap_padding = None

        padding = make_2tuple(padding)
        self.sampler = nn.Unfold(
            kernel_size=kernel_size,
            dilation=dilation,
            padding=padding,
            stride=stride)
        if norm_layer is not None:
            self.norm = norm_layer(max_in_channels * self.num_scale)
        else:
            self.norm = None

        self.reduction = ScaledDynamicLinear(
            max_in_channels, max_out_channels, self.num_scale, mode, bias=bias
        )

    def get_index(self, idx_in, out_features):
        idx_in, idx_out = self.reduction.get_index(idx_in, out_features)
        if self.norm is not None:
            self.norm.set_index(idx_in)
        self.idx_in = idx_in
        return idx_out

    def forward(self, x, input_size):
        """
        Args:
            x (Tensor): Has shape (B, H*W, C_in).
            input_size (tuple[int]): The spatial shape of x, arrange as (H, W).
                Default: None.

        Returns:
            tuple: Contains merged results and its spatial shape.

                - x (Tensor): Has shape (B, Merged_H * Merged_W, C_out)
                - out_size (tuple[int]): Spatial shape of x, arrange as
                    (Merged_H, Merged_W).
        """
        B, L, C = x.shape
        assert isinstance(input_size, Sequence), f'Expect ' \
                                                 f'input_size is ' \
                                                 f'`Sequence` ' \
                                                 f'but get {input_size}'

        H, W = input_size
        assert L == H * W, 'input feature has wrong size'

        x = x.view(B, H, W, C).permute([0, 3, 1, 2])  # B, C, H, W
        # Use nn.Unfold to merge patch. About 25% faster than original method,
        # but need to modify pretrained model for compatibility

        if self.adap_padding:
            x = self.adap_padding(x)
            H, W = x.shape[-2:]

        x = self.sampler(x)
        # if kernel_size=2 and stride=2, x should has shape (B, C*4, H/2*W/2)

        out_h = (H + 2 * self.sampler.padding[0] - self.sampler.dilation[0] *
                 (self.sampler.kernel_size[0] - 1) -
                 1) // self.sampler.stride[0] + 1
        out_w = (W + 2 * self.sampler.padding[1] - self.sampler.dilation[1] *
                 (self.sampler.kernel_size[1] - 1) -
                 1) // self.sampler.stride[1] + 1

        output_size = (out_h, out_w)
        x = x.transpose(1, 2)  # B, H/2*W/2, C*4
        # process for unfold: C * 4 -> 4 * C
        num = x.shape[-1] // self.num_scale
        x = torch.stack(x.chunk(num, dim=-1), dim=-1).flatten(-2)

        x = self.norm(x) if self.norm is not None else x
        x = self.reduction(x)
        return x, output_size

