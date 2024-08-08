# Copyright 2024 Ant Group.
import torch.nn.functional as F
from typing import Union
import torch
from torch import Tensor
from torch import nn


class DynamicLayerScale(nn.Module):
    def __init__(
        self,
        max_dim: int,
        init_values: Union[float, Tensor] = 1e-5,
        inplace: bool = False,
    ) -> None:
        super().__init__()
        self.inplace = inplace
        self.gamma = nn.Parameter(init_values * torch.ones(max_dim))
        self.max_dim = max_dim
        self.idx_in = None

    def set_index(self, idx_in):
        self.idx_in = idx_in

    def get_gamma(self) -> Tensor:
        num = len(self.idx_in) if isinstance(self.idx_in, Tensor) else self.idx_in
        if num == self.max_dim:
            gamma = self.gamma
        elif isinstance(self.idx_in, Tensor):
            gamma = self.gamma[self.idx_in]
        else:
            gamma = self.gamma[:self.idx_in]
        return gamma

    def forward(self, x: Tensor) -> Tensor:
        gamma = self.get_gamma()
        return x.mul_(gamma) if self.inplace else x * gamma


class DynamicLinear(nn.Module):

    def __init__(self, max_in_features, max_out_features, mode, bias=True):
        super(DynamicLinear, self).__init__()
        assert mode in ['order', 'sort', 'random'], mode
        self.max_in_features = max_in_features
        self.max_out_features = max_out_features
        self.bias = bias
        self.linear = nn.Linear(self.max_in_features, self.max_out_features, self.bias)
        self.mode = mode
        self.idx_in = None
        self.idx_out = None
        self.offset_scale = None

    def get_index(self, idx_in, out_features):
        self.idx_in = idx_in
        if self.mode == 'order':
            if isinstance(idx_in, Tensor):
                self.idx_in = len(idx_in)
            idx_out = out_features
        elif self.mode == 'sort':
            assert isinstance(idx_in, Tensor)
            w = self.linear.weight[:, idx_in]
            _, idx_out = torch.topk(torch.abs(w).sum(1), out_features)
        elif self.mode == 'random':
            assert isinstance(idx_in, Tensor)
            idx_out = torch.randperm(
                self.max_out_features,
                device=self.linear.weight.device
            )[:out_features]
        else:
            raise TypeError(f'Unsupported mode: {self.mode}')
        self.idx_out = idx_out
        self.offset_scale = self.max_in_features / len(idx_in) if isinstance(idx_in, Tensor) else \
            self.max_in_features / idx_in
        return idx_in, idx_out

    def set_index(self, idx_in, idx_out):
        self.idx_in = idx_in
        if self.mode == 'order':
            if isinstance(idx_in, Tensor):
                self.idx_in = len(idx_in)
        self.idx_out = idx_out
        self.offset_scale = self.max_in_features / len(idx_in) if isinstance(idx_in, Tensor) else \
            self.max_in_features / idx_in

    def get_active_weight_bias(self):
        if self.mode == 'order':
            w = self.linear.weight if self.idx_in == self.max_in_features and \
                self.idx_out == self.max_out_features else \
                self.linear.weight[:self.idx_out, :self.idx_in]
            if self.bias:
                b = self.linear.bias if self.idx_out == self.max_out_features else \
                    self.linear.bias[:self.idx_out]
            else:
                b = None
        else:
            w = self.linear.weight if len(self.idx_in) == self.max_in_features and \
                len(self.idx_out) == self.max_out_features else \
                self.linear.weight[self.idx_out, :][:, self.idx_in]
            if self.bias:
                b = self.linear.bias if len(self.idx_out) == self.max_out_features else \
                    self.linear.bias[self.idx_out]
            else:
                b = None
        return w, b

    def forward(self, x):
        weight, bias = self.get_active_weight_bias()
        x = F.linear(x * self.offset_scale, weight, bias)
        return x


class GroupedDynamicLinear(DynamicLinear):

    def __init__(self, max_in_features, max_out_features, num_groups, mode, bias=True):
        total_out_features = max_out_features * num_groups
        super(GroupedDynamicLinear, self).__init__(max_in_features, total_out_features, mode, bias)
        self.max_out_features = max_out_features
        self.num_groups = num_groups

    def get_index(self, idx_in, out_features):
        self.idx_in = idx_in
        idx_outs = []
        if self.mode == 'order':
            if isinstance(idx_in, Tensor):
                self.idx_in = len(idx_in)
            for i in range(self.num_groups):
                idx_outs.append(
                    torch.arange(
                        i * self.max_out_features,
                        i * self.max_out_features + out_features,
                        device=self.linear.weight.device
                    )
                )
            idx_out = torch.cat(idx_outs)
        elif self.mode == 'sort':
            assert isinstance(idx_in, Tensor)
            w = torch.abs(
                self.linear.weight.reshape(self.num_groups, self.max_out_features, -1)
            ).mean(0)
            _, idx_out_ = torch.topk(w.sum(1), out_features)
            for i in range(self.num_groups):
                idx_outs.append(idx_out_ + i * self.max_out_features)
            idx_out = torch.cat(idx_outs)
        elif self.mode == 'random':
            assert isinstance(idx_in, Tensor)
            # idx_in / idx_out -> Tensor
            idx_out_ = torch.randperm(
                self.max_out_features)[:out_features].to(self.linear.weight.device)
            for i in range(self.num_groups):
                idx_outs.append(idx_out_ + i * self.max_out_features)
            idx_out = torch.cat(idx_outs)
        else:
            raise TypeError(f'Unsupported mode: {self.mode}')
        self.idx_out = idx_out
        self.offset_scale = self.max_in_features / len(idx_in) if isinstance(idx_in, Tensor) else \
            self.max_in_features / idx_in
        return idx_in, idx_out

    def get_active_weight_bias(self):
        if self.mode == 'order':
            w = self.linear.weight if self.idx_in == self.max_in_features and \
                len(self.idx_out) == self.max_out_features * self.num_groups else \
                self.linear.weight[self.idx_out, :][:, :self.idx_in]
        else:
            w = self.linear.weight if len(self.idx_in) == self.max_in_features and \
                len(self.idx_out) == self.max_out_features * self.num_groups else \
                self.linear.weight[self.idx_out, :][:, self.idx_in]
        if self.bias:
            b = self.linear.bias if len(self.idx_out) == self.max_out_features * \
                self.num_groups else self.linear.bias[self.idx_out]
        else:
            b = None
        return w, b


class DynamicLayerNorm(nn.Module):

    def __init__(self, max_dim: int, eps: float = 1e-5, elementwise_affine: bool = True,
                 device=None, dtype=None):
        super(DynamicLayerNorm, self).__init__()
        self.elementwise_affine = elementwise_affine
        self.max_dim = max_dim
        self.eps = eps
        self.norm = nn.LayerNorm(self.max_dim, eps, elementwise_affine, device, dtype)
        self.idx_in = None

    def set_index(self, idx_in):
        self.idx_in = idx_in

    def get_active_weight_bias(self):
        num = len(self.idx_in) if isinstance(self.idx_in, Tensor) else self.idx_in
        if num == self.max_dim:
            w = self.norm.weight
            b = self.norm.bias
        elif isinstance(self.idx_in, Tensor):
            w = self.norm.weight[self.idx_in] if self.elementwise_affine else None
            b = self.norm.bias[self.idx_in] if self.elementwise_affine else None
        else:
            w = self.norm.weight[:self.idx_in] if self.elementwise_affine else None
            b = self.norm.bias[:self.idx_in] if self.elementwise_affine else None
        return w, b, num

    def forward(self, x):
        w, b, num = self.get_active_weight_bias()
        normalized_shape = (num, )
        return F.layer_norm(x, normalized_shape, w, b, self.eps)



class ScaledDynamicLinear(DynamicLinear):

    def __init__(self, max_in_features, max_out_features, num_scale, mode, bias=True):
        total_in_features = max_in_features * num_scale
        super(ScaledDynamicLinear, self).__init__(total_in_features, max_out_features, mode, bias)
        self.max_in_features = max_in_features
        self.num_scale = num_scale

    def get_index(self, idx_in, out_features):
        idx_ins = []
        if self.mode == 'order':
            assert isinstance(idx_in, int)
            for i in range(self.num_scale):
                idx_ins.append(
                    torch.arange(
                        i * self.max_in_features,
                        i * self.max_in_features + idx_in,
                        device=self.linear.weight.device
                    )
                )
            self.idx_in = torch.cat(idx_ins)
            idx_out = out_features
        elif self.mode == 'random':
            assert isinstance(idx_in, Tensor)
            for i in range(self.num_scale):
                idx_ins.append(idx_in + i * self.max_in_features)
            self.idx_in = torch.cat(idx_ins)
            idx_out = torch.randperm(
                self.max_out_features,
                device=self.linear.weight.device
            )[:out_features]
        else:
            raise TypeError(f'Unsupported mode: {self.mode}')
        self.idx_out = idx_out
        self.offset_scale = self.max_in_features * self.num_scale / len(self.idx_in)
        return self.idx_in, self.idx_out

    def get_active_weight_bias(self):
        if self.mode == 'order':
            w = self.linear.weight if len(self.idx_in) == self.max_in_features * self.num_scale and \
                self.idx_out == self.max_out_features else \
                self.linear.weight[:self.idx_out, :][:, self.idx_in]
        else:
            w = self.linear.weight if len(self.idx_in) == self.max_in_features * self.num_scale and \
                len(self.idx_out) == self.max_out_features else \
                self.linear.weight[self.idx_out, :][:, self.idx_in]
        if self.bias: 
            if self.mode == 'order':
                b = self.linear.bias if self.idx_out == self.max_out_features else \
                    self.linear.bias[:self.idx_out]
            else:
                b = self.linear.bias if len(self.idx_out) == self.max_out_features else \
                    self.linear.bias[self.idx_out]
        else:
            b = None
        return w, b


class DynamicConv2d(nn.Module):

    def __init__(self, max_in_channels, max_out_channels, mode, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1, bias=True):
        super(DynamicConv2d, self).__init__()
        assert mode in ['order', 'random'], mode
        self.max_in_channels = max_in_channels
        self.max_out_channels = max_out_channels
        self.mode = mode
        self.bias = bias
        self.conv = nn.Conv2d(max_in_channels, max_out_channels, kernel_size, stride, padding,
                              dilation, groups, bias)
        self.idx_in = None
        self.idx_out = None
        self.offset_scale = None

    def get_index(self, idx_in, out_channels):
        self.idx_in = idx_in
        if self.mode == 'order':
            if isinstance(idx_in, Tensor):
                self.idx_in = len(idx_in)
            idx_out = out_channels
        elif self.mode == 'random':
            assert isinstance(idx_in, Tensor)
            idx_out = torch.randperm(
                self.max_out_channels,
                device=self.conv.weight.device
            )[:out_channels]
        else:
            raise TypeError(f'Unsupported mode: {self.mode}')
        self.idx_out = idx_out
        self.offset_scale = self.max_in_channels / len(idx_in) if isinstance(idx_in, Tensor) else \
            self.max_in_channels / idx_in
        return idx_out

    def get_active_weight_bias(self):
        if self.mode == 'order':
            w = self.conv.weight if self.idx_in == self.max_in_channels and \
                self.idx_out == self.max_out_channels else \
                self.conv.weight[:self.idx_out, :self.idx_in, :, :]
            w = w.contiguous()
            if self.bias:
                b = self.conv.bias if self.idx_out == self.max_out_channels else \
                    self.conv.bias[:self.idx_out]
                b = b.contiguous()
            else:
                b = None
        else:
            w = self.conv.weight if len(self.idx_in) == self.max_in_channels and \
                len(self.idx_out) == self.max_out_channels else \
                self.conv.weight[self.idx_out, :, :, :][:, self.idx_in, :, :]
            w = w.contiguous()
            if self.bias:
                b = self.conv.bias if len(self.idx_out) == self.max_out_channels else \
                    self.conv.bias[self.idx_out]
                b = b.contiguous()
            else:
                b = None
        return w, b

    def forward(self, x):
        weight, bias = self.get_active_weight_bias()
        x = self.conv._conv_forward(x * self.offset_scale, weight, bias)
        return x


class DynamicGroupNorm(nn.Module):

    def __init__(self, num_groups, max_num_channels, mode, eps=1e-5, affine=True,
                 device=None, dtype=None):
        factory_kwargs = {'device': device, 'dtype': dtype}
        super(DynamicGroupNorm, self).__init__()
        if max_num_channels % num_groups != 0:
            raise ValueError('num_channels must be divisible by num_groups')
        self.num_groups = num_groups
        self.max_num_channels = max_num_channels
        self.eps = eps
        self.affine = affine
        self.mode = mode
        if self.affine:
            self.weight = nn.Parameter(torch.empty(max_num_channels, **factory_kwargs))
            self.bias = nn.Parameter(torch.empty(max_num_channels, **factory_kwargs))
        else:
            self.register_parameter('weight', None)
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        if self.affine:
            nn.init.ones_(self.weight)
            nn.init.zeros_(self.bias)

    def set_index(self, idx_in):
        if self.mode == 'order':
            assert isinstance(idx_in, int)
        else:
            assert isinstance(idx_in, Tensor)
        self.idx_in = idx_in

    def get_active_weight_bias(self):
        if self.affine:
            if self.mode == 'order':
                w = self.weight if self.idx_in == self.max_num_channels else self.weight[:self.idx_in]
            else:
                w = self.weight if len(self.idx_in) == self.max_num_channels else self.weight[self.idx_in]
        else:
            w = None
        if self.affine:
            if self.mode == 'order':
                b = self.bias if self.idx_in == self.max_num_channels else self.bias[:self.idx_in]
            else:
                b = self.bias if len(self.idx_in) == self.max_num_channels else self.bias[self.idx_in]
        else:
            b = None
        return w, b

    def forward(self, input: Tensor) -> Tensor:
        weight, bias = self.get_active_weight_bias()
        return F.group_norm(input, self.num_groups, weight, bias, self.eps)
