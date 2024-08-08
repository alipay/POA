# Copyright 2024 Ant Group.
import torch
import torch.nn as nn
from torch.nn.init import trunc_normal_
from torch.nn.utils import weight_norm
from .dynamic_ops import DynamicLinear 


class MDINOHead(nn.Module):
    def __init__(
        self,
        max_in_dim,
        out_dims,
        mode='order',
        use_bn=False,
        nlayers=3,
        hidden_dim=2048,
        bottleneck_dim=256,
        mlp_bias=True,
    ):
        super().__init__()
        nlayers = max(nlayers, 1)
        self.nlayers = nlayers
        self.bottleneck_dim = bottleneck_dim
        self.hidden_dim = hidden_dim
        self.out_dims = list(out_dims)
        self.num = len(self.out_dims)
        self.max_in_dim = max_in_dim
        for i in range(self.num):
            mlp = _build_mlp(nlayers, max_in_dim, mode, bottleneck_dim, hidden_dim=hidden_dim,
                                  use_bn=use_bn, bias=mlp_bias)
            name = f'mlp{i}'
            self.add_module(name, mlp)
        self.apply(self._init_weights)
        for i in range(self.num):
            out_dim = self.out_dims[i]
            last_layer = weight_norm(nn.Linear(bottleneck_dim, out_dim, bias=False))
            last_layer.weight_g.data.fill_(1)
            name = f'last_layer{i}'
            self.add_module(name, last_layer)

    def get_index(self, idx_in):
        if self.nlayers == 1:
            for i in range(self.num):
                idx_in, idx_out = getattr(self, f'mlp{i}').get_index(idx_in, self.bottleneck_dim)
        else:
            for i in range(self.num):
                idx_in, idx_out = getattr(self, f'mlp{i}')[0].get_index(idx_in, self.hidden_dim)
        return idx_in, idx_out

    def set_index(self, idx_in, idx_out):
        if self.nlayers == 1:
            for i in range(self.num):
                getattr(self, f'mlp{i}').set_index(idx_in, idx_out)
        else:
            for i in range(self.num):
                getattr(self, f'mlp{i}')[0].set_index(idx_in, idx_out)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        out = []
        for i in range(self.num):
            x_ = getattr(self, f'mlp{i}')(x)
            eps = 1e-6 if x_.dtype == torch.float16 else 1e-12
            x_ = nn.functional.normalize(x_, dim=-1, p=2, eps=eps)
            x_ = getattr(self, f'last_layer{i}')(x_)
            out.append(x_)
        return out


class MDINOSHead(nn.Module):
    '''share mlp'''
    def __init__(
        self,
        max_in_dim,
        out_dims,
        mode='order',
        use_bn=False,
        nlayers=3,
        hidden_dim=2048,
        bottleneck_dim=256,
        mlp_bias=True,
    ):
        super().__init__()
        nlayers = max(nlayers, 1)
        self.nlayers = nlayers
        self.bottleneck_dim = bottleneck_dim
        self.hidden_dim = hidden_dim
        self.out_dims = list(out_dims)
        self.num = len(self.out_dims)
        self.max_in_dim = max_in_dim
        self.mlp = _build_mlp(nlayers, max_in_dim, mode, bottleneck_dim, hidden_dim=hidden_dim,
                              use_bn=use_bn, bias=mlp_bias)
        self.apply(self._init_weights)
        for i in range(self.num):
            out_dim = self.out_dims[i]
            last_layer = weight_norm(nn.Linear(bottleneck_dim, out_dim, bias=False))
            last_layer.weight_g.data.fill_(1)
            name = f'last_layer{i}'
            self.add_module(name, last_layer)

    def get_index(self, idx_in):
        if self.nlayers == 1:
            idx_in, idx_out = getattr(self, f'mlp').get_index(idx_in, self.bottleneck_dim)
        else:
            idx_in, idx_out = getattr(self, f'mlp')[0].get_index(idx_in, self.hidden_dim)
        return idx_in, idx_out

    def set_index(self, idx_in, idx_out):
        if self.nlayers == 1:
            getattr(self, f'mlp').set_index(idx_in, idx_out)
        else:
            getattr(self, f'mlp')[0].set_index(idx_in, idx_out)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        out = []
        x = self.mlp(x)
        for i in range(self.num):
            eps = 1e-6 if x.dtype == torch.float16 else 1e-12
            x_ = nn.functional.normalize(x, dim=-1, p=2, eps=eps)
            x_ = getattr(self, f'last_layer{i}')(x_)
            out.append(x_)
        return out


def _build_mlp(nlayers, max_in_dim, mode, bottleneck_dim, hidden_dim=None, use_bn=False, bias=True):
    if nlayers == 1:
        return DynamicLinear(max_in_dim, bottleneck_dim, bias=bias, mode=mode)
    else:
        layers = [DynamicLinear(max_in_dim, hidden_dim, bias=bias, mode=mode)]
        if use_bn:
            layers.append(nn.BatchNorm1d(hidden_dim))
        layers.append(nn.GELU())
        for _ in range(nlayers - 2):
            layers.append(nn.Linear(hidden_dim, hidden_dim, bias=bias))
            if use_bn:
                layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(nn.GELU())
        layers.append(nn.Linear(hidden_dim, bottleneck_dim, bias=bias))
        return nn.Sequential(*layers)
