# Copyright 2024 Ant Group.
import torch
import torch.nn as nn
from torch.nn.init import trunc_normal_
from torch.nn.utils import weight_norm
from .dynamic_ops import DynamicLinear 

class DINOHead(nn.Module):
    def __init__(
        self,
        max_in_dim,
        out_dim,
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
        self.out_dim = out_dim
        self.max_in_dim = max_in_dim
        self.mlp = _build_mlp(nlayers, max_in_dim, mode, bottleneck_dim, hidden_dim=hidden_dim,
                              use_bn=use_bn, bias=mlp_bias)
        self.apply(self._init_weights)
        self.last_layer = weight_norm(nn.Linear(bottleneck_dim, out_dim, bias=False))
        self.last_layer.weight_g.data.fill_(1)

    def get_index(self, idx_in):
        if self.nlayers == 1:
            idx_in, idx_out = self.mlp.get_index(idx_in, self.bottleneck_dim)
        else:
            idx_in, idx_out = self.mlp[0].get_index(idx_in, self.hidden_dim)
        return idx_in, idx_out

    def set_index(self, idx_in, idx_out):
        if self.nlayers == 1:
            self.mlp.set_index(idx_in, idx_out)
        else:
            self.mlp[0].set_index(idx_in, idx_out)

    def get_flat_param_mask(self, idx_in):
        param_mask = []
        if self.nlayers == 1:
            w_mask = torch.zeros(self.bottleneck_dim, self.max_in_dim)
        else:
            w_mask = torch.zeros(self.hidden_dim, self.max_in_dim)
        if isinstance(idx_in, int):
            idx_in = torch.arange(idx_in)
        else:
            idx_in = idx_in.cpu()
        w_mask[:, idx_in] = 1
        param_mask.append(w_mask.reshape(-1))
        if self.nlayers > 1:
            num_hidden = (self.nlayers - 2) * (self.hidden_dim + 1) * self.hidden_dim + \
                         (self.bottleneck_dim + 1) * self.hidden_dim

            res_mask = torch.ones(num_hidden)
            param_mask.append(res_mask)
        last_mask = torch.ones(self.out_dim * (self.bottleneck_dim + 1))
        param_mask.append(last_mask)
        return torch.cat(param_mask)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.mlp((x))
        eps = 1e-6 if x.dtype == torch.float16 else 1e-12
        x = nn.functional.normalize(x, dim=-1, p=2, eps=eps)
        x = self.last_layer(x)
        return x


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
