# Copyright 2024 Ant Group.
import os
from typing import Callable, Optional, Tuple
import warnings
import torch
from .dynamic_ops import GroupedDynamicLinear, DynamicLinear
from torch import Tensor, nn
import torch.nn.functional as F


class DynamicSwiGLUFFN(nn.Module):
    def __init__(
            self,
            max_in_features: int,
            max_hidden_features: Optional[int] = None,
            max_out_features: Optional[int] = None,
            act_layer: Callable[..., nn.Module] = None,
            drop: float = 0.0,
            bias: bool = True,
            mode: str = 'order',
    ) -> None:
        super().__init__()
        assert mode in ['order', 'sort', 'random']
        self.mode = mode
        max_out_features = max_out_features or max_in_features
        self.max_out_features = max_out_features
        max_hidden_features = max_hidden_features or max_in_features
        self.max_hidden_features = max_hidden_features
        self.w12 = GroupedDynamicLinear(
            max_in_features, max_hidden_features,
            num_groups=2, bias=bias, mode=mode)
        self.w3 = DynamicLinear(max_hidden_features, max_out_features, bias=bias, mode=mode)

    def get_index(self, idx_in, out_features):
        idx_ins, idx_outs = [], []
        hidden_features = int(out_features / self.max_out_features * self.max_hidden_features)
        idx_in, idx_out = self.w12.get_index(idx_in, hidden_features)
        idx_ins.append(idx_in)
        idx_outs.append(idx_out)
        idx_in = torch.chunk(idx_out, 2)[0]
        idx_in, idx_out = self.w3.get_index(idx_in, out_features)
        idx_ins.append(idx_in)
        idx_outs.append(idx_out)
        return idx_ins, idx_outs

    def set_index(self, idx_ins, idx_outs):
        assert len(idx_ins) == len(idx_outs)
        assert len(idx_ins) == 2
        self.w12.set_index(idx_ins[0], idx_outs[0])
        self.w3.set_index(idx_ins[1], idx_outs[1])

    def forward(self, x: Tensor) -> Tensor:
        x12 = self.w12(x)
        x1, x2 = x12.chunk(2, dim=-1)
        hidden = F.silu(x1) * x2
        return self.w3(hidden)


XFORMERS_ENABLED = os.environ.get("XFORMERS_DISABLED") is None
try:
    if XFORMERS_ENABLED:
        from xformers.ops import swiglu, unbind

        class DynamicSwiGLU:

            def __init__(
                    self,
                    max_in_features: int,
                    max_hidden_features: int,
                    max_out_features: Optional[int] = None,
                    bias: bool = True,
                    mode: str = 'order',
            ) -> None:

                super().__init__()
                max_out_features = max_out_features or max_in_features
                self.max_out_features = max_out_features
                max_hidden_features = max_hidden_features or max_in_features
                self.max_hidden_features = max_hidden_features

                self.w12 = GroupedDynamicLinear(
                    max_in_features, max_hidden_features, 2, bias, mode
                )
                self.w3 = DynamicLinear(
                    max_hidden_features, max_out_features, bias, mode
                )

                self.max_hidden_features = max_hidden_features
                self.max_out_features = max_out_features
                self.max_in_features = max_in_features
                self.op = None

            def get_index(self, idx_in, out_features):
                idx_ins, idx_outs = [], []
                hidden_features = int(
                    out_features / self.max_out_features * self.max_hidden_features)
                idx_in, idx_out = self.w12.get_index(idx_in, hidden_features)
                idx_ins.append(idx_in)
                idx_outs.append(idx_out)
                idx_in = torch.chunk(idx_out, 2)[0]
                idx_in, idx_out = self.w3.get_index(idx_in, out_features)
                idx_ins.append(idx_in)
                idx_outs.append(idx_out)
                return idx_ins, idx_outs

            def set_index(self, idx_ins, idx_outs):
                assert len(idx_ins) == len(idx_outs)
                assert len(idx_ins) == 2
                self.w12.set_index(idx_ins[0], idx_outs[0])
                self.w3.set_index(idx_ins[1], idx_outs[1])

            def forward(self, x: Tensor):
                wbs = self._ordered_params()
                x = swiglu(x, *wbs, op=self.op)
                return x

            def _ordered_params(self,):
                w1w2, b1b2 = self.w12.get_active_weight_bias()
                w1w2 = w1w2.contiguous()
                b1b2 = b1b2.contiguous()
                w1, w2 = unbind(
                    w1w2.view([2, w1w2.shape[0] // 2, w1w2.shape[1]]),
                    dim=0,
                )
                if b1b2 is not None:
                    b1, b2 = unbind(b1b2.view([2, b1b2.shape[0] // 2]), dim=0)
                else:
                    b1, b2 = None, None
                w3, b3, = self.w3.get_active_weight_bias()
                w3 = w3.contiguous()
                b3 = b3.contiguous()
                return [
                    w1,
                    b1,
                    w2,
                    b2,
                    w3,
                    b3,
                ]

        XFORMERS_AVAILABLE = True
        warnings.warn("xFormers is available (DynamicSwiGLU)")
    else:
        warnings.warn("xFormers is disabled (DynamicSwiGLU)")
        raise ImportError
except ImportError:
    DynamicSwiGLU = DynamicSwiGLUFFN
    XFORMERS_AVAILABLE = False

    warnings.warn("xFormers is not available (SwiGLU)")


class DynamicSwiGLUFFNFused(DynamicSwiGLU):
    def __init__(
        self,
        max_in_features: int,
        max_hidden_features: Optional[int] = None,
        max_out_features: Optional[int] = None,
        mode: str = 'order',
        act_layer: Callable[..., nn.Module] = None,
        drop: float = 0.0,
        bias: bool = True,
    ) -> None:
        assert mode in ['order', 'sort', 'random']
        self.mode = mode
        max_out_features = max_out_features or max_in_features
        max_hidden_features = max_hidden_features or max_in_features
        max_hidden_features = (int(max_hidden_features * 2 / 3) + 7) // 8 * 8
        super().__init__(
            max_in_features=max_in_features,
            max_hidden_features=max_hidden_features,
            max_out_features=max_out_features,
            bias=bias,
            mode=mode,
        )
