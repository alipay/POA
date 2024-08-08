# Copyright 2024 Ant Group.
from typing import Callable, Optional
from torch import Tensor, nn
from .dynamic_ops import DynamicLinear


class DynamicMlp(nn.Module):
    def __init__(
        self,
        max_in_features: int,
        max_hidden_features: Optional[int] = None,
        max_out_features: Optional[int] = None,
        mode: str = 'order',
        act_layer: Callable[..., nn.Module] = nn.GELU,
        drop: float = 0.0,
        bias: bool = True,
    ) -> None:
        super().__init__()
        max_out_features = max_out_features or max_in_features
        max_hidden_features = max_hidden_features or max_in_features
        self.fc1 = DynamicLinear(max_in_features, max_hidden_features, bias=bias, mode=mode)
        self.act = act_layer()
        self.fc2 = DynamicLinear(max_hidden_features, max_out_features, bias=bias, mode=mode)
        self.drop = nn.Dropout(drop)
        self.max_in_features = max_in_features
        self.max_hidden_features = max_hidden_features
        self.max_out_features = max_out_features

    def get_index(self, idx_in, out_features):
        out_features_1 = int(out_features / self.max_out_features * self.max_hidden_features)
        idx_ins, idx_outs = [], []
        idx_in, idx_out = self.fc1.get_index(idx_in, out_features_1)
        idx_ins.append(idx_in)
        idx_outs.append(idx_out)
        idx_in, idx_out = self.fc2.get_index(idx_out, out_features)
        idx_ins.append(idx_in)
        idx_outs.append(idx_out)
        return idx_ins, idx_outs

    def set_index(self, idx_ins, idx_outs):
        assert len(idx_ins) == len(idx_outs)
        assert len(idx_ins) == 2
        self.fc1.set_index(idx_ins[0], idx_outs[0])
        self.fc2.set_index(idx_ins[1], idx_outs[1])

    def forward(self, x: Tensor) -> Tensor:
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class DynamicFFN(DynamicMlp):

    def __init__(self,
                 max_embed_dim,
                 max_feedforward_channels,
                 mode,
                 act_func=nn.GELU,
                 ffn_drop=0.,
                 dropout_layer=None,
                 add_identity=True):
        super(DynamicFFN, self).__init__(
            max_in_features=max_embed_dim,
            max_hidden_features=max_feedforward_channels,
            mode=mode,
            act_layer=act_func,
            drop=ffn_drop,
        )
        self.dropout_layer = dropout_layer if dropout_layer else nn.Identity()
        self.add_identity = add_identity

    def forward(self, x, identity=None):
        """Forward function for `FFN`.

        The function would add x to the output tensor if residue is None.
        """
        out = super(DynamicFFN, self).forward(x)
        if not self.add_identity:
            return self.dropout_layer(out)
        if identity is None:
            identity = x
        return identity + self.dropout_layer(out)

    def get_index(self, idx_in, out_features):
        out_features_1 = int(out_features / self.max_out_features * self.max_hidden_features)
        idx_in, idx_out = self.fc1.get_index(idx_in, out_features_1)
        idx_in, idx_out = self.fc2.get_index(idx_out, out_features)
        return idx_out
