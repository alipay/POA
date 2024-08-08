# Copyright 2024 Ant Group.
from __future__ import print_function

import torch
import torch.nn.functional as F

def cosine_similarity(a, b, eps=1e-6):
    return (a * b).sum(1) / (a.norm(dim=1) * b.norm(dim=1) + eps)

def pearson_correlation(a, b, eps=1e-6):
    return cosine_similarity(a - a.mean(-1).unsqueeze(-1),
                                b - b.mean(-1).unsqueeze(-1), eps)
def pcl_loss(y_s, y_t):
    return 1 - pearson_correlation(y_s, y_t).mean()


class FCDLoss(object):

    def __init__(self, 
                alpha=1.0, 
                beta=1.0, 
                gamma=0., 
                K=128,
                loss_weight: float = 1.0,
    ):
        self.K = K
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.loss_weight = loss_weight

    def __call__(self, feature_S, feature_T):
        if not isinstance(feature_S, (list, tuple)):
            feature_S = [feature_S]
            feature_T = [feature_T]
        else:
            assert len(feature_S) == len(feature_T)
        loss, num  = 0, len(feature_S)
        for feat_s, feat_t in zip(feature_S, feature_T):
            loss += self.forward_single(feat_s, feat_t)
        loss /= num
        return loss


    def forward_single(self, feature_S, feature_T: torch.Tensor):
        """FCDLoss forward function.

        Args:
            feature_S (torch.Tensor): The student transformer block feature, shape (B, N, C).
            feature_T (torch.Tensor): The teacher transformer block feature, shape (B, N, C).

        Return:
            torch.Tensor: The calculated loss value.
        """
        B, N, _ = feature_T.shape

        if self.alpha > 0:
            feature_S = F.normalize(feature_S, dim=-1)
            feature_T = F.normalize(feature_T, dim=-1)
            R_S = feature_S.bmm(feature_S.transpose(-1, -2))
            R_T = feature_T.bmm(feature_T.transpose(-1, -2))
            R_S = R_S.view(B, -1)
            R_T = R_T.view(B, -1)
            loss_token = pcl_loss(R_S, R_T)
        else:
            loss_token = 0

        if self.beta > 0:
            feature_S = feature_S.permute(1, 0, 2)  # N, B, D
            feature_T = feature_T.permute(1, 0, 2)
            R_S = feature_S.bmm(feature_S.transpose(-1, -2))  # N, B, B
            R_T = feature_T.bmm(feature_T.transpose(-1, -2))
            R_S = R_S.view(B, -1)
            R_T = R_T.view(B, -1)
            loss_sample = pcl_loss(R_S, R_T)
        else:
            loss_sample = 0

        if self.gamma > 0:
            sampler = torch.randperm(B * N)[:self.K]
            feature_S = feature_S.reshape(B * N, -1)[sampler]
            feature_T = feature_T.reshape(B * N, -1)[sampler]
            R_S = feature_S.mm(feature_S.T)
            R_T = feature_T.mm(feature_T.T)
            loss_rand = pcl_loss(R_S, R_T)
        else:
            loss_rand = 0

        kd_loss = self.alpha * loss_token + self.beta * loss_sample + self.gamma * loss_rand  # noqa

        return kd_loss * self.loss_weight

