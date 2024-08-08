# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the Apache License, Version 2.0
# found in the LICENSE file in the root directory of this source tree.

import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch import nn


class DINOLoss(nn.Module):
    def __init__(
        self,
        out_dims,
        student_temp=0.1,
        center_momentum=0.9,
    ):
        super().__init__()
        self.student_temp = student_temp
        self.center_momentum = center_momentum
        if isinstance(out_dims, int):
            out_dims = [out_dims]
        for i, out_dim in enumerate(out_dims):
            self.register_buffer(f"center_{i}", torch.zeros(1, out_dim))
        self.num_centers = len(out_dims)
        self.updated = True
        self.reduce_handle = [None] * self.num_centers
        self.len_teacher_output = None
        self.async_batch_center = [None] * self.num_centers

    @torch.no_grad()
    def vanilla_teacher(self, teacher_outputs, teacher_temp):
        if isinstance(teacher_outputs, torch.Tensor):
            return F.softmax((teacher_outputs) / teacher_temp, dim=-1)
        return [self.vanilla_teacher(x) for x in teacher_outputs]

    @torch.no_grad()
    def softmax_center_teacher(self, teacher_outputs, teacher_temp):
        if isinstance(teacher_outputs, torch.Tensor):
            teacher_outputs = [teacher_outputs]
        self.apply_center_update()
        # teacher centering and sharpening
        softmax_center_teacher = []
        for i, teacher_output in enumerate(teacher_outputs):
            center = getattr(self, f"center_{i}")
            softmax_center_teacher.append(F.softmax((teacher_output - center) / teacher_temp, dim=-1))
        if len(softmax_center_teacher) == 0:
            softmax_center_teacher = softmax_center_teacher[0]
        return softmax_center_teacher

    @torch.no_grad()
    def sinkhorn_knopp_teacher(self, teacher_output, teacher_temp, n_iterations=3):
        teacher_output = teacher_output.float()
        world_size = dist.get_world_size() if dist.is_initialized() else 1
        Q = torch.exp(teacher_output / teacher_temp).t()  # Q is K-by-B for consistency with notations from our paper
        B = Q.shape[1] * world_size  # number of samples to assign
        K = Q.shape[0]  # how many prototypes

        # make the matrix sums to 1
        sum_Q = torch.sum(Q)
        if dist.is_initialized():
            dist.all_reduce(sum_Q)
        Q /= sum_Q

        for it in range(n_iterations):
            # normalize each row: total weight per prototype must be 1/K
            sum_of_rows = torch.sum(Q, dim=1, keepdim=True)
            if dist.is_initialized():
                dist.all_reduce(sum_of_rows)
            Q /= sum_of_rows
            Q /= K

            # normalize each column: total weight per sample must be 1/B
            Q /= torch.sum(Q, dim=0, keepdim=True)
            Q /= B

        Q *= B  # the columns must sum to 1 so that Q is an assignment
        return Q.t()

    def forward(self, student_output_list, teacher_out_softmaxed_centered_list):
        """
        Cross-entropy between softmax outputs of the teacher and student networks.
        """
        # TODO: Use cross_entropy_distribution here
        total_loss = 0
        for s in student_output_list:
            lsm = F.log_softmax(s / self.student_temp, dim=-1)
            for t in teacher_out_softmaxed_centered_list:
                loss = torch.sum(t * lsm, dim=-1)
                total_loss -= loss.mean()
        return total_loss

    @torch.no_grad()
    def update_center(self, teacher_outputs):
        self.reduce_center_update(teacher_outputs)

    @torch.no_grad()
    def reduce_center_update(self, teacher_outputs):
        self.updated = False
        if isinstance(teacher_outputs, torch.Tensor):
            teacher_outputs = [teacher_outputs]
        self.len_teacher_output = len(teacher_outputs[0])
        for i, teacher_output in enumerate(teacher_outputs):
            self.async_batch_center[i] = torch.sum(teacher_output, dim=0, keepdim=True)
            if dist.is_initialized():
                self.reduce_handle[i] = dist.all_reduce(self.async_batch_center[i], async_op=True)

    @torch.no_grad()
    def apply_center_update(self):
        if self.updated is False:
            world_size = dist.get_world_size() if dist.is_initialized() else 1
            for i in range(self.num_centers):
                if self.reduce_handle[i] is not None:
                    self.reduce_handle[i].wait()
                _t = self.async_batch_center[i] / (self.len_teacher_output * world_size)
                center = getattr(self, f"center_{i}")
                setattr(self, f"center_{i}", center * self.center_momentum + _t * (1 - self.center_momentum))
            self.updated = True
