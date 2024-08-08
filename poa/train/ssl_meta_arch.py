# Copyright 2024 Ant Group.
import random
from functools import partial
import logging
import torch
from torch import nn
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from poa.loss import DINOLoss, iBOTPatchLoss, KoLeoLoss
from poa.models import build_model_from_cfg
from poa.layers import MDINOHead
from poa.utils.utils import has_batchnorms
from poa.utils.param_groups import get_params_groups_with_decay, fuse_params_groups
from poa.fsdp import get_fsdp_wrapper, ShardedGradScaler, get_fsdp_modules, reshard_fsdp_model


try:
    from xformers.ops import fmha
except ImportError:
    raise AssertionError("xFormers is required for training")


logger = logging.getLogger("poa")


class SSLMetaArch(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.arch = cfg.student.arch.split('_')[0]
        self.fp16_scaler = ShardedGradScaler() if cfg.compute_precision.grad_scaler else None

        student_model_dict = dict()
        teacher_model_dict = dict()

        student_backbone, teacher_backbone, max_embed_dim, num_networks = build_model_from_cfg(cfg)
        student_model_dict["backbone"] = student_backbone
        teacher_model_dict["backbone"] = teacher_backbone
        logger.info(f"OPTIONS -- architecture : max_embed_dim: {max_embed_dim}")
        logger.info(f"OPTIONS -- architecture : total number of networks: {num_networks}")
        logger.info("All networks: ")
        for i, (w, d) in enumerate(teacher_backbone.networks):
            logger.info(f"Network-{i}: width--{w}, depth--{d}")
        if cfg.student.pretrained_weights:
            chkpt = torch.load(cfg.student.pretrained_weights)
            logger.info(f"OPTIONS -- pretrained weights: loading from {cfg.student.pretrained_weights}")
            student_backbone.load_state_dict(chkpt["model"], strict=False)

        self.max_embed_dim = max_embed_dim
        self.dino_out_dim = list(cfg.dino.head_n_prototypes)
        self.num_networks = num_networks
        self.network_ids = list(range(num_networks - 1)) # the last network is instact
        random.shuffle(self.network_ids)
        self.nidx = 0
        self.idx = 0

        self.do_dino = cfg.dino.loss_weight > 0
        self.do_koleo = cfg.dino.koleo_loss_weight > 0
        self.do_ibot = cfg.ibot.loss_weight > 0
        self.ibot_separate_head = cfg.ibot.separate_head
        self.lambda_ie = cfg.dino.lambda_ie
        logger.info(f"LOSS -- OFA -- lambda_ie: {self.lambda_ie}")

        logger.info("OPTIONS -- DINO")
        if self.do_dino:
            logger.info(f"OPTIONS -- DINO -- loss_weight: {cfg.dino.loss_weight}")
            logger.info(f"OPTIONS -- DINO -- head_n_prototypes: {cfg.dino.head_n_prototypes}")
            logger.info(f"OPTIONS -- DINO -- head_bottleneck_dim: {cfg.dino.head_bottleneck_dim}")
            logger.info(f"OPTIONS -- DINO -- head_hidden_dim: {cfg.dino.head_hidden_dim}")
            self.dino_loss_weight = cfg.dino.loss_weight
            dino_head = partial(
                MDINOHead,
                max_in_dim=max_embed_dim,
                out_dims=cfg.dino.head_n_prototypes,
                mode=cfg.student.mode,
                hidden_dim=cfg.dino.head_hidden_dim,
                bottleneck_dim=cfg.dino.head_bottleneck_dim,
                nlayers=cfg.dino.head_nlayers,
            )
            self.dino_loss = DINOLoss(self.dino_out_dim)
            if self.do_koleo:
                logger.info("OPTIONS -- DINO -- applying KOLEO regularization")
                self.koleo_loss = KoLeoLoss()
        else:
            logger.info("OPTIONS -- DINO -- not using DINO")

        if self.do_dino or self.do_ibot:
            student_model_dict["dino_head"] = dino_head()
            teacher_model_dict["dino_head"] = dino_head()
            self.with_dino_head = True
        else:
            self.with_dino_head = False

        logger.info("OPTIONS -- IBOT")
        logger.info(f"OPTIONS -- IBOT -- loss_weight: {cfg.ibot.loss_weight}")
        logger.info(f"OPTIONS -- IBOT masking -- ibot_mask_ratio_tuple: {cfg.ibot.mask_ratio_min_max}")
        logger.info(f"OPTIONS -- IBOT masking -- ibot_mask_sample_probability: {cfg.ibot.mask_sample_probability}")
        if self.do_ibot:
            self.ibot_loss_weight = cfg.ibot.loss_weight
            assert max(cfg.ibot.mask_ratio_min_max) > 0, "please provide a positive mask ratio tuple for ibot"
            assert cfg.ibot.mask_sample_probability > 0, "please provide a positive mask probability for ibot"
            self.ibot_out_dim = list(cfg.ibot.head_n_prototypes) if self.ibot_separate_head else list(cfg.dino.head_n_prototypes)
            self.ibot_patch_loss = iBOTPatchLoss(self.ibot_out_dim)
            if self.ibot_separate_head:
                logger.info(f"OPTIONS -- IBOT -- loss_weight: {cfg.ibot.loss_weight}")
                logger.info(f"OPTIONS -- IBOT -- head_n_prototypes: {cfg.ibot.head_n_prototypes}")
                logger.info(f"OPTIONS -- IBOT -- head_bottleneck_dim: {cfg.ibot.head_bottleneck_dim}")
                logger.info(f"OPTIONS -- IBOT -- head_hidden_dim: {cfg.ibot.head_hidden_dim}")
                ibot_head = partial(
                    MDINOHead,
                    max_in_dim=max_embed_dim,
                    out_dims=cfg.ibot.head_n_prototypes,
                    mode=cfg.student.mode,
                    hidden_dim=cfg.ibot.head_hidden_dim,
                    bottleneck_dim=cfg.ibot.head_bottleneck_dim,
                    nlayers=cfg.ibot.head_nlayers,
                )
                student_model_dict["ibot_head"] = ibot_head()
                teacher_model_dict["ibot_head"] = ibot_head()
                self.with_ibot_head = True
            else:
                self.with_ibot_head = False
                logger.info("OPTIONS -- IBOT -- head shared with DINO")

        self.need_to_synchronize_fsdp_streams = True
        self.need_to_set_teacher = True

        self.student = nn.ModuleDict(student_model_dict)
        self.teacher = nn.ModuleDict(teacher_model_dict)

        # there is no backpropagation through the teacher, so no need for gradients
        for p in self.teacher.parameters():
            p.requires_grad = False
        logger.info(f"Student and Teacher are built: they are both {cfg.student.arch} network.")

    def get_network_id(self):
        netowrk_id = self.network_ids[self.nidx]
        if self.nidx == self.num_networks - 2:
            self.nidx = 0
            random.shuffle(self.network_ids)
        else:
            self.nidx += 1
        return netowrk_id

    def forward(self, inputs):
        raise NotImplementedError

    def backprop_loss(self, loss):
        if self.fp16_scaler is not None:
            self.fp16_scaler.scale(loss).backward()
        else:
            loss.backward()

    def forward_backward(self, images, teacher_temp, activate_ibot):
        n_global_crops = 2
        assert n_global_crops == 2
        n_local_crops = self.cfg.crops.local_crops_number
        network_id = self.get_network_id()

        global_crops = images["collated_global_crops"].cuda(non_blocking=True)
        local_crops = images["collated_local_crops"].cuda(non_blocking=True)

        masks = images["collated_masks"].cuda(non_blocking=True)
        mask_indices_list = images["mask_indices_list"].cuda(non_blocking=True)
        n_masked_patches_tensor = images["n_masked_patches"].cuda(non_blocking=True)
        n_masked_patches = mask_indices_list.shape[0]
        upperbound = images["upperbound"]
        masks_weight = images["masks_weight"].cuda(non_blocking=True)

        n_local_crops_loss_terms = max(n_local_crops * n_global_crops, 1)
        n_global_crops_loss_terms = (n_global_crops - 1) * n_global_crops

        do_dino = self.do_dino
        do_ibot = self.do_ibot and activate_ibot

        # set subnetworks
        if self.need_to_set_teacher:
            idx_out = self.teacher.backbone.get_index(self.num_networks - 1)
            if self.with_dino_head:
                self.teacher.dino_head.get_index(idx_out)
            if self.with_ibot_head:
                self.teacher.ibot_head.get_index(idx_out)
            self.need_to_set_teacher = False

        # loss scales
        ibot_loss_scale = 1.0 / n_global_crops

        # teacher output
        @torch.no_grad()
        def get_teacher_output():
            x, n_global_crops_teacher = global_crops, n_global_crops
            teacher_backbone_output_dict = self.teacher.backbone(x, None, is_training=True)
            teacher_cls_tokens = teacher_backbone_output_dict["x_norm_clstoken"]
            teacher_cls_tokens = teacher_cls_tokens.chunk(n_global_crops_teacher)
            # watch out: these are chunked and cat'd in reverse so A is matched to B in the global crops dino loss
            teacher_cls_tokens = torch.cat((teacher_cls_tokens[1], teacher_cls_tokens[0]))
            ibot_teacher_patch_tokens = teacher_backbone_output_dict["x_norm_patchtokens"]
            _dim = ibot_teacher_patch_tokens.shape[-1]
            n_cls_tokens = teacher_cls_tokens.shape[0]

            if do_ibot and not self.ibot_separate_head:
                buffer_tensor_teacher = ibot_teacher_patch_tokens.new_zeros(upperbound + n_cls_tokens, _dim)
                buffer_tensor_teacher[:n_cls_tokens].copy_(teacher_cls_tokens)
                torch.index_select(
                    ibot_teacher_patch_tokens.flatten(0, 1),
                    dim=0,
                    index=mask_indices_list,
                    out=buffer_tensor_teacher[n_cls_tokens : n_cls_tokens + n_masked_patches],
                )
                tokens_after_heads = self.teacher.dino_head(buffer_tensor_teacher)
                teacher_cls_tokens_after_heads = [x[:n_cls_tokens] for x in tokens_after_heads]
                masked_teacher_patch_tokens_after_heads = [
                   x[n_cls_tokens : n_cls_tokens + n_masked_patches] for x in tokens_after_heads
                ]
            elif do_ibot and self.ibot_separate_head:
                buffer_tensor_teacher = ibot_teacher_patch_tokens.new_zeros(upperbound, _dim)
                torch.index_select(
                    ibot_teacher_patch_tokens.flatten(0, 1),
                    dim=0,
                    index=mask_indices_list,
                    out=buffer_tensor_teacher[:n_masked_patches],
                )
                teacher_cls_tokens_after_heads = self.teacher.dino_head(teacher_cls_tokens)
                masked_teacher_patch_tokens_after_heads = [
                    x[:n_masked_patches] for x in self.teacher.ibot_head(buffer_tensor_teacher)
                ]
            else:
                teacher_cls_tokens_after_heads = self.teacher.dino_head(teacher_cls_tokens)
                masked_teacher_ibot_softmaxed_centered = None

            if self.cfg.train.centering == "centering":
                teacher_dino_softmaxed_centered_lists = self.dino_loss.softmax_center_teacher(
                    teacher_cls_tokens_after_heads, teacher_temp=teacher_temp
                )
                teacher_dino_softmaxed_centered_lists = [
                    x.view(n_global_crops_teacher, -1, *x.shape[1:]) for x in teacher_dino_softmaxed_centered_lists
                ]
                self.dino_loss.update_center(teacher_cls_tokens_after_heads)
                if do_ibot:
                    masked_teacher_patch_tokens_after_heads = [
                        masked_teacher_patch_tokens_after_head.unsqueeze(0)[:, :n_masked_patches]
                        for masked_teacher_patch_tokens_after_head in masked_teacher_patch_tokens_after_heads
                    ]
                    masked_teacher_ibot_softmaxed_centereds = self.ibot_patch_loss.softmax_center_teacher(
                        masked_teacher_patch_tokens_after_heads, teacher_temp=teacher_temp
                    )
                    self.ibot_patch_loss.update_center(masked_teacher_patch_tokens_after_heads)
                else:
                    masked_teacher_ibot_softmaxed_centereds = None

            elif self.cfg.train.centering == "sinkhorn_knopp":
                teacher_dino_softmaxed_centered_lists = [self.dino_loss.sinkhorn_knopp_teacher(
                    teacher_cls_tokens_after_head, teacher_temp=teacher_temp
                    ).view(n_global_crops_teacher, -1, *teacher_cls_tokens_after_head.shape[1:])
                    for teacher_cls_tokens_after_head in teacher_cls_tokens_after_heads
                ]
                if do_ibot:
                    masked_teacher_ibot_softmaxed_centereds = [self.ibot_patch_loss.sinkhorn_knopp_teacher(
                        masked_teacher_patch_tokens_after_head, teacher_temp=teacher_temp,
                        n_masked_patches_tensor=n_masked_patches_tensor) for masked_teacher_patch_tokens_after_head
                        in masked_teacher_patch_tokens_after_heads]
                else:
                    masked_teacher_ibot_softmaxed_centereds = None

            else:
                raise NotImplementedError

            return teacher_dino_softmaxed_centered_lists, masked_teacher_ibot_softmaxed_centereds

        def get_student_output(network_id, activate_ibot):
            idx_out = self.student.backbone.get_index(network_id)
            if self.with_dino_head:
                self.student.dino_head.get_index(idx_out)
            if self.with_ibot_head:
                self.student.ibot_head.get_index(idx_out)
            if activate_ibot:
                student_global_backbone_output_dict = self.student.backbone(global_crops, masks=masks, is_training=True)
            else:
                student_global_backbone_output_dict = self.student.backbone(global_crops, masks=None, is_training=True)
            student_local_backbone_output_dict = self.student.backbone(local_crops, masks=None, is_training=True)

            inputs_for_student_head_list = []

            # 1a: local crops cls tokens
            student_local_cls_tokens = student_local_backbone_output_dict["x_norm_clstoken"]
            inputs_for_student_head_list.append(student_local_cls_tokens.unsqueeze(0))

            # 1b: global crops cls tokens
            student_global_cls_tokens = student_global_backbone_output_dict["x_norm_clstoken"]
            inputs_for_student_head_list.append(student_global_cls_tokens.unsqueeze(0))
            student_patch_tokens = student_global_backbone_output_dict["x_norm_patchtokens"]

            # 1c: global crops patch tokens
            if do_ibot:
                _dim = student_global_backbone_output_dict["x_norm_clstoken"].shape[-1]
                ibot_student_patch_tokens = student_global_backbone_output_dict["x_norm_patchtokens"]
                buffer_tensor_patch_tokens = ibot_student_patch_tokens.new_zeros(upperbound, _dim)
                buffer_tensor_patch_tokens[:n_masked_patches].copy_(
                    torch.index_select(ibot_student_patch_tokens.flatten(0, 1), dim=0, index=mask_indices_list)
                )
                if not self.ibot_separate_head:
                    inputs_for_student_head_list.append(buffer_tensor_patch_tokens.unsqueeze(0))
                else:
                    student_global_masked_patch_tokens_after_head = self.student.ibot_head(
                        buffer_tensor_patch_tokens)[
                        :n_masked_patches
                    ]

            # 2: run
            _attn_bias, cat_inputs = fmha.BlockDiagonalMask.from_tensor_list(inputs_for_student_head_list)
            outputs_lists = [_attn_bias.split(x) for x in self.student.dino_head(cat_inputs)]

            # 3a: local crops cls tokens
            student_local_cls_tokens_after_heads = [outputs_list.pop(0).squeeze(0) for outputs_list in outputs_lists]

            # 3b: global crops cls tokens
            student_global_cls_tokens_after_heads = [outputs_list.pop(0).squeeze(0) for outputs_list in outputs_lists]

            # 3c: global crops patch tokens
            if do_ibot and not self.ibot_separate_head:
                student_global_masked_patch_tokens_after_heads = [
                        outputs_list.pop(0).squeeze(0)[:n_masked_patches] for outputs_list in outputs_lists]
            else:
                student_global_masked_patch_tokens_after_heads = None
            return student_local_cls_tokens_after_heads, student_global_cls_tokens_after_heads, \
                    student_global_masked_patch_tokens_after_heads, student_global_cls_tokens

        teacher_dino_softmaxed_centered_lists, masked_teacher_ibot_softmaxed_centereds = get_teacher_output()
        reshard_fsdp_model(self.teacher)
        student_local_cls_tokens_after_heads_1, student_global_cls_tokens_after_heads_1, \
                student_global_masked_patch_tokens_after_heads_1, student_global_cls_tokens_1 \
                = get_student_output(self.num_networks - 1, activate_ibot)
        student_local_cls_tokens_after_heads_2, student_global_cls_tokens_after_heads_2, \
                student_global_masked_patch_tokens_after_heads_2, student_global_cls_tokens_2 \
                = get_student_output(network_id, activate_ibot)

        loss_dict = {}
        loss_accumulator = 0  # for backprop
        if n_local_crops > 0:
            num = len(teacher_dino_softmaxed_centered_lists)
            for i in range(num):
                dino_local_crops_loss = self.dino_loss(
                    student_output_list=student_local_cls_tokens_after_heads_1[i].chunk(n_local_crops),
                    teacher_out_softmaxed_centered_list=teacher_dino_softmaxed_centered_lists[i],
                ) / (n_global_crops_loss_terms + n_local_crops_loss_terms) * self.lambda_ie
                dino_local_crops_loss += self.dino_loss(
                    student_output_list=student_local_cls_tokens_after_heads_2[i].chunk(n_local_crops),
                    teacher_out_softmaxed_centered_list=teacher_dino_softmaxed_centered_lists[i],
                ) / (n_global_crops_loss_terms + n_local_crops_loss_terms) * (1.0 - self.lambda_ie)
                #dino_local_crops_loss /= 2
                # store for display
                loss_dict[f"dino_local_crops_loss_{i}"] = dino_local_crops_loss
                # accumulate loss
                loss_accumulator += self.dino_loss_weight * dino_local_crops_loss / num
                # local distill loss
                vanilla_target = self.dino_loss.vanilla_teacher(student_local_cls_tokens_after_heads_1[i].detach(), 0.1)
                distill_local_crops_loss = self.dino_loss(
                    student_output_list=[student_local_cls_tokens_after_heads_2[i]],
                    teacher_out_softmaxed_centered_list=[vanilla_target],
                )
                loss_dict[f"distill_local_crops_loss_{i}"] = distill_local_crops_loss
                loss_accumulator += (1.0 - self.lambda_ie) * distill_local_crops_loss / num



        # process global crops
        loss_scales = 2  # this is here since we process global crops together

        if do_dino:
            num = len(student_global_cls_tokens_after_heads_1)
            # compute loss
            for i in range(num):
                dino_global_crops_loss = (
                    self.dino_loss(
                        student_output_list=[student_global_cls_tokens_after_heads_1[i]],
                        teacher_out_softmaxed_centered_list=[
                            teacher_dino_softmaxed_centered_lists[i].flatten(0, 1)
                        ],  # these were chunked and stacked in reverse so A is matched to B
                    )
                    * loss_scales
                    / (n_global_crops_loss_terms + n_local_crops_loss_terms)
                ) * self.lambda_ie
                dino_global_crops_loss += (
                    self.dino_loss(
                        student_output_list=[student_global_cls_tokens_after_heads_2[i]],
                        teacher_out_softmaxed_centered_list=[
                            teacher_dino_softmaxed_centered_lists[i].flatten(0, 1)
                        ],  # these were chunked and stacked in reverse so A is matched to B
                    )
                    * loss_scales
                    / (n_global_crops_loss_terms + n_local_crops_loss_terms)
                ) * (1.0 - self.lambda_ie)
                #dino_global_crops_loss /= 2
                # store for display
                loss_dict[f"dino_global_crops_loss_{i}"] = dino_global_crops_loss
                # accumulate loss
                loss_accumulator += self.dino_loss_weight * dino_global_crops_loss / num
                # global distill loss
                vanilla_target = self.dino_loss.vanilla_teacher(student_global_cls_tokens_after_heads_1[i].detach(), 0.1)
                distill_global_crops_loss = self.dino_loss(
                    student_output_list=[student_global_cls_tokens_after_heads_2[i]],
                    teacher_out_softmaxed_centered_list=[vanilla_target],
                )
                loss_dict[f"distill_global_crops_loss_{i}"] = distill_global_crops_loss
                loss_accumulator += (1.0 - self.lambda_ie) * distill_global_crops_loss / num

            if self.do_koleo:
                koleo_loss = self.cfg.dino.koleo_loss_weight * sum(
                    self.koleo_loss(p) for p in student_global_cls_tokens_1.chunk(2)
                )  # we don't apply koleo loss between cls tokens of a same image
                koleo_loss += self.cfg.dino.koleo_loss_weight * sum(
                    self.koleo_loss(p) for p in student_global_cls_tokens_2.chunk(2)
                )  # we don't apply koleo loss between cls tokens of a same image
                koleo_loss /= 2
                loss_accumulator += koleo_loss
                loss_dict["koleo_loss"] = (
                    koleo_loss / loss_scales
                )  # this is to display the same losses as before but we can remove eventually

        if do_ibot:
            # compute loss
            num = len(student_global_masked_patch_tokens_after_heads_1)
            for i in range(num):
                ibot_patch_loss = (
                    self.ibot_patch_loss.forward_masked(
                        student_global_masked_patch_tokens_after_heads_1[i],
                        masked_teacher_ibot_softmaxed_centereds[i],
                        student_masks_flat=masks,
                        n_masked_patches=n_masked_patches,
                        masks_weight=masks_weight,
                    )
                    * loss_scales
                    * ibot_loss_scale
                ) * self.lambda_ie
                ibot_patch_loss += (
                    self.ibot_patch_loss.forward_masked(
                        student_global_masked_patch_tokens_after_heads_2[i],
                        masked_teacher_ibot_softmaxed_centereds[i],
                        student_masks_flat=masks,
                        n_masked_patches=n_masked_patches,
                        masks_weight=masks_weight,
                    )
                    * loss_scales
                    * ibot_loss_scale
                ) * (1.0 - self.lambda_ie)
                #ibot_patch_loss /= 2
                # store for display
                loss_dict[f"ibot_loss_{i}"] = ibot_patch_loss / 2
                # accumulate loss
                loss_accumulator += self.ibot_loss_weight * ibot_patch_loss / num

        self.backprop_loss(loss_accumulator)

        self.fsdp_synchronize_streams()

        return loss_dict

    def fsdp_synchronize_streams(self):
        if self.need_to_synchronize_fsdp_streams:
            torch.cuda.synchronize()
            self.student.dino_head._streams = (
                self.teacher.dino_head._streams
            ) = self.student.backbone._streams = self.teacher.backbone._streams
            self.need_to_synchronize_fsdp_streams = False

    def update_teacher(self, m):
        student_param_list = []
        teacher_param_list = []
        with torch.no_grad():
            for k in self.student.keys():
                for ms, mt in zip(get_fsdp_modules(self.student[k]), get_fsdp_modules(self.teacher[k])):
                    student_param_list += ms.params
                    teacher_param_list += mt.params
            torch._foreach_mul_(teacher_param_list, m)
            torch._foreach_add_(teacher_param_list, student_param_list, alpha=1 - m)

    def train(self):
        super().train()
        self.teacher.eval()

    def get_maybe_fused_params_for_submodel(self, m):
        params_groups = get_params_groups_with_decay(
            model=m,
            lr_decay_rate=self.cfg.optim.layerwise_decay,
            patch_embed_lr_mult=self.cfg.optim.patch_embed_lr_mult,
            arch=self.arch,
        )
        fused_params_groups = fuse_params_groups(params_groups)
        logger.info("fusing param groups")

        for g in fused_params_groups:
            g["foreach"] = True
        return fused_params_groups

    def get_params_groups(self):
        all_params_groups = []
        for m in self.student.values():
            all_params_groups += self.get_maybe_fused_params_for_submodel(m)
        return all_params_groups

    def get_subnetwork(self, network_id):
        with FSDP.summon_full_params(module=self.teacher, writeback=False):
            patch_idx, block_idx_ins, block_idx_outs = self.teacher.backbone.get_index(network_id)
            self.student.backbone.set_index(network_id, patch_idx, block_idx_ins, block_idx_outs)
            if self.with_dino_head:
                idx_in, idx_out = self.teacher.dino_head.get_index(block_idx_outs[-1])
                self.student.dino_head.set_index(idx_in, idx_out)
            if self.with_ibot_head:
                idx_in, idx_out = self.teacher.ibot_head.get_index(block_idx_outs[-1])
                self.student.ibot_head.set_index(idx_in, idx_out)

    def prepare_for_distributed_training(self):
        logger.info("DISTRIBUTED FSDP -- preparing model for distributed training")
        if has_batchnorms(self.student):
            raise NotImplementedError
        # below will synchronize all student subnetworks across gpus:
        for k, v in self.student.items():
            self.teacher[k].load_state_dict(self.student[k].state_dict())
            student_model_cfg = self.cfg.compute_precision.student[k]
            self.student[k] = get_fsdp_wrapper(student_model_cfg)(self.student[k])
            teacher_model_cfg = self.cfg.compute_precision.teacher[k]
            self.teacher[k] = get_fsdp_wrapper(teacher_model_cfg)(self.teacher[k])
