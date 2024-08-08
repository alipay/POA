# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the Apache License, Version 2.0
# found in the LICENSE file in the root directory of this source tree.

import logging
from typing import Dict, Optional

import os
import io
import sys
import json
import math
import subprocess
import numpy as np
import time
import datetime
from collections import deque, defaultdict
import torch.distributed as dist
from numpy import inf
from PIL import Image
from torch import optim as optim
import torch
from torch import nn
from torchmetrics import MetricCollection
import torchvision.transforms as transforms
from timm.data.random_erasing import RandomErasing
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from poa.data import DatasetWithEnumeratedTargets, SamplerType, make_data_loader
from poa.eval.rand_aug import rand_augment_transform
import poa.distributed as distributed
from poa.logging import MetricLogger


logger = logging.getLogger("poa")


class ModelWithNormalize(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, samples):
        return nn.functional.normalize(self.model(samples), dim=1, p=2)


class ModelWithIntermediateLayers(nn.Module):
    def __init__(self, feature_model, n_last_blocks, autocast_ctx):
        super().__init__()
        self.feature_model = feature_model
        self.feature_model.eval()
        self.n_last_blocks = n_last_blocks
        self.autocast_ctx = autocast_ctx

    def forward(self, images):
        with torch.inference_mode():
            with self.autocast_ctx():
                features = self.feature_model.get_intermediate_layers(
                    images, self.n_last_blocks, return_class_token=True
                )
        return features


@torch.inference_mode()
def evaluate(
    model: nn.Module,
    data_loader,
    postprocessors: Dict[str, nn.Module],
    metrics: Dict[str, MetricCollection],
    device: torch.device,
    criterion: Optional[nn.Module] = None,
):
    model.eval()
    if criterion is not None:
        criterion.eval()

    for metric in metrics.values():
        metric = metric.to(device)

    metric_logger = MetricLogger(delimiter="  ")
    header = "Test:"

    for samples, targets, *_ in metric_logger.log_every(data_loader, 10, header):
        outputs = model(samples.to(device))
        targets = targets.to(device)

        if criterion is not None:
            loss = criterion(outputs, targets)
            metric_logger.update(loss=loss.item())

        for k, metric in metrics.items():
            metric_inputs = postprocessors[k](outputs, targets)
            metric.update(**metric_inputs)

    metric_logger.synchronize_between_processes()
    logger.info(f"Averaged stats: {metric_logger}")

    stats = {k: metric.compute() for k, metric in metrics.items()}
    metric_logger_stats = {k: meter.global_avg for k, meter in metric_logger.meters.items()}
    return metric_logger_stats, stats


def all_gather_and_flatten(tensor_rank):
    tensor_all_ranks = torch.empty(
        distributed.get_global_size(),
        *tensor_rank.shape,
        dtype=tensor_rank.dtype,
        device=tensor_rank.device,
    )
    tensor_list = list(tensor_all_ranks.unbind(0))
    torch.distributed.all_gather(tensor_list, tensor_rank.contiguous())
    return tensor_all_ranks.flatten(end_dim=1)


def extract_features(model, net_ids, dataset, batch_size, num_workers, gather_on_cpu=False):
    dataset_with_enumerated_targets = DatasetWithEnumeratedTargets(dataset)
    sample_count = len(dataset_with_enumerated_targets)
    data_loader = make_data_loader(
        dataset=dataset_with_enumerated_targets,
        batch_size=batch_size,
        num_workers=num_workers,
        sampler_type=SamplerType.DISTRIBUTED,
        drop_last=False,
        shuffle=False,
    )
    return extract_features_with_dataloader(model, net_ids, data_loader, sample_count, gather_on_cpu)


@torch.inference_mode()
def extract_features_with_dataloader(model, net_ids, data_loader, sample_count, gather_on_cpu=False):
    gather_device = torch.device("cpu") if gather_on_cpu else torch.device("cuda")
    metric_logger = MetricLogger(delimiter="  ")
    num_nets = len(net_ids)
    features, all_labels = [None] * num_nets, [None] * num_nets
    for samples, (index, labels_rank) in metric_logger.log_every(data_loader, 10):
        samples = samples.cuda(non_blocking=True)
        labels_rank = labels_rank.cuda(non_blocking=True)
        index = index.cuda(non_blocking=True)
        for i, net_id in enumerate(net_ids):
            #model.model.get_index(net_id)
            model.model.get_by_type(net_id)
            features_rank = model(samples).float()
            # init storage feature matrix
            if features[i] is None:
                features[i] = torch.zeros(sample_count, features_rank.shape[-1], device=gather_device)
                labels_shape = list(labels_rank.shape)
                labels_shape[0] = sample_count
                all_labels[i] = torch.full(labels_shape, fill_value=-1, device=gather_device)
                logger.info(f"Storing features into tensor of shape {features[i].shape}")

            # share indexes, features and labels between processes
            index_all = all_gather_and_flatten(index).to(gather_device)
            features_all_ranks = all_gather_and_flatten(features_rank).to(gather_device)
            labels_all_ranks = all_gather_and_flatten(labels_rank).to(gather_device)

            # update storage feature matrix
            if len(index_all) > 0:
                features[i].index_copy_(0, index_all, features_all_ranks)
                all_labels[i].index_copy_(0, index_all, labels_all_ranks)

    logger.info(f"Features shape: {tuple(features[0].shape)}")
    logger.info(f"Labels shape: {tuple(all_labels[0].shape)}")

    for i in range(num_nets):
        assert torch.all(all_labels[i] > -1)

    return features, all_labels


class ModelWithClassifier(nn.Module):

    def __init__(self, feature_model, embed_dim, num_classes=1000, use_multi_stage_feat=False, use_cls=False):
        super().__init__()
        self.feature_model = feature_model
        self.feature_model.use_mean_pooling = not use_cls
        self.head = nn.Linear(embed_dim, num_classes)
        self.use_multi_stage_feat = use_multi_stage_feat
        self.use_cls = use_cls
        if self.use_cls:
            logger.info(f"Using features: x_norm_clstoken")
        else:
            logger.info(f"Using features: x_mean_pooling")

    def forward(self, images):
        if self.use_cls:
            features = self.feature_model.forward_features(images)["x_norm_clstoken"]
        else:
            if self.use_multi_stage_feat:
                features = self.feature_model.forward_multistage_features(images)["x_mean_pooling"]
            else:
                features = self.feature_model.forward_features(images)["x_mean_pooling"]
        logit = self.head(features)
        return logit

    def get_depths(self):
        return self.feature_model.get_depths()

    def no_weight_decay(self):
        names = self.feature_model.no_weight_decay()
        names = ['feature_model.' + name for name in names]
        return names


def get_parameter_groups(
        model, weight_decay=1e-5, skip_list=(), get_num_layer=None, get_layer_scale=None
    ):
    parameter_group_names = {}
    parameter_group_vars = {}

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue  # frozen weights
        if len(param.shape) == 1 or name.endswith(".bias") or name in skip_list:
            group_name = "no_decay"
            this_weight_decay = 0.
        else:
            group_name = "decay"
            this_weight_decay = weight_decay
        if get_num_layer is not None:
            layer_id = get_num_layer(name)
            group_name = "layer_%d_%s" % (layer_id, group_name)
        else:
            layer_id = None

        if group_name not in parameter_group_names:
            if get_layer_scale is not None:
                scale = get_layer_scale(layer_id)
            else:
                scale = 1.

            parameter_group_names[group_name] = {
                "weight_decay": this_weight_decay,
                "params": [],
                "lr_scale": scale
            }
            parameter_group_vars[group_name] = {
                "weight_decay": this_weight_decay,
                "params": [],
                "lr_scale": scale
            }

        parameter_group_vars[group_name]["params"].append(param)
        parameter_group_names[group_name]["params"].append(name)
    print("Param groups = %s" % json.dumps(parameter_group_names, indent=2))
    return list(parameter_group_vars.values())


def create_optimizer(
        args, model, get_num_layer=None, get_layer_scale=None,
        filter_bias_and_bn=True, skip_list=None
    ):
    opt_lower = args.opt.lower()
    weight_decay = args.weight_decay
    if weight_decay and filter_bias_and_bn:
        skip = {}
        if skip_list is not None:
            skip = skip_list
        elif hasattr(model, 'no_weight_decay'):
            skip = model.no_weight_decay()
        parameters = get_parameter_groups(model, weight_decay, skip, get_num_layer, get_layer_scale)
        weight_decay = 0.
    else:
        parameters = model.parameters()

    opt_args = dict(lr=args.learning_rate, weight_decay=weight_decay)
    opt_args['eps'] = 1e-8

    opt_split = opt_lower.split('_')
    opt_lower = opt_split[-1]
    if opt_lower == 'sgd' or opt_lower == 'nesterov':
        opt_args.pop('eps', None)
        optimizer = optim.SGD(parameters, momentum=args.momentum, nesterov=True, **opt_args)
    elif opt_lower == 'momentum':
        opt_args.pop('eps', None)
        optimizer = optim.SGD(parameters, momentum=args.momentum, nesterov=False, **opt_args)
    elif opt_lower == 'adam':
        optimizer = optim.Adam(parameters, **opt_args)
    elif opt_lower == 'adamw':
        optimizer = optim.AdamW(parameters, **opt_args)
    else:
        assert False and "Invalid optimizer"
        raise ValueError

    return optimizer


class LayerDecayValueAssigner(object):

    def __init__(self, values, prefix, net_type, actived_block_idx, depths=None):
        assert net_type in ['swin', 'vit', 'resnet']
        self.values = values
        self.depths = depths
        self.prefix = prefix
        self.net_type = net_type
        if net_type == 'resnet':
            assert isinstance(actived_block_idx, list)
            assert len(actived_block_idx) == 2
            self.block_id_map = []
            for actived_block_idx_i in actived_block_idx:
                self.block_id_map.append({str(x): i for i, x in enumerate(actived_block_idx_i)})
        else:
            self.block_id_map = {str(x): i for i, x in enumerate(actived_block_idx)}

    def get_scale(self, layer_id):
        return self.values[layer_id]

    def get_num_layer_for_resnet(self, var_name, num_max_layer, depths):
        if var_name == f"{self.prefix}.mask_token":
            return 0
        elif var_name.startswith(f"{self.prefix}.conv1"):
            return 0
        elif var_name.startswith(f"{self.prefix}.norm1"):
            return 0
        elif var_name.startswith(f"{self.prefix}.layer"):
            stage_id = int(var_name.split('.')[1].replace('layer', '')) - 1
            if stage_id == 1:
                block_id = self.block_id_map[0].get(var_name.split('.')[3], -1)
            elif stage_id == 2:  
                block_id = self.block_id_map[1].get(var_name.split('.')[3], -1)
            else:
                block_id = int(var_name.split('.')[3])
            layer_id = sum(depths[:stage_id]) + block_id
            if block_id != -1:
                print(f'resnet-{stage_id}-{layer_id}', var_name)
            else:
                return 0 # not activated parameters
            return layer_id + 1
        else:
            return num_max_layer - 1

    def get_num_layer_for_swin(self, var_name, num_max_layer, depths):
        if var_name in (
            f"{self.prefix}.mask_token", f"{self.prefix}.pos_embed"
        ):
            return 0
        elif var_name.startswith(f"{self.prefix}.patch_embed"):
            return 0
        elif var_name.startswith(f"{self.prefix}.stages"):
            stage_id = int(var_name.split('.')[2])
            if stage_id == 2:
                if 'blocks' in var_name:
                    block_id = self.block_id_map.get(var_name.split('.')[4], -1)
                    if block_id != -1:
                        self.cur_block_id = block_id
                else:
                    block_id = self.cur_block_id
            else:
                if 'blocks' in var_name: 
                    block_id = int(var_name.split('.')[4])
                    self.cur_block_id = block_id
                else: 
                    block_id = self.cur_block_id
            layer_id = sum(depths[:stage_id]) + block_id
            if block_id != -1:
                print(f'swin-{layer_id}', var_name)
            else:
                return 0 # not activated parameters
            return layer_id + 1
        else:
            return num_max_layer - 1

    def get_num_layer_for_vit(self, var_name, num_max_layer):
        if var_name in (
            f"{self.prefix}.cls_token", f"{self.prefix}.mask_token", f"{self.prefix}.pos_embed"
        ):
            return 0
        elif var_name.startswith(f"{self.prefix}.patch_embed"):
            return 0
        elif var_name.startswith(f"{self.prefix}.blocks"):
            layer_id = self.block_id_map.get(var_name.split('.')[2], -1)
            return layer_id + 1
        else:
            return num_max_layer - 1

    def get_layer_id(self, var_name):
        if self.net_type == 'swin':
            return self.get_num_layer_for_swin(var_name, len(self.values), self.depths) 
        if self.net_type == 'resnet':
            return self.get_num_layer_for_resnet(var_name, len(self.values), self.depths) 
        if self.net_type == 'vit':
            return self.get_num_layer_for_vit(var_name, len(self.values))   


def cosine_scheduler(base_value, final_value, epochs, niter_per_ep, warmup_epochs=0,
                     start_warmup_value=0, warmup_steps=-1):
    warmup_schedule = np.array([])
    warmup_iters = warmup_epochs * niter_per_ep
    if warmup_steps > 0:
        warmup_iters = warmup_steps
    print("Set warmup steps = %d" % warmup_iters)
    if warmup_epochs > 0:
        warmup_schedule = np.linspace(start_warmup_value, base_value, warmup_iters)

    iters = np.arange(epochs * niter_per_ep - warmup_iters)
    schedule = np.array(
        [final_value + 0.5 * (base_value - final_value) * (1 + math.cos(math.pi * i / (len(iters)))) for i in iters])

    schedule = np.concatenate((warmup_schedule, schedule))

    assert len(schedule) == epochs * niter_per_ep
    return schedule


class SmoothedValue(object):
    """Track a series of values and provide access to smoothed values over a
    window or the global series average.
    """

    def __init__(self, window_size=20, fmt=None):
        if fmt is None:
            fmt = "{median:.4f} ({global_avg:.4f})"
        self.deque = deque(maxlen=window_size)
        self.total = 0.0
        self.count = 0
        self.fmt = fmt

    def update(self, value, n=1):
        self.deque.append(value)
        self.count += n
        self.total += value * n

    def synchronize_between_processes(self):
        """
        Warning: does not synchronize the deque!
        """
        t = torch.tensor([self.count, self.total], dtype=torch.float64, device='cuda')
        dist.barrier()
        dist.all_reduce(t)
        t = t.tolist()
        self.count = int(t[0])
        self.total = t[1]

    @property
    def median(self):
        d = torch.tensor(list(self.deque))
        return d.median().item()

    @property
    def avg(self):
        d = torch.tensor(list(self.deque), dtype=torch.float32)
        return d.mean().item()

    @property
    def global_avg(self):
        return self.total / self.count

    @property
    def max(self):
        return max(self.deque)

    @property
    def value(self):
        return self.deque[-1]

    def __str__(self):
        return self.fmt.format(
            median=self.median,
            avg=self.avg,
            global_avg=self.global_avg,
            max=self.max,
            value=self.value)


class NativeScalerWithGradNormCount:
    state_dict_key = "amp_scaler"

    def __init__(self):
        self._scaler = torch.cuda.amp.GradScaler()

    def __call__(self, loss, optimizer, clip_grad=None, parameters=None, create_graph=False, update_grad=True):
        self._scaler.scale(loss).backward(create_graph=create_graph)
        if update_grad:
            if clip_grad is not None:
                assert parameters is not None
                self._scaler.unscale_(optimizer)  # unscale the gradients of optimizer's assigned params in-place
                norm = torch.nn.utils.clip_grad_norm_(parameters, clip_grad)
            else:
                self._scaler.unscale_(optimizer)
                norm = self.get_grad_norm_(parameters)
            self._scaler.step(optimizer)
            self._scaler.update()
        else:
            norm = None
        return norm

    def state_dict(self):
        return self._scaler.state_dict()

    def load_state_dict(self, state_dict):
        self._scaler.load_state_dict(state_dict)

    def get_grad_norm_(self, parameters, norm_type: float = 2.0) -> torch.Tensor:
        if isinstance(parameters, torch.Tensor):
            parameters = [parameters]
        parameters = [p for p in parameters if p.grad is not None]
        norm_type = float(norm_type)
        if len(parameters) == 0:
            return torch.tensor(0.)
        device = parameters[0].grad.device
        if norm_type == inf:
            total_norm = max(p.grad.detach().abs().max().to(device) for p in parameters)
        else:
            total_norm = torch.norm(torch.stack(
                [torch.norm(p.grad.detach(), norm_type).to(device) for p in parameters]), norm_type)
        return total_norm


class MetricLogger2(object):
    def __init__(self, delimiter="\t"):
        self.meters = defaultdict(SmoothedValue)
        self.delimiter = delimiter

    def update(self, **kwargs):
        for k, v in kwargs.items():
            if v is None:
                continue
            if isinstance(v, torch.Tensor):
                v = v.item()
            assert isinstance(v, (float, int))
            self.meters[k].update(v)

    def __getattr__(self, attr):
        if attr in self.meters:
            return self.meters[attr]
        if attr in self.__dict__:
            return self.__dict__[attr]
        raise AttributeError("'{}' object has no attribute '{}'".format(
            type(self).__name__, attr))

    def __str__(self):
        loss_str = []
        for name, meter in self.meters.items():
            loss_str.append(
                "{}: {}".format(name, str(meter))
            )
        return self.delimiter.join(loss_str)

    def synchronize_between_processes(self):
        for meter in self.meters.values():
            meter.synchronize_between_processes()

    def add_meter(self, name, meter):
        self.meters[name] = meter

    def log_every(self, iterable, print_freq, header=None):
        i = 0
        if not header:
            header = ''
        start_time = time.time()
        end = time.time()
        iter_time = SmoothedValue(fmt='{avg:.4f}')
        data_time = SmoothedValue(fmt='{avg:.4f}')
        space_fmt = ':' + str(len(str(len(iterable)))) + 'd'
        log_msg = [
            header,
            '[{0' + space_fmt + '}/{1}]',
            'eta: {eta}',
            '{meters}',
            'time: {time}',
            'data: {data}'
        ]
        if torch.cuda.is_available():
            log_msg.append('max mem: {memory:.0f}')
        log_msg = self.delimiter.join(log_msg)
        MB = 1024.0 * 1024.0
        for obj in iterable:
            data_time.update(time.time() - end)
            yield obj
            iter_time.update(time.time() - end)
            if i % print_freq == 0 or i == len(iterable) - 1:
                eta_seconds = iter_time.global_avg * (len(iterable) - i)
                eta_string = str(datetime.timedelta(seconds=int(eta_seconds)))
                if torch.cuda.is_available():
                    print(log_msg.format(
                        i, len(iterable), eta=eta_string,
                        meters=str(self),
                        time=str(iter_time), data=str(data_time),
                        memory=torch.cuda.max_memory_allocated() / MB))
                else:
                    print(log_msg.format(
                        i, len(iterable), eta=eta_string,
                        meters=str(self),
                        time=str(iter_time), data=str(data_time)))
            i += 1
            end = time.time()
        total_time = time.time() - start_time
        total_time_str = str(datetime.timedelta(seconds=int(total_time)))
        print('{} Total time: {} ({:.4f} s / it)'.format(
            header, total_time_str, total_time / len(iterable)))


def get_logger(file_path_name):
    logger = logging.getLogger()
    logger.setLevel('INFO')
    BASIC_FORMAT = "%(levelname)s:%(message)s"
    DATE_FORMAT = ''
    formatter = logging.Formatter(BASIC_FORMAT, DATE_FORMAT)
    chlr = logging.StreamHandler()
    chlr.setFormatter(formatter)
    chlr.setLevel('INFO')
    fhlr = logging.FileHandler(file_path_name)
    fhlr.setFormatter(formatter)
    logger.addHandler(chlr)
    logger.addHandler(fhlr)
    return logger


def setup_for_distributed(is_master):
    """
    This function disables printing when not in master process
    """
    import builtins as __builtin__
    builtin_print = __builtin__.print

    def print(*args, **kwargs):
        force = kwargs.pop('force', False)
        if is_master or force:
            builtin_print(*args, **kwargs)

    __builtin__.print = print



def _load_checkpoint_for_ema(model_ema, checkpoint):
    """
    Workaround for ModelEma._load_checkpoint to accept an already-loaded object
    """
    mem_file = io.BytesIO()
    torch.save(checkpoint, mem_file)
    mem_file.seek(0)
    model_ema._load_checkpoint(mem_file)


def _pil_interp(method):
    if method == 'bicubic':
        return Image.BICUBIC
    elif method == 'lanczos':
        return Image.LANCZOS
    elif method == 'hamming':
        return Image.HAMMING
    else:
        # default bilinear, do we want to allow nearest?
        return Image.BILINEAR


def strong_transforms(
    img_size=224,
    scale=(0.08, 1.0),
    ratio=(0.75, 1.3333333333333333),
    hflip=0.5,
    vflip=0.0,
    color_jitter=0.4,
    auto_augment="rand-m9-mstd0.5-inc1",
    interpolation="random",
    use_prefetcher=True,
    mean=IMAGENET_DEFAULT_MEAN,  # (0.485, 0.456, 0.406)
    std=IMAGENET_DEFAULT_STD,  # (0.229, 0.224, 0.225)
    re_prob=0.25,
    re_mode="pixel",
    re_count=1,
    re_num_splits=0,
    color_aug=False,
    strong_ratio=0.45,
):
    """
    for use in a mixing dataset that passes
     * all data through the first (primary) transform, called the 'clean' data
     * a portion of the data through the secondary transform
     * normalizes and converts the branches above with the third, final transform
    """

    scale = tuple(scale or (0.08, 1.0))  # default imagenet scale range
    ratio = tuple(ratio or (3.0 / 4.0, 4.0 / 3.0))  # default imagenet ratio range

    primary_tfl = []
    if hflip > 0.0:
        primary_tfl += [transforms.RandomHorizontalFlip(p=hflip)]
    if vflip > 0.0:
        primary_tfl += [transforms.RandomVerticalFlip(p=vflip)]

    secondary_tfl = []
    if auto_augment:
        assert isinstance(auto_augment, str)
        if isinstance(img_size, tuple):
            img_size_min = min(img_size)
        else:
            img_size_min = img_size
        aa_params = dict(
            translate_const=int(img_size_min * strong_ratio),
            img_mean=tuple([min(255, round(255 * x)) for x in mean]),
        )
        if interpolation and interpolation != "random":
            aa_params["interpolation"] = _pil_interp(interpolation)
        if auto_augment.startswith("rand"):
            secondary_tfl += [rand_augment_transform(auto_augment, aa_params)]
    if color_jitter is not None and color_aug:
        # color jitter is enabled when not using AA
        flip_and_color_jitter = [
            transforms.RandomApply(
                [
                    transforms.ColorJitter(
                        brightness=0.4, contrast=0.4, saturation=0.2, hue=0.1
                    )
                ],
                p=0.8,
            ),
            transforms.RandomGrayscale(p=0.2),
        ]
        secondary_tfl += flip_and_color_jitter

    if interpolation == "random":
        interpolation = (Image.BILINEAR, Image.BICUBIC)
    else:
        interpolation = _pil_interp(interpolation)
    final_tfl = [
        transforms.RandomResizedCrop(
            size=img_size, scale=scale, ratio=ratio, interpolation=Image.BICUBIC
        )
    ]
    if use_prefetcher:
        # prefetcher and collate will handle tensor conversion and norm
        final_tfl += [transforms.ToTensor()]
    else:
        final_tfl += [
            transforms.ToTensor(),
            transforms.Normalize(mean=torch.tensor(mean), std=torch.tensor(std)),
        ]
    if re_prob > 0.0:
        final_tfl.append(
            RandomErasing(
                re_prob,
                mode=re_mode,
                max_count=re_count,
                num_splits=re_num_splits,
                device="cpu",
            )
        )
    return transforms.Compose(primary_tfl + secondary_tfl + final_tfl)


def restart_from_checkpoint(ckp_path, run_variables=None, **kwargs):
    """
    Re-start from checkpoint
    """
    if not os.path.isfile(ckp_path):
        return
    print("Found checkpoint at {}".format(ckp_path))

    # open checkpoint file
    checkpoint = torch.load(ckp_path, map_location="cpu")
    # key is what to look for in the checkpoint file
    # value is the object to load
    # example: {'state_dict': model}
    for key, value in kwargs.items():
        if key in checkpoint and value is not None:
            try:
                msg = value.load_state_dict(checkpoint[key], strict=False)
                print(
                    "=> loaded '{}' from checkpoint '{}' with msg {}".format(
                        key, ckp_path, msg
                    )
                )
            except TypeError:
                try:
                    msg = value.load_state_dict(checkpoint[key])
                    print("=> loaded '{}' from checkpoint: '{}'".format(key, ckp_path))
                except ValueError:
                    print(
                        "=> failed to load '{}' from checkpoint: '{}'".format(
                            key, ckp_path
                        )
                    )
        else:
            print("=> key '{}' not found in checkpoint: '{}'".format(key, ckp_path))

    # reload variable important for the run
    if run_variables is not None:
        for var_name in run_variables:
            if var_name in checkpoint:
                run_variables[var_name] = checkpoint[var_name]


def accuracy(output, target, topk=(1,)):
    """
    Computes the accuracy over the k top predictions for the specified values of k
    """
    maxk = max(topk)
    batch_size = target.size(0)
    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.reshape(1, -1).expand_as(pred))
    return [correct[:k].reshape(-1).float().sum(0) * 100.0 / batch_size for k in topk]
