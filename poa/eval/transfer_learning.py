# Copyright 2024 Ant Group.
import argparse
import datetime
import logging
import time
import torch
import torch.backends.cudnn as cudnn
import json
import os
import math
import sys
import numpy as np
import torch.distributed as dist
from pathlib import Path
from typing import Iterable, Optional
from torch.nn import functional as F
from timm.models import create_model
from timm.loss import LabelSmoothingCrossEntropy, SoftTargetCrossEntropy
from timm.scheduler import create_scheduler
#from timm.optim import create_optimizer
from timm.utils import NativeScaler, get_state_dict, ModelEma, accuracy
from timm.data import Mixup
from transfer_datasets import build_dataset, RASampler

from poa.eval.utils import MetricLogger2 as MetricLogger
from poa.eval.setup import get_args_parser as get_setup_args_parser
from poa.eval.setup import setup_and_build_model
from poa.eval.utils import (SmoothedValue, get_logger, ModelWithClassifier, _load_checkpoint_for_ema,
                            create_optimizer, LayerDecayValueAssigner)
logger = logging.getLogger("poa")


def get_args_parser():
    setup_args_parser = get_setup_args_parser(add_help=False)
    parents = [setup_args_parser]
    parser = argparse.ArgumentParser(
        description='Transfer Learning Evaluation',
        parents=parents,
        add_help=False
    )
    parser.add_argument('--batch-size', default=128, type=int)
    parser.add_argument('--epochs', default=300, type=int)
    parser.add_argument('--no-resume', action='store_true', help='do not resume from checkpoint')

    # Model parameters
    parser.add_argument("--net-type", type=str,
        help="Network type for vit or swin, for example: samll, base, large",
    )
    parser.add_argument('--input-size', default=224, type=int, help='images input size')
    parser.add_argument('--model-ema', action='store_true')
    parser.add_argument('--no-model-ema', action='store_false', dest='model_ema')
    parser.set_defaults(model_ema=True)
    parser.add_argument('--model-ema-decay', type=float, default=0.7, help='')
    parser.add_argument('--model-ema-force-cpu', action='store_true', default=False, help='')

    # Optimizer parameters
    parser.add_argument('--opt', default='adamw', type=str, metavar='OPTIMIZER',
                        help='Optimizer (default: "adamw"')
    parser.add_argument('--opt-eps', default=1e-8, type=float, metavar='EPSILON',
                        help='Optimizer Epsilon (default: 1e-8)')
    parser.add_argument('--opt-betas', default=None, type=float, nargs='+', metavar='BETA',
                        help='Optimizer Betas (default: None, use opt default)')
    parser.add_argument('--clip-grad', type=float, default=None, metavar='NORM',
                        help='Clip gradient norm (default: None, no clipping)')
    parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                        help='SGD momentum (default: 0.9)')
    parser.add_argument('--weight-decay', type=float, default=0.05,
                        help='weight decay (default: 0.05)')
    # Learning rate schedule parameters
    parser.add_argument('--sched', default='cosine', type=str, metavar='SCHEDULER',
                        help='LR scheduler (default: "cosine"')
    parser.add_argument('--learning-rate', type=float, default=5e-4, metavar='LR',
                        help='learning rate (default: 5e-4)')
    parser.add_argument('--lr-noise', type=float, nargs='+', default=None, metavar='pct, pct',
                        help='learning rate noise on/off epoch percentages')
    parser.add_argument('--lr-noise-pct', type=float, default=0.67, metavar='PERCENT',
                        help='learning rate noise limit percent (default: 0.67)')
    parser.add_argument('--lr-noise-std', type=float, default=1.0, metavar='STDDEV',
                        help='learning rate noise std-dev (default: 1.0)')
    parser.add_argument('--warmup-lr', type=float, default=1e-6, metavar='LR',
                        help='warmup learning rate (default: 1e-6)')
    parser.add_argument('--min-lr', type=float, default=1e-5, metavar='LR',
                        help='lower lr bound for cyclic schedulers that hit 0 (1e-5)')

    parser.add_argument('--decay-epochs', type=float, default=30, metavar='N',
                        help='epoch interval to decay LR')
    parser.add_argument('--warmup-epochs', type=int, default=5, metavar='N',
                        help='epochs to warmup LR, if scheduler supports')
    parser.add_argument('--cooldown-epochs', type=int, default=10, metavar='N',
                        help='epochs to cooldown LR at min_lr, after cyclic schedule ends')
    parser.add_argument('--patience-epochs', type=int, default=10, metavar='N',
                        help='patience epochs for Plateau LR scheduler (default: 10')
    parser.add_argument('--decay-rate', '--dr', type=float, default=0.1, metavar='RATE',
                        help='LR decay rate (default: 0.1)')

    # Augmentation parameters
    parser.add_argument('--color-jitter', type=float, default=0.4, metavar='PCT',
                        help='Color jitter factor (default: 0.4)')
    parser.add_argument('--aa', type=str, default='rand-m9-mstd0.5-inc1', metavar='NAME',
                        help='Use AutoAugment policy. "v0" or "original". " + \
                             "(default: rand-m9-mstd0.5-inc1)'),
    parser.add_argument('--smoothing', type=float, default=0.1,
                        help='Label smoothing (default: 0.1)')
    parser.add_argument('--train-interpolation', type=str, default='bicubic',
                        help='Training interpolation (random, bilinear, bicubic default: "bicubic")')

    parser.add_argument('--repeated-aug', action='store_true')
    parser.add_argument('--no-repeated-aug', action='store_false', dest='repeated_aug')
    parser.set_defaults(repeated_aug=True)

    # * Random Erase params
    parser.add_argument('--reprob', type=float, default=0.25, metavar='PCT',
                        help='Random erase prob (default: 0.25)')
    parser.add_argument('--remode', type=str, default='pixel',
                        help='Random erase mode (default: "pixel")')
    parser.add_argument('--recount', type=int, default=1,
                        help='Random erase count (default: 1)')
    parser.add_argument('--resplit', action='store_true', default=False,
                        help='Do not random erase first (clean) augmentation split')

    # * Mixup params
    parser.add_argument('--mixup', type=float, default=0.8,
                        help='mixup alpha, mixup enabled if > 0. (default: 0.8)')
    parser.add_argument('--cutmix', type=float, default=1.0,
                        help='cutmix alpha, cutmix enabled if > 0. (default: 1.0)')
    parser.add_argument('--cutmix-minmax', type=float, nargs='+', default=None,
                        help='cutmix min/max ratio, overrides alpha and enables cutmix if set (default: None)')
    parser.add_argument('--mixup-prob', type=float, default=1.0,
                        help='Probability of performing mixup or cutmix when either/both is enabled')
    parser.add_argument('--mixup-switch-prob', type=float, default=0.5,
                        help='Probability of switching to cutmix when both mixup and cutmix enabled')
    parser.add_argument('--mixup-mode', type=str, default='batch',
                        help='How to apply mixup/cutmix params. Per "batch", "pair", or "elem"')

    # Distillation parameters
    parser.add_argument('--teacher-model', default='regnety_160', type=str, metavar='MODEL',
                        help='Name of teacher model to train (default: "regnety_160"')
    parser.add_argument('--teacher-path', type=str, default='')
    parser.add_argument('--distillation-type', default='none', choices=['none', 'soft', 'hard'],
                        type=str, help="")
    parser.add_argument('--distillation-alpha', default=0.5, type=float, help="")
    parser.add_argument('--distillation-tau', default=1.0, type=float, help="")

    # Dataset parameters
    parser.add_argument('--data_path', default='/home/aiops/zhoupan/code/dataset/flowers', type=str,
                        help='dataset path')
    parser.add_argument('--data_set', default='flowers',
                        choices=['CIFAR', 'Cars', 'CIFAR10', 'IMNET', 'INAT19', 'INAT', 'flowers',
                                 'image_folder'],
                        type=str, help='ImageNet dataset path')
    parser.add_argument('--inat-category', default='name',
                        choices=['kingdom', 'phylum', 'class', 'order', 'supercategory', 'family',
                                 'genus', 'name'],
                        type=str, help='semantic granularity')
    parser.add_argument('--device', default='cuda',
                        help='device to use for training / testing')
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--resume', default='checkpoint.pth', help='resume from checkpoint')
    parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                        help='start epoch')
    parser.add_argument('--eval', action='store_true', help='Perform evaluation only')
    parser.add_argument('--dist-eval', action='store_true', default=False,
                        help='Enabling distributed evaluation')
    parser.add_argument('--num_workers', default=16, type=int)
    parser.add_argument('--pin-mem', action='store_true',
                        help='Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU.')
    parser.add_argument('--no-pin-mem', action='store_false', dest='pin_mem',
                        help='')
    parser.set_defaults(pin_mem=True)
    parser.add_argument("--local-rank", default=0, type=int, help="Please ignore and do not set this argument.")

    parser.add_argument('--configure', type=str, default="job.yaml")
    parser.add_argument('--layer-decay', type=float, default=1.0)
    parser.add_argument('--drop-path', type=float, default=0.1)
    return parser


class DistillationLoss(torch.nn.Module):
    """
    This module wraps a standard criterion and adds an extra knowledge distillation loss by
    taking a teacher model prediction and using it as additional supervision.
    """

    def __init__(self, base_criterion: torch.nn.Module, teacher_model: torch.nn.Module,
                 distillation_type: str, alpha: float, tau: float):
        super().__init__()
        self.base_criterion = base_criterion
        self.teacher_model = teacher_model
        assert distillation_type in ['none', 'soft', 'hard']
        self.distillation_type = distillation_type
        self.alpha = alpha
        self.tau = tau

    def forward(self, inputs, outputs, labels):
        """
        Args:
            inputs: The original inputs that are feed to the teacher model
            outputs: the outputs of the model to be trained. It is expected to be
                either a Tensor, or a Tuple[Tensor, Tensor], with the original output
                in the first position and the distillation predictions as the second output
            labels: the labels for the base criterion
        """
        outputs_kd = None
        if not isinstance(outputs, torch.Tensor):
            # assume that the model outputs a tuple of [outputs, outputs_kd]
            outputs, outputs_kd = outputs
        base_loss = self.base_criterion(outputs, labels)
        if self.distillation_type == 'none':
            return base_loss

        if outputs_kd is None:
            raise ValueError("When knowledge distillation is enabled, the model is "
                             "expected to return a Tuple[Tensor, Tensor] with the output of the "
                             "class_token and the dist_token")
        # don't backprop throught the teacher
        with torch.no_grad():
            teacher_outputs = self.teacher_model(inputs)

        if self.distillation_type == 'soft':
            T = self.tau
            # taken from https://github.com/peterliht/knowledge-distillation-pytorch/blob/master/model/net.py#L100
            # with slight modifications
            distillation_loss = F.kl_div(
                F.log_softmax(outputs_kd / T, dim=1),
                F.log_softmax(teacher_outputs / T, dim=1),
                reduction='sum',
                log_target=True
            ) * (T * T) / outputs_kd.numel()
        elif self.distillation_type == 'hard':
            distillation_loss = F.cross_entropy(outputs_kd, teacher_outputs.argmax(dim=1))

        loss = base_loss * (1 - self.alpha) + distillation_loss * self.alpha
        return loss


def train_one_epoch(model: torch.nn.Module, criterion: DistillationLoss,
                    data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, loss_scaler, max_norm: float = 0,
                    model_ema: Optional[ModelEma] = None, mixup_fn: Optional[Mixup] = None,
                    set_training_mode=True):
    model.train(set_training_mode)
    metric_logger = MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 20
    # pdb.set_trace()
    for samples, targets in metric_logger.log_every(data_loader, print_freq, header):
        samples = samples.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        if mixup_fn is not None:
            samples, targets = mixup_fn(samples, targets)

        with torch.cuda.amp.autocast():
            outputs = model(samples)
            #outputs = model.module.head(output)
            loss = criterion(samples, outputs, targets)

        loss_value = loss.item()

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            sys.exit(1)

        optimizer.zero_grad()

        # this attribute is added by timm on one optimizer (adahessian)
        is_second_order = hasattr(optimizer, 'is_second_order') and optimizer.is_second_order
        loss_scaler(loss, optimizer, clip_grad=max_norm,
                    parameters=model.parameters(), create_graph=is_second_order)

        torch.cuda.synchronize()
        if model_ema is not None:
            model_ema.update(model)

        metric_logger.update(loss=loss_value)
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])
    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


@torch.no_grad()
def evaluate(data_loader, model, device):
    criterion = torch.nn.CrossEntropyLoss()

    metric_logger = MetricLogger(delimiter="  ")
    header = 'Test:'

    # switch to evaluation mode
    model.eval()
    # pdb.set_trace()
    for images, target in metric_logger.log_every(data_loader, 10, header):
        images = images.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True, dtype=torch.long)

        # compute output
        with torch.cuda.amp.autocast():
            output = model(images)
            #output = model.module.head(output)
            loss = criterion(output, target)

        acc1, acc5 = accuracy(output, target, topk=(1, 5))

        batch_size = images.shape[0]
        metric_logger.update(loss=loss.item())
        metric_logger.meters['acc1'].update(acc1.item(), n=batch_size)
        metric_logger.meters['acc5'].update(acc5.item(), n=batch_size)
    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print('* Acc@1 {top1.global_avg:.3f} Acc@5 {top5.global_avg:.3f} loss {losses.global_avg:.3f}'
          .format(top1=metric_logger.acc1, top5=metric_logger.acc5, losses=metric_logger.loss))

    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


def main(args, logger):

    if args.distillation_type != 'none' and not args.eval:
        raise NotImplementedError("Finetuning with distillation not yet supported")
    device = torch.device(args.device)

    #setup and build model
    drop_path_dict = {'small': 0.1, 'base': 0.2}
    args.drop_path = drop_path_dict[args.net_type]
    feature_model, _ = setup_and_build_model(args, is_train=True)

    # fix the seed for reproducibility
    seed=1024
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)

    # training setting (follow iBot)
    epochs_dict = dict(
        CIFAR10=1000,
        CIFAR=1000,
        INAT=360,
        INAT19=360,
        flowers=1000,
        Cars=1000,
    )
    args.epochs = epochs_dict[args.data_set]
    lr_dict = dict(
        CIFAR10={'small': 8.0e-6, 'base': 7.5e-6},
        CIFAR={'small': 9.0e-6, 'base': 7.5e-6},
        INAT={'small': 4.0e-4, 'base': 2.0e-4},
        INAT19={'small': 7.0e-5, 'base': 4.0e-5},
        flowers={'small': 6.0e-4, 'base': 5.0e-4},
        Cars={'small': 4.0e-4, 'base': 3.0e-4},
    )
    args.learning_rate = lr_dict[args.data_set][args.net_type]
    layer_decay_dict = dict(
        CIFAR10={'small': 0.9, 'base': 0.9},
        CIFAR={'small': 0.85, 'base': 0.8},
        INAT={'small': 0.8, 'base': 0.75},
        INAT19={'small': 0.85, 'base': 0.75},
        flowers={'small': 0.60, 'base': 0.7},
        Cars={'small': 0.8, 'base': 0.7},
    )
    args.layer_decay = layer_decay_dict[args.data_set][args.net_type]
    
    # dataset path setting
    data_path_dict = dict(
        CIFAR10='/gruntdata/rs_nas/workspace/qichu.zyy/Dataset/CIFAR_10',
        CIFAR='/gruntdata/rs_nas/workspace/qichu.zyy/Dataset/CIFAR_100',
        INAT='/gruntdata/rs_nas/workspace/qichu.zyy/Dataset/INAT_18',
        INAT19='/gruntdata/rs_nas/workspace/qichu.zyy/Dataset/INAT_19',
        flowers='/gruntdata/rs_nas/workspace/qichu.zyy/Dataset/Flowers',
        Cars='/gruntdata/rs_nas/workspace/qichu.zyy/Dataset/Cars',
    )
    args.data_path = data_path_dict[args.data_set]
    dataset_train, args.nb_classes = build_dataset(is_train=True, args=args)
    dataset_val, _ = build_dataset(is_train=False, args=args)
    logger.info("training sample %d, test sampels %d" % (len(dataset_train), len(dataset_val)))

    # network setting
    feature_model.get_by_type(args.net_type)
    embed_dim = feature_model.feat_dim
    actived_block_idx = feature_model.block_idx
    model = ModelWithClassifier(feature_model, embed_dim, args.nb_classes, use_cls=True)
    model.to(device)
    cudnn.benchmark = True
    if dist.get_rank() == 0:
        print(args)

    num_tasks = dist.get_world_size()
    global_rank = dist.get_rank()
    if args.repeated_aug:
        sampler_train = RASampler(
            dataset_train, num_replicas=num_tasks, rank=global_rank, shuffle=True
        )
    else:
        sampler_train = torch.utils.data.DistributedSampler(
            dataset_train, num_replicas=num_tasks, rank=global_rank, shuffle=True
        )
    if args.dist_eval:
        if len(dataset_val) % num_tasks != 0:
            print(
                'Warning: Enabling distributed evaluation with an eval dataset not divisible by process number. '
                'This will slightly alter validation results as extra duplicate entries are added to achieve '
                'equal num of samples per-process.')
        sampler_val = torch.utils.data.DistributedSampler(
            dataset_val, num_replicas=num_tasks, rank=global_rank, shuffle=False)
    else:
        sampler_val = torch.utils.data.SequentialSampler(dataset_val)

    data_loader_train = torch.utils.data.DataLoader(
        dataset_train, sampler=sampler_train,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=False,
    )
    data_loader_val = torch.utils.data.DataLoader(
        dataset_val, sampler=sampler_val,
        batch_size=int(1.5 * args.batch_size),
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=False
    )
    # logger.info("training data %d, test data %d"%(len(data_loader_train), len(data_loader_val)))
    mixup_fn = None
    mixup_active = args.mixup > 0 or args.cutmix > 0. or args.cutmix_minmax is not None
    if mixup_active:
        mixup_fn = Mixup(
            mixup_alpha=args.mixup, cutmix_alpha=args.cutmix, cutmix_minmax=args.cutmix_minmax,
            prob=args.mixup_prob, switch_prob=args.mixup_switch_prob, mode=args.mixup_mode,
            label_smoothing=args.smoothing, num_classes=args.nb_classes)



    model_ema = None
    if args.model_ema:
        # Important to create EMA model after cuda(), DP wrapper, and AMP but before SyncBN and DDP wrapper
        model_ema = ModelEma(
            model,
            decay=args.model_ema_decay,
            device='cpu' if args.model_ema_force_cpu else '',
            resume='')
    model = torch.nn.parallel.DistributedDataParallel(model, find_unused_parameters=True)
    model_without_ddp = model.module
    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info('number of params: %d' % n_parameters)
    linear_scaled_lr = args.learning_rate * args.batch_size * dist.get_world_size() / 512.0
    args.learning_rate = linear_scaled_lr

    num_layers_dict = {'small': 12, 'base': 12}
    num_layers = num_layers_dict[args.net_type]
    if args.layer_decay < 1.0:
        assigner = LayerDecayValueAssigner(
            list(args.layer_decay ** (num_layers + 1 - i) for i in range(num_layers + 2)), 
            prefix = 'feature_model', net_type='vit', actived_block_idx=actived_block_idx
        )
    else:
        assigner = None

    skip_weight_decay_list = model_without_ddp.no_weight_decay()
    optimizer = create_optimizer(
        args, model_without_ddp, skip_list=skip_weight_decay_list,
        get_num_layer=assigner.get_layer_id if assigner is not None else None,
        get_layer_scale=assigner.get_scale if assigner is not None else None)
    #optimizer = create_optimizer(args, model_without_ddp)

    loss_scaler = NativeScaler()
    lr_scheduler, _ = create_scheduler(args, optimizer)

    if args.mixup > 0.:
        # smoothing is handled with mixup label transform
        criterion = SoftTargetCrossEntropy()
    elif args.smoothing:
        criterion = LabelSmoothingCrossEntropy(smoothing=args.smoothing)
    else:
        criterion = torch.nn.CrossEntropyLoss()

    teacher_model = None
    if args.distillation_type != 'none':
        assert args.teacher_path, 'need to specify teacher-path when using distillation'
        logger.info(f"Creating teacher model: {args.teacher_model}")
        teacher_model = create_model(
            args.teacher_model,
            pretrained=False,
            num_classes=args.nb_classes,
            global_pool='avg',
        )
        if args.teacher_path.startswith('https'):
            checkpoint = torch.hub.load_state_dict_from_url(
                args.teacher_path, map_location='cpu', check_hash=True)
        else:
            checkpoint = torch.load(args.teacher_path, map_location='cpu')
        teacher_model.load_state_dict(checkpoint['model'])
        teacher_model.to(device)
        teacher_model.eval()
    # wrap the criterion in our custom DistillationLoss, which
    # just dispatches to the original criterion if args.distillation_type is 'none'
    criterion = DistillationLoss(
        criterion, teacher_model, args.distillation_type, args.distillation_alpha,
        args.distillation_tau
    )

    # pdb.set_trace()
    output_dir = Path(args.output_dir)
    if args.resume and not args.no_resume:
        checkpoint_path = os.path.join(output_dir, args.resume)
        if os.path.exists(checkpoint_path):
            checkpoint = torch.load(checkpoint_path, map_location='cpu')
            model_without_ddp.load_state_dict(checkpoint['model'])
            if not args.eval and 'optimizer' in checkpoint and 'lr_scheduler' in checkpoint and 'epoch' in checkpoint:
                optimizer.load_state_dict(checkpoint['optimizer'])
                lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
                args.start_epoch = checkpoint['epoch'] + 1
                if args.model_ema:
                    _load_checkpoint_for_ema(model_ema, checkpoint['model_ema'])
                if 'scaler' in checkpoint:
                    loss_scaler.load_state_dict(checkpoint['scaler'])
    if args.eval:
        test_stats = evaluate(data_loader_val, model, device)
        logger.info(
            f"Accuracy of the network on the {len(dataset_val)} test images: {test_stats['acc1']:.1f}%")
        return

    logger.info(f"Start training for {args.epochs} epochs")
    start_time = time.time()
    max_accuracy = 0.0
    is_main = dist.get_rank() == 0
    for epoch in range(args.start_epoch, args.epochs):
        if True:  # args.distributed:
            data_loader_train.sampler.set_epoch(epoch)

        train_stats = train_one_epoch(
            model, criterion, data_loader_train,
            optimizer, device, epoch, loss_scaler,
            args.clip_grad, model_ema, mixup_fn,
        )

        lr_scheduler.step(epoch)
        eval_model = model_ema.ema if args.model_ema else model
        test_stats = evaluate(data_loader_val, eval_model, device)
        if is_main:
            logger.info(
                f"Epoch [{epoch + 1}/{args.epochs}] Accuracy of the network on the {len(dataset_val)} test images: {test_stats['acc1']:.1f}%")
        is_main_process = dist.get_rank() == 0
        if args.output_dir and is_main_process:
            save_dic = {
                'model': model_without_ddp.state_dict(),
                'optimizer': optimizer.state_dict(),
                'lr_scheduler': lr_scheduler.state_dict(),
                'epoch': epoch,
                'model_ema': get_state_dict(model_ema),
                'scaler': loss_scaler.state_dict(),
                'args': args,
            }
            output_dir = Path(args.output_dir)
            checkpoint_path = os.path.join(output_dir, "checkpoint_bak.pth")
            if os.path.exists(checkpoint_path):
                checkpoint_bak_path = os.path.join(output_dir, "checkpoint.pth")
                os.rename(checkpoint_path, checkpoint_bak_path)
            #if is_main_process:
            #    torch.save(save_dic, checkpoint_path)
            #if (test_stats["acc1"] >= max_accuracy):
            #    # always only save best checkpoint till now
            #    checkpoint_path = os.path.join(output_dir, "checkpoint_best.pth")
            #    if is_main_process:
            #        torch.save(save_dic, checkpoint_path)

        max_accuracy = max(max_accuracy, test_stats["acc1"])
        
        if is_main:
            logger.info(f'Epoch [{epoch + 1}/{args.epochs}] Max accuracy: {max_accuracy:.2f}%')

        log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                     **{f'test_{k}': v for k, v in test_stats.items()},
                     'epoch': epoch,
                     'n_parameters': n_parameters}

        if args.output_dir and is_main_process:
            with (output_dir / "log.txt").open("a") as f:
                f.write(json.dumps(log_stats) + "\n")

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))


if __name__ == '__main__':
    parser = argparse.ArgumentParser('evaluation script for transfer learning',
                                     parents=[get_args_parser()])
    args = parser.parse_args()

    (filepath, tempfilename) = os.path.split(args.pretrained_weights)
    filename = filepath + '/transfer_learning/%s/%s_b_%d_lr_%.2e_wd_%.2e-ep_%d' % (
        args.data_set, args.opt, args.batch_size, args.learning_rate, args.weight_decay, args.epochs
    )
    if filename:
        Path(filename).mkdir(parents=True, exist_ok=True)
    #logger = get_logger(filename + '/train.log')

    args.output_dir = filename
    main(args, logger)

