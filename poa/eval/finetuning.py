# Copyright (c) ByteDance, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import json
import logging
import time
import math
import os
import sys
import datetime
from typing import List, Optional, Iterable
from pathlib import Path
import torch
import torch.distributed as dist
from timm.loss import LabelSmoothingCrossEntropy, SoftTargetCrossEntropy
from timm.utils import accuracy, ModelEma
from timm.data.mixup import Mixup
from poa.data import make_dataset
from poa.data.transforms import make_finetuning_transform
from poa.eval.metrics import MetricType
from poa.eval.setup import get_args_parser as get_setup_args_parser
from poa.eval.setup import setup_and_build_model
from poa.eval.utils import (ModelWithClassifier, create_optimizer, cosine_scheduler,
                            LayerDecayValueAssigner, SmoothedValue, NativeScalerWithGradNormCount, MetricLogger2)
logger = logging.getLogger("poa")


def get_args_parser(
    description: Optional[str] = None,
    parents: Optional[List[argparse.ArgumentParser]] = [],
    add_help: bool = True,
):
    setup_args_parser = get_setup_args_parser(parents=parents, add_help=False)
    parents = [setup_args_parser]
    parser = argparse.ArgumentParser(
        description=description,
        parents=parents,
        add_help=add_help,
    )
    parser.add_argument(
        "--arch-name",
        type=str,
        help="Architecture name: swin, vit, or resnet",
    )
    parser.add_argument(
        "--train-dataset",
        dest="train_dataset_str",
        type=str,
        help="Training dataset",
    )
    parser.add_argument(
        "--val-dataset",
        dest="val_dataset_str",
        type=str,
        help="Validation dataset",
    )
    parser.add_argument(
        "--test-datasets",
        dest="test_dataset_strs",
        type=str,
        nargs="+",
        help="Test datasets, none to reuse the validation dataset",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        help="Number of training epochs",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        help="Batch Size (per GPU)",
    )
    parser.add_argument(
        '--input-size',
        type=int,
        help='images input size')
    parser.add_argument(
        "--num-workers",
        type=int,
        help="Number de Workers",
    )
    parser.add_argument(
        "--epoch-length",
        type=int,
        help="Length of an epoch in number of iterations",
    )
    parser.add_argument(
        "--save-checkpoint-frequency",
        type=int,
        help="Number of epochs between two named checkpoint saves.",
    )
    parser.add_argument(
        "--eval-period-iterations",
        type=int,
        help="Number of iterations between two evaluations.",
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        help="Learning rate for finetuning.",
    )
    parser.add_argument(
        "--no-resume",
        action="store_true",
        help="Whether to not resume from existing checkpoints",
    )
    parser.add_argument(
        "--val-metric-type",
        type=MetricType,
        choices=list(MetricType),
        help="Validation metric",
    )
    parser.add_argument(
        "--test-metric-types",
        type=MetricType,
        choices=list(MetricType),
        nargs="+",
        help="Evaluation metric",
    )
    parser.add_argument(
        "--classifier-fpath",
        type=str,
        help="Path to a file containing pretrained linear classifiers",
    )
    parser.add_argument(
        "--val-class-mapping-fpath",
        type=str,
        help="Path to a file containing a mapping to adjust classifier outputs",
    )
    parser.add_argument(
        "--test-class-mapping-fpaths",
        nargs="+",
        type=str,
        help="Path to a file containing a mapping to adjust classifier outputs",
    )
    parser.add_argument(
        '--color-jitter',
        type=float,
        help='Color jitter factor (default: 0.4)'
    )
    parser.add_argument(
        '--aa', type=str, default='rand-m9-mstd0.5-inc1',
        help='Use AutoAugment policy. "v0" or "original". " + "(default: rand-m9-mstd0.5-inc1)'
    )
    parser.add_argument(
        '--train-interpolation',
        type=str,
        help='Training interpolation (random, bilinear, bicubic default: "bicubic")'
    )
    parser.add_argument(
        '--reprob',
        type=float,
        help='Random erase prob (default: 0.25)'
    )
    parser.add_argument(
        "--net-type",
        type=str,
        help="Network type for vit or swin, for example: samll, base, large",
    )
    parser.add_argument(
        '--opt',
        type=str,
        help='Optimizer (default: "adamw")'
    )
    parser.add_argument(
        '--weight-decay',
        type=float,
        help='Weight decay (default: 0.05)'
    )
    parser.add_argument(
        '--momentum',
        type=float,
        help='SGD momentum (default: 0.9)'
    )
    parser.add_argument(
        '--min-lr',
        type=float,
        help='lower lr bound for cyclic schedulers that hit 0 (1e-6)'
    )
    parser.add_argument(
        '--warmup-epochs',
        type=int,
        help='epochs to warmup LR, if scheduler supports'
    )
    parser.add_argument(
        '--warmup-steps',
        type=int, default=-1,
        help='num of steps to warmup LR, will overload warmup_epochs if set > 0'
    )
    parser.add_argument(
        '--layer-decay',
        type=float,
        help='layer lr decay rate (default: 0.9)'
   )
    # * Mixup params
    parser.add_argument(
        '--mixup',
        type=float,
        help='mixup alpha, mixup enabled if > 0.'
    )
    parser.add_argument(
        '--cutmix',
        type=float,
        help='cutmix alpha, cutmix enabled if > 0.'
    )
    parser.add_argument(
        '--cutmix-minmax',
        type=float, nargs='+', default=None,
        help='cutmix min/max ratio, overrides alpha and enables cutmix if set (default: None)'
    )
    parser.add_argument(
        '--mixup-prob',
        type=float, default=1.0,
        help='Probability of performing mixup or cutmix when either/both is enabled'
    )
    parser.add_argument(
        '--mixup-switch-prob',
        type=float, default=0.5,
        help='Probability of switching to cutmix when both mixup and cutmix enabled'
    )
    parser.add_argument(
        '--mixup-mode',
        type=str, default='batch',
        help='How to apply mixup/cutmix params. Per "batch", "pair", or "elem"'
    )
    parser.add_argument(
        '--model-ema', action='store_true', 
        help='Using model EMA in training',
    )
    parser.add_argument(
        '--model-ema-decay', type=float, 
    )
    parser.add_argument(
        '--smoothing',
        type=float,
        help='Label smoothing (default: 0.1)'
    )
    parser.add_argument(
        '--pin_mem',
        action='store_true',
        help='Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU.'
    )
    parser.add_argument(
        '--clip_grad',
        type=float, default=None,
         help='Clip gradient norm (default: None, no clipping)'
    )
    parser.add_argument(
        '--drop_path', 
        type=float, default=0.1, 
        help='Drop path rate (default: 0.1)'
    )
    parser.set_defaults(
        train_dataset_str="ImageNet:split=TRAIN",
        val_dataset_str="ImageNet:split=VAL",
        test_dataset_strs=None,
        input_size=224,
        epochs=200,
        warmup_epochs=20,
        batch_size=128,
        num_workers=32,
        pin_mem=True,
        epoch_length=1250,
        color_jitter=0.4,
        reprob=0.25,
        train_interpolation="bicubic",
        save_checkpoint_frequency=20,
        eval_period_iterations=1250,
        opt='adamw',
        weight_decay=0.05,
        momentum=0.9,
        learning_rate=0.0012,
        min_lr=1e-6,
        layer_decay=0.9,
        mixup=0.8,
        cutmix=1.0,
        smoothing=0.1,
        model_ema=True,
        model_ema_decay=0.7,
        val_metric_type=MetricType.MEAN_ACCURACY,
        test_metric_types=None,
        classifier_fpath=None,
        val_class_mapping_fpath=None,
        test_class_mapping_fpaths=[None],
    )
    return parser

def train_class_batch(model, samples, target, criterion):
    outputs = model(samples)
    loss = criterion(outputs, target)
    return loss, outputs


@torch.no_grad()
def evaluate(data_loader, model, device):
    criterion = torch.nn.CrossEntropyLoss()

    metric_logger = MetricLogger2(delimiter="  ")
    header = 'Test:'

    # switch to evaluation mode
    model.eval()

    for batch in metric_logger.log_every(data_loader, 10, header):
        images = batch[0]
        target = batch[-1]
        images = images.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)

        # compute output
        with torch.cuda.amp.autocast():
            output = model(images)
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


def train_one_epoch(
        model: torch.nn.Module,
        criterion: torch.nn.Module,
        data_loader: Iterable,
        optimizer: torch.optim.Optimizer,
        device: torch.device,
        epoch: int,
        loss_scaler,
        max_norm: float = 0,
        mixup_fn: Optional[Mixup] = None,
        start_steps=None,
        lr_schedule_values=None,
        wd_schedule_values=None,
        num_training_steps_per_epoch=None,
        update_freq=None,
        model_ema: Optional[ModelEma] = None,
    ):
    model.train(True)
    metric_logger = MetricLogger2(delimiter="  ")
    metric_logger.add_meter('lr', SmoothedValue(window_size=1, fmt='{value:.6f}'))
    metric_logger.add_meter('min_lr', SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 10
    if loss_scaler is None:
        model.zero_grad()
        model.micro_steps = 0
    else:
        optimizer.zero_grad()
    for data_iter_step, (samples, targets) in enumerate(metric_logger.log_every(data_loader, print_freq, header)):
        step = data_iter_step // update_freq
        if step >= num_training_steps_per_epoch:
            continue
        it = start_steps + step  # global training iteration
        # Update LR & WD for the first acc
        if lr_schedule_values is not None or wd_schedule_values is not None and data_iter_step % update_freq == 0:
            for i, param_group in enumerate(optimizer.param_groups):
                if lr_schedule_values is not None:
                    param_group["lr"] = lr_schedule_values[it] * param_group["lr_scale"]
                if wd_schedule_values is not None and param_group["weight_decay"] > 0:
                    param_group["weight_decay"] = wd_schedule_values[it]

        samples = samples.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        if mixup_fn is not None:
            samples, targets = mixup_fn(samples, targets)

        if loss_scaler is None:
            samples = samples.half()
            loss, output = train_class_batch(
                model, samples, targets, criterion)
        else:
            with torch.cuda.amp.autocast():
                loss, output = train_class_batch(
                    model, samples, targets, criterion)

        loss_value = loss.item()

        if not math.isfinite(loss_value):
            logger.info("Loss is {}, stopping training".format(loss_value))
            sys.exit(1)
        # this attribute is added by timm on one optimizer (adahessian)
        is_second_order = hasattr(optimizer, 'is_second_order') and optimizer.is_second_order
        loss /= update_freq
        grad_norm = loss_scaler(loss, optimizer, clip_grad=max_norm,
                                parameters=model.parameters(), create_graph=is_second_order,
                                update_grad=(data_iter_step + 1) % update_freq == 0)
        if (data_iter_step + 1) % update_freq == 0:
            optimizer.zero_grad()
            if model_ema is not None:
                model_ema.update(model)
        loss_scale_value = loss_scaler.state_dict()["scale"]

        torch.cuda.synchronize()

        if mixup_fn is None:
            class_acc = (output.max(-1)[-1] == targets).float().mean()
            metric_logger.update(class_acc=class_acc)
        else:
            class_acc = None
        metric_logger.update(loss=loss_value)
        metric_logger.update(loss_scale=loss_scale_value)
        min_lr = 10.
        max_lr = 0.
        for group in optimizer.param_groups:
            min_lr = min(min_lr, group["lr"])
            max_lr = max(max_lr, group["lr"])

        metric_logger.update(lr=max_lr)
        metric_logger.update(min_lr=min_lr)
        weight_decay_value = None
        for group in optimizer.param_groups:
            if group["weight_decay"] > 0:
                weight_decay_value = group["weight_decay"]
        metric_logger.update(weight_decay=weight_decay_value)
        metric_logger.update(grad_norm=grad_norm)

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}



def run_finetnuing(args):

    if not Path(args.pretrained_weights).exists():
        raise Exception(f'Pretrained model not found: {args.pretrained_weights}')

    # training setting 
    assert args.arch_name in ['vit', 'swin', 'resnet']
    if args.arch_name == 'vit':
        batch_size_dict = {'small': 256, 'base': 256, 'large': 256}
    elif args.arch_name == 'swin':
        batch_size_dict = {'tiny': 128, 'small': 128, 'base': 64}
    elif args.arch_name == 'resnet':
        batch_size_dict = {'R50': 128, 'R101': 128, 'R152': 64}
    args.batch_size = batch_size_dict[args.net_type]
    if args.arch_name == 'vit':
        training_epochs_dict = {'small': 200, 'base': 100, 'large': 50}
    elif args.arch_name == 'swin':
        training_epochs_dict = {'tiny': 200, 'small': 100, 'base': 50}
    elif args.arch_name == 'resnet':
        training_epochs_dict = {'R50': 100, 'R101': 100, 'R152': 100}
    training_epochs = training_epochs_dict[args.net_type]
    if args.arch_name == 'vit':
        learning_rate_dict = {'small': 0.002, 'base': 0.0007, 'large': 0.0018}
    elif args.arch_name == 'swin':
        learning_rate_dict = {'tiny': 0.0014, 'small': 0.001, 'base': 0.0007}
    elif args.arch_name == 'resnet':
        learning_rate_dict = {'R50': 0.0014, 'R101': 0.001, 'R152': 0.0007}
    learning_rate = learning_rate_dict[args.net_type]
    if args.arch_name == 'vit':
        layer_decay_dict = {'small': 0.55, 'base': 0.4, 'large': 0.6}
    elif args.arch_name == 'swin':
        layer_decay_dict = {'tiny': 0.6, 'small': 0.4, 'base': 0.55}
    elif args.arch_name == 'resnet':
        layer_decay_dict = {'R50': 0.6, 'R101': 0.4, 'R152': 0.45}
    layer_decay = layer_decay_dict[args.net_type]
    warmup_epochs = int(training_epochs * 0.1)

    # network setting
    if args.arch_name == 'vit':
        drop_path_dict = {'small': 0.1, 'base': 0.2, 'large': 0.2}
    elif args.arch_name == 'swin':
        drop_path_dict = {'tiny': 0.1, 'small': 0.2, 'base': 0.2}
    elif args.arch_name == 'resnet':
        drop_path_dict = {'R50': None, 'R101': None, 'R152': None}
    args.drop_path = drop_path_dict[args.net_type]
    feature_model, _ = setup_and_build_model(args, is_train=True)
    feature_model.get_by_type(args.net_type)
    feature_model.use_mean_pooling = True
    embed_dim = feature_model.feat_dim
    actived_block_idx = feature_model.block_idx
    num_layers = feature_model.get_num_layers()

    train_transform = make_finetuning_transform(is_train=True, args=args)
    train_dataset = make_dataset(dataset_str=args.train_dataset_str, transform=train_transform)
    num_tasks = dist.get_world_size()
    global_rank = dist.get_rank()
    sampler_train = torch.utils.data.DistributedSampler(
        train_dataset, num_replicas=num_tasks, rank=global_rank, shuffle=True
    )
    logger.info("Sampler_train = %s" % str(sampler_train))
    val_transform = make_finetuning_transform(is_train=False, args=args)
    val_dataset = make_dataset(dataset_str=args.val_dataset_str,transform=val_transform)
    sampler_val = torch.utils.data.DistributedSampler(
        val_dataset, num_replicas=num_tasks, rank=global_rank, shuffle=False
    )
    train_data_loader = torch.utils.data.DataLoader(
        train_dataset, sampler=sampler_train,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=True,
    )
    val_data_loader = torch.utils.data.DataLoader(
        val_dataset, sampler=sampler_val,
        batch_size=int(1.5 * args.batch_size),
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=False
    )

    training_num_classes = len(torch.unique(torch.Tensor(train_dataset.get_targets().astype(int))))
    use_multi_stage_feat = args.arch_name == 'resnet'
    use_cls = args.arch_name  == 'vit'
    model = ModelWithClassifier(feature_model, embed_dim, training_num_classes, use_multi_stage_feat, use_cls)
    device = torch.device('cuda')
    model.to(device)
    model_ema = None  
    if args.model_ema:
        # Important to create EMA model after cuda(), DP wrapper, and AMP but before SyncBN and DDP wrapper
        model_ema = ModelEma(
            model, decay=args.model_ema_decay, device='', resume=''
        )
        print("Using EMA with decay = %.8f" % args.model_ema_decay)


    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_batch_size = args.batch_size * dist.get_world_size()
    num_training_steps_per_epoch = len(train_dataset) // total_batch_size
    model = torch.nn.parallel.DistributedDataParallel(model, find_unused_parameters=True)
    model_without_ddp = model.module

    logger.info(f"Finetuning network: {args.net_type}")
    logger.info("Model = %s" % str(model_without_ddp))
    logger.info("Number of params: %d " % n_parameters)
    logger.info("LR = %.8f" % learning_rate)
    logger.info("Total batch size = %d" % total_batch_size)
    logger.info("Number of training examples = %d" % len(train_dataset))
    logger.info("Number of training step per epoch = %d" % num_training_steps_per_epoch)


    if layer_decay < 1.0 and args.arch_name == 'resnet':
        assigner = LayerDecayValueAssigner(
            list(layer_decay ** (num_layers + 1 - i) for i in range(num_layers + 2)),
            prefix = 'feature_model', net_type='resnet', actived_block_idx=actived_block_idx, 
            depths=model_without_ddp.get_depths()
        )
    elif layer_decay < 1.0 and args.arch_name == 'swin':
        assigner = LayerDecayValueAssigner(
            list(layer_decay ** (num_layers + 1 - i) for i in range(num_layers + 2)),
            prefix = 'feature_model', net_type='swin', actived_block_idx=actived_block_idx, 
            depths=model_without_ddp.get_depths()
        )
    elif layer_decay < 1.0 and args.arch_name == 'vit':
        assigner = LayerDecayValueAssigner(
            list(layer_decay ** (num_layers + 1 - i) for i in range(num_layers + 2)), 
            prefix = 'feature_model', net_type='vit', actived_block_idx=actived_block_idx
        )
    else:
        assigner = None
    skip_weight_decay_list = model_without_ddp.no_weight_decay()
    optimizer = create_optimizer(
        args, model_without_ddp, skip_list=skip_weight_decay_list,
        get_num_layer=assigner.get_layer_id if assigner is not None else None,
        get_layer_scale=assigner.get_scale if assigner is not None else None)

    logger.info("Use step level LR scheduler!")
    lr_schedule_values = cosine_scheduler(
        learning_rate, args.min_lr, args.epochs, num_training_steps_per_epoch,
        warmup_epochs=warmup_epochs, warmup_steps=args.warmup_steps,
    )
    # mix up training
    mixup_fn = None
    mixup_active = args.mixup > 0 or args.cutmix > 0. or args.cutmix_minmax is not None
    if mixup_active:
        logger.info("Mixup is activated!")
        mixup_fn = Mixup(
            mixup_alpha=args.mixup, cutmix_alpha=args.cutmix, cutmix_minmax=args.cutmix_minmax,
            prob=args.mixup_prob, switch_prob=args.mixup_switch_prob, mode=args.mixup_mode,
            label_smoothing=args.smoothing, num_classes=training_num_classes)
    if mixup_fn is not None:
        # smoothing is handled with mixup label transform
        criterion = SoftTargetCrossEntropy()
    elif args.smoothing > 0.:
        criterion = LabelSmoothingCrossEntropy(smoothing=args.smoothing)
    else:
        criterion = torch.nn.CrossEntropyLoss()
    logger.info("Criterion = %s" % str(criterion))

    logger.info(f"Start training for {args.epochs} epochs")
    start_time = time.time()
    max_accuracy = 0.0
    loss_scaler =  NativeScalerWithGradNormCount()
    for epoch in range(training_epochs):
        train_data_loader.sampler.set_epoch(epoch)
        # Training
        train_stats = train_one_epoch(
            model, criterion, train_data_loader, optimizer,
            device, epoch, loss_scaler, args.clip_grad, mixup_fn,
            start_steps=epoch * num_training_steps_per_epoch,
            lr_schedule_values=lr_schedule_values, wd_schedule_values=None,
            num_training_steps_per_epoch=num_training_steps_per_epoch, update_freq=1,
            model_ema=model_ema,
        )
        # Evaluation
        eval_model = model_ema.ema if args.model_ema else model
        test_stats = evaluate(val_data_loader, eval_model, device)
        logger.info(
            f"Accuracy of the network on the {len(val_dataset)} test images: {test_stats['acc1']:.1f}%"
        )
        if max_accuracy < test_stats["acc1"]:
            max_accuracy = test_stats["acc1"]
        logger.info(f'Epoch[{epoch + 1}/{training_epochs}] Max accuracy: {max_accuracy:.2f}%')
        log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                     **{f'test_{k}': v for k, v in test_stats.items()},
                     'epoch': epoch,
                     'n_parameters': n_parameters}
        if args.output_dir and dist.get_rank() == 0:
            with open(os.path.join(args.output_dir, "log.txt"), mode="a", encoding="utf-8") as f:
                f.write(json.dumps(log_stats) + "\n")

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))


def main(args):
    run_finetnuing(args)
    return 0


if __name__ == "__main__":
    description = "Finetuning evaluation"
    args_parser = get_args_parser(description=description)
    args = args_parser.parse_args()
    sys.exit(main(args))


