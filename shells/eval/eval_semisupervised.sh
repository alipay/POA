# Copyright 2024 Ant Group.

DATASET_ROOT=../Dataset/imagenet

## semi-supervised learning for pretrained ViT-small on 1% labeled data via fine-tuning
ARCH=small   
NPROC_PER_NODE=8
BATCH_SIZE_PER_GPU=128
PRETRAINED_MODEL=work_dirs/vit_ee1200/eval/training_276799/teacher_checkpoint_${ARCH}.pth

export PYTHONPATH=./
python -m torch.distributed.launch \
    --nproc_per_node=$NPROC_PER_NODE \
    --master_port=12345 \
    --nnodes=1 \
    --node_rank=0 \
    --master_addr=$MASTER_ADDR \
    poa/eval/semi_supervised_learning/eval_semi_supervised_learning.py \
    --data_path $DATASET_ROOT \
    --pretrained_weights $PRETRAINED_MODEL \
    --arch vit_${ARCH} \
    --avgpool_patchtokens 0 \
    --finetune_head_layer 1 \
    --checkpoint_key teacher \
    --epochs 1000 \
    --opt adamw \
    --batch_size $BATCH_SIZE_PER_GPU \
    --lr 5.0e-6 \
    --weight-decay 0.05 \
    --drop-path 0.1 \
    --class_num 1000 \
    --dist-eval \
    --target_list_path poa/eval/semi_supervised_learning/subset/1percent.txt

## semi-supervised learning for pretrained ViT-small on 10% labeled data via fine-tuning
NPROC_PER_NODE=8
BATCH_SIZE_PER_GPU=128
PRETRAINED_MODEL=work_dirs/vit_ee1200/eval/training_276799/teacher_checkpoint_${ARCH}.pth
python -m torch.distributed.launch --nproc_per_node=$NPROC_PER_NODE \
    poa/eval/semi_supervised_learning/eval_semi_supervised_learning.py \
    --data_path $DATASET_ROOT \
    --pretrained_weights $PRETRAINED_MODEL \
    --arch vit_small \
    --avgpool_patchtokens 0 \
    --finetune_head_layer 1 \
    --checkpoint_key teacher \
    --epochs 1000 \
    --opt adamw \
    --batch_size $BATCH_SIZE_PER_GPU \
    --lr 5.0e-6 \
    --weight-decay 0.05 \
    --drop-path 0.1 \
    --class_num 1000 \
    --dist-eval \
    --target_list_path poa/eval/semi_supervised_learning/subset/10percent.txt


## semi-supervised learning for pretrained resnet50 on 1% labeled data via fine-tuning
NPROC_PER_NODE=8
BATCH_SIZE_PER_GPU=128
PRETRAINED_MODEL=work_dirs/resnet_ee2400/eval/training_664319/teacher_checkpoint.pth
mkdir -p logs/eval/semi_cls
export PYTHONPATH=./
export TORCH_DISTRIBUTED_DEBUG=DETAIL
python -m torch.distributed.launch \
    --nproc_per_node=$NPROC_PER_NODE \
    --master_port=12345 \
    --nnodes=1 \
    --node_rank=0 \
    --master_addr=$MASTER_ADDR \
    poa/eval/semi_supervised_learning/eval_semi_supervised_learning.py \
    --data_path $DATASET_ROOT \
    --pretrained_weights $PRETRAINED_MODEL \
    --arch resnet \
    --avgpool_patchtokens 0 \
    --finetune_head_layer 1 \
    --checkpoint_key teacher \
    --epochs 1000 \
    --opt adamw \
    --batch_size $BATCH_SIZE_PER_GPU \
    --lr 5.0e-6 \
    --weight-decay 0.05 \
    --class_num 1000 \
    --dist-eval \
    --target_list_path poa/eval/semi_supervised_learning/subset/1percent.txt

## semi-supervised learning for pretrained resnet50 on 10% labeled data via fine-tuning
NPROC_PER_NODE=8
BATCH_SIZE_PER_GPU=128
PRETRAINED_MODEL=work_dirs/resnet_ee2400/eval/training_664319/teacher_checkpoint.pth
python -m torch.distributed.launch --nproc_per_node=$NPROC_PER_NODE \
    poa/eval/semi_supervised_learning/eval_semi_supervised_learning.py \
    --data_path $DATASET_ROOT \
    --pretrained_weights $PRETRAINED_MODEL \
    --arch resnet \
    --avgpool_patchtokens 0 \
    --finetune_head_layer 1 \
    --checkpoint_key teacher \
    --epochs 1000 \
    --opt adamw\
    --batch_size $BATCH_SIZE_PER_GPU \
    --lr 5.0e-6 \
    --weight-decay 0.05 \
    --class_num 1000 \
    --num_workers 32 \
    --dist-eval \
    --target_list_path poa/eval/semi_supervised_learning/subset/10percent.txt