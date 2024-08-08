# Copyright 2024 Ant Group.

DATASET_ROOT=../Dataset/imagenet

export PYTHONPATH=./
python=/conda/envs/dinov2/bin/python

# for vit
PRETRAINED=$1
$python -m torch.distributed.launch --nproc_per_node=1 \
    --master_port=12345 \
    poa/eval/unsupervised_cls/unsupervised_cls.py \
    --pretrained_weights $PRETRAINED \
    --arch vit_small \
    --data_path ${DATASET_ROOT}

# for resnet
PRETRAINED=$1
$python -m torch.distributed.launch --nproc_per_node=1 \
    --master_port=12345 \
    poa/eval/unsupervised_cls/unsupervised_cls.py \
    --pretrained_weights $PRETRAINED \
    --arch resnet50 \
    --data_path ${DATASET_ROOT}
