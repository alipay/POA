# Copyright 2024 Ant Group.

# config of resnet
CONFIG=poa/configs/train/resnet_ee2400.yaml
WORK_DIR=resnet_ee2400

# config of swin
CONFIG=poa/configs/train/swin_ee1200.yaml
WORK_DIR=swin_ee1200

# cofing of vit
CONFIG=poa/configs/train/vit_ee1200.yaml
WORK_DIR=vit_ee1200

DATA_ROOT=../Dataset/imagenet 
python=/conda/envs/dinov2/bin/python

ADDR=$MASTER_ADDR
PORT=$MASTER_PORT
NODE=$WORLD_SIZE
RANK=$RANK

export PYTHONPATH=$PYTHONPATH:./
$python -u -m torch.distributed.launch \
  --nproc_per_node=8 \
  --master_port=$PORT \
  --nnodes=$NODE \
  --node_rank=$RANK \
  --master_addr=$ADDR \
  --use-env poa/train/train.py \
  --config-file $CONFIG \
  --output-dir work_dirs/$WORK_DIR \
  train.dataset_path=ImageNet:split=TRAIN:root=${DATA_ROOT}:extra=extra

