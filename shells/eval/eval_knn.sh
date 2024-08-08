# Copyright 2024 Ant Group.

NUM_GPUS=8
DATA_ROOT=../Dataset/imagenet
python=/opt/conda/envs/dinov2/bin/python
dir=vit_ee1200
step=training_276799

export PYTHONPATH=./:$PYTHONPATH
$python -u -m torch.distributed.launch \
  --nproc_per_node=${NUM_GPUS} \
  --master_port=12345 \
  --nnodes=1 \
  --node_rank=0 \
  --master_addr=$MASTER_ADDR \
  --use-env \
  poa/eval/knn.py \
  --config-file work_dirs/${dir}/config.yaml \
  --pretrained-weights work_dirs/${dir}/eval/${step}/teacher_checkpoint.pth \
  --output-dir work_dirs/${dir}/eval/${step}/knn \
  --train-dataset ImageNet:split=TRAIN:root=${DATA_ROOT}:extra=extra \
  --val-dataset ImageNet:split=VAL:root=${DATA_ROOT}:extra=extra
