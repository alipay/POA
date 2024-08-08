# Copyright 2024 Ant Group.

dir=vit_ee1200
step=training_276799

NUM_GPUS=8
DATA_ROOT=../Dataset/imagenet

export PYTHONPATH=./:$PYTHONPATHd
python=/conda/envs/dinov2/bin/python
for net_type in large base small 
do 
  $python -u -m torch.distributed.launch \
    --nproc_per_node=${NUM_GPUS} \
    --master_port=12345 \
    --nnodes=1 \
    --node_rank=0 \
    --master_addr=$MASTER_ADDR \
    --use-env \
    poa/eval/linear.py \
    --net-type ${net_type} \
    --config-file work_dirs/${dir}/config.yaml \
    --pretrained-weights work_dirs/${dir}/eval/${step}/teacher_checkpoint.pth \
    --num-workers 4 \
    --output-dir work_dirs/${dir}/eval/${step}/linear \
    --train-dataset ImageNet:split=TRAIN:root=${DATA_ROOT}:extra=extra \
    --val-dataset ImageNet:split=VAL:root=${DATA_ROOT}:extra=extra
done
