# Copyright 2024 Ant Group.

dir=vit_ee1200
step=training_276799
net_type=small # or base
NUM_GPUS=8

export PYTHONPATH=./:$PYTHONPATH
python=/opt/conda/envs/dinov2/bin/python

for dataset in CIFAR10 CIFAR Cars flowers INAT19 INAT
do
    $python -u -m torch.distributed.launch \
        --nproc_per_node=${NUM_GPUS} \
        --master_port=12345 \
        --nnodes=1 \
        --node_rank=0 \
        --master_addr=$MASTER_ADDR \
        poa/eval/transfer_learning.py \
        --net-type ${net_type} \
        --config-file work_dirs/${dir}/config.yaml \
        --pretrained-weights work_dirs/${dir}/eval/${step}/teacher_checkpoint.pth \
        --no-resume \
        --resume checkpoint.pth \
        --batch-size 96 \
        --reprob 0.1 \
        --data_set ${dataset}
done


