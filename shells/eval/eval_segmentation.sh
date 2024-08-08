# Copyright 2024 Ant Group.

ARCH=base
WEIGHT_FILE=work_dirs/vit_ee1200/eval/training_276799/teacher_checkpoint_${ARCH}.pth
export PYTHONPATH=./poa/eval/semantic_segmentation
python -m torch.distributed.launch --nproc_per_node=8 \
    --master_port=12345 \
    ./poa/eval/semantic_segmentation/train.py \
    ./poa/eval/semantic_segmentation/configs/upernet/vit_${ARCH}_512_ade20k_160k.py \
    --launcher pytorch \
    --deterministic \
    --options model.backbone.use_checkpoint=True \
    model.pretrained=$WEIGHT_FILE

