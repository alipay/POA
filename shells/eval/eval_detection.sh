# Copyright 2024 Ant Group.

ARCH=base
CONFIG=vit_${ARCH}_giou_4conv1f_coco_3x
WEIGHT_FILE=work_dirs/vit_ee1200/eval/training_276799/teacher_checkpoint_${ARCH}.pth
python -m torch.distributed.launch --nproc_per_node=8 \
    --master_port=${PORT:-29500} \
    poa/eval/object_detection/train.py \
    poa/eval/object_detection/configs/cascade_rcnn/${CONFIG}.py \
    --launcher pytorch \
    --deterministic \
    --options model.pretrained=$WEIGHT_FILE \
    data.samples_per_gpu=2