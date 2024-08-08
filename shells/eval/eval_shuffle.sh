# Copyright 2024 Ant Group.

CHECKPOINT=$1

export PYTHONPATH=./
python poa/eval/occlusion/eval.py \
    --model_name vit_small \
    --test_data ../Dataset/imagenet/val \
    --shuffle \
    --shuffle_h 2 2 4 4 8 14 16 \
    --shuffle_w 2 4 4 8 8 14 16 \
    --pretrained_weights $CHECKPOINT 
