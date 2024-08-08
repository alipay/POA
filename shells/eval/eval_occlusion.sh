# Copyright 2024 Ant Group.

CHECKPOINT=$1
export PYTHONPATH=./
python poa/eval/occlusion/eval.py \
    --model_name vit_small \
    --pretrained_weights $CHECKPOINT \
    --test_data ../Dataset/imagenet/val \
    --random_drop 

