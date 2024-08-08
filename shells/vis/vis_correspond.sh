# Copyright 2024 Ant Group.

DATA_ROOT=../Dataset/imagenet
CKPT=$1

SAVE_PATH=$2
mkdir -p ${SAVE_PATH}_instance
python tools/vis_attention/vis_correspondence.py \
    --arch vit_small \
    --patch_size 16 \
    --sample_type instance \
    --data_path ${DATA_ROOT}/val \
    --pretrained_weights $CKPT \
    --output_path ${SAVE_PATH}_instance \
    --show_nums 300 

mkdir -p ${SAVE_PATH}
python tools/vis_attention/vis_correspondence.py \
    --arch vit_small \
    --patch_size 16 \
    --data_path ${DATA_ROOT}/val \
    --pretrained_weights $CKPT \
    --output_path ${SAVE_PATH} \
    --show_nums 300 

