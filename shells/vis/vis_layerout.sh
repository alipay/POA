# Copyright 2024 Ant Group.

DATA_ROOT=../Dataset/imagenet
CKPT=$1
OUT_DIM=16384
SAVE_PATH=$2_${OUT_DIM}

mkdir -p ${SAVE_PATH}
export CUDA_VISIBLE_DEVICES=0
export PYTHONPATH=./
python tools/vis_attention/vis_topk_cluster.py \
    --arch vit_small \
    --data_path ${DATA_ROOT}/val \
    --pretrained_path $CKPT \
    --save_path ${SAVE_PATH} \
    --type cls \
    --out_dim ${OUT_DIM}


