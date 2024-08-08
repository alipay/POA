# Copyright 2024 Ant Group.

DATA_ROOT=../Dataset/imagenet

# visaulization for poa small
CKPT=work_dirs/vit_ee1200/eval/teacher_checkpoint_small.pth
SAVE_PATH=vis_att_vit_small_ee1200_poa
mkdir -p ${SAVE_PATH}
python  tools/vis_attention/visualize_attention.py \
        --arch vit_small \
        --patch_size 16 \
        --batch_size 64 \
        --data_path ${DATA_ROOT}/val \
        --pretrained_weights $CKPT \
        --output_dir ${SAVE_PATH} \
        --show_pics 500 


# visaulization for dinov2
CKPT=../dinov2_baseline/work_dirs/vit_small_ee1200/teacher_checkpoint.pth
SAVE_PATH=vis_att_vit_small_ee1200_dinov2
mkdir -p ${SAVE_PATH}
python  tools/vis_attention/visualize_attention.py \
        --arch vit_small \
        --patch_size 16 \
        --batch_size 64 \
        --data_path ${DATA_ROOT}/val \
        --pretrained_weights $CKPT \
        --output_dir ${SAVE_PATH} \
        --show_pics 500 \
        --remove_chunk_id
