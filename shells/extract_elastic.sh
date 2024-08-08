# Copyright 2024 Ant Group.

CKPT_PATH=work_dirs/vit_ee1200/eval/training_276799

CKPT=${CKPT_PATH}/teacher_checkpoint.pth
ARCH=vit
INTACT=large
ELASTIC='small base large'
export PYTHONPATH=$PYTHONPATH:./
python tools/extract_elastic_net.py \
  --intact-ckpt $CKPT \
  --arch $ARCH \
  --intact-net $INTACT \
  --block-chunks 0 \
  --elastic-nets $ELASTIC \
  --output-dir ${CKPT_PATH} 
  

