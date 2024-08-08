# Copyright 2024 Ant Group.

CHECKPOINT=$1
export PYTHONPATH=./poa/eval/backgrounds_challenge:./
python poa/eval/backgrounds_challenge/challenge_eval.py \
    --arch vit_small \
    --checkpoint ${CHECKPOINT} \
    --data-path data/bg_challenge \

python poa/eval/backgrounds_challenge/in9_eval.py \
    --arch vit_small \
    --checkpoint ${CHECKPOINT} \
    --data-path data/bg_challenge \

