#!/bin/bash
ASR_MODEL="visemefix"

python main.py data/${DATA}/ \
  --workspace logs/${NAME}/ \
  -O \
  --iters 200000 \
  --asr_model ${ASR_MODEL} || exit 1

python main.py data/${DATA}/ \
  --workspace logs/${NAME}/ \
  -O \
  --iters 250000 \
  --finetune_lips \
  --asr_model ${ASR_MODEL} || exit 1

python main.py data/${DATA}/ \
  --workspace logs/${NAME}_torso/ \
  -O \
  --torso \
  --head_ckpt logs/${NAME}/checkpoints/ngp.pth \
  --iters 200000 \
  --asr_model ${ASR_MODEL} || exit 1