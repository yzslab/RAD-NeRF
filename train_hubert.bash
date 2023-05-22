#!/bin/bash
ASR_MODEL="facebook/hubert-large-ls960-ft"

python main.py data/${DATA}/ \
  --workspace logs/${NAME}/ \
  -O \
  --iters 300000 \
  --asr_model ${ASR_MODEL} || exit 1

python main.py data/${DATA}/ \
  --workspace logs/${NAME}/ \
  -O \
  --iters 400000 \
  --finetune_lips \
  --asr_model ${ASR_MODEL} || exit 1

python main.py data/${DATA}/ \
  --workspace logs/${NAME}_torso/ \
  -O \
  --torso \
  --head_ckpt logs/${NAME}/checkpoints/ngp.pth \
  --iters 200000 \
  --asr_model ${ASR_MODEL} || exit 1
