#!/bin/bash
if [ "${ASR_MODEL}" == "" ]; then
    ASR_MODEL="cpierse/wav2vec2-large-xlsr-53-esperanto"
fi

if [ "${DATA}" == "" ] || [ "${NAME}" == "" ]; then
  echo "DATA and NAME are required"
fi

echo "DATA=${DATA}
NAME=${NAME}
ASR_MODEL=${ASR_MODEL}"

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