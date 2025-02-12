#!/bin/bash
if [ "${ASR_MODEL}" == "" ]; then
    ASR_MODEL="cpierse/wav2vec2-large-xlsr-53-esperanto"
fi

if [ "${DATA}" == "" ] || [ "${NAME}" == "" ] || [ "${AUD}" == "" ]; then
  echo "DATA and NAME and AUD are required"
  exit 1
fi

python main.py \
  data/${DATA}/ \
  --workspace logs/${NAME}_torso/ \
  -O \
  --torso \
  --test \
  --test_train \
  --asr_model ${ASR_MODEL} \
  --aud "${AUD}" || exit 1
ffmpeg \
  -i "$(ls -1rt logs/${NAME}_torso/results/*.mp4)" \
  -i "$(echo ${AUD} | sed -r 's/(.*)_[a-zA-Z_]+\.npy$/\1.wav/g')" \
  -c:v copy \
  -c:a aac \
  "logs/${NAME}_torso/$(basename ${AUD}).mp4" || exit 1