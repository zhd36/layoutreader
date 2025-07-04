#!/usr/bin/env bash

set -x
set -e

DIR="$( cd "$( dirname "$0" )" && cd .. && pwd )"
OUTPUT_DIR="${DIR}/checkpoint/v3/$(date +%F-%H)"
DATA_DIR="${DIR}/Data"

mkdir -p "${OUTPUT_DIR}"

#--model_dir 'microsoft/layoutlmv3-large' \
#  --model_dir '/workspace/paddlex/Model/layoutreader' \
# --model_dir '/workspace/paddlex/Model/MonkeyOCR/Relation' \
deepspeed train.py \
  --model_dir '/workspace/paddlex/Model/layoutreader' \
  --dataset_dir "${DATA_DIR}" \
  --shuffle_probability 1.0 \
  --bbox_noise_level 0.0 \
  --dataloader_num_workers 1 \
  --deepspeed ds_config.json \
  --per_device_train_batch_size 64 \
  --per_device_eval_batch_size 32 \
  --do_train \
  --do_eval \
  --logging_steps 5 \
  --bf16 \
  --seed 42 \
  --num_train_epochs 400 \
  --learning_rate 5e-5 \
  --warmup_steps 1000 \
  --save_strategy epoch \
  --eval_strategy epoch \
  --remove_unused_columns False \
  --output_dir "${OUTPUT_DIR}" \
  --overwrite_output_dir \
  --save_total_limit 5 \
  "$@"
