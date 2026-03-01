#!/bin/sh


model_name_list=(
    "Llama33_70B"
    "Qwen3_32B"
    "Gemma3_27B"
    "Mistral_7B"
    "Qwen25_7B"
    "Llama31_8B"
)

export CUDA_VISIBLE_DEVICES=0

for model_name in "${model_name_list[@]}"; do

  python3 model_inference.py \
  --model_name "$model_name"

done;