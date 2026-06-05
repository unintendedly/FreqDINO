#!/bin/bash

# use one GPU for testing
export CUDA_VISIBLE_DEVICES='7'

# 获取可见 GPU 数量
NUM_GPUS=$(echo $CUDA_VISIBLE_DEVICES | tr ',' ' ' | wc -w)
echo "Using GPUs: $CUDA_VISIBLE_DEVICES"
echo "Number of GPUs: $NUM_GPUS"

torchrun \
    --standalone \
    --nproc_per_node="$NUM_GPUS" \
    --master_port=12381 \
./freqdino.py \
    --model_dir "freqdino" \
    --weights "pretrained.tar" \
    --batch_size 32 \
    --image_size 512 \
    --backbone_size "large" \
    --num_samples 21000 \
    --adapter_interval 6 \
    --hidden_dim_ratio 1 \
    --adapter_dim_ratio 4 \
    --learning_rate 2e-5 \
    --wavelet 'sym4' \
    --test_mode \



