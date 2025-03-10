#!/bin/bash

# Create required directories
mkdir -p output

# Run training with optimized settings
python train.py \
    --msp_file "./datasets/human_hcd_tryp_best.msp" \
    --output_dir "./output" \
    --epochs 10 \
    --batch_size 64 \
    --lr 1e-3 \
    --warmup_steps 1000 \
    --dropout 0.2 \
    --label_smoothing 0.1 \
    --max_grad_norm 1.0 \
    --teacher_forcing_start 1.0 \
    --teacher_forcing_end 0.5 \
    --temperature 0.7 \
    --seq_loss_weight 1.0 \
    --ptm_local_loss_weight 0.5 \
    --ptm_global_loss_weight 0.3