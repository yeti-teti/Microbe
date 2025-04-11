#!/bin/bash

python main.py train src/datasets/train_set.mgf -p src/datasets/val_set.mgf -o model_run

python main.py train src/datasets/train_set.mgf -p src/datasets/val_set.mgf -o model_run --model outputs/loss=2.7084.ckpt --config src/config.yaml --verbosity debug

python main.py evaluate src/datasets/val_set.mgf --model outputs/epoch=21-step=216000.ckpt --config src/config.yaml --output eval_results.txt

python main.py sequence src/datasets/P_putida_KT2440_glu_2_DP_Bane_06Jul18_18-06-01.mgf --model outputs/epoch=21-step=216000.ckpt --config src/config.yaml --output predictions.mztab --verbosity info