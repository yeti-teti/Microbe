#!/bin/bash

# python main.py train src/datasets/train_set.mgf -p src/datasets/val_set.mgf -o model_run

# python main.py train src/datasets/train_set.mgf -p src/datasets/val_set.mgf -o model_run --model outputs/epoch=7-step=72500.ckpt --config src/config.yaml --verbosity debug

# python main.py evaluate src/datasets/val_set.mgf --model outputs/epoch=0-step=7500.ckpt --config src/config.yaml --output eval_results.txt --verbosity info

python main.py sequence src/datasets/unknown_spectra.mgf --model outputs/epoch=9-step=97500.ckpt --config src/config.yaml --output predictions.mztab --verbosity info