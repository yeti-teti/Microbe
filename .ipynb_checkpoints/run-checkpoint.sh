#!/bin/bash

python main.py train src/datasets/train_set.mgf -p src/datasets/val_set.mgf -o model_run

# python main.py evaluate src/datasets/test_set.mgf -m path/to/trained_model.pt -o evaluation_results