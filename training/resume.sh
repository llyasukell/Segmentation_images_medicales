#!/bin/bash
export nnUNet_raw="/workspace/nnUNet_raw"
export nnUNet_preprocessed="/workspace/nnUNet_preprocessed"
export nnUNet_results="/workspace/nnUNet_results"

# Install dependencies
pip install nnunetv2 -q

for fold in 0 1 2 3 4; do
    nnUNetv2_train 2 3d_fullres $fold -tr nnUNetTrainer_100epochs --c
done
