#!/bin/bash
export nnUNet_raw="/workspace/nnUNet_raw"
export nnUNet_preprocessed="/workspace/nnUNet_preprocessed"
export nnUNet_results="/workspace/nnUNet_results"

# Install dependencies
pip install nnunetv2 -q

# Nettoyer les fichiers Mac AVANT la conversion
find /workspace/dataset/Task02_Heart -name "._*" -delete
find /workspace/dataset/Task02_Heart -name ".DS_Store" -delete

# Supprimer le dataset mal converti si existant
rm -rf /workspace/nnUNet_raw/Dataset002_Heart

# Convert dataset
nnUNetv2_convert_MSD_dataset /workspace/dataset/Task02_Heart Dataset002_Heart

# Preprocess
nnUNetv2_plan_and_preprocess -d 2 --verify_dataset_integrity

# Train all 5 folds
for fold in 0 1 2 3 4; do
    nnUNetv2_train 2 3d_fullres $fold -tr nnUNetTrainer_100epochs
done
