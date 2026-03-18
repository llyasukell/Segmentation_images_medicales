#!/bin/bash
#SBATCH --job-name=nnUnet_train
#SBATCH --time=48:00:00
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=8G
#SBATCH --output=/scratch/$USER/logs/slurm-%j-%x.out 
#SBATCH --error=/scratch/$USER/logs/slurm-%j-%x.error
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=walid.bouhazza@umontreal.ca

module purge
module load python/3.12

echo -e "\nSetting up Python environment..."
python -m venv $SLURM_TMPDIR/venv
source $SLURM_TMPDIR/venv/bin/activate

pip install -r requirements.txt 

echo -e "\nSetting up nnUNet variables..."
export nnUNet_raw="/scratch/$USER/nnUNet_raw"
export nnUNet_preprocessed="/scratch/$USER/nnUNet_preprocessed"
export nnUNet_results="/scratch/$USER/nnUNet_results"
 
mkdir -p /scratch/$USER/logs
 
echo "BEGIN : $(date)"

python preprocessing/convert_to_nnunet.py \
    --task-dir /scratch/$USER/data/Task02_Heart \
    --images-tr-out /scratch/$USER/nnUNet_raw/Dataset002_Heart/imagesTr \
    --labels-tr-out /scratch/$USER/nnUNet_raw/Dataset002_Heart/labelsTr \
    --images-ts-out /scratch/$USER/nnUNet_raw/Dataset002_Heart/imagesTs \
    --output-json /scratch/$USER/nnUNet_raw/Dataset002_Heart/dataset.json

nnUNetv2_plan_and_preprocess -d 2 --verify_dataset_integrity
 
nnUNetv2_train 2 3d_fullres 0 
nnUNetv2_train 2 3d_fullres 1 
nnUNetv2_train 2 3d_fullres 2 
nnUNetv2_train 2 3d_fullres 3 
nnUNetv2_train 2 3d_fullres 4 
 
echo "END : $(date)"
 