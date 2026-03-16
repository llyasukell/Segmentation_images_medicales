#!/bin/bash
#SBATCH --job-name=nnUnet_train
#SBATCH --time=04:00:00
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=8G
#SBATCH --output=/scratch/$USER/logs/slurm-%j-%x.out 
#SBATCH --error=/scratch/$USER/logs/slurm-%j-%x.error

cd $HOME/projects/ift3710

module purge
module load python/3.12

echo -e "\nSetting up Python environment..."
python -m venv $SLURM_TMPDIR/venv
source $SLURM_TMPDIR/venv/bin/activate

pip install -r requirements.txt 

echo -e "\nSetting up the nnUnet variables..."
export nnUNet_raw="$HOME/projects/ift3710/nnUNet_raw"
export nnUNet_preprocessed="$HOME/projects/ift3710/nnUNet_preprocessed"
export nnUNet_results="$HOME/projects/ift3710/nnUNet_results"
 
mkdir -p /scratch/$USER/logs
 
echo "BEGIN : $(date)"

python $HOME/projects/ift3710/preprocessing/convert_to_nnunet.py
nnUNetv2_plan_and_preprocess -d 2 --verify_dataset_integrity
 
nnUNetv2_train 2 3d_fullres 0 

echo "END : $(date)"
 