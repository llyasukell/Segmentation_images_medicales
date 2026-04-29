#!/bin/bash

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

export nnUNet_raw="$SCRIPT_DIR/data/nnUNet_raw"
export nnUNet_preprocessed="$SCRIPT_DIR/preprocessing/nnUNet_preprocessed"
export nnUNet_results="$SCRIPT_DIR/result/nnUNet_results"

echo "Variables nnU-Net configurées :"
echo "  nnUNet_raw          = $nnUNet_raw"
echo "  nnUNet_preprocessed = $nnUNet_preprocessed"
echo "  nnUNet_results      = $nnUNet_results"

python3 - <<'EOF'
import sys
import torch

version = tuple(int(x) for x in torch.__version__.split(".")[:2])
if version >= (2, 9):
    print(f"ERREUR : PyTorch {torch.__version__} >= 2.9 — version non supportée.")
    sys.exit(1)

if not torch.cuda.is_available():
    print("ERREUR : CUDA non disponible — GPU NVIDIA requis.")
    sys.exit(1)

print(f"PyTorch {torch.__version__} | CUDA {torch.version.cuda} | GPU : {torch.cuda.get_device_name(0)}")
EOF
