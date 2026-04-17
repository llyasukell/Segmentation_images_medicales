# IFT-3710 — 3D Medical Image Segmentation

Binary segmentation of the **left atrium** from MRI scans using [nnU-Net](https://github.com/MIC-DKFZ/nnUNet) on the [Medical Segmentation Decathlon](http://medicaldecathlon.com/) dataset (Task02_Heart / Dataset002_Heart).

**Team:** Mamour Ndiaye (20257780) · Haoran Sun (20260543) · Walid Bouhazza (20280620)

---

## Repository Structure

```
IFT-3710-Projet/
├── preprocessing/
│   ├── convert_to_nnunet.py          # Converts raw MRI data → nnU-Net format (manually)
│   └── nnUNet_preprocessed/          # nnU-Net planning output (plans, fingerprint, fold splits)
├── training/
│   ├── nnUnet_train.sh               # SLURM job script for the cluster (full 5-fold run)
│   ├── train.sh                      # Standalone training script
│   └── resume.sh                     # Resumes an interrupted training run
├── analysis/
│   ├── quantitative_analysis.ipynb   # Dice scores, volumes, FP/FN, morphology, voxel spacing
│   └── qualitative_analysis.ipynb    # TP/FP/FN overlays per case, interactive 3D render
├── utils/
│   └── visualize.ipynb               # Interactive slice viewer (image + label)
├── evaluation/
│   └── predictions/                  # Model predictions on unseen data (inferance)
├── result/
│   └── nnUNet_results/Dataset002_Heart/
│       ├── nnUNetTrainer_100epochs__nnUNetPlans__3d_fullres/
│       │   └── fold_{0..4}/          # Checkpoints, per-fold validation predictions, summary.json
│       └── nnUNetTrainer_250epochs__nnUNetPlans__3d_fullres/
│           └── fold_0/               # Longer run (in progress)
├── configs/                          # Reserved for custom nnU-Net plans/overrides
├── report/
│   └── IFT3710.html                  # Final report
├── data/                             # Raw NIfTI data — gitignored, place here locally
├── requirements.txt                  # Python dependencies
└── setup_env.sh                      # Sets the nnU-Net environment variables
```

> **Note:** `.pth` checkpoint files are gitignored. Download them from the GitHub Releases page.

---

## Results

Model: `nnUNetTrainer_100epochs__nnUNetPlans__3d_fullres` — 20 training cases, 5-fold CV.

| Fold | Mean Dice | Worst case |
|------|-----------|------------|
| fold_0 | 0.9357 | la_007 (0.929) |
| fold_1 | 0.9289 | la_020 (0.896) |
| fold_2 | 0.9183 | la_019 (0.869) |
| fold_3 | 0.9385 | la_029 (0.927) |
| fold_4 | 0.9153 | la_009 (0.879) |
| **Overall** | **0.9274** | **la_019 (0.869)** |

Outliers (Dice < 0.90): `la_019`, `la_009`, `la_020`.

---

## Data Notes

- Format: NIfTI (`.nii.gz`), loaded via `nibabel`
- Labels: `0` = background, `1` = left atrium
- Raw data is excluded from git — place it in `data/Task02_Heart/` at the repo root
- Checkpoints (`.pth`) are gitignored — distribute via **GitHub Releases** (`gh release upload`)
