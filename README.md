# IFT-3710 — Segmentation 3D d'images médicales

Segmentation binaire de l'**oreillette gauche** à partir d'IRM cardiaques, en utilisant le framework [nnU-Net](https://github.com/MIC-DKFZ/nnUNet) sur le dataset [Medical Segmentation Decathlon](http://medicaldecathlon.com/) (tâche Task02_Heart / Dataset002_Heart).

**Équipe :** Mamour Ndiaye (20257780) · Haoran Sun (20260543) · Walid Bouhazza (20280620)

---

## Contexte du projet

L'objectif est de segmenter automatiquement l'oreillette gauche sur des volumes IRM 3D. La sortie du modèle est un masque binaire : `0` = fond, `1` = oreillette gauche. Nous utilisons **nnU-Net v2**, un framework de segmentation médicale auto-configurant qui ajuste automatiquement l'architecture, le prétraitement et les hyperparamètres selon les propriétés du dataset.

Le dataset contient **20 cas d'entraînement** et des cas de test, chacun étant un volume 3D en format NIfTI (`.nii.gz`). Nous entraînons et évaluons avec une **validation croisée à 5 folds**.

---

## Structure du dépôt

```
IFT-3710-Projet/
│
├── preprocessing/
│   ├── convert_to_nnunet.py          # Convertit les données brutes → format nnU-Net
│   └── nnUNet_preprocessed/          # Sorties du planning nnU-Net (plans, fingerprint, splits)
│
├── training/
│   ├── nnUnet_train.sh               # Script SLURM pour le cluster (run complet 5 folds)
│   ├── train.sh                      # Entraînement en local
│   └── resume.sh                     # Reprend un entraînement interrompu
│
├── analysis/
│   ├── quantitative_analysis.ipynb   # Métriques : Dice, volumes, FP/FN, morphologie, espacement voxel
│   └── visualisationFP_FN.ipynb      # Overlays TP/FP/FN par cas, rendu 3D interactif
│
├── utils/
│   └── visualize.ipynb               # Visionneur de coupes interactif (image + label)
│
├── unet_baseline.ipynb               # Implémentation baseline U-Net 3D (comparaison)
│
├── evaluation/
│   └── predictions/                  # Prédictions du modèle sur données de test (inférence)
│
├── result/
│   └── nnUNet_results/Dataset002_Heart/
│       ├── nnUNetTrainer_100epochs__nnUNetPlans__3d_fullres/
│       │   └── fold_{0..4}/          # Checkpoints, prédictions de validation, summary.json
│       └── nnUNetTrainer_250epochs__nnUNetPlans__3d_fullres/
│           └── fold_0/               # Run plus long (exploratoire)
│
├── configs/                          # Réservé pour les plans/overrides nnU-Net personnalisés
├── report/
│   └── IFT3710.html                  # Rapport final du projet
├── data/                             # Données NIfTI brutes — gitignorées, à placer localement
├── requirements.txt                  # Dépendances Python
└── setup_env.sh                      # Configure les variables d'environnement nnU-Net
```

> **Note :** Les fichiers de checkpoint `.pth` sont gitignorés (taille trop grande). Ils sont disponibles sur la page **GitHub Releases** du dépôt.

---

## Installation

### 1. Dépendances Python

```bash
pip install -r requirements.txt
```

Packages installés : `nibabel`, `napari`, `numpy`, `matplotlib`, `ipywidgets`, `pandas`, `jupyter`, `scikit-image`, `nnunetv2`.

### 2. Variables d'environnement nnU-Net

nnU-Net requiert trois variables d'environnement pointant vers les dossiers de données :

**Linux / Mac :**
```bash
source setup_env.sh
# ou manuellement :
export nnUNet_raw="./nnUNet_raw"
export nnUNet_preprocessed="./nnUNet_preprocessed"
export nnUNet_results="./result/nnUNet_results"
```

**Windows :**
```bat
setup_env.bat
```

### 3. Données

Télécharger le dataset Task02_Heart depuis [medicaldecathlon.com](http://medicaldecathlon.com/) et le placer dans :
```
data/Task02_Heart/
├── imagesTr/      # Volumes IRM d'entraînement
├── labelsTr/      # Masques de segmentation ground truth
└── imagesTs/      # Volumes IRM de test (sans labels)
```

---

## Pipeline complet

### Étape 1 — Conversion des données

Le script `preprocessing/convert_to_nnunet.py` convertit les données brutes du format Decathlon vers le format attendu par nnU-Net :
- Renomme les fichiers avec le suffixe `_0000.nii.gz` (convention nnU-Net pour les modalités)
- Recadre les volumes pour supprimer le rembourrage vide (zero-padding)
- Génère le fichier `dataset.json` avec les métadonnées (noms des canaux, labels)

```bash
python preprocessing/convert_to_nnunet.py \
    --task-dir data/Task02_Heart \
    --images-tr-out nnUNet_raw/Dataset002_Heart/imagesTr \
    --labels-tr-out nnUNet_raw/Dataset002_Heart/labelsTr \
    --images-ts-out nnUNet_raw/Dataset002_Heart/imagesTs \
    --output-json nnUNet_raw/Dataset002_Heart/dataset.json
```

### Étape 2 — Planning et prétraitement nnU-Net

nnU-Net analyse le dataset et configure automatiquement l'architecture (taille des patches, batch size, augmentations, etc.) :

```bash
nnUNetv2_plan_and_preprocess -d 2 --verify_dataset_integrity
```

### Étape 3 — Entraînement (validation croisée 5 folds)

```bash
nnUNetv2_train 2 3d_fullres 0   # Fold 0
nnUNetv2_train 2 3d_fullres 1   # Fold 1
nnUNetv2_train 2 3d_fullres 2   # Fold 2
nnUNetv2_train 2 3d_fullres 3   # Fold 3
nnUNetv2_train 2 3d_fullres 4   # Fold 4
```

Sur le cluster de calcul (Mila/Calcul Québec) : tout est géré par `training/nnUnet_train.sh` (job SLURM, 48h max, 1 GPU, 12 CPUs, 8 GB RAM).

### Étape 4 — Analyse des résultats

Les notebooks dans `analysis/` permettent d'évaluer les performances du modèle (voir section [Notebooks](#notebooks) ci-dessous).

---

## Résultats

Modèle : `nnUNetTrainer_100epochs__nnUNetPlans__3d_fullres` — 20 cas, validation croisée 5 folds.

| Fold   | Dice moyen | Pire cas           |
|--------|------------|--------------------|
| fold_0 | 0.9357     | la_007 (0.929)     |
| fold_1 | 0.9289     | la_020 (0.896)     |
| fold_2 | 0.9183     | la_019 (0.869)     |
| fold_3 | 0.9385     | la_029 (0.927)     |
| fold_4 | 0.9153     | la_009 (0.879)     |
| **Global** | **0.9274** | **la_019 (0.869)** |

Cas outliers (Dice < 0.90) : `la_019`, `la_009`, `la_020`.

---

## Notebooks

### `analysis/quantitative_analysis.ipynb` — Analyse quantitative

Calcule des métriques détaillées sur les 20 cas de validation :

| Métrique | Description |
|----------|-------------|
| **Dice** | Coefficient de chevauchement entre prédiction et ground truth |
| **Volume GT / prédit** | Volume de l'oreillette en mL (ground truth vs. modèle) |
| **FP / FN ratio** | Proportion de faux positifs et faux négatifs |
| **Élongation** | Rapport entre les axes du volume segmenté |
| **Sphéricité** | Compacité de la forme 3D (via marching cubes) |
| **Contraste Michelson** | Contraste local oreillette vs. tissu adjacent |
| **Espacement voxel / Anisotropie** | Résolution et déséquilibre entre les axes |

Produit des visualisations : scatter plots Dice vs. volume, raincloud plots, détection des outliers.

### `analysis/visualisationFP_FN.ipynb` — Analyse qualitative

Visualisation interactive des erreurs de segmentation sur les coupes IRM :

- **Overlay 2D** : TP en vert, FP en rouge, FN en bleu — sur coupes axiale, coronale, sagittale
- **Slider interactif** : navigation coupe par coupe avec `ipywidgets`
- **Rendu 3D** : nuage de points Plotly (TP/FP/FN en couleur) pour chaque cas

### `utils/visualize.ipynb` — Visionneur de coupes

Outil simple pour visualiser n'importe quel volume NIfTI avec son masque de segmentation, utile pour l'exploration manuelle des données.

### `unet_baseline.ipynb` — Baseline U-Net 3D

Implémentation d'un U-Net 3D minimal entraîné directement sur les données, sans le pipeline nnU-Net. Sert de point de comparaison pour évaluer l'apport de nnU-Net.

---

## Format des données

- Format : **NIfTI** (`.nii.gz`), chargé via `nibabel`
- Labels : `0` = fond, `1` = oreillette gauche
- Le dossier `data/` est exclu du git (`.gitignore`) — à placer localement
- Les dossiers `imagesTr` contiennent à la fois `la_XXX.nii.gz` (original) et `la_XXX_0000.nii.gz` (format nnU-Net) — les notebooks gèrent les deux variantes
- Les checkpoints `.pth` sont gitignorés — les distribuer via **GitHub Releases** (`gh release upload`)
