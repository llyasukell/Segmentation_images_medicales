# IFT-3710 – Projet : Segmentation d’images médicales

## Équipe
- **Mamour Ndiaye** – 20257780  
- **Haoran Sun** – 20260543  
- **Walid Bouhazza** – 20280620  

---

## Description du projet
Ce projet porte sur le problème de la segmentation d'images médicales 3D, c'est-à-dire la classification des voxels de ces images en différentes catégories. Il propose d'utiliser le Décathlon de segmentation médicale, qui comprend 10 jeux de données d'images issues de différentes modalités, telles que l'imagerie par résonance magnétique (IRM) ou la tomodensitométrie (TDM), provenant de différentes parties du corps humain.

---

## Structure du projet

```
project/
├── preprocessing/          # conversion des données
│   └── convert_to_nnunet.py
├── utils/                  # acripts utilitaires réutilisables
│   └── visualize.ipynb
├── configs/                # paramètres nnU-Net 
├── training/               # scripts de lancement des folds
├── evaluation/             # inférence, Dice, visualisations
├── results/
│   ├── figures/            # PNG des visualisations
│   └── scores.csv          # Scores Dice par cas
├── report/                 # rapport final
│   └── IFT3710.html
└── data/                   # Données brutes (gitignored)

```

---

## Installation

```bash
pip install -r requirements.txt
```

