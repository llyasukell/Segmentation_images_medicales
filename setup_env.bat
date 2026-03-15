@echo off

echo Configuration des variables pour nnU-Net
call .venv\Scripts\activate

set nnUNet_raw=%~dp0nnUNet_raw
set nnUNet_preprocessed=%~dp0nnUNet_preprocessed
set nnUNet_results=%~dp0nnUNet_results

echo Fini.
echo Dossier raw actuel : %nnUNet_raw%

cmd /k