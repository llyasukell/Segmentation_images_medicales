import os
from pathlib import Path

import nibabel as nib
import numpy as np
from scipy.ndimage import label


def keep_largest_connected_component(mask):
    labeled_mask, num_features = label(mask)

    if num_features == 0:
        return mask.astype(np.uint8)

    largest_cc_label = 0
    max_size = 0

    for i in range(1, num_features + 1):
        size = np.sum(labeled_mask == i)
        if size > max_size:
            max_size = size
            largest_cc_label = i

    cleaned_mask = (labeled_mask == largest_cc_label).astype(np.uint8)
    return cleaned_mask


def calculate_dice(pred, target):
    pred = pred.astype(bool)
    target = target.astype(bool)

    intersection = np.logical_and(pred, target).sum()
    pred_sum = pred.sum()
    target_sum = target.sum()

    if pred_sum + target_sum == 0:
        return 1.0

    return 2.0 * intersection / (pred_sum + target_sum)


# =========================================================
# 1. Path config
# =========================================================
pred_root = Path(
    "results/nnUNet_results/Dataset002_Heart/"
    "nnUNetTrainer_100epochs__nnUNetPlans__3d_fullres"
)

gt_dir = Path("results/gt_segmentations")


save_root = Path(
    "results/nnUNet_results_postprocessed/Dataset002_Heart/"
    "nnUNetTrainer_100epochs__nnUNetPlans__3d_fullres"
)
save_root.mkdir(parents=True, exist_ok=True)


pred_paths = sorted(pred_root.glob("fold_*/validation/*.nii.gz"))

print(f"Found {len(pred_paths)} prediction files.")


# =========================================================
# 2. All prediction files
# =========================================================
all_original_dice = []
all_cleaned_dice = []

for pred_path in pred_paths:
    
    fold_name = pred_path.parts[-3]

    
    filename = pred_path.name

    
    gt_path = gt_dir / filename

    if not gt_path.exists():
        print(f"Missing GT for {filename}, skipped.")
        continue

   
    pred_img = nib.load(str(pred_path))
    pred_data = pred_img.get_fdata() > 0
    gt_data = nib.load(str(gt_path)).get_fdata() > 0

  
    original_dice = calculate_dice(pred_data, gt_data)

    
    cleaned_data = keep_largest_connected_component(pred_data)

   
    cleaned_dice = calculate_dice(cleaned_data, gt_data)

    all_original_dice.append(original_dice)
    all_cleaned_dice.append(cleaned_dice)

    print(f"Fold: {fold_name} | Case: {filename}")
    print(f"  Original Dice:      {original_dice:.4f}")
    print(f"  Post-process Dice:  {cleaned_dice:.4f}")

    if cleaned_dice > original_dice:
        print("  Largest CC removed false positives; Dice improved.")
    elif cleaned_dice < original_dice:
        print("  Post-processing removed useful regions; Dice dropped.")
    else:
        print("  No change.")
    print("-" * 50)

   
    save_dir = save_root / fold_name / "validation"
    save_dir.mkdir(parents=True, exist_ok=True)

    cleaned_img = nib.Nifti1Image(
        cleaned_data.astype(np.uint8),
        affine=pred_img.affine,
        header=pred_img.header
    )
    nib.save(cleaned_img, str(save_dir / filename))


# =========================================================
# 3. summary
# =========================================================
if len(all_original_dice) > 0:
    mean_original = np.mean(all_original_dice)
    mean_cleaned = np.mean(all_cleaned_dice)

    print("\n===== SUMMARY =====")
    print(f"Mean Original Dice:     {mean_original:.4f}")
    print(f"Mean Post-process Dice: {mean_cleaned:.4f}")
    print(f"Mean Improvement:       {mean_cleaned - mean_original:.4f}")
else:
    print("No valid prediction/GT pairs found.")