import os
import json
from pathlib import Path

import nibabel as nib
import numpy as np
import torch
from sklearn.model_selection import KFold
from tqdm import tqdm

from monai.data import Dataset, DataLoader
from monai.inferers import sliding_window_inference
from monai.losses import DiceCELoss
from monai.networks.nets import UNet
from monai.transforms import (
    Compose,
    LoadImaged,
    EnsureChannelFirstd,
    NormalizeIntensityd,
    RandCropByPosNegLabeld,
    RandFlipd,
    RandRotate90d,
    EnsureTyped,
)


# =========================================================
IMAGES_TR = "./data/heart/Task02_Heart/imagesTr"
LABELS_TR = "./data/heart/Task02_Heart/labelsTr"
RESULTS_ROOT = "./results/unet_baseline_predictions"

os.makedirs(RESULTS_ROOT, exist_ok=True)

# =========================================================

# =========================================================
N_SPLITS = 5
RANDOM_STATE = 42

MAX_EPOCHS = 100
TRAIN_BATCH_SIZE = 1
NUM_WORKERS = 0   
LR = 1e-4
WEIGHT_DECAY = 1e-5


ROI_SIZE = (96, 96, 80)
NUM_SAMPLES = 2
SW_BATCH_SIZE = 1
OVERLAP = 0.5

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)


# =========================================================
def build_datalist(images_dir, labels_dir):
    images_dir = Path(images_dir)
    labels_dir = Path(labels_dir)

    items = []
    for img_path in sorted(images_dir.glob("*.nii.gz")):
        case_id = img_path.name.replace("_0000.nii.gz", "")
        label_path = labels_dir / f"{case_id}.nii.gz"
        if label_path.exists():
            items.append({
                "case_id": case_id,
                "image": str(img_path),
                "label": str(label_path),
            })
    return items



# =========================================================
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
# 5. transforms
# =========================================================
train_transforms = Compose([
    LoadImaged(keys=["image", "label"]),
    EnsureChannelFirstd(keys=["image", "label"]),
    NormalizeIntensityd(keys=["image"], nonzero=True, channel_wise=True),

    RandCropByPosNegLabeld(
        keys=["image", "label"],
        label_key="label",
        spatial_size=ROI_SIZE,
        pos=1,
        neg=1,
        num_samples=NUM_SAMPLES,
        image_key="image",
        image_threshold=0,
        allow_smaller=True,
    ),

    RandFlipd(keys=["image", "label"], spatial_axis=0, prob=0.5),
    RandFlipd(keys=["image", "label"], spatial_axis=1, prob=0.5),
    RandFlipd(keys=["image", "label"], spatial_axis=2, prob=0.5),
    RandRotate90d(keys=["image", "label"], prob=0.5, max_k=3),

    EnsureTyped(keys=["image", "label"]),
])

# validation/export predictions
val_transforms = Compose([
    LoadImaged(keys=["image", "label"]),
    EnsureChannelFirstd(keys=["image", "label"]),
    NormalizeIntensityd(keys=["image"], nonzero=True, channel_wise=True),
    EnsureTyped(keys=["image", "label"]),
])


# =========================================================
# models
# =========================================================
def build_model():
    model = UNet(
        spatial_dims=3,
        in_channels=1,
        out_channels=2,
        channels=(16, 32, 64, 128, 256),
        strides=(2, 2, 2, 2),
        num_res_units=2,
    ).to(device)
    return model


# =========================================================
# referencee and return prediction
# =========================================================
def infer_case(model, item):
    sample = val_transforms(item)
    image_tensor = sample["image"].unsqueeze(0).to(device)  # [1, C, H, W, D]

    with torch.no_grad():
        output = sliding_window_inference(
            inputs=image_tensor,
            roi_size=ROI_SIZE,
            sw_batch_size=SW_BATCH_SIZE,
            predictor=model,
            overlap=OVERLAP,
        )

    pred = torch.argmax(output, dim=1).cpu().numpy()[0].astype(np.uint8)
    return pred


# =========================================================

# =========================================================
def validate_mean_dice(model, val_files):
    model.eval()
    case_dices = []

    for item in val_files:
        pred = infer_case(model, item)
        gt = nib.load(item["label"]).get_fdata() > 0
        dice = calculate_dice(pred > 0, gt)
        case_dices.append(dice)

    return float(np.mean(case_dices)) if case_dices else 0.0


# =========================================================
# 9. 导出当前 fold 的验证集预测 + case_dice.json
# =========================================================
def export_fold_predictions_and_metrics(model, val_files, fold_dir):
    model.eval()

    validation_dir = fold_dir / "validation"
    validation_dir.mkdir(parents=True, exist_ok=True)

    case_results = []

    for item in val_files:
        case_id = item["case_id"]
        image_path = item["image"]
        label_path = item["label"]

        pred = infer_case(model, item)

       
        orig_img = nib.load(image_path)
        pred_img = nib.Nifti1Image(pred.astype(np.uint8), affine=orig_img.affine, header=orig_img.header)

        pred_path = validation_dir / f"{case_id}.nii.gz"
        nib.save(pred_img, str(pred_path))

        gt = nib.load(label_path).get_fdata() > 0
        dice = calculate_dice(pred > 0, gt)

        case_results.append({
            "case": case_id,
            "dice": float(dice),
            "prediction_path": str(pred_path),
            "ground_truth_path": label_path,
        })

        print(f"Saved prediction: {pred_path} | Dice: {dice:.4f}")

    with open(fold_dir / "case_dice.json", "w") as f:
        json.dump(case_results, f, indent=2)

    return case_results


# =========================================================

# =========================================================
def main():
    all_files = build_datalist(IMAGES_TR, LABELS_TR)
    print(f"Total cases found: {len(all_files)}")

    if len(all_files) == 0:
        raise RuntimeError("No cases found. Check IMAGES_TR / LABELS_TR.")

    kf = KFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)

    all_fold_summaries = []

    for fold, (train_idx, val_idx) in enumerate(kf.split(all_files)):
        print("\n" + "=" * 80)
        print(f"Starting fold_{fold}")
        print("=" * 80)

        fold_dir = Path(RESULTS_ROOT) / f"fold_{fold}"
        fold_dir.mkdir(parents=True, exist_ok=True)

        train_files = [all_files[i] for i in train_idx]
        val_files = [all_files[i] for i in val_idx]

        print(f"Train cases: {len(train_files)}")
        print(f"Val cases:   {len(val_files)}")
        print("Val case IDs:", [x["case_id"] for x in val_files])

        train_ds = Dataset(data=train_files, transform=train_transforms)
        train_loader = DataLoader(
            train_ds,
            batch_size=TRAIN_BATCH_SIZE,
            shuffle=True,
            num_workers=NUM_WORKERS,
        )

        model = build_model()
        loss_function = DiceCELoss(to_onehot_y=True, softmax=True)
        optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)

        best_dice = -1.0
        best_epoch = -1
        history = []

        for epoch in range(MAX_EPOCHS):
            model.train()
            epoch_loss = 0.0
            step = 0

            for batch_data in tqdm(train_loader, desc=f"fold_{fold} Epoch {epoch+1}/{MAX_EPOCHS}"):
                step += 1

                inputs = batch_data["image"].to(device)
                labels = batch_data["label"].to(device)

                optimizer.zero_grad()
                outputs = model(inputs)
                loss = loss_function(outputs, labels)
                loss.backward()
                optimizer.step()

                epoch_loss += loss.item()

            epoch_loss /= max(step, 1)
            val_dice = validate_mean_dice(model, val_files)

            print(f"fold_{fold} | Epoch {epoch+1}/{MAX_EPOCHS} | Loss: {epoch_loss:.4f} | Val Dice: {val_dice:.4f}")

            history.append({
                "epoch": epoch + 1,
                "train_loss": float(epoch_loss),
                "val_dice": float(val_dice),
            })

            with open(fold_dir / "history.json", "w") as f:
                json.dump(history, f, indent=2)

            if val_dice > best_dice:
                best_dice = val_dice
                best_epoch = epoch + 1
                torch.save(model.state_dict(), fold_dir / "best_model.pth")
                print(f"Saved best model for fold_{fold}.")


        model.load_state_dict(torch.load(fold_dir / "best_model.pth", map_location=device))

     
        case_results = export_fold_predictions_and_metrics(model, val_files, fold_dir)

        fold_summary = {
            "fold": fold,
            "best_dice": float(best_dice),
            "best_epoch": int(best_epoch),
            "num_train_cases": len(train_files),
            "num_val_cases": len(val_files),
            "mean_case_dice": float(np.mean([x["dice"] for x in case_results])) if case_results else 0.0,
            "val_case_ids": [x["case_id"] for x in val_files],
        }

        with open(fold_dir / "summary.json", "w") as f:
            json.dump(fold_summary, f, indent=2)

        all_fold_summaries.append(fold_summary)

    global_summary = {
        "num_folds": N_SPLITS,
        "folds": all_fold_summaries,
        "mean_best_dice": float(np.mean([x["best_dice"] for x in all_fold_summaries])),
        "mean_case_dice_across_folds": float(np.mean([x["mean_case_dice"] for x in all_fold_summaries])),
    }

    with open(Path(RESULTS_ROOT) / "summary.json", "w") as f:
        json.dump(global_summary, f, indent=2)

    print("\nDone.")
    print("Global summary saved to:", Path(RESULTS_ROOT) / "summary.json")


if __name__ == "__main__":
    main()