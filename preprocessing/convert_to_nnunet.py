import json
import os
import pandas as pd


def renommage(training, chemin_tr):
    t = training.copy()
    for index, row in t.iterrows():
        nom_fichier_brut = os.path.basename(row["image"])
        if "_0000.nii.gz" in nom_fichier_brut:
            t.at[index, "image"] = "./imagesTr/" + nom_fichier_brut
            continue
        nom_fichier_neuf = nom_fichier_brut.replace(".nii.gz", "_0000.nii.gz")
        chemin_ancien = os.path.join(chemin_tr, nom_fichier_brut)
        chemin_nouveau = os.path.join(chemin_tr, nom_fichier_neuf)
        chemin_json_nouveau = "./imagesTr/" + nom_fichier_neuf
        if os.path.exists(chemin_ancien):
            os.rename(chemin_ancien, chemin_nouveau)
            print(f"OK Train : {nom_fichier_brut} -> {nom_fichier_neuf}")
        else:
            print("INTROUVABLE Train:", chemin_ancien)
        t.at[index, "image"] = chemin_json_nouveau
    return t

def renommage_test(test_list, chemin_ts):
    updated = []
    for image_path in test_list:
        nom_fichier_brut = os.path.basename(image_path)
        if "_0000.nii.gz" in nom_fichier_brut:
            updated.append("./imagesTs/" + nom_fichier_brut)
            continue
        nom_fichier_neuf = nom_fichier_brut.replace(".nii.gz", "_0000.nii.gz")
        chemin_ancien = os.path.join(chemin_ts, nom_fichier_brut)
        chemin_nouveau = os.path.join(chemin_ts, nom_fichier_neuf)
        if os.path.exists(chemin_ancien):
            os.rename(chemin_ancien, chemin_nouveau)
            print(f"OK Test : {nom_fichier_brut} -> {nom_fichier_neuf}")
        else:
            print("INTROUVABLE Test:", chemin_ancien)
        updated.append("./imagesTs/" + nom_fichier_neuf)
    return updated






#########################Crop

import argparse
import json
import os
from dataclasses import dataclass
from typing import Dict, List, Tuple

import nibabel as nib
import numpy as np


@dataclass
class CropBox:
    z_min: int
    z_max: int
    y_min: int
    y_max: int
    x_min: int
    x_max: int

    def to_dict(self) -> Dict[str, int]:
        return {
            "z_min": self.z_min,
            "z_max": self.z_max,
            "y_min": self.y_min,
            "y_max": self.y_max,
            "x_min": self.x_min,
            "x_max": self.x_max,
        }


def compute_nonzero_crop_box(image_data: np.ndarray) -> CropBox:
    nonzero = np.argwhere(image_data != 0)
    if nonzero.size == 0:
        shape = image_data.shape
        return CropBox(0, shape[0] - 1, 0, shape[1] - 1, 0, shape[2] - 1)

    z_min, y_min, x_min = nonzero.min(axis=0)
    z_max, y_max, x_max = nonzero.max(axis=0)
    return CropBox(int(z_min), int(z_max), int(y_min), int(y_max), int(x_min), int(x_max))


def apply_crop(data: np.ndarray, box: CropBox) -> np.ndarray:
    return data[
        box.z_min : box.z_max + 1,
        box.y_min : box.y_max + 1,
        box.x_min : box.x_max + 1,
    ]


def save_nifti_like(reference_nii: nib.Nifti1Image, cropped_data: np.ndarray, output_path: str) -> None:
    cropped_nii = nib.Nifti1Image(cropped_data, reference_nii.affine, reference_nii.header)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    nib.save(cropped_nii, output_path)


def load_dataset_json(dataset_json_path: str) -> Dict:
    with open(dataset_json_path, "r", encoding="utf-8") as f:
        return json.load(f)


def write_dataset_json(data: Dict, output_path: str) -> None:
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4)


def abs_from_rel(task_dir: str, rel_path: str) -> str:
    return os.path.normpath(os.path.join(task_dir, rel_path.lstrip("./")))


def rel_from_abs(task_dir: str, abs_path: str) -> str:
    rel = os.path.relpath(abs_path, task_dir).replace("\\", "/")
    return f"./{rel}"


def process_training_case(task_dir: str, item: Dict[str, str], out_images_dir: str, out_labels_dir: str) -> Tuple[Dict[str, str], Dict]:
    image_abs = abs_from_rel(task_dir, item["image"])
    label_abs = abs_from_rel(task_dir, item["label"])

    image_nii = nib.load(image_abs)
    image_data = image_nii.get_fdata()
    box = compute_nonzero_crop_box(image_data)

    cropped_img = apply_crop(image_data, box)

    label_nii = nib.load(label_abs)
    label_data = label_nii.get_fdata()
    cropped_lbl = apply_crop(label_data, box)

    image_name = os.path.basename(image_abs)

    label_name = os.path.basename(label_abs).replace("_0000.nii.gz", ".nii.gz")
    out_image_abs = os.path.join(out_images_dir, image_name)
    out_label_abs = os.path.join(out_labels_dir, label_name)

    save_nifti_like(image_nii, cropped_img, out_image_abs)
    save_nifti_like(label_nii, cropped_lbl, out_label_abs)

    new_item = {
        "image": rel_from_abs(task_dir, out_image_abs),
        "label": rel_from_abs(task_dir, out_label_abs),
    }
    crop_info = {
        "image": item["image"],
        "label": item["label"],
        "crop_box": box.to_dict(),
        "original_shape": list(image_data.shape),
        "cropped_shape": list(cropped_img.shape),
    }
    return new_item, crop_info


def process_test_case(task_dir: str, test_image_rel: str, out_images_ts_dir: str) -> Tuple[str, Dict]:
    image_abs = abs_from_rel(task_dir, test_image_rel)
    image_nii = nib.load(image_abs)
    image_data = image_nii.get_fdata()

    box = compute_nonzero_crop_box(image_data)
    cropped_img = apply_crop(image_data, box)

    image_name = os.path.basename(image_abs)
    out_image_abs = os.path.join(out_images_ts_dir, image_name)
    save_nifti_like(image_nii, cropped_img, out_image_abs)

    crop_info = {
        "image": test_image_rel,
        "crop_box": box.to_dict(),
        "original_shape": list(image_data.shape),
        "cropped_shape": list(cropped_img.shape),
    }
    return rel_from_abs(task_dir, out_image_abs), crop_info


def main() -> None:
    
    parser = argparse.ArgumentParser(description="Crop Decathlon Heart data to nnU-Net format.")
    parser.add_argument("--task-dir", default="data/heart/Task02_Heart", help="Task02_Heart directory")
    parser.add_argument("--dataset-json", default="dataset.json", help="Dataset JSON filename under task-dir")
    
    # Chemins de sortie 
    parser.add_argument("--images-tr-out", default="nnUNet_raw/Dataset002_Heart/imagesTr", help="Output for training images")
    parser.add_argument("--labels-tr-out", default="nnUNet_raw/Dataset002_Heart/labelsTr", help="Output for training labels")
    parser.add_argument("--images-ts-out", default="nnUNet_raw/Dataset002_Heart/imagesTs", help="Output for test images")
    parser.add_argument("--output-json", default="nnUNet_raw/Dataset002_Heart/dataset.json", help="Final dataset json")
    parser.add_argument("--report-json", default="crop_report.json", help="Crop report filename under task-dir")
    args = parser.parse_args()
    out_images_tr = args.images_tr_out
    out_labels_tr = args.labels_tr_out
    out_images_ts = args.images_ts_out


    task_dir = args.task_dir
    json_path = os.path.join(task_dir, args.dataset_json)

    #renommage
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    tr_dir = os.path.join(task_dir, "imagesTr")
    ts_dir = os.path.join(task_dir, "imagesTs")

    train_df = pd.DataFrame(data["training"])
    data["training"] = renommage(train_df, tr_dir).to_dict(orient="records")
    data["test"] = renommage_test(data.get("test", []), ts_dir)

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4)


    print("Fichier renommé")




    os.makedirs(out_images_tr, exist_ok=True)
    os.makedirs(out_labels_tr, exist_ok=True)
    os.makedirs(out_images_ts, exist_ok=True)

    new_training: List[Dict[str, str]] = []
    training_report: List[Dict] = []
    for item in data.get("training", []):
        new_item, info = process_training_case(task_dir, item, out_images_tr, out_labels_tr)
        new_training.append(new_item)
        training_report.append(info)

    new_test: List[str] = []
    test_report: List[Dict] = []
    for test_image_rel in data.get("test", []):
        new_test_rel, info = process_test_case(task_dir, test_image_rel, out_images_ts)
        new_test.append(new_test_rel)
        test_report.append(info)

    out_data = {
        "channel_names": {  
            "0": "MRI"
        },
        "labels": {         
            "background": 0,
            "heart": 1
        },
        "numTraining": len(new_training),
        "file_ending": ".nii.gz",            
        "training": new_training,
        "test": new_test
    }

    output_json_path = args.output_json
    report_json_path = os.path.join(task_dir, args.report_json)

    write_dataset_json(out_data, output_json_path)
    write_dataset_json({"training": training_report, "test": test_report}, report_json_path)

    print(f"Wrote cropped dataset json: {os.path.abspath(output_json_path)}")
    print(f"Wrote crop report: {report_json_path}")


if __name__ == "__main__":
    main()
