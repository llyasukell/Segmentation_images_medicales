import json
import os
import pandas as pd

PATH_BASE = os.path.join("..", "data", "heart", "Task02_Heart")
PATH_JSON = os.path.join(PATH_BASE, "dataset.json")
IMAGES_TR_DIR = os.path.join(PATH_BASE, "imagesTr")
IMAGES_TS_DIR = os.path.join(PATH_BASE, "imagesTs")

def renommage(training):
    t = training.copy()
    for index, row in t.iterrows():
        nom_fichier_brut = os.path.basename(row["image"])
        if "_0000.nii.gz" in nom_fichier_brut:
            t.at[index, "image"] = "./imagesTr/" + nom_fichier_brut
            continue
        nom_fichier_neuf = nom_fichier_brut.replace(".nii.gz", "_0000.nii.gz")
        chemin_ancien = os.path.join(IMAGES_TR_DIR, nom_fichier_brut)
        chemin_nouveau = os.path.join(IMAGES_TR_DIR, nom_fichier_neuf)
        chemin_json_nouveau = "./imagesTr/" + nom_fichier_neuf
        if os.path.exists(chemin_ancien):
            os.rename(chemin_ancien, chemin_nouveau)
            print(f"OK Train : {nom_fichier_brut} -> {nom_fichier_neuf}")
        else:
            print("INTROUVABLE Train:", chemin_ancien)
        t.at[index, "image"] = chemin_json_nouveau
    return t

def renommage_test(test_list):
    updated = []
    for image_path in test_list:
        nom_fichier_brut = os.path.basename(image_path)
        if "_0000.nii.gz" in nom_fichier_brut:
            updated.append("./imagesTs/" + nom_fichier_brut)
            continue
        nom_fichier_neuf = nom_fichier_brut.replace(".nii.gz", "_0000.nii.gz")
        chemin_ancien = os.path.join(IMAGES_TS_DIR, nom_fichier_brut)
        chemin_nouveau = os.path.join(IMAGES_TS_DIR, nom_fichier_neuf)
        if os.path.exists(chemin_ancien):
            os.rename(chemin_ancien, chemin_nouveau)
            print(f"OK Test : {nom_fichier_brut} -> {nom_fichier_neuf}")
        else:
            print("INTROUVABLE Test:", chemin_ancien)
        updated.append("./imagesTs/" + nom_fichier_neuf)
    return updated

with open(PATH_JSON, "r", encoding="utf-8") as f:
    data = json.load(f)

train_df = pd.DataFrame(data["training"])
data["training"] = renommage(train_df).to_dict(orient="records")
data["test"] = renommage_test(data.get("test", []))

with open(PATH_JSON, "w", encoding="utf-8") as f:
    json.dump(data, f, indent=4)

print("Fini")