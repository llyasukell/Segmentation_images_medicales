import json
import pandas as pd
import os

PATH_BASE = "../data/heart/Task02_Heart/"
PATH_JSON = os.path.join(PATH_BASE, "dataset.json")
PATH_IMAGES = os.path.join(PATH_BASE, "imagesTr/")

with open(PATH_JSON) as f:
    data = json.load(f)

train_df = pd.DataFrame(data["training"])

def renommage(training):
    t = training
    
    for index, row in t.iterrows():
        nom_fichier_brut = os.path.basename(row["image"]) 

        if "_0000.nii.gz" in nom_fichier_brut:
            continue

        nom_fichier_neuf = nom_fichier_brut.replace(".nii.gz", "_0000.nii.gz")

        chemin_ancien = os.path.join(PATH_IMAGES, nom_fichier_brut)  
        chemin_nouveau = os.path.join(PATH_IMAGES, nom_fichier_neuf)  
        chemin_json_nouveau = "./imagesTr/" + nom_fichier_neuf

        if os.path.exists(chemin_ancien):
            os.rename(chemin_ancien, chemin_nouveau)
            print(f"OK : {nom_fichier_brut} -> {nom_fichier_neuf}")
        
        t.at[index, 'image'] = chemin_json_nouveau

    return t

data["training"] = renommage(train_df).to_dict(orient="records") 

with open(PATH_JSON, "w") as f:
    json.dump(data, f, indent=4)


