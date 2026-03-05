import json
import pandas as pd
import os


with open("data/heart/Task02_Heart/dataset.json") as f:
    data = json.load(f)

train_df = pd.DataFrame(data["training"])
print(train_df.head())

# training":[{"image":"./imagesTr/la_007.nii.gz","label":"./labelsTr/la_007.nii.gz"}

def renommage(training):
    t = training
    dossier_reel = "data/heart/Task02_Heart/imagesTr/"

    for index, row in t.iterrows():
        nom_fichier_brut = os.path.basename(row["image"]) 

        if "_0000.nii.gz" in nom_fichier_brut:
            continue

        nom_fichier_neuf = nom_fichier_brut.replace(".nii.gz", "_0000.nii.gz")

        chemin_ancien = os.path.join(dossier_reel, nom_fichier_brut)  
        chemin_nouveau = os.path.join(dossier_reel, nom_fichier_neuf)  

        chemin_json_nouveau = "./imagesTr/" + nom_fichier_neuf

        if os.path.exists(chemin_ancien):
            os.rename(chemin_ancien, chemin_nouveau)

        else:
            print("INTROUVABLE:", chemin_ancien)

        t.at[index, 'image'] = chemin_json_nouveau

    return t


data["training"] = renommage(train_df).to_dict(orient="records") 

with open("data/heart/Task02_Heart/dataset.json", "w") as f:
    json.dump(data, f, indent=4)
    
