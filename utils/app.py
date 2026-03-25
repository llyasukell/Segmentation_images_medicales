import streamlit as st
import nibabel as nib
import matplotlib.pyplot as plt
import os

st.title("Visualisateur de Segmentation - IFT-3710")


DATA_DIR = "../data/heart/Task02_Heart"
image_path = os.path.join(DATA_DIR, "imagesTr_cropped", "la_003_0000.nii.gz")
label_path = os.path.join(DATA_DIR, "labelsTr_cropped", "la_003.nii.gz")


@st.cache_data
def load_data():
    img = nib.load(image_path).get_fdata()
    lbl = nib.load(label_path).get_fdata()
    return img, lbl

img_vol, lbl_vol = load_data()


z = st.slider("Sélectionner la coupe (Slice)", 0, img_vol.shape[2] - 1, img_vol.shape[2] // 2)

fig, ax = plt.subplots()
ax.imshow(img_vol[:, :, z].T, cmap="gray", origin="lower")
ax.imshow(lbl_vol[:, :, z].T, cmap="Reds", origin="lower", alpha=0.35)
ax.axis("off")

st.pyplot(fig)