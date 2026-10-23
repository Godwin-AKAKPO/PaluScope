"""PaluScope Studio V0 — démo Streamlit.

Owners V0 :
- Godwin : upload + affichage image / gris + histogramme  (godwin/streamlit-upload)
- Bénoit : panneau QC + candidats + export                (benoit/streamlit-qc-panel)

Cette version implémente UNIQUEMENT la partie Godwin.
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import streamlit as st
from PIL import Image

# ---------------------------------------------------------------------------
# Import du package local src/paluscope sans installation pip editable
# ---------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from paluscope.preprocess import to_gray  # noqa: E402


# ---------------------------------------------------------------------------
# Configuration de la page
# ---------------------------------------------------------------------------
st.set_page_config(page_title="PaluScope V0", layout="wide")

# Titre + disclaimer médical (obligatoire cahier des charges)
st.title("PaluScope Studio — V0 Quality Analyzer")
st.caption(
    "Prototype de recherche / formation / assistance à la lecture. "
    "**Ne constitue pas un diagnostic médical.** "
    "Aucun traitement n'est prescrit."
)

# ---------------------------------------------------------------------------
# Godwin : import d'une image de frottis
# ---------------------------------------------------------------------------
uploaded = st.file_uploader(
    "Importer une image de frottis (png / jpg)",
    type=["png", "jpg", "jpeg"],
)

# Si aucune image : on s'arrête proprement (évite d'exécuter la suite)
if uploaded is None:
    st.info("Chargez une image pour commencer l'analyse visuelle.")
    st.stop()

# ---------------------------------------------------------------------------
# Godwin : chargement en RGB (PIL → NumPy)
# ---------------------------------------------------------------------------
# convert("RGB") force 3 canaux même si le fichier est en niveaux de gris / RGBA
pil = Image.open(uploaded).convert("RGB")
rgb = np.asarray(pil)  # uint8 (H, W, 3)

# Conversion en niveaux de gris float [0, 1] via notre module preprocess
gray = to_gray(rgb)

# ---------------------------------------------------------------------------
# Godwin : affichage côte à côte (RGB + gris)
# ---------------------------------------------------------------------------
col1, col2 = st.columns(2)

with col1:
    st.subheader("Image RGB")
    st.image(rgb, use_container_width=True)
    st.caption(f"shape={rgb.shape} | dtype={rgb.dtype}")

with col2:
    st.subheader("Niveaux de gris")
    # clamp=True : interprète bien les floats [0,1]
    st.image(gray, clamp=True, use_container_width=True)
    st.caption(
        f"shape={gray.shape} | "
        f"min={gray.min():.3f} | max={gray.max():.3f} | mean={gray.mean():.3f}"
    )

# ---------------------------------------------------------------------------
# Godwin : histogramme grayscale (aperçu distribution des intensités)
# ---------------------------------------------------------------------------
st.subheader("Histogramme (niveaux de gris)")
fig, ax = plt.subplots(figsize=(7, 2.8))
# .ravel() aplatit l'image 2D → 1D pour plt.hist
ax.hist(gray.ravel(), bins=64, range=(0, 1), color="steelblue")
ax.set_xlabel("Intensité")
ax.set_ylabel("Nombre de pixels")
ax.set_title("Distribution des intensités (float [0, 1])")
st.pyplot(fig)
plt.close(fig)  # libère la figure Matplotlib (important dans Streamlit)

# ---------------------------------------------------------------------------
# Bénoit (TODO) : panneau QC, masque candidats, export JSON
# ---------------------------------------------------------------------------
st.divider()
st.info(
    "Prochaine étape (Bénoit) : indicateurs QC (netteté, exposition, contraste), "
    "zones candidates et export du rapport JSON."
)
