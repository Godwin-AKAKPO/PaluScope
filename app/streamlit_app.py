"""PaluScope Studio V0 — démo Streamlit.

Owners V0 :
- Godwin : upload + affichage image / gris + histogramme
- Bénoit : panneau QC + candidats + export JSON

Les deux parties sont assemblées ici pour la démo V0 complète.
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
from paluscope.report import (  # noqa: E402
    build_report,
    report_to_dataframe,
    save_report_json,
)


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
# Bénoit : construire le rapport QC + candidats (une seule fois)
# ---------------------------------------------------------------------------
report = build_report(rgb, source=uploaded.name)
mask = report["_mask"]  # masque booléen pour l'affichage

# ---------------------------------------------------------------------------
# Affichage : RGB (Godwin) + gris (Godwin) + candidats (Bénoit)
# ---------------------------------------------------------------------------
col1, col2, col3 = st.columns(3)

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

with col3:
    # Bénoit : visualisation du masque de zones candidates
    st.subheader("Zones candidates")
    st.image(mask.astype(float), clamp=True, use_container_width=True)
    st.caption(f"composantes connexes = {report['candidates']['n_candidates']}")

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
# Bénoit : panneau indicateurs QC
# ---------------------------------------------------------------------------
st.divider()
st.subheader("Indicateurs de qualité (QC)")
st.metric(
    "Candidats (composantes connexes)",
    report["candidates"]["n_candidates"],
)
# Tableau aplati (une ligne) pour lire mean/std/michelson/sharpness/expo…
df = report_to_dataframe(report)
st.dataframe(df, use_container_width=True)

# ---------------------------------------------------------------------------
# Bénoit : export du rapport JSON vers outputs/reports/
# ---------------------------------------------------------------------------
out_dir = ROOT / "outputs" / "reports"
out_path = out_dir / f"report_{Path(uploaded.name).stem}.json"

if st.button("Exporter le rapport JSON"):
    saved = save_report_json(report, out_path)
    st.success(f"Rapport écrit : `{saved}`")