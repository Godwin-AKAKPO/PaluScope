"""PaluScope Studio V0 — démo Streamlit.

Owners V0 :
- Godwin : upload + affichage image / gris + histogramme
- Bénoit : panneau QC + candidats + export JSON

Les deux parties sont assemblées ici pour la démo V0 complète.
"""

from __future__ import annotations

import base64
import io
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

from paluscope.candidates import candidate_stages, overlay_candidates  # noqa: E402
from paluscope.preprocess import to_gray  # noqa: E402
from paluscope.quality import threshold_manual, threshold_otsu_mask  # noqa: E402
from paluscope.report import (  # noqa: E402
    build_report,
    interpret_report,
    save_report_json,
)


# ---------------------------------------------------------------------------
# Configuration de la page
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="PaluScope",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
        .stApp { background-color: #faf4f3; color: #5c3438; }
        [data-testid="stHeader"] { background-color: rgba(250, 244, 243, 0.94); }
        [data-testid="stSidebar"] {
            background-color: #f3e2e0;
            border-right: 1px solid #e6d0cd;
        }
        [data-testid="stSidebar"] h1,
        [data-testid="stSidebar"] h2,
        [data-testid="stSidebar"] h3,
        [data-testid="stSidebar"] p,
        [data-testid="stSidebar"] span,
        [data-testid="stSidebar"] label,
        [data-testid="stSidebar"] small {
            color: #5c3438 !important;
        }
        [data-testid="stSidebar"] hr,
        [data-testid="stSidebar"] [data-testid="stDivider"] {
            border-color: #e4ccc9;
        }
        [data-testid="stSidebar"] .stButton button {
            background: #fffaf9;
            color: #8d454c !important;
            border: 1.5px solid #e4c4c0;
            border-radius: 999px;
        }
        [data-testid="stSidebar"] .stButton button p {
            color: #8d454c !important;
        }
        .block-container { padding-top: 1.6rem; max-width: 1180px; }
        h1, h2, h3 { font-weight: 650; letter-spacing: -0.03em; color: #5c3438; }
        [data-testid="stFileUploaderDropzone"] {
            background: #ffffff;
            border: 1.5px dashed #e0c8c5;
            border-radius: 12px;
        }
        .st-key-gallery .gallery-title {
            margin: 1.15rem 0 0.6rem;
            font-weight: 650;
            color: #6b4044;
        }
        .st-key-gallery .cell-frame {
            width: 100%;
            box-sizing: border-box;
            background: #faf4f3;
            border: 1px solid #ead4d1;
            border-radius: 14px;
            padding: 0;
            overflow: hidden;
            box-shadow: 0 1px 2px rgba(92, 52, 56, 0.06);
        }
        .st-key-gallery .cell-frame img {
            display: block;
            width: 100%;
            height: auto;
            aspect-ratio: 1;
            object-fit: cover;
            border-radius: 0;
        }
        .st-key-gallery .cell-label {
            height: 2.6rem;
            margin: 0.55rem 0 0.2rem;
            color: #7a5558;
            font-size: 0.95rem;
            line-height: 1.25rem;
        }
        div.stButton > button {
            border-radius: 999px;
            font-weight: 600;
            min-height: 2.45rem;
            padding: 0.35rem 1rem;
            box-shadow: none;
        }
        div.stButton > button[kind="primary"] {
            background: #a8565e;
            border: none;
            color: #fffaf9 !important;
            box-shadow: 0 1px 2px rgba(92, 52, 56, 0.14);
        }
        div.stButton > button[kind="primary"] p {
            color: #fffaf9 !important;
        }
        div.stButton > button[kind="primary"]:hover {
            background: #8d454c;
            border: none;
            color: #fffaf9 !important;
        }
        div.stButton > button[kind="primary"]:focus {
            box-shadow: 0 0 0 3px rgba(216, 158, 154, 0.55);
        }
        div.stButton > button[kind="secondary"] {
            background: #fffaf9;
            border: 1.5px solid #e4c4c0;
            color: #8d454c;
        }
        div.stButton > button[kind="secondary"] p {
            color: #8d454c !important;
        }
        div.stButton > button[kind="secondary"]:hover {
            background: #f6e8e6;
            border-color: #d89e9a;
            color: #8d454c;
        }
    </style>
    """,
    unsafe_allow_html=True,
)


def _sample_images() -> list[Path]:
    """Images d'exemple déjà présentes dans data/raw (png / jpg)."""
    folder = ROOT / "data" / "raw"
    files: list[Path] = []
    for pattern in ("*.png", "*.jpg", "*.jpeg"):
        files.extend(folder.glob(pattern))
    return sorted(files, key=lambda path: path.name)


def _example_family(path: Path) -> str | None:
    if "parasitized" in path.name:
        return "parasitée"
    if "uninfected" in path.name:
        return "non parasitée"
    return None


def _sample_label(path: Path) -> str:
    family = _example_family(path)
    nice = path.stem.replace("parasitized_", "").replace("uninfected_", "").replace("_", " ")
    if family:
        return f"{family.capitalize()} — {nice}"
    return path.name


if "selection" not in st.session_state:
    st.session_state.selection = None
if "uploader_key" not in st.session_state:
    st.session_state.uploader_key = 0

samples = _sample_images()

with st.sidebar:
    st.header("PaluScope")
    st.caption("Frottis mince · aide à la lecture · V0")
    st.divider()
    st.markdown("**Réglage**")
    t_manual = st.slider(
        "Seuil manuel",
        0.0,
        1.0,
        0.45,
        0.05,
        help="Sur l'image en gris : au-dessus de ce seuil, le pixel devient blanc.",
    )
    st.divider()
    st.markdown("**Comment lire**")
    st.markdown(
        "Le cadre rouge entoure une tache plus magenta que le cytoplasme. "
        "Ce n'est pas un parasite confirmé."
    )
    st.markdown(
        "L'étiquette « parasitée » ou « non parasitée » vient du jeu d'exemple. "
        "Elle sert à comparer votre lecture, pas à valider un diagnostic."
    )
    st.divider()
    st.caption(
        "Prototype de recherche et de formation. "
        "Aucun diagnostic, aucun traitement."
    )
    if st.session_state.selection is not None and st.button("Changer d'image", width="stretch"):
        st.session_state.selection = None
        st.session_state.uploader_key += 1
        st.rerun()


def _remember_upload(uploaded) -> None:
    payload = uploaded.getvalue()
    token = f"{uploaded.name}:{len(payload)}"
    if st.session_state.get("upload_token") == token:
        return
    st.session_state.upload_token = token
    st.session_state.selection = {"kind": "upload", "name": uploaded.name, "bytes": payload}
    st.rerun()


def _square_preview(path: Path, side: int = 480) -> str:
    """Recadre la cellule pour qu'elle couvre tout le carré, sans la déformer."""
    image = Image.open(path).convert("RGB")
    scale = max(side / image.width, side / image.height)
    resized = image.resize(
        (max(1, round(image.width * scale)), max(1, round(image.height * scale))),
        Image.Resampling.LANCZOS,
    )
    left = max(0, (resized.width - side) // 2)
    top = max(0, (resized.height - side) // 2)
    canvas = resized.crop((left, top, left + side, top + side))
    buffer = io.BytesIO()
    canvas.save(buffer, format="PNG")
    encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
    return f"data:image/png;base64,{encoded}"


def _show_gallery(title: str, paths: list[Path]) -> None:
    if not paths:
        return
    st.markdown(f"<p class='gallery-title'>{title}</p>", unsafe_allow_html=True)
    for start in range(0, len(paths), 3):
        row = paths[start : start + 3]
        columns = st.columns(3, gap="medium")
        for column, path in zip(columns, row):
            with column:
                st.markdown(
                    f"<div class='cell-frame'><img src='{_square_preview(path)}' alt=''></div>",
                    unsafe_allow_html=True,
                )
                st.markdown(
                    f"<p class='cell-label'>{_sample_label(path)}</p>",
                    unsafe_allow_html=True,
                )
                if st.button(
                    "Tester cette cellule",
                    key=f"test-{path.name}",
                    type="primary",
                    width="stretch",
                ):
                    st.session_state.selection = {"kind": "sample", "path": str(path)}
                    st.rerun()


selection = st.session_state.selection
if selection is None:
    st.title("Lecture d'un frottis")
    st.markdown(
        "Déposez une image, ou choisissez une cellule du jeu d'exemple. "
        "PaluScope enchaîne la qualité de la photo et les taches à regarder. "
        "**Ce n'est pas un diagnostic médical.**"
    )
    uploaded = st.file_uploader(
        "Déposez une image de frottis",
        type=["png", "jpg", "jpeg"],
        key=f"upload-{st.session_state.uploader_key}",
    )
    if uploaded is not None:
        _remember_upload(uploaded)
    st.markdown("… ou choisissez une cellule du jeu d'exemple :")
    if not samples:
        st.info("Aucune image d'exemple dans data/raw pour le moment.")
    else:
        with st.container(key="gallery"):
            _show_gallery(
                "Cellules étiquetées parasitées",
                [path for path in samples if _example_family(path) == "parasitée"],
            )
            _show_gallery(
                "Cellules étiquetées non parasitées",
                [path for path in samples if _example_family(path) == "non parasitée"],
            )
            other = [path for path in samples if _example_family(path) is None]
            _show_gallery("Autres images", other)
    st.stop()

if selection["kind"] == "sample":
    chosen_path = Path(selection["path"])
    pil = Image.open(chosen_path).convert("RGB")
    source_name = chosen_path.name
    example_family = _example_family(chosen_path)
else:
    pil = Image.open(io.BytesIO(selection["bytes"])).convert("RGB")
    source_name = selection["name"]
    example_family = None

st.title("Lecture en cours")
st.caption(f"Image : `{source_name}`")

# ---------------------------------------------------------------------------
# Godwin : chargement en RGB (PIL → NumPy)
# ---------------------------------------------------------------------------
rgb = np.asarray(pil)  # uint8 (H, W, 3)

# Conversion en niveaux de gris float [0, 1] via notre module preprocess
gray = to_gray(rgb)

# ---------------------------------------------------------------------------
# Bénoit : rapport QC + étapes de détection (une seule fois)
# ---------------------------------------------------------------------------
report = build_report(rgb, source=source_name)
stages = candidate_stages(rgb)
mask = stages["mask"]
hsv = stages["hsv"]
n_stains = int(report["candidates"]["n_candidates"])

st.subheader("Fiche de lecture")
metric_a, metric_b, metric_c = st.columns(3)
metric_a.metric("Taches encadrées", n_stains)
metric_b.metric("Luminosité", f"{report['quality']['mean']:.2f} sur 1")
metric_c.metric("Netteté", f"{report['quality']['sharpness_laplacian']:.0f}")

if example_family == "parasitée" and n_stains == 0:
    st.warning(
        "Cet exemple est étiqueté parasitée, mais aucun cadre n'a été posé. "
        "La tache est peut-être trop pâle pour le filtre. Regardez la cellule quand même."
    )
elif example_family == "parasitée":
    st.info(
        "Cet exemple est étiqueté parasitée, et des cadres ont été posés. "
        "Comparez-les à ce que vous voyez : un cadre n'est pas une confirmation."
    )
elif example_family == "non parasitée" and n_stains == 0:
    st.info(
        "Cet exemple est étiqueté non parasitée, et aucun cadre n'a été posé. "
        "Une tache très discrète peut tout de même être présente."
    )
elif example_family == "non parasitée":
    st.warning(
        "Cet exemple est étiqueté non parasitée, et pourtant des cadres ont été posés. "
        "Ce peut être un repère en trop : un noyau ou un dépôt de colorant."
    )
else:
    st.info(
        "Fichier personnel : il n'y a pas d'étiquette de référence. "
        "Les cadres sont seulement des zones à regarder."
    )

with col3:
    # Bénoit : visualisation du masque de zones candidates
    st.subheader("Zones candidates")
    st.image(mask.astype(float), clamp=True, use_container_width=True)
    st.caption(f"composantes connexes = {report['candidates']['n_candidates']}")

# ---------------------------------------------------------------------------
# Succession des opérations, dans l'ordre du code
# ---------------------------------------------------------------------------
st.subheader("Parcours de l'image")
st.caption("Les étapes suivent l'ordre du code, de la photo jusqu'aux cadres.")


def show_pair(steps: list[dict]) -> None:
    """Titres sur une ligne, images sur la suivante, pour qu'elles soient alignées."""
    title_cols = st.columns(len(steps))
    for col, step in zip(title_cols, steps):
        with col:
            st.markdown(f"**{step['index']}. {step['title']}** ({step['explanation']})")
    image_cols = st.columns(len(steps))
    for col, step in zip(image_cols, steps):
        with col:
            st.image(step["image"], width="stretch", clamp=step.get("clamp", False))
            if step.get("caption"):
                st.caption(step["caption"])


show_pair(
    [
        {
            "index": 1,
            "title": "Image RGB",
            "explanation": "le fichier est lu et converti en trois canaux rouge, vert, bleu",
            "image": rgb,
            "caption": f"shape={rgb.shape} | dtype={rgb.dtype}",
        },
        {
            "index": 2,
            "title": "Niveaux de gris",
            "explanation": "to_gray mélange rouge, vert et bleu en une intensité entre 0 et 1",
            "image": gray,
            "clamp": True,
        },
    ]
)

bin_manual = threshold_manual(gray, t_manual)
t_otsu, bin_otsu = threshold_otsu_mask(gray)
show_pair(
    [
        {
            "index": 3,
            "title": "Seuillage manuel",
            "explanation": "sur le gris : un pixel plus clair que T devient blanc, sinon noir",
            "image": bin_manual.astype(float),
            "clamp": True,
            "caption": f"pixel > {t_manual:.2f} → blanc",
        },
        {
            "index": 4,
            "title": "Seuillage d'Otsu",
            "explanation": "sur le même gris : T est choisi automatiquement pour séparer foncé et clair",
            "image": bin_otsu.astype(float),
            "clamp": True,
            "caption": f"pixel > {t_otsu:.2f} → blanc",
        },
    ]
)

st.markdown(
    "**5. Histogramme** "
    "(compte combien de pixels du gris ont chaque intensité)"
)
fig, ax = plt.subplots(figsize=(7, 2.8))
ax.hist(gray.ravel(), bins=64, range=(0, 1), color="steelblue")
ax.axvline(t_manual, color="crimson", linestyle="--", label=f"Manuel {t_manual:.2f}")
ax.axvline(t_otsu, color="darkorange", linestyle="-", label=f"Otsu {t_otsu:.2f}")
ax.set_xlabel("Intensité")
ax.set_ylabel("Nombre de pixels")
ax.legend()
st.pyplot(fig)
plt.close(fig)

st.markdown(
    "**6. HSV** "
    "(to_hsv sépare teinte, saturation et luminosité ; la détection repart de la couleur, pas du gris)"
)
show_pair(
    [
        {
            "index": "H",
            "title": "Teinte",
            "explanation": "le rose est près de 0, le magenta près de 1",
            "image": hsv[:, :, 0],
            "clamp": True,
        },
        {
            "index": "S",
            "title": "Saturation",
            "explanation": "couleur vive ou pâle",
            "image": hsv[:, :, 1],
            "clamp": True,
        },
        {
            "index": "V",
            "title": "Luminosité",
            "explanation": "le fond noir est proche de 0",
            "image": hsv[:, :, 2],
            "clamp": True,
        },
    ]
)

show_pair(
    [
        {
            "index": 7,
            "title": "Cellule",
            "explanation": "on écarte le fond noir, là où la luminosité V est trop basse",
            "image": stages["cell"].astype(float),
            "clamp": True,
        },
        {
            "index": 8,
            "title": "Pixels magenta",
            "explanation": "dans la cellule, teinte magenta et saturation plus forte que le cytoplasme rose",
            "image": stages["color"].astype(float),
            "clamp": True,
        },
    ]
)
show_pair(
    [
        {
            "index": 9,
            "title": "Masque nettoyé",
            "explanation": "on enlève les trous et les taches trop petites ou trop grandes",
            "image": mask.astype(float),
            "clamp": True,
        },
        {
            "index": 10,
            "title": "Zones candidates",
            "explanation": "un cadre rouge entoure chaque tache, qui garde sa couleur",
            "image": overlay_candidates(rgb, mask),
            "caption": f"composantes connexes = {report['candidates']['n_candidates']}",
        },
    ]
)

# ---------------------------------------------------------------------------
# Bénoit : panneau indicateurs QC
# ---------------------------------------------------------------------------
st.divider()
st.subheader("Que disent les chiffres ?")
st.caption(
    "Lecture simple de la photo. Ce n'est pas un diagnostic, "
    "et le nombre de taches n'est pas un nombre de parasites."
)
reading = interpret_report(report)
st.dataframe(
    reading,
    hide_index=True,
    width="stretch",
    column_config={
        "Indicateur": st.column_config.TextColumn("Indicateur", width="small"),
        "Valeur": st.column_config.TextColumn("Valeur", width="small"),
        "En clair": st.column_config.TextColumn("En clair", width="large"),
    },
)

# ---------------------------------------------------------------------------
# Bénoit : export du rapport JSON vers outputs/reports/
# ---------------------------------------------------------------------------
out_dir = ROOT / "outputs" / "reports"
out_path = out_dir / f"report_{Path(source_name).stem}.json"

if st.button("Exporter le rapport JSON"):
    saved = save_report_json(report, out_path)
    st.success(f"Rapport écrit : `{saved}`")
