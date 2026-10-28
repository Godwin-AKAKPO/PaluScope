"""Agrégation et export du rapport expérimental V0 (Bénoit).

Owner V0 : Bénoit
Branche type : benoit/report-json

Assemble :
    QC (analyze_quality) + candidats + disclaimer + timestamp
puis permet l'affichage (DataFrame) et l'export JSON.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from .candidates import candidate_summary
from .quality import analyze_quality


def build_report(rgb: np.ndarray, source: str = "upload") -> dict:
    """Construit un rapport QC + candidats pour une image RGB.

    Paramètres
    ----------
    rgb :
        Image RGB.
    source :
        Origine lisible (nom de fichier uploadé, chemin, etc.).

    Retour
    ------
    dict
        - champs sérialisables (quality, candidates scalaires, disclaimer…)
        - clé privée `_mask` pour l'UI Streamlit (non exportée en JSON)
    """
    # 1) Indicateurs techniques (Bénoit quality)
    quality = analyze_quality(rgb)
    # 2) Zones candidates (Bénoit candidates)
    cands = candidate_summary(rgb)

    report = {
        # Horodatage UTC ISO (traçabilité de l'expérience)
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "source": source,
        # Rappel médical obligatoire
        "disclaimer": (
            "Prototype de recherche PaluScope — pas un diagnostic clinique."
        ),
        "quality": {k: v for k, v in quality.items()},
        "candidates": {
            "n_candidates": cands["n_candidates"],
            "mask_area_px": cands["mask_area_px"],
        },
        # Objet volumineux : utile à l'affichage, exclu de l'export JSON
        "_mask": cands["mask"],
    }
    return _attach_interpretations(report)


def _attach_interpretations(report: dict) -> dict:
    """Ajoute les phrases lisibles au rapport, pour l'écran et le JSON."""
    report["interpretations"] = interpret_report(report).to_dict(orient="records")
    return report


def _reading_brightness(mean: float) -> str:
    if mean < 0.35:
        return "L'image est plutôt sombre. Les détails dans les zones foncées sont plus difficiles à voir."
    if mean > 0.75:
        return "L'image est plutôt claire. Si elle tire vers le blanc, les taches peuvent s'effacer."
    return "La luminosité est moyenne : ni trop sombre, ni trop blanche."


def _reading_contrast(michelson: float) -> str:
    if michelson < 0.25:
        return "Peu d'écart entre le pixel le plus sombre et le plus clair. L'image paraît plate."
    if michelson > 0.8:
        return (
            "Grand écart entre le plus sombre et le plus clair. "
            "Sur une cellule détourée, le fond noir fait souvent monter ce chiffre, "
            "même si l'intérieur du globule est assez uniforme."
        )
    return "On distingue des zones plus claires et des zones plus sombres."


def _reading_spread(std: float) -> str:
    if std < 0.08:
        return "Les pixels se ressemblent beaucoup. L'image a peu de relief."
    if std > 0.25:
        return "Les pixels couvrent des gris très différents. L'image a du relief."
    return "Les gris varient de façon modérée d'un pixel à l'autre."


def _reading_sharpness() -> str:
    return (
        "Plus ce nombre est élevé, plus les contours sont francs et l'image nette. "
        "Plus il est bas, plus l'image est floue, souvent à cause de la mise au point. "
        "Il n'y a pas encore de seuil « trop flou, à refaire »."
    )


def _reading_over(pct: float) -> str:
    if pct < 1:
        return "Presque aucun pixel n'est blanc pur. La lumière ne grille pas l'image."
    if pct >= 5:
        return "Une part notable de l'image est presque blanche. Ces zones sont trop éclairées pour y lire un détail."
    return "Quelques pixels sont très clairs, sans que toute l'image soit grillée."


def _reading_under(pct: float) -> str:
    if pct < 2:
        return "Très peu de pixels sont presque noirs."
    if pct > 15:
        return (
            "Beaucoup de pixels sont presque noirs. "
            "Sur les cellules du jeu NIH, ce chiffre vient souvent du fond noir autour du globule, "
            "pas forcément d'une photo trop sombre."
        )
    return "Quelques pixels sont très sombres."


def _reading_candidates(n: int) -> str:
    if n == 0:
        return (
            "Aucune tache assez magenta n'a été retenue. "
            "Cela ne prouve pas l'absence de parasite : une tache trop pâle peut échapper au filtre."
        )
    if n == 1:
        return "Une tache plus magenta que le reste de la cellule a été encadrée. C'est une zone à regarder, pas un diagnostic."
    return (
        f"{n} taches plus magenta que le reste de la cellule ont été encadrées. "
        "Ce sont des zones à regarder, pas un nombre de parasites."
    )


def _reading_area(area: int, n: int) -> str:
    if n == 0 or area == 0:
        return "Aucune surface n'est marquée."
    return "C'est la surface totale des taches encadrées. Plus le nombre est grand, plus la zone marquée est étendue."


def interpret_report(report: dict) -> pd.DataFrame:
    """Tableau lisible : nom courant, valeur, phrase d'interprétation.

    Les nombres bruts restent dans le JSON. Ici, on explique ce qu'ils veulent dire.
    """
    quality = report["quality"]
    candidates = report["candidates"]
    n = int(candidates["n_candidates"])
    area = int(candidates["mask_area_px"])
    rows = [
        {
            "Indicateur": "Luminosité",
            "Valeur": f"{quality['mean']:.2f} sur 1",
            "En clair": _reading_brightness(float(quality["mean"])),
        },
        {
            "Indicateur": "Contraste",
            "Valeur": f"{quality['michelson']:.2f} sur 1",
            "En clair": _reading_contrast(float(quality["michelson"])),
        },
        {
            "Indicateur": "Variation des gris",
            "Valeur": f"{quality['std']:.2f}",
            "En clair": _reading_spread(float(quality["std"])),
        },
        {
            "Indicateur": "Netteté",
            "Valeur": f"{quality['sharpness_laplacian']:.0f}",
            "En clair": _reading_sharpness(),
        },
        {
            "Indicateur": "Zones presque blanches",
            "Valeur": f"{quality['pct_overexposed']:.1f} %",
            "En clair": _reading_over(float(quality["pct_overexposed"])),
        },
        {
            "Indicateur": "Zones presque noires",
            "Valeur": f"{quality['pct_underexposed']:.1f} %",
            "En clair": _reading_under(float(quality["pct_underexposed"])),
        },
        {
            "Indicateur": "Taches repérées",
            "Valeur": str(n),
            "En clair": _reading_candidates(n),
        },
        {
            "Indicateur": "Surface des taches",
            "Valeur": f"{area} pixels",
            "En clair": _reading_area(area, n),
        },
    ]
    return pd.DataFrame(rows)


def report_to_dataframe(report: dict) -> pd.DataFrame:
    """Aplatit quality + candidates en une seule ligne Pandas (UI Streamlit)."""
    row = {
        "timestamp_utc": report["timestamp_utc"],
        "source": report["source"],
        **report["quality"],
        **report["candidates"],
    }
    return pd.DataFrame([row])


def save_report_json(report: dict, path: str | Path) -> Path:
    """Sauvegarde le rapport en JSON (sans les arrays `_...`).

    Crée le dossier parent si besoin.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    # On ignore les clés techniques internes (préfixe "_")
    payload = {k: v for k, v in report.items() if not k.startswith("_")}

    # Conversion de types NumPy → types JSON-friendly si besoin
    def _default(obj):
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        raise TypeError(f"Type non sérialisable: {type(obj)}")

    path.write_text(
        json.dumps(payload, indent=2, default=_default),
        encoding="utf-8",
    )
    return path
