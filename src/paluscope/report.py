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

    return {
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