"""Lecture / description d'images.

Owner V0 : Godwin
Branche type : godwin/io-load-rgb

Ce module sert à :
- charger une image depuis le disque en RGB (format standard du projet) ;
- résumer un tableau image (shape, dtype, statistiques).
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np


def load_rgb(path: str | Path) -> np.ndarray:
    """Charge une image depuis le disque en RGB uint8 de forme (H, W, 3).

    Pourquoi RGB et pas BGR ?
        OpenCV lit en BGR par défaut. Dans PaluScope / Matplotlib / scikit-image,
        on travaille en RGB. On convertit donc immédiatement après la lecture.

    Paramètres
    ----------
    path :
        Chemin vers un fichier image (png, jpg, ...).

    Retour
    ------
    np.ndarray
        Image RGB, dtype uint8, valeurs typiquement dans [0, 255].

    Exceptions
    ----------
    FileNotFoundError
        Si le fichier n'existe pas.
    ValueError
        Si OpenCV ne parvient pas à décoder l'image.
    """
    # 1) Normaliser le chemin (str ou Path → Path)
    path = Path(path)

    # 2) Vérifier l'existence avant d'appeler OpenCV (message d'erreur plus clair)
    if not path.exists():
        raise FileNotFoundError(f"Image introuvable: {path}")

    # 3) Lecture disque. IMREAD_COLOR force 3 canaux (pas de niveaux de gris seuls).
    bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if bgr is None:
        # OpenCV renvoie None si le fichier est corrompu / non supporté
        raise ValueError(f"Impossible de lire l'image: {path}")

    # 4) BGR → RGB pour le reste du pipeline PaluScope
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    return rgb


def describe_array(name: str, image: np.ndarray) -> dict:
    """Résume un tableau image (définition BIP : image = tableau de nombres).

    Utile pour l'inspection Phase 0 / V0 : shape, dtype, min, max, mean, std.

    Paramètres
    ----------
    name :
        Libellé lisible (ex. "parasitized_1 (RGB)").
    image :
        Tableau NumPy 2D (gris) ou 3D (couleur).

    Retour
    ------
    dict
        Dictionnaire de métadonnées numériques.
    """
    # Conversion en float64 pour des stats stables (évite les surprises uint8)
    arr = image.astype(np.float64)

    # Aplatir tous les pixels (et tous les canaux si RGB) en 1D
    flat = arr.reshape(-1)

    return {
        "name": name,
        "shape": tuple(image.shape),
        "dtype": str(image.dtype),
        "min": float(flat.min()),
        "max": float(flat.max()),
        "mean": float(flat.mean()),
        "std": float(flat.std()),
    }
