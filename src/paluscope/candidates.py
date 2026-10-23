"""Zones candidates — baseline Phase 0 (Bénoit B1 / B5).

Owner V0 : Bénoit
Branche type : benoit/candidates-mask

Objectif :
    proposer un masque simple de régions "intéressantes" (souvent sombres),
    puis compter les composantes connexes.

Ce n'est PAS un détecteur final de parasites — juste une première pierre V0.
"""

from __future__ import annotations

import numpy as np
from skimage import measure, morphology
from skimage.filters import threshold_otsu

# to_hsv vient du module Godwin (preprocess) — on ne le réécrit pas ici
from .preprocess import to_hsv


def candidate_mask(rgb: np.ndarray, min_size: int = 20) -> np.ndarray:
    """Construit un masque binaire de régions candidates.

    Pipeline naïf (baseline) :
        1) RGB → HSV
        2) récupérer le canal V (luminosité / value)
        3) seuiller V avec Otsu (zones plus sombres que T)
        4) nettoyer : enlever petits blobs / petits trous

    Paramètres
    ----------
    rgb :
        Image RGB uint8 ou float.
    min_size :
        Taille minimale (en pixels) pour garder un objet / combler un trou.

    Retour
    ------
    np.ndarray bool (H, W)
    """
    # 1) Passage en HSV (H, S, V ∈ [0,1] avec skimage)
    hsv = to_hsv(rgb)
    # 2) Canal V = luminosité : les parasites / structures sombres y apparaissent souvent
    v = hsv[:, :, 2]

    # 3) Otsu sur V : pixels plus sombres que le seuil → candidats
    mask = v < threshold_otsu(v)

    # 4) Nettoyage morphologique léger (réduit le bruit)
    # skimage >= 0.26 : paramètre max_size (objets/trous <= max_size supprimés)
    mask = morphology.remove_small_objects(mask.astype(bool), max_size=min_size)
    mask = morphology.remove_small_holes(mask, max_size=min_size)
    return mask


def label_candidates(mask: np.ndarray) -> tuple[np.ndarray, int]:
    """Étiquette chaque composante connexe du masque.

    Retour
    ------
    labeled : np.ndarray int
        0 = fond, 1..n = identifiants d'objets
    n : int
        Nombre de composantes connexes (= candidats)
    """
    labeled = measure.label(mask)
    n = int(labeled.max())
    return labeled, n


def candidate_summary(rgb: np.ndarray) -> dict:
    """Résumé simple pour l'UI / le rapport.

    Contient :
        - n_candidates : nombre de blobs
        - mask_area_px : aire totale du masque (nb de pixels True)
        - mask / labeled : objets utiles à l'affichage (pas à exporter en JSON brut)
    """
    mask = candidate_mask(rgb)
    labeled, n = label_candidates(mask)
    return {
        "n_candidates": n,
        "mask_area_px": int(mask.sum()),
        "mask": mask,
        "labeled": labeled,
    }