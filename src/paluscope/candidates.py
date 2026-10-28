"""Zones candidates — baseline Phase 0 (Bénoit B1 / B5).

Owner V0 : Bénoit
Branche type : benoit/candidates-mask

Objectif :
    proposer des taches à regarder dans la cellule (coloration Giemsa),
    puis compter les composantes connexes.

Ce n'est PAS un détecteur final de parasites — juste une aide à la lecture V0.
Le fond noir des cellules recadrées (dataset NIH) n'est pas une zone candidate.
"""

from __future__ import annotations

import cv2
import numpy as np
from skimage import measure, morphology

# to_hsv vient du module Godwin (preprocess) — on ne le réécrit pas ici
from .preprocess import to_hsv

# Teinte magenta / violette (H skimage dans [0, 1]). Le cytoplasme rose est près de 0.
# 0,65 plutôt que 0,75 : une tache pâle est moins franchement magenta.
HUE_MAGENTA_MIN = 0.65
# Écart de saturation au-dessus du cytoplasme. 0,04 laisse passer une tache légère.
SATURATION_DELTA = 0.04
# En dessous, on considère le pixel comme le fond noir du recadrage NIH.
VALUE_BACKGROUND_MAX = 0.15
# Une tache plus grande que cette fraction de la cellule n'est pas un petit dépôt.
MAX_AREA_FRACTION = 0.12


def candidate_stages(rgb: np.ndarray, min_size: int = 12) -> dict:
    """Étapes intermédiaires de la détection, dans l'ordre du code.

    Retour
    ------
    dict
        hsv, cell, color, mask
    """
    hsv = to_hsv(rgb)
    hue = hsv[:, :, 0]
    saturation = hsv[:, :, 1]
    value = hsv[:, :, 2]

    cell = value > VALUE_BACKGROUND_MAX
    empty = np.zeros(value.shape, dtype=bool)
    color = empty.copy()
    mask = empty.copy()
    if int(cell.sum()) < min_size:
        return {"hsv": hsv, "cell": cell, "color": color, "mask": mask}

    # Référence : couleur typique du cytoplasme (rose, peu saturé).
    saturation_limit = float(np.median(saturation[cell])) + SATURATION_DELTA
    color = cell & (hue >= HUE_MAGENTA_MIN) & (saturation > saturation_limit)
    if not color.any():
        return {"hsv": hsv, "cell": cell, "color": color, "mask": mask}

    # skimage >= 0.26 : max_size = trous de cette taille comblés
    filled = morphology.remove_small_holes(color, max_size=min_size)
    labeled = measure.label(filled)
    cell_area = int(cell.sum())
    max_area = max(min_size, int(MAX_AREA_FRACTION * cell_area))
    for region in measure.regionprops(labeled):
        if region.area < min_size or region.area > max_area:
            continue
        mask[labeled == region.label] = True
    return {"hsv": hsv, "cell": cell, "color": color, "mask": mask}


def candidate_mask(rgb: np.ndarray, min_size: int = 12) -> np.ndarray:
    """Construit un masque binaire de régions candidates.

    Pipeline :
        1) RGB → HSV
        2) ignorer le fond noir (hors cellule)
        3) garder les pixels magenta plus saturés que le cytoplasme
        4) jeter le bruit trop petit et les taches trop grandes

    Paramètres
    ----------
    rgb :
        Image RGB uint8 ou float.
    min_size :
        Taille minimale (en pixels) pour garder une tache.

    Retour
    ------
    np.ndarray bool (H, W)
    """
    return candidate_stages(rgb, min_size=min_size)["mask"]


def overlay_candidates(rgb: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Laisse les taches dans leur couleur et trace un cadre rouge autour.

    Retour
    ------
    np.ndarray uint8 (H, W, 3)
    """
    if np.issubdtype(rgb.dtype, np.floating):
        base = (np.clip(rgb, 0.0, 1.0) * 255.0).astype(np.uint8)
    else:
        base = np.asarray(rgb, dtype=np.uint8)
    if base.ndim == 2:
        base = np.stack([base, base, base], axis=-1)

    out = np.ascontiguousarray(base).copy()
    if not mask.any():
        return out

    height, width = out.shape[:2]
    thickness = int(np.clip(round(min(height, width) * 0.012), 2, 4))
    pad = thickness + 1
    labeled, _ = label_candidates(mask)
    for region in measure.regionprops(labeled):
        min_row, min_col, max_row, max_col = region.bbox
        top = max(0, min_row - pad)
        left = max(0, min_col - pad)
        bottom = min(height - 1, max_row - 1 + pad)
        right = min(width - 1, max_col - 1 + pad)
        cv2.rectangle(out, (left, top), (right, bottom), (255, 30, 30), thickness)
    return out


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
