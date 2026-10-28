"""Contrôle qualité d'image (fichier partagé Godwin + Bénoit).

Owners V0 :
- Godwin  : seuillage manuel / Otsu          (branche godwin/quality-threshold)
- Bénoit  : mean/variance/contraste, sharp, expo, analyze_quality

Règle collab : ne modifier que SA section, sauf accord.
"""

from __future__ import annotations

import cv2
import numpy as np
from skimage.filters import threshold_otsu
from skimage.util import img_as_ubyte

from .preprocess import to_gray

# =============================================================================
# --- Godwin : SEUILLAGE (Phase 0 · A5) ---
# =============================================================================


def threshold_manual(gray01: np.ndarray, t: float) -> np.ndarray:
    """Seuillage manuel → image binaire (bool).

    Principe BIP :
        si pixel > T  → True  (classe 1, souvent affiché en blanc)
        sinon         → False (classe 0, noir)

    Paramètres
    ----------
    gray01 :
        Image grayscale float, idéalement dans [0, 1].
    t :
        Seuil choisi à la main (ex. 0.45), souvent en regardant l'histogramme.

    Retour
    ------
    np.ndarray bool de même shape que gray01
    """
    return gray01 > t


def threshold_otsu_mask(gray01: np.ndarray) -> tuple[float, np.ndarray]:
    """Seuillage automatique d'Otsu.

    Otsu choisit T en maximisant la variance inter-classes
    (meilleure séparation "foncé vs clair" au sens statistique).

    Retour
    ------
    t : float
        Seuil trouvé.
    mask : np.ndarray bool
        gray01 > t
    """
    # Calcul du seuil optimal selon Otsu
    t = float(threshold_otsu(gray01))
    # Application du même principe que le seuillage manuel
    mask = gray01 > t
    return t, mask


# =============================================================================
# --- Bénoit : MÉTRIQUES QC + analyze_quality (Phase 0 · B3 / B4 / B5) ---
# =============================================================================


def mean_variance_contrast(gray01: np.ndarray) -> dict:
    """Calcule moyenne, variance, écart-type et contraste de Michelson.

    Formules (cahier des charges / BIP) :
        mu        = moyenne des pixels
        variance  = variance des pixels
        std       = sqrt(variance)  → contraste "simple"
        michelson = (Imax - Imin) / (Imax + Imin)
                    → contraste basé sur les extrêmes (range normalisé)

    Paramètres
    ----------
    gray01 :
        Image grayscale float, idéalement dans [0, 1].

    Retour
    ------
    dict avec les clés mean, variance, std, michelson
    """
    # float64 pour des stats numériques stables
    g = gray01.astype(np.float64)
    i_min = float(g.min())
    i_max = float(g.max())
    # +1e-8 évite la division par zéro si l'image est parfaitement plate
    michelson = (i_max - i_min) / (i_max + i_min + 1e-8)

    return {
        "mean": float(g.mean()),
        "variance": float(g.var()),
        "std": float(g.std()),
        "michelson": float(michelson),
    }


def sharpness_laplacian(gray01: np.ndarray) -> float:
    """Score de netteté = variance du Laplacien (OpenCV).

    Principe :
        le Laplacien détecte les contours / hautes fréquences.
        - score ÉLEVÉ  → image nette (beaucoup de détails)
        - score FAIBLE → image floue

    Utile pour le QC PaluScope : rejeter les frottis mal focalisés.
    """
    # OpenCV Laplacian attend souvent du uint8
    g8 = img_as_ubyte(np.clip(gray01, 0.0, 1.0))
    # CV_64F : calcul en float64 pour ne pas saturer
    return float(cv2.Laplacian(g8, cv2.CV_64F).var())


def exposure_ratios(
    gray01: np.ndarray,
    over_thr: float = 0.95,
    under_thr: float = 0.05,
) -> dict:
    """Pourcentages de pixels surexposés / sous-exposés.

    Idée simple (baseline V0) :
        - pixel > 0.95 → presque blanc  → surexposition
        - pixel < 0.05 → presque noir   → sous-exposition

    Ces seuils sont ajustables selon le dataset.
    """
    g = gray01
    return {
        "pct_overexposed": float((g > over_thr).mean() * 100.0),
        "pct_underexposed": float((g < under_thr).mean() * 100.0),
    }


def analyze_quality(rgb: np.ndarray) -> dict:
    """Rapport QC complet pour une image RGB (cœur V0 Quality Analyzer).

    Enchaîne :
        RGB → gris → stats contraste → nettete → exposition

    Retour
    ------
    dict prêt à être affiché / exporté (sans les gros tableaux image).
    """
    # Réutilise le preprocess Godwin (to_gray)
    gray = to_gray(rgb)
    stats = mean_variance_contrast(gray)
    expo = exposure_ratios(gray)

    return {
        "shape": tuple(rgb.shape),
        **stats,
        "sharpness_laplacian": sharpness_laplacian(gray),
        **expo,
    }
