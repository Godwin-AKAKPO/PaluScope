"""Contrôle qualité d'image (fichier partagé Godwin + Bénoit).

Owners V0 :
- Godwin  : seuillage manuel / Otsu          (branche godwin/quality-threshold)
- Bénoit  : mean/variance/contraste, sharp, expo, analyze_quality

Règle collab : ne modifier que SA section, sauf accord.
"""

from __future__ import annotations

import numpy as np
from skimage.filters import threshold_otsu

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
# --- Bénoit : MÉTRIQUES QC + analyze_quality (à implémenter) ---
# =============================================================================
# TODO Bénoit :
#   - mean_variance_contrast(gray01)
#   - sharpness_laplacian(gray01)
#   - exposure_ratios(gray01)
#   - analyze_quality(rgb)
