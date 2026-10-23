"""Tests unitaires V0.

Godwin  : preprocess + seuillage   (godwin/tests-preprocess)
Bénoit  : sharpness / candidates   (à ajouter plus tard)
"""

from __future__ import annotations

import numpy as np

from paluscope.preprocess import gamma_correct, negative, to_gray
from paluscope.quality import threshold_manual, threshold_otsu_mask


def _fake_rgb(h: int = 64, w: int = 64) -> np.ndarray:
    """Image RGB synthétique : fond clair + tache sombre au centre.

    Sert de fixture légère (pas besoin d'image NIH pour les tests unitaires).
    """
    img = np.full((h, w, 3), 200, dtype=np.uint8)
    img[20:40, 20:40] = 40
    return img


# =============================================================================
# --- Tests Godwin : preprocess ---
# =============================================================================


def test_to_gray_shape_and_range():
    """to_gray doit renvoyer (H, W) avec des valeurs dans [0, 1]."""
    rgb = _fake_rgb()
    gray = to_gray(rgb)

    assert gray.shape == (64, 64)
    assert gray.dtype == np.float64 or np.issubdtype(gray.dtype, np.floating)
    assert float(gray.min()) >= 0.0
    assert float(gray.max()) <= 1.0


def test_negative_inverts_intensities():
    """Négatif : neg + gray ≈ 1 pour chaque pixel."""
    gray = to_gray(_fake_rgb())
    neg = negative(gray)
    assert np.allclose(neg + gray, 1.0)


def test_gamma_correct_shape():
    """Gamma conserve la shape."""
    gray = to_gray(_fake_rgb())
    bright = gamma_correct(gray, 0.4)  # gamma < 1 → éclaircit
    assert bright.shape == gray.shape


# =============================================================================
# --- Tests Godwin : seuillage ---
# =============================================================================


def test_threshold_manual_bool_mask():
    """Le seuillage manuel renvoie un masque booléen."""
    gray = to_gray(_fake_rgb())
    mask = threshold_manual(gray, t=0.45)
    assert mask.dtype == bool
    assert mask.shape == gray.shape


def test_threshold_otsu_mask_range():
    """Otsu renvoie un seuil dans [0, 1] et un masque booléen."""
    gray = to_gray(_fake_rgb())
    t, mask = threshold_otsu_mask(gray)

    assert 0.0 <= t <= 1.0
    assert mask.dtype == bool
    assert mask.shape == gray.shape


# =============================================================================
# --- Tests Bénoit (TODO) ---
# =============================================================================
# TODO Bénoit : analyze_quality keys, sharpness blur, candidate_mask
