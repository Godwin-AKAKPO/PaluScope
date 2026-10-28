"""Tests unitaires V0.

Godwin  : preprocess + seuillage
Bénoit  : analyze_quality, sharpness, candidates
"""

from __future__ import annotations

import cv2
import numpy as np
from skimage.util import img_as_float, img_as_ubyte

from paluscope.candidates import candidate_mask, label_candidates, overlay_candidates
from paluscope.preprocess import gamma_correct, negative, sharpen, to_gray
from paluscope.report import build_report, interpret_report
from paluscope.quality import (
    analyze_quality,
    sharpness_laplacian,
    threshold_manual,
    threshold_otsu_mask,
)


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


def test_sharpen_increases_sharpness():
    """Le masque flou rend une image floue plus nette, sans changer la taille."""
    gray = to_gray(_fake_rgb())
    blurred = img_as_float(cv2.GaussianBlur(img_as_ubyte(gray), (11, 11), 0))
    sharp = sharpen(blurred, amount=1.5)
    assert sharp.shape == blurred.shape
    assert sharpness_laplacian(sharp) > sharpness_laplacian(blurred)


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
# --- Tests Bénoit : QC + candidats ---
# =============================================================================


def test_analyze_quality_keys():
    """analyze_quality doit exposer les indicateurs QC attendus."""
    q = analyze_quality(_fake_rgb())
    for key in (
        "shape",
        "mean",
        "variance",
        "std",
        "michelson",
        "sharpness_laplacian",
        "pct_overexposed",
        "pct_underexposed",
    ):
        assert key in q


def test_interpret_report_explains_each_indicator():
    """Le tableau lisible a un nom, une valeur et une phrase pour chaque ligne."""
    table = interpret_report(build_report(_fake_rgb(), source="test"))
    assert list(table.columns) == ["Indicateur", "Valeur", "En clair"]
    assert len(table) == 8
    assert table["En clair"].str.len().min() > 20


def test_build_report_stores_interpretations():
    """Le rapport exportable contient les mêmes phrases que le tableau."""
    report = build_report(_fake_rgb(), source="test")
    notes = report["interpretations"]
    assert len(notes) == 8
    assert notes[0]["Indicateur"]
    assert notes[0]["En clair"]


def test_sharpness_blur_reduces_score():
    """Une image floutée doit avoir un score de netteté plus faible."""
    gray = to_gray(_fake_rgb())
    blurred = img_as_float(cv2.GaussianBlur(img_as_ubyte(gray), (11, 11), 0))
    assert sharpness_laplacian(blurred) < sharpness_laplacian(gray)


def test_candidate_mask_and_labels():
    """Le masque candidats est booléen ; le nombre de CC est >= 0."""
    rgb = _fake_rgb()
    mask = candidate_mask(rgb)
    assert mask.dtype == bool
    assert mask.shape == (64, 64)

    _, n = label_candidates(mask)
    assert n >= 0


def _cell_on_black(color: tuple[int, int, int], blob: tuple[int, int, int] | None = None) -> np.ndarray:
    """Cellule rose sur fond noir, avec une tache optionnelle au centre."""
    rgb = np.zeros((80, 80, 3), dtype=np.uint8)
    rgb[15:65, 15:65] = color
    if blob is not None:
        rgb[34:50, 34:50] = blob
    return rgb


def test_candidate_mask_marks_magenta_inside_cell_only():
    """La tache magenta est retenue ; le fond noir et le cytoplasme rose non."""
    rgb = _cell_on_black((210, 150, 145), blob=(170, 80, 140))
    mask = candidate_mask(rgb)

    assert not mask[0, 0]
    assert not mask[20, 20]
    assert mask[40, 40]
    _, n = label_candidates(mask)
    assert n == 1


def test_overlay_draws_a_box_around_the_stain():
    """Le cadre dépasse la tache : un pixel juste au-dessus est rouge."""
    rgb = _cell_on_black((210, 150, 145), blob=(170, 80, 140))
    mask = candidate_mask(rgb)
    over = overlay_candidates(rgb, mask)
    rows = np.where(mask)[0]
    top = int(rows.min())
    col = int(np.where(mask[top])[0][0])
    assert over[top - 1, col, 0] > 200
    assert not mask[top - 1, col]
    inside = tuple(np.argwhere(mask)[len(np.argwhere(mask)) // 2])
    assert np.array_equal(over[inside], rgb[inside])


def test_candidate_mask_pink_cell_has_no_candidate():
    """Une cellule seulement rose ne produit pas de zone candidate."""
    rgb = _cell_on_black((210, 150, 145))
    mask = candidate_mask(rgb)
    assert int(mask.sum()) == 0
