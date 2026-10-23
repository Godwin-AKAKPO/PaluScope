"""Prétraitements vision classique (Godwin — Phase 0 A1 / A2 / A4).

Owner V0 : Godwin
Branche type : godwin/preprocess-hist

Ce module convertit et transforme les intensités SANS changer la géométrie :
- RGB → float / gris / HSV
- séparation des canaux R, G, B
- transformations d'histogramme (négatif, gamma, étalement, égalisation)

Note : to_hsv() est aussi consommé par Bénoit (candidates.py).
"""

from __future__ import annotations

import numpy as np
from skimage.color import rgb2gray, rgb2hsv
from skimage.exposure import equalize_hist, rescale_intensity
from skimage.util import img_as_float


def to_float_rgb(rgb: np.ndarray) -> np.ndarray:
    """Convertit une image RGB (souvent uint8 [0,255]) en float [0, 1].

    Pourquoi ?
        Beaucoup de formules BIP (gamma, HSV skimage, etc.) supposent [0, 1].
    """
    return img_as_float(rgb)


def to_gray(rgb: np.ndarray) -> np.ndarray:
    """RGB → niveaux de gris float [0, 1], shape (H, W).

    Principe :
        combinaison pondérée des canaux R, G, B (le vert pèse plus,
        car l'œil humain y est plus sensible).
    """
    return rgb2gray(to_float_rgb(rgb))


def to_hsv(rgb: np.ndarray) -> np.ndarray:
    """RGB → HSV float, shape (H, W, 3), canaux H/S/V dans [0, 1] (skimage).

    H = teinte, S = saturation, V = luminosité (value).
    Utile ensuite pour le masque de candidats (Bénoit).
    """
    return rgb2hsv(to_float_rgb(rgb))


def split_rgb(rgb: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Sépare les 3 canaux couleur.

    Retour
    ------
    (R, G, B) : trois tableaux (H, W)
        img[:, :, 0] = Rouge, 1 = Vert, 2 = Bleu
    """
    return rgb[:, :, 0], rgb[:, :, 1], rgb[:, :, 2]


def negative(gray01: np.ndarray) -> np.ndarray:
    """Négatif d'histogramme : T(i) = 1 - i.

    Les zones sombres deviennent claires et inversement.
    Ne change pas la forme des objets, seulement les intensités.
    """
    return 1.0 - gray01


def gamma_correct(gray01: np.ndarray, gamma: float) -> np.ndarray:
    """Correction gamma : T(i) = i ** gamma (image normalisée [0, 1]).

    - gamma < 1  → éclaircit (détails sombres plus visibles)
    - gamma > 1  → assombrit
    """
    # clip pour rester dans [0,1] avant la puissance
    return np.clip(gray01, 0.0, 1.0) ** gamma


def stretch(gray01: np.ndarray) -> np.ndarray:
    """Étalement d'histogramme : étire [Imin, Imax] vers [0, 1].

    Améliore le contraste si l'image n'utilise pas toute la plage dynamique.
    """
    return rescale_intensity(gray01)


def equalize(gray01: np.ndarray) -> np.ndarray:
    """Égalisation d'histogramme : redistribue les intensités.

    Tend à "aplatir" l'histogramme et augmenter le contraste global.
    Attention : peut aussi amplifier le bruit.
    """
    return equalize_hist(gray01)
