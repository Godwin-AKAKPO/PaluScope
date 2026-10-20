#!/usr/bin/env python3
"""Build PaluScope Phase 0 Colab notebook (commentaires explicites)."""
import json
from pathlib import Path


def md(source: str) -> dict:
    lines = source.strip("\n").split("\n")
    src = [l + "\n" for l in lines[:-1]] + ([lines[-1] + "\n"] if lines else [])
    return {"cell_type": "markdown", "metadata": {}, "source": src}


def code(source: str) -> dict:
    lines = source.strip("\n").split("\n")
    src = [l + "\n" for l in lines[:-1]] + ([lines[-1] + "\n"] if lines else [])
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": src,
    }


cells = []

cells.append(md(r"""# PaluScope — Phase 0 · Digital Images (BIP)

**Projet :** PaluScope (AMA · PIIA Cohorte 2)
**Équipe :** Godwin & Bénoit
**Référence mentor :** [Basics of Image Processing — Digital images](https://vincmazet.github.io/bip/digital-images/intro/)
**Livrable cible :** premières briques du *Blood Smear Quality Analyzer* (V0)

---

## Objectif de ce notebook

Charger **quelques images clés** d'un dataset ouvert de frottis sanguins (ici NIH/NLM *Malaria*), puis appliquer les opérations du chapitre **Digital images** :

| Section BIP | Opérations |
|---|---|
| [Definition](https://vincmazet.github.io/bip/digital-images/definition) | image = tableau de nombres, shape, dtype, plage d'intensités |
| [Color models](https://vincmazet.github.io/bip/digital-images/color-models) | RGB, HSV |
| [Displaying](https://vincmazet.github.io/bip/digital-images/displaying) | affichage, colormaps, canaux |
| [Arithmetic operations](https://vincmazet.github.io/bip/digital-images/operations) | addition, soustraction, division |
| [Histogram](https://vincmazet.github.io/bip/digital-images/histogram) | histogrammes, transformations, seuillage |

Sans CNN. On reste en **vision classique + maths**.

---

## Répartition du travail (équitable)

| Qui | Blocs | Thème |
|---|---|---|
| **Commun** | 0 | Setup, chargement des images, utilitaires |
| **Godwin** | A1 → A5 | Inspection, affichage, histogrammes & transformations, seuillage |
| **Bénoit** | B1 → B5 | Espaces couleur HSV, opérations arithmétiques, stats QC, quantification/flou, synthèse V0 |
| **Commun** | C | Checklist finale & conclusions à deux |

**Règle :** chaque personne remplit **ses cellules marquées TODO**, laisse un **commentaire court** sous chaque figure (1–3 phrases), et n'écrase pas le travail de l'autre.

**Comment utiliser sur Google Colab**
1. Uploader ce fichier `.ipynb` sur [Google Colab](https://colab.research.google.com/) (*Fichier → Importer un notebook*).
2. Exécuter d'abord **toutes les cellules du Bloc 0** (Commun).
3. Godwin exécute/complète les blocs **A**, Bénoit les blocs **B**.
4. Partager le Colab en édition (lien) pour travailler à deux.

---

## Comment les images sont chargées ? (lire avant de coder)

Par défaut, on utilise **TensorFlow Datasets (`tfds`)** et le jeu nommé **`malaria`** :

- c'est le dataset NIH/NLM de **patches de frottis minces** (cellules) ;
- chaque exemple est un couple `(image, label)` ;
- `label = 0` → **parasitized** (cellule parasitée) ;
- `label = 1` → **uninfected** (cellule non parasitée) ;
- l'image arrive en **RGB `uint8`**, forme typique `(H, W, 3)`.

Le Bloc 0 construit un dictionnaire Python :

```text
IMAGES = {
  "parasitized_1": ...,
  "parasitized_2": ...,
  "parasitized_3": ...,
  "uninfected_1": ...,
  "uninfected_2": ...,
  "uninfected_3": ...,
}
```

**Tous les exercices A et B piochent dans `IMAGES[...]`.**
Pour changer d'image dans un exercice, remplacez simplement la clé, par exemple :

```python
img = IMAGES["parasitized_2"]   # au lieu de parasitized_1
```

Trois façons de changer les images sont détaillées dans les cellules **0.2 / 0.2b / 0.2c**."""))

cells.append(md(r"""---
# Bloc 0 — Commun (Godwin prépare le socle, Bénoit vérifie)

> À faire **avant** les blocs A et B. Une seule personne peut l'exécuter ; les deux doivent comprendre le contenu."""))

cells.append(md(r"""## 0.1 Installer / importer les librairies

Cette cellule installe (si besoin) et importe tout ce dont le notebook a besoin.
Exécutez-la **une fois** au début de la session Colab."""))

cells.append(code(r'''# =============================================================================
# BLOC 0.1 — INSTALLATION + IMPORTS (COMMUN)
# =============================================================================
# Pourquoi cette cellule ?
#   - Sur Google Colab, certaines libs sont déjà là, d'autres non.
#   - On réinstalle "au cas où" pour que Godwin et Bénoit aient le même environnement.
# =============================================================================

# %pip : commande spéciale notebook (= pip install). Le -q rend la sortie plus silencieuse.
%pip install -q scikit-image opencv-python-headless tensorflow-datasets pillow

# --- Imports Python standard ---
import os          # chemins / fichiers
import random      # tirage aléatoire (si on veut re-échantillonner)
from pathlib import Path  # chemins modernes

# --- Imports scientifiques ---
import numpy as np                 # tableaux numériques (= les images sont des arrays)
import matplotlib.pyplot as plt    # affichage des figures
import cv2                         # OpenCV : flou, Laplacien, etc.

# --- Imports scikit-image (vision classique) ---
from skimage import exposure, filters, measure, morphology
from skimage.color import rgb2gray, rgb2hsv
from skimage.util import img_as_float, img_as_ubyte

# Réglages d'affichage Matplotlib (taille des figures / polices)
plt.rcParams.update({
    "figure.figsize": (10, 4),
    "axes.titlesize": 11,
    "axes.labelsize": 10,
})

print("Imports OK")
print("NumPy :", np.__version__)
print("OpenCV:", cv2.__version__)'''))

cells.append(md(r"""## 0.2 Charger quelques images clés (méthode par défaut = TFDS Malaria)

### Ce que fait le code
1. Télécharge (1ère fois) le dataset `malaria` via `tensorflow_datasets`.
2. Parcourt le split `train`.
3. Garde **3 images parasitée** + **3 images non parasitée**.
4. Les range dans le dictionnaire global `IMAGES`.

### Comment changer les images chargées ?
Dans la cellule suivante, modifiez ces paramètres clairement commentés :

| Paramètre | Effet |
|---|---|
| `N_PER_CLASS` | nombre d'images par classe (ex. 3 → 5) |
| `SEED` | graine aléatoire : changez-la pour tirer **d'autres** images |
| clés de `IMAGES` | renommer / réordonner les exemples |

### Autres méthodes (plus bas)
- **0.2b** : upload manuel depuis votre PC
- **0.2c** : charger des fichiers déjà présents dans Colab (`/content/...`)"""))

cells.append(code(r'''# =============================================================================
# BLOC 0.2 — CHARGEMENT DES IMAGES VIA TENSORFLOW DATASETS (COMMUN)
# =============================================================================
# Dataset utilisé : "malaria" (NIH/NLM cell images)
# Convention OFFICIELLE TFDS (_NAMES = ["parasitized", "uninfected"]) :
#   label 0 = parasitized
#   label 1 = uninfected
# On utilise info.features['label'].int2str(y) pour ne plus se tromper.
#
# COMMENT CHANGER LES IMAGES ?
#   1) Changez N_PER_CLASS  -> plus/moins d'images
#   2) Changez SEED         -> un autre tirage d'exemples
#   3) Ou passez à la cellule 0.2b (upload) / 0.2c (fichiers locaux)
# =============================================================================

import tensorflow_datasets as tfds

# --- PARAMÈTRES À MODIFIER SI BESOIN ---
N_PER_CLASS = 3     # <-- changez ici : 2, 3, 5, ...
SEED = 42           # <-- changez ici pour obtenir d'autres images

# Charge le dataset + ses metadonnees (noms des classes)
# as_supervised=True => chaque element = (image, label)
ds, info = tfds.load(
    "malaria",
    split="train",
    as_supervised=True,
    shuffle_files=True,
    with_info=True,
    read_config=tfds.ReadConfig(shuffle_seed=SEED),
)

print("Classes TFDS malaria :", info.features["label"].names)
# attendu : ['parasitized', 'uninfected']  => 0=parasitized, 1=uninfected

# On parcourt le flux jusqu'a avoir N_PER_CLASS images de chaque classe
parasitized, uninfected = [], []

for img, label in ds:
    # img est un tenseur TF ; .numpy() le convertit en array NumPy uint8 (H, W, 3)
    arr = img.numpy()
    y = int(label.numpy())
    # Conversion robuste : 0 -> "parasitized", 1 -> "uninfected"
    class_name = info.features["label"].int2str(y)

    if class_name == "parasitized" and len(parasitized) < N_PER_CLASS:
        parasitized.append(arr)
    elif class_name == "uninfected" and len(uninfected) < N_PER_CLASS:
        uninfected.append(arr)

    # Stop des qu'on a assez d'exemples des deux classes
    if len(parasitized) >= N_PER_CLASS and len(uninfected) >= N_PER_CLASS:
        break

# --- Dictionnaire GLOBAL utilisé par tout le notebook ---
# Clé = nom lisible ; valeur = image RGB uint8
IMAGES = {}
for i, arr in enumerate(parasitized, start=1):
    IMAGES[f"parasitized_{i}"] = arr
for i, arr in enumerate(uninfected, start=1):
    IMAGES[f"uninfected_{i}"] = arr

# Variables de confort (image "par défaut" pour tester rapidement)
img_rgb = IMAGES["parasitized_1"].copy()
img_gray = rgb2gray(img_rgb)  # float64 dans [0, 1]

print("=" * 60)
print("CHARGEMENT OK")
print("Clés disponibles dans IMAGES :", list(IMAGES.keys()))
print("Exemple parasitized_1 -> shape:", img_rgb.shape, "| dtype:", img_rgb.dtype)
print("Grayscale associé      -> shape:", img_gray.shape,
      "| min/max:", float(img_gray.min()), float(img_gray.max()))
print("=" * 60)
print("Astuce: pour utiliser une autre image dans un exo, faites:")
print('        img = IMAGES["uninfected_2"]')'''))

cells.append(md(r"""## 0.2b — Alternative A : upload manuel depuis votre ordinateur

Utilisez cette cellule si :
- TFDS est trop lent / bloqué, **ou**
- vous voulez vos propres images NIH / Uganda / Makerere téléchargées ailleurs.

**Comment faire :**
1. Commentez / ignorez 0.2 si besoin.
2. Exécutez la cellule ci-dessous.
3. Dans la fenêtre Colab, sélectionnez 4–6 fichiers `.png` / `.jpg`.
4. Les images seront placées dans `IMAGES` sous leur nom de fichier."""))

cells.append(code(r'''# =============================================================================
# BLOC 0.2b — UPLOAD MANUEL (OPTIONNEL)
# =============================================================================
# Par défaut cette cellule est INACTIVE pour ne pas bloquer le notebook.
# Pour l'activer :
#   1) Passez USE_UPLOAD = True
#   2) Exécutez la cellule
#   3) Choisissez vos fichiers dans la boîte de dialogue Colab
# =============================================================================

USE_UPLOAD = False  # <-- passez à True pour activer l'upload

if USE_UPLOAD:
    from google.colab import files

    print("Selectionnez vos images (png/jpg)...")
    uploaded = files.upload()  # dict {nom_fichier: bytes}

    IMAGES = {}
    for name in uploaded:
        # OpenCV lit en BGR : on convertit immédiatement en RGB
        bgr = cv2.imread(name)
        if bgr is None:
            print(f"[WARN] Impossible de lire {name}, ignore.")
            continue
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        IMAGES[name] = rgb
        print(f"Charge: {name} -> shape={rgb.shape}, dtype={rgb.dtype}")

    if not IMAGES:
        raise RuntimeError("Aucune image chargee. Reessayez l'upload.")

    img_rgb = next(iter(IMAGES.values())).copy()
    img_gray = rgb2gray(img_rgb)
    print("IMAGES cles:", list(IMAGES.keys()))
else:
    print("Upload manuel desactive (USE_UPLOAD=False). On garde le chargement TFDS de 0.2.")'''))

cells.append(md(r"""## 0.2c — Alternative B : fichiers déjà dans Colab (`/content/...`)

Utile si vous avez monté Google Drive, ou dézippé un dataset dans `/content`.

**Exemple :**
```python
FOLDER = "/content/mes_frottis"
```
Le code charge tous les `.png` / `.jpg` du dossier dans `IMAGES`."""))

cells.append(code(r'''# =============================================================================
# BLOC 0.2c — CHARGER DEPUIS UN DOSSIER LOCAL COLAB (OPTIONNEL)
# =============================================================================
# Activez avec USE_FOLDER = True et indiquez le chemin FOLDER.
# Exemple apres unzip :
#   !unzip -q cell_images.zip -d /content/cell_images
#   FOLDER = "/content/cell_images/Parasitized"
# =============================================================================

USE_FOLDER = False  # <-- passez a True pour activer
FOLDER = "/content/mes_frottis"  # <-- changez le chemin
MAX_IMAGES = 6  # limite pour ne pas tout charger

if USE_FOLDER:
    folder = Path(FOLDER)
    if not folder.exists():
        raise FileNotFoundError(f"Dossier introuvable: {folder}")

    paths = sorted(
        list(folder.glob("*.png"))
        + list(folder.glob("*.jpg"))
        + list(folder.glob("*.jpeg"))
    )[:MAX_IMAGES]

    IMAGES = {}
    for p in paths:
        bgr = cv2.imread(str(p))
        if bgr is None:
            print(f"[WARN] Lecture impossible: {p.name}")
            continue
        IMAGES[p.stem] = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        print(f"Charge: {p.name} -> {IMAGES[p.stem].shape}")

    if not IMAGES:
        raise RuntimeError(f"Aucune image trouvee dans {folder}")

    img_rgb = next(iter(IMAGES.values())).copy()
    img_gray = rgb2gray(img_rgb)
    print("IMAGES cles:", list(IMAGES.keys()))
else:
    print("Chargement dossier desactive (USE_FOLDER=False). On garde IMAGES actuel.")'''))

cells.append(md(r"""## 0.3 Utilitaires d'affichage (ne pas modifier sans se concerter)

Deux fonctions partagées :
- `show_images(...)` : affiche plusieurs images côte à côte
- `describe_image(...)` : imprime shape / dtype / min / max / mean / std

Elles évitent de recopier le même code Matplotlib partout."""))

cells.append(code(r'''# =============================================================================
# BLOC 0.3 — FONCTIONS UTILITAIRES D'AFFICHAGE (COMMUN)
# =============================================================================

def show_images(images, titles=None, cmap=None, ncols=3, figsize=None):
    """Affiche une liste d'images cote a cote.

    Parametres
    ----------
    images : list d'arrays
        Chaque array est soit HxW (gris), soit HxWx3 (couleur RGB).
    titles : list de str ou None
        Titre au-dessus de chaque image.
    cmap : str ou None
        Colormap Matplotlib pour les images 2D (ex: "gray", "jet").
    ncols : int
        Nombre de colonnes dans la grille.
    figsize : tuple ou None
        Taille de la figure (largeur, hauteur) en pouces.
    """
    n = len(images)
    ncols = min(ncols, n)
    nrows = int(np.ceil(n / ncols))
    figsize = figsize or (4 * ncols, 3.5 * nrows)

    fig, axs = plt.subplots(nrows, ncols, figsize=figsize)
    axs = np.array(axs).reshape(-1)

    for i, im in enumerate(images):
        if im.ndim == 2:
            # Image en niveaux de gris -> on impose une colormap
            axs[i].imshow(im, cmap=cmap or "gray")
        else:
            # Image couleur RGB
            axs[i].imshow(im)
        if titles:
            axs[i].set_title(titles[i])
        axs[i].axis("off")

    # Cache les cases vides s'il en reste
    for j in range(i + 1, len(axs)):
        axs[j].axis("off")

    plt.tight_layout()
    plt.show()


def describe_image(name, im):
    """Affiche un resume numerique d'une image (= definition BIP: tableau de nombres)."""
    imf = img_as_float(im)  # convertit en float [0,1] pour comparer facilement
    print(f"=== {name} ===")
    print(f"shape={im.shape} | dtype={im.dtype} | ndim={im.ndim}")
    print(
        f"min={imf.min():.4f} | max={imf.max():.4f} | "
        f"mean={imf.mean():.4f} | std={imf.std():.4f}"
    )


# Apercu rapide : on visualise TOUTES les images actuellement dans IMAGES
print("Apercu des images chargees :")
show_images(
    list(IMAGES.values()),
    titles=list(IMAGES.keys()),
    ncols=3,
)
'''))

# ========================= BLOC A GODWIN =========================

cells.append(md(r"""---
# Bloc A — Godwin

**Focus BIP :** Definition · Displaying · Histogram (+ transformations & thresholding)

À la fin du bloc A, Godwin doit pouvoir répondre :
1. Quelle est la forme mathématique d'une image numérique ici ?
2. Que montre l'histogramme d'un frottis parasité vs non parasité ?
3. Quelle transformation / quel seuil améliore le contraste ou sépare mieux les structures ?

> Pour changer d'image dans un exercice : `IMAGES["parasitized_2"]`, `IMAGES["uninfected_1"]`, etc."""))

cells.append(md(r"""## A1 — Godwin · Inspection d'une image (Definition)

**Lire :** [What is a digital image?](https://vincmazet.github.io/bip/digital-images/definition)

**À faire :**
1. Choisir 2 images (`parasitized_1` et `uninfected_1`) — changez les clés si besoin.
2. Afficher `shape`, `dtype`, min/max, moyenne.
3. Vérifier que c'est bien $d=2$, $B=3$ (couleur) puis $B=1$ en niveaux de gris.
4. Commenter : *que représente un pixel ?*"""))

cells.append(code(r'''# =============================================================================
# A1 — GODWIN · INSPECTION (Definition BIP)
# =============================================================================
# But : comprendre qu'une image = un tableau NumPy de nombres.
#   - RGB  : shape (H, W, 3)  -> d=2 dimensions spatiales, B=3 bandes
#   - Gray : shape (H, W)     -> d=2, B=1
#
# POUR CHANGER D'IMAGE :
#   remplacez "parasitized_1" / "uninfected_1" par d'autres cles de IMAGES
# =============================================================================

# 1) Recuperer deux images du dictionnaire IMAGES
img_p = IMAGES["parasitized_1"]   # <-- changez la cle si besoin
img_u = IMAGES["uninfected_1"]    # <-- changez la cle si besoin

# 2) Decrire les tableaux RGB
describe_image("parasitized_1 (RGB)", img_p)
describe_image("uninfected_1 (RGB)", img_u)

# 3) Conversion RGB -> niveaux de gris (float dans [0, 1])
#    rgb2gray fait une combinaison ponderee des canaux R, G, B
gray_p = rgb2gray(img_p)
gray_u = rgb2gray(img_u)
describe_image("parasitized_1 (gray float)", gray_p)
describe_image("uninfected_1 (gray float)", gray_u)

# 4) Afficher RGB et gris cote a cote
show_images(
    [img_p, gray_p, img_u, gray_u],
    titles=["P RGB", "P gray", "U RGB", "U gray"],
    ncols=4,
)

# TODO GODWIN : ecrire 2-3 phrases dans la cellule markdown "Notes Godwin — A1"
# Questions guides :
#   - Que signifient H, W, 3 dans shape ?
#   - Les intensites sont-elles dans [0,255] ou [0,1] ? Pourquoi ?'''))

cells.append(md(r"""### Notes Godwin — A1
> *À compléter :* Que représentent shape / dtype / intensités ? Différence visuelle parasité vs non parasité ?"""))

cells.append(md(r"""## A2 — Godwin · Affichage & colormaps (Displaying)

**Lire :** [Displaying an image](https://vincmazet.github.io/bip/digital-images/displaying)

**À faire :**
1. Afficher la **même** image grayscale avec plusieurs colormaps (`gray`, `viridis`, `jet`, `hot`).
2. Séparer et afficher les canaux **R, G, B**.
3. Commenter : *la colormap change-t-elle l'information ou seulement la perception ?*"""))

cells.append(code(r'''# =============================================================================
# A2 — GODWIN · AFFICHAGE + COLORMAPS + CANAUX RGB
# =============================================================================
# Idee BIP : la colormap change la PERCEPTION, pas les valeurs du tableau.
# Les canaux R, G, B montrent comment la couleur est codee (B=3).
#
# POUR CHANGER D'IMAGE : modifiez la cle ci-dessous.
# =============================================================================

img = IMAGES["parasitized_2"]  # <-- changez ici si besoin
gray = rgb2gray(img)

# --- 1) Meme image, colormaps differentes ---
fig, axs = plt.subplots(1, 4, figsize=(14, 3))
for ax, cmap in zip(axs, ["gray", "viridis", "jet", "hot"]):
    # cmap ne modifie PAS gray : c'est seulement le rendu a l'ecran
    ax.imshow(gray, cmap=cmap)
    ax.set_title(cmap)
    ax.axis("off")
plt.suptitle("Meme image, colormaps differentes (l'info numerique est identique)")
plt.tight_layout()
plt.show()

# --- 2) Separaration des canaux RGB ---
# img[:, :, 0] = rouge, 1 = vert, 2 = bleu
r, g, b = img[:, :, 0], img[:, :, 1], img[:, :, 2]
show_images([img, r, g, b], titles=["RGB", "R", "G", "B"], ncols=4)

# TODO GODWIN (optionnel) :
#   img = IMAGES["uninfected_2"]
#   recommencer et comparer quel canal est le plus contraste.'''))

cells.append(md(r"""### Notes Godwin — A2
> *À compléter :* Effet des colormaps ? Quel canal semble le plus informatif pour le frottis ?"""))

cells.append(md(r"""## A3 — Godwin · Histogrammes (Histogram)

**Lire :** [Histogram](https://vincmazet.github.io/bip/digital-images/histogram)

**À faire :**
1. Tracer l'histogramme grayscale (128 bins puis 16 bins).
2. Tracer les histogrammes des canaux R, G, B.
3. Comparer **parasitized** vs **uninfected**.
4. Rappeler : *deux images différentes peuvent avoir le même histogramme* (perte de l'info spatiale)."""))

cells.append(code(r'''# =============================================================================
# A3 — GODWIN · HISTOGRAMMES
# =============================================================================
# Histogramme h(i) = nombre de pixels d'intensite i
# Attention : l'histogramme perd l'information SPATIALE.
#
# POUR CHANGER D'IMAGES COMPAREES : changez les 2 cles ci-dessous.
# =============================================================================

def plot_gray_hist(ax, gray, title, bins=128):
    """Trace l'histogramme d'une image grayscale float [0,1]."""
    # .ravel() aplatit l'image 2D en 1D : obligatoire pour plt.hist
    ax.hist(gray.ravel(), bins=bins, range=(0, 1), color="steelblue")
    ax.set_title(title)
    ax.set_xlabel("Intensite")
    ax.set_ylabel("Nombre de pixels")


def plot_rgb_hist(ax, rgb, title):
    """Superpose les histogrammes des 3 canaux R, G, B (uint8 0..255)."""
    for name, c, idx in [("R", "r", 0), ("G", "g", 1), ("B", "b", 2)]:
        ax.hist(
            rgb[:, :, idx].ravel(),
            bins=64,
            range=(0, 255),
            alpha=0.45,   # transparence pour voir les chevauchements
            color=c,
            label=name,
        )
    ax.set_title(title)
    ax.legend()


# --- Choix des images a comparer ---
img_p = IMAGES["parasitized_1"]  # <-- changez si besoin
img_u = IMAGES["uninfected_1"]   # <-- changez si besoin
g_p, g_u = rgb2gray(img_p), rgb2gray(img_u)

# Figure 1 : hist gray (128 bins) + hist RGB
fig, axs = plt.subplots(2, 2, figsize=(12, 7))
plot_gray_hist(axs[0, 0], g_p, "Parasitized — hist gray (128 bins)", bins=128)
plot_gray_hist(axs[0, 1], g_u, "Uninfected — hist gray (128 bins)", bins=128)
plot_rgb_hist(axs[1, 0], img_p, "Parasitized — hist RGB")
plot_rgb_hist(axs[1, 1], img_u, "Uninfected — hist RGB")
plt.tight_layout()
plt.show()

# Figure 2 : meme chose avec PEU de bins (vue plus grossiere)
fig, axs = plt.subplots(1, 2, figsize=(10, 3))
plot_gray_hist(axs[0], g_p, "Parasitized — 16 bins", bins=16)
plot_gray_hist(axs[1], g_u, "Uninfected — 16 bins", bins=16)
plt.tight_layout()
plt.show()

# TODO GODWIN : reperez les "modes" (bosses) et dites a quoi ils correspondent
# (fond sombre ? globule clair ? ...).'''))

cells.append(md(r"""### Notes Godwin — A3
> *À compléter :* Où sont les modes ? Différences parasité / non parasité ?"""))

cells.append(md(r"""## A4 — Godwin · Transformations d'histogramme

**Lire :** section *Histogram transformations* du même chapitre.

**À faire (sur une image grayscale float $[0,1]$) :**
1. **Négatif** : $T(i)=1-i$
2. **Gamma** : $T(i)=i^\gamma$ avec $\gamma=0.4$ puis $\gamma=2.0$
3. **Étalement** (rescaling / stretching)
4. **Égalisation** d'histogramme
5. Comparer histogrammes avant/après"""))

cells.append(code(r'''# =============================================================================
# A4 — GODWIN · TRANSFORMATIONS D'HISTOGRAMME
# =============================================================================
# Chaque transformation applique j = T(i) a TOUS les pixels.
# Ca change le contraste / la luminosite, pas la geometrie de l'image.
#
# POUR CHANGER D'IMAGE : modifiez la cle ci-dessous.
# =============================================================================

gray = rgb2gray(IMAGES["parasitized_1"])  # <-- changez ici si besoin

# 1) Negatif : les noirs deviennent blancs et inversement
neg = 1.0 - gray

# 2) Gamma : <1 eclaircit, >1 assombrit (sur image normalisee [0,1])
gamma_bright = gray ** 0.4   # gamma < 1
gamma_dark = gray ** 2.0     # gamma > 1

# 3) Etalement : etire [min, max] vers [0, 1]
stretched = exposure.rescale_intensity(gray)

# 4) Egalisation : aplatit l'histogramme pour augmenter le contraste global
equalized = exposure.equalize_hist(gray)

show_images(
    [gray, neg, gamma_bright, gamma_dark, stretched, equalized],
    titles=["Original", "Negatif", "Gamma 0.4", "Gamma 2.0", "Etalement", "Egalisation"],
    ncols=3,
)

# Comparaison d'histogrammes avant / apres egalisation
fig, ax = plt.subplots(1, 1, figsize=(8, 3))
ax.hist(gray.ravel(), bins=64, range=(0, 1), alpha=0.5, label="original")
ax.hist(equalized.ravel(), bins=64, range=(0, 1), alpha=0.5, label="egalisation")
ax.legend()
ax.set_title("Avant / apres egalisation — TODO Godwin : interpreter")
ax.set_xlabel("Intensite")
ax.set_ylabel("Nombre de pixels")
plt.show()

# TODO GODWIN : quelle transformation aide le plus pour VOIR le parasite ?
# Attention : ameliorer la visu != ameliorer automatiquement la detection.'''))

cells.append(md(r"""### Notes Godwin — A4
> *À compléter :* Quelle transformation aide le plus pour un frottis ? Risque de créer des artefacts ?"""))

cells.append(md(r"""## A5 — Godwin · Seuillage (Thresholding)

**Lire :** fin du chapitre Histogram (*Thresholding*, méthode d'Otsu).

**À faire :**
1. Choisir un seuil manuel $T$ (ex. 0.45) → image binaire.
2. Calculer le seuil d'**Otsu** automatiquement.
3. Afficher original / manuel / Otsu.
4. Commenter : *le seuil segmente-t-il le parasite, le globule, ou un artefact ?*"""))

cells.append(code(r'''# =============================================================================
# A5 — GODWIN · SEUILLAGE MANUEL + OTSU
# =============================================================================
# Principe :
#   si pixel > T  -> classe 1 (blanc)
#   sinon         -> classe 0 (noir)
# Otsu choisit T automatiquement en maximisant la variance inter-classes.
#
# POUR CHANGER D'IMAGE : modifiez les cles ci-dessous.
# POUR CHANGER LE SEUIL MANUEL : modifiez T_manual.
# =============================================================================

gray = rgb2gray(IMAGES["parasitized_1"])  # <-- changez ici si besoin

# --- Seuil manuel (a ajuster en regardant l'histogramme de A3) ---
T_manual = 0.45  # <-- TODO GODWIN : essayez 0.30, 0.45, 0.60...
bin_manual = gray > T_manual

# --- Seuil automatique d'Otsu ---
T_otsu = filters.threshold_otsu(gray)
bin_otsu = gray > T_otsu

print(f"Seuil manuel = {T_manual:.3f}")
print(f"Seuil Otsu   = {T_otsu:.3f}")

show_images(
    [gray, bin_manual, bin_otsu],
    titles=["Gray", f"Manuel T={T_manual}", f"Otsu T={T_otsu:.2f}"],
    ncols=3,
)

# Comparaison sur une image non parasitee
gray_u = rgb2gray(IMAGES["uninfected_1"])  # <-- changez ici si besoin
T_u = filters.threshold_otsu(gray_u)
show_images(
    [gray_u, gray_u > T_u],
    titles=["Uninfected gray", f"Otsu T={T_u:.2f}"],
    ncols=2,
)

# TODO GODWIN :
#   Le masque binaire isole-t-il vraiment le parasite ?
#   Ou plutot le globule / le fond / des artefacts de coloration ?'''))

cells.append(md(r"""### Notes Godwin — A5
> *À compléter :* Le seuillage suffit-il pour détecter un parasite ? Limites ?"""))

# ========================= BLOC B BENOIT =========================

cells.append(md(r"""---
# Bloc B — Bénoit

**Focus BIP :** Color models · Arithmetic operations · Acquisition (quantization / blur / exposition) · pont vers le QC PaluScope

À la fin du bloc B, Bénoit doit pouvoir répondre :
1. Pourquoi passer en HSV pour un frottis coloré ?
2. À quoi servent addition / soustraction / division d'images ?
3. Quels indicateurs simples détectent flou, surexposition, mauvais contraste ?

> Pour changer d'image : `IMAGES["parasitized_3"]`, etc."""))

cells.append(md(r"""## B1 — Bénoit · Modèles de couleur RGB ↔ HSV

**Lire :** [Color models](https://vincmazet.github.io/bip/digital-images/color-models)

**À faire :**
1. Convertir une image RGB → HSV.
2. Afficher les canaux **H**, **S**, **V** séparément.
3. Calculer moyenne / écart-type de chaque canal HSV.
4. Commenter : *quel canal aide à trouver des régions sombres / teintées ?*"""))

cells.append(code(r'''# =============================================================================
# B1 — BENOIT · RGB -> HSV
# =============================================================================
# HSV (BIP) :
#   H = teinte (type de couleur)
#   S = saturation (purete)
#   V = value / luminosite  -> utile pour retrouver l'info "niveaux de gris"
#
# Avec skimage.rgb2hsv, H, S, V sont dans [0, 1].
# POUR CHANGER D'IMAGE : modifiez la cle.
# =============================================================================

img = IMAGES["parasitized_1"]  # <-- changez ici si besoin
img_f = img_as_float(img)      # RGB float [0,1] (necessaire pour rgb2hsv)
hsv = rgb2hsv(img_f)

# Separation des 3 canaux HSV
h, s, v = hsv[:, :, 0], hsv[:, :, 1], hsv[:, :, 2]

show_images(
    [img, h, s, v],
    titles=["RGB", "H (hue)", "S (sat)", "V (value)"],
    ncols=4,
)

print("--- Stats HSV sur parasitized_1 ---")
for name, ch in [("H", h), ("S", s), ("V", v)]:
    print(
        f"{name}: mean={ch.mean():.4f} | std={ch.std():.4f} | "
        f"min={ch.min():.4f} | max={ch.max():.4f}"
    )

# Comparaison avec une image non parasitee
img2 = img_as_float(IMAGES["uninfected_1"])  # <-- changez ici si besoin
hsv2 = rgb2hsv(img2)
print("\n--- Stats HSV sur uninfected_1 ---")
for name, ch in [("H", hsv2[:, :, 0]), ("S", hsv2[:, :, 1]), ("V", hsv2[:, :, 2])]:
    print(f"{name}: mean={ch.mean():.4f} | std={ch.std():.4f}")

# TODO BENOIT : quel canal (H/S/V) semble le plus utile pour des candidats parasites ?'''))

cells.append(md(r"""### Notes Bénoit — B1
> *À compléter :* Intérêt de HSV vs RGB pour la microscopie ?"""))

cells.append(md(r"""## B2 — Bénoit · Opérations arithmétiques

**Lire :** [Arithmetic operations](https://vincmazet.github.io/bip/digital-images/operations)

**À faire (en float $[0,1]$) :**
1. **Addition** de deux images.
2. **Soustraction** absolute entre deux images (même taille).
3. **Division** d'une image par une version lissée (illumination).
4. Commenter l'usage possible pour PaluScope."""))

cells.append(code(r'''# =============================================================================
# B2 — BENOIT · ADDITION / SOUSTRACTION / DIVISION
# =============================================================================
# Important :
#   - les operations sont PIXEL A PIXEL
#   - les images doivent avoir la MEME taille -> on recadre au min H/W
#   - on travaille en float [0,1] pour eviter les debordements uint8
#
# POUR CHANGER LES 2 IMAGES : modifiez les cles a et b.
# =============================================================================

a = img_as_float(IMAGES["parasitized_1"])  # <-- image A
b = img_as_float(IMAGES["uninfected_1"])   # <-- image B

# Recadrage pour avoir exactement la meme taille
h = min(a.shape[0], b.shape[0])
w = min(a.shape[1], b.shape[1])
a2, b2 = a[:h, :w], b[:h, :w]

# 1) Addition : peut saturer au-dessus de 1 -> on clippe pour l'affichage
add = np.clip(a2 + b2, 0, 1)

# 2) Soustraction absolue : met en evidence les differences locales
#    ATTENTION : ici A et B sont des cellules DIFFERENTES, donc |A-B|
#    n'est PAS un vrai "avant/apres" scientifique. C'est pedagogique.
sub = np.abs(a2 - b2)

# 3) Division par une version tres lissee ~= correction d'illumination
gray = rgb2gray(a)
smooth = cv2.GaussianBlur((gray * 255).astype(np.uint8), (31, 31), 0)
smooth_f = np.maximum(img_as_float(smooth), 1e-6)  # evite division par 0
div = np.clip(gray / smooth_f, 0, 1)
div = exposure.rescale_intensity(div)  # remet dans [0,1] pour voir le contraste

show_images(
    [a2, b2, add, sub, gray, div],
    titles=["A", "B", "A+B", "|A-B|", "A gray", "A / blur(A)"],
    ncols=3,
)

# TODO BENOIT :
#   Quelle operation est la plus pertinente pour PaluScope ?
#   Dans quel cas reel ferait-on une soustraction (ex: deux captures proches) ?'''))

cells.append(md(r"""### Notes Bénoit — B2
> *À compléter :* Quelle opération est la plus pertinente ici, et pourquoi faire attention aux tailles / échelles ?"""))

cells.append(md(r"""## B3 — Bénoit · Statistiques d'image (pont PaluScope Phase 0)

Le cahier des charges demande moyenne, variance, contraste.

**À faire :** pour chaque image de `IMAGES`, calculer $\mu$, $\sigma^2$, contraste (std + Michelson)."""))

cells.append(code(r'''# =============================================================================
# B3 — BENOIT · MOYENNE / VARIANCE / CONTRASTE
# =============================================================================
# Formules (cahier des charges / BIP) :
#   mu  = moyenne des pixels
#   var = variance
#   contraste simple = std (= sqrt(var))
#   contraste Michelson = (Imax - Imin) / (Imax + Imin)
#
# On boucle sur TOUTES les cles de IMAGES -> si vous avez change N_PER_CLASS
# ou uploade d'autres fichiers, elles seront incluses automatiquement.
# =============================================================================

rows = []
for name, im in IMAGES.items():
    g = rgb2gray(im)                 # travail sur luminances
    mu = float(g.mean())
    var = float(g.var())
    std = float(g.std())
    michelson = float((g.max() - g.min()) / (g.max() + g.min() + 1e-8))
    rows.append((name, mu, var, std, michelson))

print(f"{'name':16} | {'mean':>8} | {'var':>8} | {'std':>8} | {'michelson':>10}")
print("-" * 60)
for name, mu, var, std, mic in rows:
    print(f"{name:16} | {mu:8.4f} | {var:8.4f} | {std:8.4f} | {mic:10.4f}")

# TODO BENOIT : lesquelles semblent sous-contrastees / trop sombres / trop claires ?'''))

cells.append(md(r"""### Notes Bénoit — B3
> *À compléter :* Lien entre ces stats et la "qualité technique" d'une lame."""))

cells.append(md(r"""## B4 — Bénoit · Quantification, flou, surexposition (Acquisition → QC)

**Lire :** [Image acquisition](https://vincmazet.github.io/bip/digital-images/acquisition)

**À faire :**
1. Quantification à 16 / 4 / 2 niveaux.
2. Score de netteté (variance du Laplacien) + flou simulé.
3. % surexposition / sous-exposition.
4. Appliquer le QC à toutes les images de `IMAGES`."""))

cells.append(code(r'''# =============================================================================
# B4 — BENOIT · QUANTIFICATION + FLOU + EXPOSITION (QC)
# =============================================================================
# Lien BIP "Acquisition" :
#   - quantification = reduire le nombre de niveaux de gris
#   - flou = lie a la PSF / mauvaise mise au point
#   - bruit / mauvaise expo = distorsions d'acquisition
#
# Score de nettete utilise ici :
#   variance du Laplacien (OpenCV). Plus c'est PETIT, plus c'est FLOU.
# =============================================================================

gray = rgb2gray(IMAGES["parasitized_1"])  # <-- changez ici si besoin

# --- 1) Quantification ---
def quantize(gray01, n_levels):
    """Reduit une image [0,1] a n_levels intensites distinctes."""
    # Ex: n_levels=4 -> valeurs possibles 0, 1/3, 2/3, 1
    return np.floor(gray01 * (n_levels - 1) + 1e-8) / (n_levels - 1)

q16, q4, q2 = quantize(gray, 16), quantize(gray, 4), quantize(gray, 2)
show_images(
    [gray, q16, q4, q2],
    titles=["~256", "16 niveaux", "4", "2 (binaire)"],
    ncols=4,
)

# --- 2) Score de flou ---
def sharpness_laplacian(gray01):
    """Variance du Laplacien : proxy de nettete."""
    g8 = img_as_ubyte(gray01)  # OpenCV prefere souvent uint8
    return float(cv2.Laplacian(g8, cv2.CV_64F).var())

# Simulation d'un flou Gaussien pour verifier que le score diminue
blurred = img_as_float(cv2.GaussianBlur(img_as_ubyte(gray), (11, 11), 0))
print("Sharpness original :", sharpness_laplacian(gray))
print("Sharpness floutee  :", sharpness_laplacian(blurred))

# --- 3) Exposition ---
# % de pixels "presque blancs" / "presque noirs"
over = float((gray > 0.95).mean() * 100)
under = float((gray < 0.05).mean() * 100)
print(f"% surexposition (>0.95) : {over:.2f}%")
print(f"% sous-exposition (<0.05): {under:.2f}%")

show_images([gray, blurred], titles=["Original", "Flou simule"], ncols=2)

# --- 4) Mini rapport QC sur TOUTES les images chargees ---
print("\n--- QC rapide toutes images ---")
for name, im in IMAGES.items():
    g = rgb2gray(im)
    print(
        f"{name:16} sharp={sharpness_laplacian(g):8.1f} "
        f"over={((g > 0.95).mean() * 100):5.2f}% "
        f"under={((g < 0.05).mean() * 100):5.2f}%"
    )

# TODO BENOIT : proposer un seuil "image trop floue" pour CE dataset
# (ex: sharp < XXX => reject).'''))

cells.append(md(r"""### Notes Bénoit — B4
> *À compléter :* Seuils proposés pour "image floue" / "image surexposée" sur ce dataset ?"""))

cells.append(md(r"""## B5 — Bénoit · Mini Quality Analyzer + premiers candidats

**Objectif PaluScope V0 (première pierre) :** pour une image importée, produire des indicateurs QC + des zones candidates simples.

**À faire :**
1. `analyze_quality(rgb) -> dict`
2. `candidate_mask(rgb)` (seuillage simple)
3. Afficher image / masque / nombre de composantes connexes
4. Tester sur parasité + non parasité"""))

cells.append(code(r'''# =============================================================================
# B5 — BENOIT · MINI QUALITY ANALYZER + CANDIDATS (pont V0)
# =============================================================================
# Cette cellule assemble B3 + B4 (+ idees de seuillage de Godwin A5)
# pour produire une premiere version du "Blood Smear Quality Analyzer".
#
# Ce n'est PAS un diagnostic. C'est un prototype de recherche / QC.
# POUR CHANGER LES IMAGES TESTEES : modifiez les cles dans preview_candidates(...)
# =============================================================================

def analyze_quality(rgb):
    """Retourne des indicateurs techniques simples sur une image RGB."""
    gray = rgb2gray(rgb)
    return {
        "shape": tuple(rgb.shape),
        "mean": float(gray.mean()),
        "variance": float(gray.var()),
        "contrast_std": float(gray.std()),
        "michelson": float((gray.max() - gray.min()) / (gray.max() + gray.min() + 1e-8)),
        "sharpness_laplacian": float(cv2.Laplacian(img_as_ubyte(gray), cv2.CV_64F).var()),
        "pct_overexposed": float((gray > 0.95).mean() * 100),
        "pct_underexposed": float((gray < 0.05).mean() * 100),
    }


def candidate_mask(rgb):
    """Masque simple de regions candidates (baseline Phase 0).

    Idee naive :
      - passer en HSV
      - seuiller le canal V (luminosite) avec Otsu
      - nettoyer un peu avec morphologie
    A ajuster apres discussion avec Godwin (ses seuils A5).
    """
    hsv = rgb2hsv(img_as_float(rgb))
    v = hsv[:, :, 2]
    mask = v < filters.threshold_otsu(v)
    # Supprime les toutes petites regions (bruit)
    mask = morphology.remove_small_objects(mask, min_size=20)
    # Comble les tout petits trous
    mask = morphology.remove_small_holes(mask, area_threshold=20)
    return mask


def preview_candidates(name, rgb):
    """Affiche QC + masque + nombre de composantes connexes."""
    q = analyze_quality(rgb)
    mask = candidate_mask(rgb)
    labeled = measure.label(mask)   # etiquette chaque blob connecte
    n = int(labeled.max())          # nombre de composantes connexes

    print(f"\n=== {name} ===")
    for k, val in q.items():
        print(f"  {k}: {val}")
    print(f"  n_candidates (composantes connexes): {n}")

    show_images(
        [rgb, rgb2gray(rgb), mask],
        titles=[name, "gray", f"mask ({n} CC)"],
        ncols=3,
    )


# Tests sur 2 images cles (changez les cles pour en tester d'autres)
preview_candidates("parasitized_1", IMAGES["parasitized_1"])
preview_candidates("uninfected_1", IMAGES["uninfected_1"])

# TODO BENOIT + relecture GODWIN :
#   - le masque est-il trop gros / trop petit ?
#   - faut-il seuiller S plutot que V ?
#   - faux positifs sur uninfected ?
'''))

cells.append(md(r"""### Notes Bénoit — B5
> *À compléter :* Le masque est-il fiable ? Faux positifs ? Piste d'amélioration Phase 1 (features) ?"""))

# ========================= BLOC C =========================

cells.append(md(r"""---
# Bloc C — Commun · Synthèse à deux

## C1 — Checklist Phase 0 (Digital Images → PaluScope)

Cochez ensemble :

- [ ] Images clés chargées et documentées (parasité / non parasité)
- [ ] Vous savez **changer** les images (`SEED`, clés `IMAGES`, upload, dossier)
- [ ] Inspection shape / dtype / plages (Godwin A1)
- [ ] Affichage + colormaps + canaux RGB (Godwin A2)
- [ ] Histogrammes gray + RGB (+ comparaison classes) (Godwin A3)
- [ ] Transformations : négatif, gamma, étalement, égalisation (Godwin A4)
- [ ] Seuillage manuel + Otsu (Godwin A5)
- [ ] RGB ↔ HSV + stats canaux (Bénoit B1)
- [ ] Addition / soustraction / division (Bénoit B2)
- [ ] Moyenne, variance, contraste (Bénoit B3)
- [ ] Quantification, flou, sur/sous-exposition (Bénoit B4)
- [ ] Mini `analyze_quality` + masque candidats (Bénoit B5)
- [ ] Notes d'interprétation rédigées sous chaque bloc

## C2 — Réponses courtes (à rédiger à deux)"""))

cells.append(code(r'''# =============================================================================
# C2 — SYNTHESE ECRITE (GODWIN + BENOIT)
# =============================================================================
# Remplissez les chaines apres vos experiments.
# Gardez des reponses COURTES (2-4 phrases max par point).
# =============================================================================

answers = {
    "definition_image": "",          # Godwin : image = tableau...
    "utilite_histogramme": "",       # Godwin
    "interet_hsv": "",               # Benoit
    "ops_arithmetiques_utiles": "",  # Benoit
    "indicateurs_qc_retenus": "",    # Benoit + Godwin
    "limites_phase0": "",            # les deux
    "comment_on_change_les_images": (
        "Par defaut: TFDS malaria -> dict IMAGES. "
        "Pour changer: (1) SEED/N_PER_CLASS en 0.2, "
        "(2) cles IMAGES['...'] dans chaque exo, "
        "(3) upload 0.2b, (4) dossier 0.2c."
    ),
}

for k, v in answers.items():
    print(f"- {k}: {v if v else '(a rediger)'}")'''))

cells.append(md(r"""## C3 — Prochaine étape (Phase 1 preview)

Une fois ce notebook stable :
- documenter la **provenance** des images (dataset, device si connu, thin/thick),
- exporter un tableau de features simples (aire, circularité, ratios couleur) sur les candidats,
- préparer la séparation train/val/test **sans fuite** patient/lame.

---

### Rappels — changer les images

| Méthode | Où ? | Quand l'utiliser |
|---|---|---|
| Changer `SEED` / `N_PER_CLASS` | Cellule **0.2** | Rester sur NIH/TFDS mais tirer d'autres exemples |
| Changer `IMAGES["clé"]` | Dans chaque exo A/B | Tester une autre image déjà chargée |
| Upload PC | Cellule **0.2b** (`USE_UPLOAD=True`) | Vos propres fichiers locaux |
| Dossier Colab | Cellule **0.2c** (`USE_FOLDER=True`) | Dataset dézippé / Google Drive |

### Liens utiles
- Cahier des charges PaluScope (PDF mentor)
- BIP Digital images : https://vincmazet.github.io/bip/digital-images/intro/
- Dataset NIH/NLM (réf. cahier des charges)"""))

nb = {
    "nbformat": 4,
    "nbformat_minor": 5,
    "metadata": {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3",
        },
        "language_info": {"name": "python", "pygments_lexer": "ipython3"},
        "colab": {"provenance": [], "toc_visible": True},
    },
    "cells": cells,
}

out = Path("/home/godwin/Bureau/PaluScope/PaluScope_Phase0_DigitalImages_Godwin_Benoit.ipynb")
out.write_text(json.dumps(nb, ensure_ascii=False, indent=2), encoding="utf-8")
print("Wrote", out)
print("Cells:", len(cells))
