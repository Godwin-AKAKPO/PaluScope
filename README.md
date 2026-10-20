# PaluScope — V0 Blood Smear Quality Analyzer

Prototype de **recherche / formation / assistance à la lecture** pour la microscopie du paludisme  
(AMA · PIIA Cohorte 2 · Machine Learning).

> **Disclaimer médical :** PaluScope ne délivre **pas** de diagnostic clinique autonome,  
> ne prescrit aucun traitement et n’est **pas** un dispositif médical validé.

---

## Objectif V0

Importer une image de frottis → mesurer la **qualité technique** → proposer des **zones candidates** → afficher un mini rapport.

```text
Image → QC → Preprocessing → Candidate Detection → Human Review
```

Sans CNN. Vision classique + maths uniquement.

---

## Équipe

| Personne | Rôle V0 |
|---|---|
| **Godwin** | I/O, preprocess, histogrammes, transformations, seuillage, structure README |
| **Bénoit** | Stats QC (sharp / expo / contraste), HSV, candidats, rapport |

La répartition détaillée (fichier par fichier + branches) est dans  
**[`docs/repartition_v0.md`](docs/repartition_v0.md)**.

---

## Stack

| Usage | Techno |
|---|---|
| Langage | Python 3.11+ |
| Calcul / tableaux | NumPy, Pandas |
| Vision | OpenCV, scikit-image |
| Affichage / démo | Matplotlib, Streamlit |
| Exploration | Jupyter / Colab |
| Tests | pytest |
| Versioning | Git + GitHub |

---

## Installation

```bash
cd PaluScope
python -m venv .venv
source .venv/bin/activate          # Windows : .venv\Scripts\activate
pip install -r requirements.txt
```

Lancer la démo (quand l’app sera implémentée) :

```bash
streamlit run app/streamlit_app.py
```

Lancer les tests :

```bash
pytest -q
```

---

## Structure du dépôt — à quoi sert chaque dossier / fichier

```text
PaluScope/
├── README.md                          # ce fichier : vision projet + guide
├── requirements.txt                   # dépendances Python V0
├── .gitignore                         # fichiers exclus de Git
├── _build_phase0_nb.py                # script utilitaire (génération notebook Phase 0)
│
├── docs/                              # documentation d’équipe
│   └── repartition_v0.md              # qui fait quoi + convention de branches
│
├── app/                               # interface utilisateur (démo)
│   └── streamlit_app.py               # PaluScope Studio V0 (upload → QC → candidats)
│
├── src/                               # code métier (bibliothèque importable)
│   └── paluscope/
│       ├── __init__.py                # package + version
│       ├── io.py                      # charger / décrire une image
│       ├── preprocess.py              # RGB↔gris/HSV, transformations d’histogramme
│       ├── quality.py                 # indicateurs QC + seuillage
│       ├── candidates.py              # masque + composantes connexes
│       └── report.py                  # agréger QC + export JSON
│
├── notebooks/                         # exploration / journal d’expérience
│   └── 00_phase0_digital_images_godwin_benoit.ipynb
│
├── data/                              # données locales (peu ou pas versionnées)
│   ├── README.md                      # sources, licences, métadonnées attendues
│   └── raw/                           # images d’exemple à placer ici
│       └── .gitkeep                   # garde le dossier vide dans Git
│
├── outputs/                           # artefacts générés (ignorés en grande partie)
│   └── reports/                       # rapports QC exportés (JSON, etc.)
│       └── .gitkeep
│
└── tests/                             # tests unitaires
    ├── conftest.py                    # ajoute src/ au PYTHONPATH pour pytest
    └── test_quality.py                # tests preprocess / QC / candidats
```

### Dossiers

| Dossier | Rôle |
|---|---|
| **`app/`** | Point d’entrée **produit** : petite appli Streamlit pour démontrer le V0. |
| **`src/paluscope/`** | Cœur du projet : fonctions réutilisables (pas de logique métier dans le notebook seul). |
| **`notebooks/`** | Expérimentations Phase 0 (BIP Digital Images). Sert de référence pour porter le code vers `src/`. |
| **`data/`** | Images locales. `raw/` reçoit les frottis de test ; les gros fichiers ne vont **pas** sur Git. |
| **`outputs/`** | Résultats produits par l’app / les scripts (rapports). Régénérables → peu versionnés. |
| **`docs/`** | Docs d’organisation (répartition, conventions). |
| **`tests/`** | Vérifie que les fonctions de `src/` se comportent correctement. |

### Fichiers racine

| Fichier | Rôle |
|---|---|
| **`README.md`** | Présentation du projet, install, carte du dépôt. |
| **`requirements.txt`** | Liste des packages à installer pour faire tourner V0. |
| **`.gitignore`** | Empêche de committer `.venv/`, caches, images brutes, rapports, etc. |
| **`_build_phase0_nb.py`** | Outil interne pour (re)générer le notebook Phase 0 si besoin. |

### Fichiers `src/paluscope/`

| Fichier | Rôle |
|---|---|
| **`__init__.py`** | Déclare le package Python `paluscope` et sa version. |
| **`io.py`** | Lire une image depuis le disque, résumer shape/dtype/stats. |
| **`preprocess.py`** | Convertir RGB→gris/HSV, négatif, gamma, étalement, égalisation. |
| **`quality.py`** | Calculer netteté, exposition, contraste ; seuillage manuel / Otsu. |
| **`candidates.py`** | Produire un masque de zones suspectes + compter les composantes connexes. |
| **`report.py`** | Assembler un rapport QC + candidats et l’exporter (JSON / tableau). |

### Autres fichiers importants

| Fichier | Rôle |
|---|---|
| **`app/streamlit_app.py`** | UI : upload image, affichage, indicateurs, export. |
| **`data/README.md`** | Documente d’où viennent les datasets (NIH, Uganda…). |
| **`docs/repartition_v0.md`** | Qui implémente quoi, dans quel fichier, sur quelle branche. |
| **`tests/conftest.py`** | Config pytest (chemin d’import). |
| **`tests/test_quality.py`** | Tests des fonctions V0. |
| **`notebooks/00_phase0_...ipynb`** | Travail Phase 0 Godwin + Bénoit (Colab / Jupyter). |

> Les modules `src/` et `app/streamlit_app.py` sont pour l’instant des **stubs** (TODO).  
> Le code se porte depuis le notebook Phase 0.

---

## Données

Dataset ouvert prioritaire V0 : **NIH/NLM Malaria** (patches thin smear).  
Détails et licences : [`data/README.md`](data/README.md).

---

## Git — convention de branches (résumé)

Chacun développe **d’abord sur sa branche personnelle**, puis on fusionne vers `main` via PR.

| Branche | Qui | Usage |
|---|---|---|
| `main` | les deux | version stable intégrée |
| `godwin/*` | Godwin | tout le travail de Godwin |
| `benoit/*` | Bénoit | tout le travail de Bénoit |

Exemples : `godwin/preprocess-hist`, `benoit/qc-candidates`.

**Règle :** pas de push direct de features sur `main`.  
Détail complet : [`docs/repartition_v0.md`](docs/repartition_v0.md).

---

## Phase 0 (référence)

Notebook BIP Digital Images :  
[`notebooks/00_phase0_digital_images_godwin_benoit.ipynb`](notebooks/00_phase0_digital_images_godwin_benoit.ipynb)

Cours mentor : https://vincmazet.github.io/bip/digital-images/intro/
