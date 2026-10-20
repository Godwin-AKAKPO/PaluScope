# PaluScope — V0 Blood Smear Quality Analyzer

Prototype de **recherche / formation / assistance à la lecture** pour la microscopie du paludisme (AMA · PIIA Cohorte 2).

> **Disclaimer médical :** PaluScope ne délivre **pas** de diagnostic clinique autonome, ne prescrit aucun traitement et n’est **pas** un dispositif médical validé.

## Objectif V0

Importer une image de frottis → mesurer la **qualité technique** → proposer des **zones candidates** → afficher un mini rapport.

```text
Image → QC → Preprocessing → Candidate Detection → Human Review
```

## Équipe

| Personne | Périmètre V0 |
|---|---|
| **Godwin** | I/O, preprocess, histogrammes, transformations, seuillage, README |
| **Bénoit** | Stats QC (sharp/expo/contraste), HSV, candidats, rapport |

Détail : `docs/repartition_v0.md`

## Stack

- Python 3.11+
- NumPy, Pandas, Matplotlib
- OpenCV, scikit-image
- Streamlit (démo)
- Git / GitHub

## Installation

```bash
cd PaluScope
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## Structure (squelette)

```text
PaluScope/
├── app/streamlit_app.py      # démo V0 (à implémenter)
├── src/paluscope/            # bibliothèque (stubs)
│   ├── io.py                 # Godwin
│   ├── preprocess.py         # Godwin
│   ├── quality.py            # Godwin + Bénoit
│   ├── candidates.py         # Bénoit
│   └── report.py             # Bénoit
├── notebooks/                # Phase 0 BIP (exploration)
├── data/raw/                 # images exemples (non versionnées)
├── tests/
├── outputs/reports/
└── docs/repartition_v0.md
```

Les modules `src/` et l’app Streamlit sont des **stubs** : le code métier reste à écrire à partir du notebook Phase 0.

## Données

Dataset ouvert prioritaire V0 : **NIH/NLM Malaria** (patches thin smear).  
Voir `data/README.md`.

## Phase 0

Notebook BIP Digital Images :  
`notebooks/00_phase0_digital_images_godwin_benoit.ipynb`

Référence mentor : https://vincmazet.github.io/bip/digital-images/intro/
