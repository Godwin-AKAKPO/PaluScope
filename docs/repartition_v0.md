# Répartition V0 — Godwin & Bénoit

## Modules

| Fichier | Owner | Contenu à implémenter |
|---|---|---|
| `src/paluscope/io.py` | Godwin | lecture RGB, describe |
| `src/paluscope/preprocess.py` | Godwin | gray/HSV, gamma, égalisation… |
| `src/paluscope/quality.py` | Godwin + Bénoit | seuillage (G) + sharp/expo/contraste (B) |
| `src/paluscope/candidates.py` | Bénoit | masque HSV + composantes connexes |
| `src/paluscope/report.py` | Bénoit | agrégation + export JSON |
| `app/streamlit_app.py` | Les deux | upload/affichage (G) + panneau QC (B) |
| `tests/test_quality.py` | Les deux | 1–2 tests chacun |
| `notebooks/` | Les deux | journal Phase 0 BIP |
| `README.md` | Godwin (draft) / Bénoit (QC) | install + disclaimer |

## Règles collab

1. Travailler sur des branches `feature/...` puis PR vers `main`.
2. Ne pas pousser de grosses images dans Git (`data/raw/` ignoré).
3. Toujours rappeler : prototype ≠ diagnostic.
4. Brancher le code depuis le notebook Phase 0, module par module.
