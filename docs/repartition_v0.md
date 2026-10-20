# Répartition V0 — Godwin & Bénoit

Ce document précise **qui fait quoi**, **dans quel fichier**, et **sur quelle branche**.

---

## 1. Convention de nommage des branches

### Branches permanentes

| Branche | Rôle |
|---|---|
| `main` | Code stable, intégré, prêt à être montré / démontré |
| *(pas de `develop` pour l’instant)* | On reste simple à 2 personnes |

### Branches de travail (obligatoires)

Chaque personne pousse **d’abord** sur **sa** branche, jamais directement sur `main`.

**Format :**

```text
<prenom>/<sujet-court>
```

| Qui | Préfixe obligatoire | Exemples |
|---|---|---|
| **Godwin** | `godwin/` | `godwin/io-load-rgb`, `godwin/preprocess-hist`, `godwin/streamlit-upload` |
| **Bénoit** | `benoit/` | `benoit/qc-metrics`, `benoit/candidates-mask`, `benoit/report-json` |

### Règles

1. Créer la branche depuis `main` à jour.
2. Un sujet = une branche (évite les branches fourre-tout géantes).
3. Commits clairs, en français ou anglais, mais cohérents.
4. Pousser sur **sa** branche distante : `git push -u origin godwin/...` ou `benoit/...`.
5. Quand c’est prêt : **Pull Request** vers `main` (review croisée si possible).
6. Ne pas modifier les fichiers “owner exclusif” de l’autre sur sa branche, sauf accord.
7. Fichiers partagés (`quality.py`, `streamlit_app.py`, `test_quality.py`) : se coordonner (voir §3).

### Workflow type

```bash
git checkout main
git pull

# Exemple Godwin
git checkout -b godwin/preprocess-hist
# ... coder ...
git add ...
git commit -m "feat(preprocess): ajout rgb2gray et gamma"
git push -u origin godwin/preprocess-hist
# puis ouvrir une PR vers main
```

```bash
# Exemple Bénoit
git checkout -b benoit/qc-metrics
# ...
git push -u origin benoit/qc-metrics
# PR vers main
```

---

## 2. Répartition précise — fichier par fichier

### 2.1 Godwin — propriétaire principal

#### `src/paluscope/io.py`
**Branche type :** `godwin/io-load-rgb`

À implémenter :
- [ ] `load_rgb(path)` — lire une image disque en RGB `uint8 (H,W,3)`
- [ ] `describe_array(name, image)` — retourner shape, dtype, min, max, mean, std
- [ ] Gérer les erreurs (fichier absent, lecture impossible)

**Source notebook :** cellules A1 (inspection).

---

#### `src/paluscope/preprocess.py`
**Branche type :** `godwin/preprocess-hist`

À implémenter :
- [ ] `to_float_rgb` / `to_gray` / `to_hsv`
- [ ] `split_rgb` — séparer R, G, B
- [ ] `negative`, `gamma_correct`, `stretch`, `equalize`
- [ ] Docstrings courtes + types

**Source notebook :** A1, A2, A4 (+ HSV utilitaire utilisé aussi par Bénoit).

> `to_hsv` est dans ce fichier mais **consommé** par Bénoit dans `candidates.py`.  
> Godwin livre l’API ; Bénoit l’importe, il ne la réécrit pas.

---

#### `src/paluscope/quality.py` — partie Godwin
**Branche type :** `godwin/quality-threshold`

À implémenter **dans ce fichier partagé** (section clairement commentée `# --- Godwin ---`) :
- [ ] `threshold_manual(gray01, t)` → masque booléen
- [ ] `threshold_otsu_mask(gray01)` → `(T, masque)`

**Source notebook :** A5.

Ne pas toucher aux fonctions Bénoit du même fichier (sharpness, exposition, etc.) sans coordination.

---

#### `app/streamlit_app.py` — partie Godwin
**Branche type :** `godwin/streamlit-upload`

À implémenter :
- [ ] Titre + disclaimer médical
- [ ] `st.file_uploader` (png/jpg)
- [ ] Affichage image RGB + niveaux de gris
- [ ] Histogramme grayscale simple

**Ne pas** encore câbler le panneau QC / export (Bénoit).

---

#### `tests/test_quality.py` — partie Godwin
**Branche type :** `godwin/tests-preprocess`

À ajouter :
- [ ] Test shape / plage de `to_gray`
- [ ] Test `negative` / `gamma_correct`
- [ ] Test `threshold_otsu_mask` (seuil dans `[0,1]`, masque booléen)

---

#### `README.md` (structure & install)
**Branche type :** `godwin/docs-readme`

- [ ] Maintenir la carte des dossiers / fichiers
- [ ] Install, stack, disclaimer
- [ ] Lien vers ce document de répartition

---

### 2.2 Bénoit — propriétaire principal

#### `src/paluscope/quality.py` — partie Bénoit
**Branche type :** `benoit/qc-metrics`

À implémenter **dans ce fichier partagé** (section `# --- Benoit ---`) :
- [ ] `mean_variance_contrast(gray01)` → mean, variance, std, michelson
- [ ] `sharpness_laplacian(gray01)` → score de netteté
- [ ] `exposure_ratios(gray01)` → % sur / sous-exposition
- [ ] `analyze_quality(rgb)` → dict QC complet (appelle aussi le gray de preprocess)

**Source notebook :** B3, B4, B5 (`analyze_quality`).

---

#### `src/paluscope/candidates.py`
**Branche type :** `benoit/candidates-mask`

À implémenter :
- [ ] `candidate_mask(rgb)` — HSV + seuillage V (ou variante) + morphologie légère
- [ ] `label_candidates(mask)` — composantes connexes + `n`
- [ ] `candidate_summary(rgb)` — n_candidates, aire du masque, etc.

**Source notebook :** B1, B5.  
Utilise `to_hsv` depuis `preprocess.py` (Godwin).

---

#### `src/paluscope/report.py`
**Branche type :** `benoit/report-json`

À implémenter :
- [ ] `build_report(rgb, source=...)` — agrège QC + candidats + disclaimer + timestamp
- [ ] `report_to_dataframe(report)` — une ligne Pandas pour l’UI
- [ ] `save_report_json(report, path)` — export sans les gros arrays (`_mask`, etc.)

**Source notebook :** B5 + synthèse.

---

#### `app/streamlit_app.py` — partie Bénoit
**Branche type :** `benoit/streamlit-qc-panel`

À implémenter (après ou en parallèle de l’upload Godwin, via merge/`main`) :
- [ ] Affichage du masque candidats
- [ ] Tableau / métriques des indicateurs QC
- [ ] Bouton export rapport JSON vers `outputs/reports/`

---

#### `tests/test_quality.py` — partie Bénoit
**Branche type :** `benoit/tests-qc`

À ajouter :
- [ ] Test clés de `analyze_quality`
- [ ] Test “image floutée ⇒ sharpness plus faible”
- [ ] Test `candidate_mask` / `label_candidates` (`n >= 0`)

---

#### `data/README.md`
**Branche type :** `benoit/docs-data`

- [ ] Compléter sources / licences si besoin
- [ ] Noter comment placer 3–5 images d’exemple dans `data/raw/`

---

### 2.3 Fichiers partagés ou communs

| Fichier | Qui | Comment on évite les conflits |
|---|---|---|
| `src/paluscope/quality.py` | Godwin + Bénoit | Sections commentées séparées ; branches différentes ; merge via PR |
| `app/streamlit_app.py` | Godwin puis Bénoit | Godwin merge d’abord l’upload ; Bénoit part de `main` à jour |
| `tests/test_quality.py` | Godwin + Bénoit | Ajouter des fonctions de test distinctes (pas réécrire les tests de l’autre) |
| `src/paluscope/__init__.py` | Godwin (léger) | Version uniquement, sauf besoin d’exports publics |
| `requirements.txt` | Les deux | Toute nouvelle dépendance = PR dédiée + accord |
| `notebooks/00_phase0_...ipynb` | Les deux | Journal d’expérience ; pas la source de vérité du produit |
| `_build_phase0_nb.py` | Godwin | Outil interne ; ne pas modifier sans besoin |

---

## 3. Ordre de travail suggéré

```text
1. Godwin  : io.py + preprocess.py          → PR → main
2. Bénoit  : quality.py (metrics)           → PR → main   (importe to_gray)
3. Godwin  : quality.py (seuillage)         → PR → main
4. Bénoit  : candidates.py + report.py      → PR → main
5. Godwin  : streamlit upload/affichage     → PR → main
6. Bénoit  : streamlit QC + export          → PR → main
7. Les deux: tests manquants + smoke Streamlit
```

Cet ordre limite les blocages : Bénoit a besoin de `to_gray` / `to_hsv` assez tôt.

---

## 4. Definition of Done V0 (équipe)

- [ ] Toutes les fonctions listées ci-dessus implémentées dans `src/`
- [ ] `streamlit run app/streamlit_app.py` : upload → QC → masque → export JSON
- [ ] `pytest -q` vert
- [ ] Disclaimer visible dans l’UI et le README
- [ ] Aucune grosse image commitée dans `data/raw/`
- [ ] Historique Git propre : travail passé par branches `godwin/*` et `benoit/*`

---

## 5. Rappels collab

1. **Chacun pousse d’abord sur sa branche** (`godwin/...` ou `benoit/...`).
2. `main` ne reçoit que des PR validées.
3. Prototype ≠ diagnostic médical.
4. Le notebook Phase 0 reste la référence pédagogique ; le produit vit dans `src/` + `app/`.
