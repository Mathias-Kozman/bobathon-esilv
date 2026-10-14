<p align="center">
  <img src="docs/figures/probabl-logo.svg" alt="Probabl" height="100">
  &nbsp;&nbsp;&nbsp;&nbsp;
  <img src="docs/figures/ibm-logo.svg" alt="IBM" height="60">
</p>
# Help science build better Parkinson's evaluation

*Predicting and correcting bias in motor score evaluation — for each visit, predict the unbiased (“true”) OFF MDS-UPDRS motor score.*

## Getting started

```bash
git clone https://github.com/probabl-ai/bobathon-esilv.git
cd bobathon-esilv
```

The lab has three steps. Do them **in order**: each one assumes the previous one is finished.


| Step | Guide                     | What you do                                                                                                |
| ---- | ------------------------- | ---------------------------------------------------------------------------------------------------------- |
| 1    | [SETUP.md](docs/SETUP.md) | Join Kaggle, form a team, install Python, create the shared Skore Hub workspace, install skore and sign in |
| 2    | [DAY1.md](docs/DAY1.md)   | Download the data, explore it (EDA), push a dummy baseline to Skore Hub                                    |
| 3    | [DAY2.md](docs/DAY2.md)   | Build real models, evaluate them with patient-grouped cross-validation, submit to Kaggle                   |


### Step 1 — Setup ([SETUP.md](docs/SETUP.md))

Follow every section of [SETUP.md](docs/SETUP.md):

1. **Join Kaggle and form a team** (maximum four people). Remember the exact team name.
2. **Install Python 3.12 or newer** and check it with `python --version` (or `python3 --version`).
3. **Create one Skore Hub workspace per team** on [skore.probabl.ai](https://skore.probabl.ai), named like the Kaggle team. One teammate creates it and invites the others.
4. **Allow Bob to use the lab skills** (tick **Skill** in Bob's Permissions menu), then **install skore and sign in** from the repo root.

#### Check your setup before moving on

Run the setup tests, with the virtual environment still active:

```bash
python -m pytest
```

See [SETUP.md step 5](docs/SETUP.md#5-install-skore-and-check-your-setup) for details.

**You cannot proceed to Day 1 until the last line says** `all passed`**.** Without a working `.skore`, you cannot push reports to Skore Hub, and every Kaggle Submission needs a Hub report URL.

### Step 2 — Day 1 ([DAY1.md](docs/DAY1.md))

1. Download the CSVs from the [competition Data tab](https://www.kaggle.com/competitions/bobathon-esilv/data) into `data/`.
2. Explore the data in Bob (EDA) and go through the workbook questions.
3. Build a **dummy mean** baseline, evaluate it with `skore.evaluate`, and `put` it to Skore Hub as `01_dummy`. Every later model must beat this score.

### Step 3 — Day 2 ([DAY2.md](docs/DAY2.md))

1. Train a **Ridge** linear model, compare it with the dummy, and make your **first Kaggle Submission**.
2. Switch to **patient-grouped cross-validation** (`GroupKFold`) so the score matches the Kaggle split.
3. Try **HistGradientBoosting** (missing values as signal), skrub `**tabular_pipeline`** (categorical columns), and skrub **DataOps** (groups baked into the graph).
4. Fit your best model on all training visits and **submit again**.

## Documents

- [SETUP.md](docs/SETUP.md) - Kaggle team, Python, Skore Hub workspace, skore install and sign-in
- [DAY1.md](docs/DAY1.md) - data, exploratory data analysis (EDA), dummy baseline
- [DAY2.md](docs/DAY2.md) - models, grouped cross-validation, Kaggle Submissions
- [CONTEXT.md](docs/CONTEXT.md) - scientific background, goals, and modelling pitfalls for the Parkinson's dataset
- Kaggle competition: [bobathon-esilv](https://www.kaggle.com/competitions/bobathon-esilv/)

## Submission rules

All code must be written and run in **Bob** (IDE or CLI), with evaluation through **skore**. Every Kaggle upload must put a Skore Hub EstimatorReport URL (`https://skore.probabl.ai/…`) in the Submission Description, and each Submission needs a **new** report key (`01_dummy`, `02_ridge`, …). Each experiment writes its file to `submissions/<report-key>.csv` (e.g. `submissions/02_ridge.csv`), so the filename matches the report key and a later model never overwrites an earlier submission. See the [competition rules](https://www.kaggle.com/competitions/bobathon-esilv/rules) for eligibility and what makes a Submission valid.

---

## Structure du projet

```
bobathon-esilv/
├── data/               # Données brutes (CSV) + EDA (eda.py, eda.md, eda_*.html)
├── docs/               # Documentation : CONTEXT.md, DAY1.md, DAY2.md, SETUP.md
├── experiments/        # Scripts d'expériences (jupytext # %%)
├── src/parkinson/      # Package Python : PROJECT_ROOT (__init__.py), hub.py
├── submissions/        # CSV Kaggle (un par expérience, jamais écrasés)
├── scratch/            # Scripts temporaires (non soumis)
├── tests/              # Tests de setup (test_setup.py)
├── WORKBOOK.md         # Réponses aux questions du workbook Day 1 / Day 2
└── README.md           # Ce fichier
```

## Reproduire les résultats

```bash
# 1. Activer l'environnement
.venv\Scripts\activate        # Windows
source .venv/bin/activate     # macOS/Linux

# 2. Vérifier le setup
python -m pytest tests/test_setup.py

# 3. Lancer une expérience (exemple : Ridge)
python experiments/02_ridge.py
```

Chaque expérience est autonome : elle charge les CSV, calcule le CV groupé et (si
applicable) pousse le report au hub et écrit le CSV de soumission.

## Expériences et RMSE (GroupKFold 5 folds, groupé par patient_id)

Toutes les valeurs sont **recalculées localement** sauf mention contraire.

| Expérience | Description | RMSE CV moyen | Source |
|------------|-------------|:-------------:|--------|
| [01_dummy](experiments/01_dummy.py) | DummyRegressor (mean) | 16.50 ± 0.31 | recalculé localement |
| [02_ridge](experiments/02_ridge.py) | Ridge α=1, imputation médiane | 10.60 ± 0.24 | recalculé localement |
| [03_groupkfold](experiments/03_groupkfold.py) | Dummy + Ridge avec CV groupé | 16.50 / 10.60 | recalculé localement |
| [04_hgbr](experiments/04_hgbr.py) | HistGradientBoostingRegressor (NaN conservés) | 7.75 ± 0.14 | recalculé localement |
| [05_skrub_tabular](experiments/05_skrub_tabular.py) | tabular_pipeline (cohort, gene, rater_id) | 7.51 ± 0.12 | recalculé localement |
| [06_dataops](experiments/06_dataops.py) | DataOps skrub (HGBR 8 num. features) | 7.75 ± 0.16 | code exp 06 |
| [07_final](experiments/07_final.py) | Meilleur modèle Day 2 (tabular_pipeline) | 7.51 ± 0.12 | code exp 07 |
| [08_error_analysis](experiments/08_error_analysis.py) | Analyse des résidus (pas de soumission) | — | — |
| [09_targeted_improvements](experiments/09_targeted_improvements.py) | Indicateurs de manquant + HGBR tuné | ~7.4 | code exp 09 |
| [11_patient_features](experiments/11_patient_features.py) | Agrégats patient (X-only, sans fuite) | ~5.2 | code exp 11 |
| [12_interaction_features](experiments/12_interaction_features.py) | Features d'interaction × agrégats patient | ~5.1 | code exp 12 |
| [13_best_model](experiments/13_best_model.py) | LightGBM + feature engineering complet + ensemble | **meilleur** | code exp 13 |
| [15_advanced_improvements_fast](experiments/15_advanced_improvements_fast.py) | Historique patient séquentiel (sans fuite) | ~4.1 | code exp 15 |
| [16_bias_correction](experiments/16_bias_correction.py) | Correction biais régressif (isotonic/Huber) | ~4.1 | code exp 16 |

> **Note :** `experiments/14_improvements.py` est en cours d'optimisation — ne pas
> modifier, ne pas lancer.
