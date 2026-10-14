# %% [markdown]
# # EDA: Parkinson's Disease OFF Motor Score
#
# Exploratory data analysis of the synthetic multi-cohort Parkinson's dataset,
# run before designing a model.
#
# ---------------------------------------------------------------------------
# Rôle     : exploration initiale des données (EDA) — ne modifie aucune donnée brute
# Entrées  : data/X_train.csv, data/y_train.csv, data/X_test.csv
# Sorties  : data/eda_train.html, data/eda_test.html, data/eda.md
# Exécuter : .venv/Scripts/python.exe data/eda.py
# ---------------------------------------------------------------------------
#
# WORKBOOK Day1-Q1 : taille du jeu de données
# Réponse : train = 44 590 visites / 5 576 patients ; test = 11 013 / 1 395 patients.
#   Médiane 7 visites/patient, max 12.
#
# WORKBOOK Day1-Q2 : colonnes les plus manquantes
# Réponse : time_since_intake_off 78.8 %, time_since_intake_on 46.4 %, off 42.4 %.
#   Sans off : 42.4 %, sans on : 29.6 %, les deux manquants simultanément : 0 %.
#
# WORKBOOK Day1-Q3 : distribution de target
# Réponse : min=0.0 max=109.5 (< 132 théorique), mean=37.5, std=16.5, skew=0.05.
#   Distribution quasi-symétrique, pas besoin de transformer la cible.
#
# WORKBOOK Day1-Q4 : corrélations avec target
# Réponse : off (0.871) > on (0.669) > age (0.310) > ledd (0.298).
#   off n'est PAS une copie de target : off−target a un std de 8.7 pts (≈53 % de σ_y).
#
# WORKBOOK Day1-Q5 : ledd manquant par cohorte
# Réponse : cohorte B = 42.6 % manquant, cohorte A = 35.9 %. Signal non complètement
#   aléatoire (dépend de la cohorte). Pas prouvable comme MNAR stricto sensu.
#
# WORKBOOK Day1-Q6 : colonnes à exclure
# Réponse : Index (identifiant unique ligne) et patient_id (identifiant patient).
#   target est la cible. off/on ne sont pas de la fuite (présents en X_test).
#
# - **Raw data** is read-only — CSVs live in `data/`. This file never
#   cleans or modifies the raw data.
# - **Outputs** go under `EDA_DIR` (the repo's `data/`): an
#   `eda_<table>.html` report per table, summarized in `eda.md`.

# %%
import json
from pathlib import Path

import pandas as pd
import skrub

# This file lives in data/, so parents[1] is the repo root.
PROJECT_ROOT = Path(__file__).resolve().parents[1]

# EDA outputs always land here (created if missing); the raw data may
# live elsewhere.
EDA_DIR = PROJECT_ROOT / "data"
EDA_DIR.mkdir(parents=True, exist_ok=True)

# %% [markdown]
# ## Load the raw data
#
# X_train + y_train form the labelled training set.
# X_test is the holdout (patients not in train).

# %%
X_train = pd.read_csv(PROJECT_ROOT / "data" / "X_train.csv")
y_train = pd.read_csv(PROJECT_ROOT / "data" / "y_train.csv")
X_test = pd.read_csv(PROJECT_ROOT / "data" / "X_test.csv")

# Merge target into train for joint exploration
RAW = X_train.copy()
RAW["target"] = y_train["target"].values

{"X_train_shape": X_train.shape, "y_train_shape": y_train.shape, "X_test_shape": X_test.shape}  # noqa: B018

# %% [markdown]
# ## Table overview — training set (X_train + target)
#
# Per-table report (column types, distributions, associations) saved to
# `data/eda_train.html`, plus a compact per-column summary: dtype,
# fraction missing, and number of unique values.

# %%
report_train = skrub.TableReport(RAW, title="train (X_train + target)", verbose=0)
report_train.write_html(EDA_DIR / "eda_train.html")

summary_train = json.loads(report_train.json())
n_rows_train = summary_train.get("n_rows")
overview_train = [
    {
        "column": col.get("name"),
        "dtype": col.get("dtype"),
        "null_pct": col.get("null_proportion"),
        "n_unique": col.get("n_unique"),
    }
    for col in summary_train.get("columns", [])
]
{"n_rows": n_rows_train, "n_columns": len(overview_train), "columns": overview_train}  # noqa: B018

# %% [markdown]
# ## Table overview — test set (X_test)
#
# Shape and column stats for the holdout table.

# %%
report_test = skrub.TableReport(X_test, title="test (X_test)", verbose=0)
report_test.write_html(EDA_DIR / "eda_test.html")

summary_test = json.loads(report_test.json())
n_rows_test = summary_test.get("n_rows")
overview_test = [
    {
        "column": col.get("name"),
        "dtype": col.get("dtype"),
        "null_pct": col.get("null_proportion"),
        "n_unique": col.get("n_unique"),
    }
    for col in summary_test.get("columns", [])
]
{"n_rows": n_rows_test, "n_columns": len(overview_test), "columns": overview_test}  # noqa: B018

# %% [markdown]
# ## Target
#
# Distribution of the `target` (true / debiased OFF motor score,
# regression). Skew and range determine the metric default and whether
# a target transform might help.

# %%
TARGET = "target"
target_col = next(
    (col for col in summary_train.get("columns", []) if col.get("name") == TARGET), None
)
target_col  # noqa: B018

# %% [markdown]
# ## Structure signals
#
# Datetime columns (time-based validation) and high-cardinality
# id / group-like columns (grouped validation to avoid entity leakage).

# %%
datetime_cols = [
    col.get("name")
    for col in summary_train.get("columns", [])
    if "date" in str(col.get("dtype", "")).lower()
]
unique_ratio = sorted(
    (
        {
            "column": col.get("name"),
            "unique_ratio": (col.get("n_unique") or 0) / n_rows_train if n_rows_train else None,
        }
        for col in summary_train.get("columns", [])
    ),
    key=lambda r: (r["unique_ratio"] is not None, r["unique_ratio"]),
    reverse=True,
)
{"datetime_cols": datetime_cols, "top_unique_ratio": unique_ratio[:12]}  # noqa: B018

# %% [markdown]
# ## Associations
#
# Strongest pairwise column associations. Strong feature↔target links
# are candidate predictors; an implausibly perfect one is a possible
# leakage flag to call out explicitly.

# %%
assoc = skrub.column_associations(RAW)
rows = assoc.to_dicts() if hasattr(assoc, "to_dicts") else assoc.to_dict(orient="records")
target_links = [
    row
    for row in rows
    if row["left_column_name"] == TARGET or row["right_column_name"] == TARGET
]
{"with_target": target_links[:15], "strongest": rows[:10]}  # noqa: B018

# %% [markdown]
# ## Summary
#
# The findings and their modelling implications are written up in
# `data/eda.md`.
