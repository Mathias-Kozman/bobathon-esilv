# %% [markdown]
# # EDA: Parkinson's Disease OFF Motor Score
#
# Exploratory data analysis of the synthetic multi-cohort Parkinson's dataset,
# run before designing a model.
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
