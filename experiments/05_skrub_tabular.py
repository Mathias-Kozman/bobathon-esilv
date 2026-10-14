# %%
# ---------------------------------------------------------------------------
# Rôle     : tabular_pipeline skrub — HGBR avec colonnes texte (cohort, gene, rater_id)
# Entrées  : data/X_train.csv, data/y_train.csv, data/X_test.csv
# Sorties  : hub "05_skrub_tabular", "05_hgbr_numeric_ref" ; submissions/05_skrub_tabular.csv
# Exécuter : .venv/Scripts/python.exe experiments/05_skrub_tabular.py
# ---------------------------------------------------------------------------
#
# WORKBOOK Day2-Q19 : gain tabular vs HGBR numérique (fold par fold)
# Réponse : skrub gagne sur 5/5 folds, delta +0.18 à +0.28 par fold. Cohérent, pas un hasard.
#   HGBR num folds : [7.83, 7.93, 7.58, 7.60, 7.82] → skrub : [7.55, 7.68, 7.33, 7.42, 7.62]
#
# WORKBOOK Day2-Q20 : encodeurs choisis par TableVectorizer (seuil cardinalité = 40)
# Réponse :
#   cohort (2 valeurs)   → ToCategorical (< 40 → type category pandas, HGBR natif)
#   gene   (4 valeurs)   → ToCategorical (< 40 → idem)
#   rater_id (50 val.)   → StringEncoder (> 40 → encodage dense)
#   ToCategorical ≠ one-hot : passe la colonne en type category, HGBR la gère directement.
import time

import pandas as pd
import skore
from sklearn.base import clone
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.model_selection import GroupKFold
from skore import ComparisonReport, CrossValidationReport, Project, login
from skrub import tabular_pipeline

from parkinson import PROJECT_ROOT
from parkinson.hub import load_skore_credentials

DATA_DIR = PROJECT_ROOT / "data"
X_train = pd.read_csv(DATA_DIR / "X_train.csv")
y_train = pd.read_csv(DATA_DIR / "y_train.csv")
X_test = pd.read_csv(DATA_DIR / "X_test.csv")

visits = X_train.merge(y_train, on="Index")
y = visits["target"]

# %% Step 1 — Feature preparation
# Drop identifiers (Index, patient_id) and target to avoid leakage.
# All remaining columns — including strings (gene, cohort, rater_id) — are kept.
X_full = visits.drop(columns=["Index", "patient_id", "target"])

print("=== X_full columns, dtypes & missing values ===")
info = pd.DataFrame(
    {"dtype": X_full.dtypes, "n_missing": X_full.isnull().sum()}
)
print(info.to_string())

print("\n=== String/categorical cardinality ===")
for col in X_full.select_dtypes(include=["object", "str", "category"]).columns:
    print(f"  {col}: {X_full[col].nunique()} unique values")

# %% Step 2 — Model
# tabular_pipeline returns TableVectorizer + HistGradientBoostingRegressor.
# No manual imputation: HGBR handles NaN natively; TableVectorizer handles strings.
model = tabular_pipeline("regressor")
print(f"\nModel: {model}")

# %% Step 3 — Grouped CV (same splits as experiments 03 & 04)
groups = visits["patient_id"]
cv_splits = list(GroupKFold(n_splits=5).split(X_full, y, groups=groups))
print(f"\nFolds: {len(cv_splits)}")
for i, (tr, val) in enumerate(cv_splits):
    print(f"  fold {i}: train={len(tr)}, val={len(val)}")

t0 = time.perf_counter()
report = skore.evaluate(model, X_full, y, splitter=cv_splits)
skrub_time = time.perf_counter() - t0

print(f"\nTabular pipeline CV [{skrub_time:.1f}s] — metrics summary:")
print(report.metrics.summarize())

skrub_rmse_vals = report.metrics.rmse()
print(f"\nskrub_tabular RMSE per fold:\n{skrub_rmse_vals}")

# %% Step 4 — Comparison with HGBR numeric (part 4)
# The two models take different feature sets, so we run two separate
# CrossValidationReport objects and combine them in a ComparisonReport.
FEATURE_COLS = [
    "sexM", "age_at_diagnosis", "age", "ledd",
    "time_since_intake_on", "time_since_intake_off", "on", "off",
]
X_numeric = visits[FEATURE_COLS]

# Rebuild the same GroupKFold splits on the numeric X (same indices, same groups).
cv_splits_num = list(GroupKFold(n_splits=5).split(X_numeric, y, groups=groups))

hgbr = HistGradientBoostingRegressor(random_state=0)

t0 = time.perf_counter()
hgbr_report = skore.evaluate(hgbr, X_numeric, y, splitter=cv_splits_num)
hgbr_time = time.perf_counter() - t0

hgbr_rmse_vals = hgbr_report.metrics.rmse()
print(f"\nHGBR numeric CV [{hgbr_time:.1f}s] — RMSE per fold:\n{hgbr_rmse_vals}")

# Comparison report (both reports are CrossValidationReport instances).
compare_report = ComparisonReport(
    {"hgbr_numeric": hgbr_report, "skrub_tabular": report}
)
print("\n=== Comparison summary ===")
print(compare_report.metrics.summarize())

# %% Recap table
import numpy as np  # noqa: E402

# Per-fold RMSE (from sub-reports, since .rmse() returns mean/std summary)
skrub_per_fold = np.array([r.metrics.rmse() for r in report.reports_])
hgbr_per_fold = np.array([r.metrics.rmse() for r in hgbr_report.reports_])

print("\n=== RMSE per fold ===")
for i, (s, h) in enumerate(zip(skrub_per_fold, hgbr_per_fold)):
    print(f"  fold {i}: skrub_tabular={s:.4f}  hgbr_numeric={h:.4f}")

print("\n=== RMSE recap ===")
print(f"skrub_tabular  mean={skrub_per_fold.mean():.4f}  std={skrub_per_fold.std():.4f}  "
      f"time={skrub_time:.1f}s")
print(f"hgbr_numeric   mean={hgbr_per_fold.mean():.4f}  std={hgbr_per_fold.std():.4f}  "
      f"time={hgbr_time:.1f}s")
print(f"delta (skrub - hgbr): {skrub_per_fold.mean() - hgbr_per_fold.mean():.4f}")

# %% Step 5 — Push to Hub
cfg = load_skore_credentials()
login(mode="hub")
project = Project(name="bobathon-esilv", mode="hub", workspace=cfg["workspace"])
project.put("05_skrub_tabular", report)
# ComparisonReport is not accepted by the Hub (same limitation as exp 04).
# Push the HGBR numeric report under a distinct key for cross-referencing.
project.put("05_hgbr_numeric_ref", hgbr_report)
print("\nReports pushed: 05_skrub_tabular, 05_hgbr_numeric_ref")
hub_url = f"{cfg['hub_url']}/workspaces/{cfg['workspace']}"
print(f"Hub URL: {hub_url}")

# %% Step 6 — Final model + Kaggle submission
final = clone(model).fit(X_full, y)

# Align test columns: drop Index and patient_id; keep only X_full columns.
test_cols = [c for c in X_full.columns if c in X_test.columns]
X_test_aligned = X_test[test_cols]
assert list(X_test_aligned.columns) == list(X_full.columns), (
    f"Column mismatch: test={list(X_test_aligned.columns)} vs train={list(X_full.columns)}"
)

preds = final.predict(X_test_aligned)
assert len(preds) == len(X_test), (
    f"Prediction count {len(preds)} != test rows {len(X_test)}"
)
assert not pd.isnull(preds).any(), "NaN detected in predictions"

# Verify against sample_submission row count.
sample = pd.read_csv(DATA_DIR / "sample_submission.csv")
assert len(preds) == len(sample), (
    f"Row count mismatch vs sample_submission: {len(preds)} != {len(sample)}"
)

submission = X_test[["Index"]].copy()
submission["target"] = preds
assert list(submission.columns) == ["Index", "target"], "Wrong submission columns"

out = PROJECT_ROOT / "submissions"
out.mkdir(exist_ok=True)
submission.to_csv(out / "05_skrub_tabular.csv", index=False)
print(f"\nSubmission: {submission.shape[0]} rows → submissions/05_skrub_tabular.csv")
print(submission.head())

# %% Step 7 — Encoder inspection for Q2
tv = final.named_steps["tablevectorizer"]
print("\n=== TableVectorizer encoders per column ===")
# In skrub 0.11 transformers_ is a dict {column_name: fitted_transformer}
for col_name, transformer in tv.transformers_.items():
    print(f"  {col_name}: {type(transformer).__name__}")
print()
print("Columns by kind:")
for kind, cols in tv.kind_to_columns_.items():
    if cols:
        print(f"  {kind}: {cols}")

print("\nOK: all checks passed.")
