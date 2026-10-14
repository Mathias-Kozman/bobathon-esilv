# %%
# ---------------------------------------------------------------------------
# Rôle     : améliorations ciblées sur le meilleur modèle Day 2 (skrub_tabular)
# Entrées  : data/X_train.csv, data/y_train.csv
# Sorties  : sorties console uniquement
# Exécuter : .venv/Scripts/python.exe experiments/09_targeted_improvements.py
# ---------------------------------------------------------------------------
#
# Experiment 09: targeted improvements on the best model (skrub_tabular, exp 05/07).
#
# Three improvements tested with the same GroupKFold(5) folds as all prior experiments:
#   A. Baseline  : tabular_pipeline("regressor")  [HGBR max_iter=100, max_leaf_nodes=31]
#   B. Miss flags: add off_is_missing + ledd_is_missing binary columns
#   C. HGBR-300  : max_iter=300, max_leaf_nodes=63  (more iterations, deeper trees)
#   D. HGBR-500  : max_iter=500, max_leaf_nodes=127 (most expressive)
#
# No hub push. No exhaustive search. No target-based feature engineering.

import warnings

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.metrics import root_mean_squared_error
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import FunctionTransformer
from skrub import tabular_pipeline

from parkinson import PROJECT_ROOT

warnings.filterwarnings("ignore", category=FutureWarning)

DATA_DIR = PROJECT_ROOT / "data"

# %%
# Load & verify
X_train = pd.read_csv(DATA_DIR / "X_train.csv")
y_train = pd.read_csv(DATA_DIR / "y_train.csv")
visits = X_train.merge(y_train, on="Index")
assert len(visits) == len(X_train) == len(y_train), "Merge mismatch"

y = visits["target"]
groups = visits["patient_id"]

# Base feature set: all columns except identifiers and target
X_base = visits.drop(columns=["Index", "patient_id", "target"])

# Verify missingness for the flag columns before using them
print("Missing value counts (base features):")
print(X_base[["off", "ledd", "time_since_intake_on", "time_since_intake_off"]].isnull().sum())
assert X_base["off"].isnull().any(), "off has no missing values — flag would be uninformative"
assert X_base["ledd"].isnull().any(), "ledd has no missing values — flag would be uninformative"

# Feature set B: base + two binary missing-value indicators
def add_miss_flags(df):
    """Add binary missingness flags for the two highest-importance sparse features."""
    df = df.copy()
    df["off_is_missing"] = df["off"].isnull().astype("int8")
    df["ledd_is_missing"] = df["ledd"].isnull().astype("int8")
    return df

X_flags = add_miss_flags(X_base)
print(f"\nX_base shape : {X_base.shape}")
print(f"X_flags shape: {X_flags.shape}  (added off_is_missing, ledd_is_missing)")

# %%
# Same deterministic folds as all previous experiments
cv_splits = list(GroupKFold(n_splits=5).split(X_base, y, groups=groups))


def run_cv(name, model, X, cv_splits, y):
    """OOF cross-validation; returns (rmse_mean, rmse_std, per_fold list)."""
    fold_rmses = []
    for i, (tr, val) in enumerate(cv_splits):
        clf = clone(model)
        clf.fit(X.iloc[tr], y.iloc[tr])
        preds = clf.predict(X.iloc[val])
        rmse = root_mean_squared_error(y.iloc[val], preds)
        fold_rmses.append(rmse)
    mean_rmse = np.mean(fold_rmses)
    std_rmse = np.std(fold_rmses)
    fold_str = "  ".join(f"{v:.4f}" for v in fold_rmses)
    print(f"{name:<30s}  RMSE={mean_rmse:.4f} +/-{std_rmse:.4f}  folds=[{fold_str}]")
    return mean_rmse, std_rmse, fold_rmses


# %%
# -- A. Baseline (exp 07 reference) -------------------------------------------
print("=" * 80)
print("RUNNING CV — 4 model variants, same GroupKFold(5) folds")
print("=" * 80)

base_model = tabular_pipeline("regressor")
rmse_A, std_A, folds_A = run_cv("A. baseline", base_model, X_base, cv_splits, y)

# %%
# -- B. Missing-value flags ---------------------------------------------------
# Same tabular_pipeline but on X_flags (2 extra binary columns).
# FunctionTransformer wraps the flag step so it fits cleanly inside a pipeline.
# tabular_pipeline handles the two new int8 columns as low-cardinality numerics.
flags_model = tabular_pipeline("regressor")
rmse_B, std_B, folds_B = run_cv("B. +miss_flags", flags_model, X_flags, cv_splits, y)

# %%
# -- C. HGBR max_iter=300, max_leaf_nodes=63 ----------------------------------
hgbr_300 = clone(base_model).set_params(
    histgradientboostingregressor__max_iter=300,
    histgradientboostingregressor__max_leaf_nodes=63,
    histgradientboostingregressor__random_state=0,
)
rmse_C, std_C, folds_C = run_cv("C. HGBR 300/63", hgbr_300, X_base, cv_splits, y)

# %%
# -- D. HGBR max_iter=500, max_leaf_nodes=127 ---------------------------------
hgbr_500 = clone(base_model).set_params(
    histgradientboostingregressor__max_iter=500,
    histgradientboostingregressor__max_leaf_nodes=127,
    histgradientboostingregressor__random_state=0,
)
rmse_D, std_D, folds_D = run_cv("D. HGBR 500/127", hgbr_500, X_base, cv_splits, y)

# %%
# -- E. B + C: flags AND HGBR 300/63 -----------------------------------------
hgbr_300_flags = clone(base_model).set_params(
    histgradientboostingregressor__max_iter=300,
    histgradientboostingregressor__max_leaf_nodes=63,
    histgradientboostingregressor__random_state=0,
)
rmse_E, std_E, folds_E = run_cv("E. +flags +HGBR300/63", hgbr_300_flags, X_flags, cv_splits, y)

# %%
# -- Summary table ------------------------------------------------------------
print()
print("=" * 80)
print("COMPARISON TABLE (reference RMSE = 7.5041)")
print("=" * 80)
ref_rmse = rmse_A  # recomputed here, not the cached value from exp 08
results = [
    ("A. baseline (exp 07)",     rmse_A, std_A),
    ("B. +miss_flags",           rmse_B, std_B),
    ("C. HGBR 300/63",           rmse_C, std_C),
    ("D. HGBR 500/127",          rmse_D, std_D),
    ("E. +flags +HGBR 300/63",   rmse_E, std_E),
]
print(f"{'Model':<30s}  {'RMSE':>8s}  {'std':>7s}  {'vs ref':>8s}")
print("-" * 60)
for name, rmse, std in results:
    delta = rmse - ref_rmse
    marker = "  <-- BEST" if rmse == min(r for _, r, _ in results) else ""
    print(f"{name:<30s}  {rmse:>8.4f}  {std:>7.4f}  {delta:>+8.4f}{marker}")

best_name, best_rmse, best_std = min(results, key=lambda x: x[1])
print()
print(f"Best model : {best_name}  RMSE={best_rmse:.4f}")
print(f"Gain vs ref: {best_rmse - ref_rmse:+.4f} RMSE")

# %%
# -- Recommendation -----------------------------------------------------------
print()
print("=" * 80)
print("RECOMMENDATION")
print("=" * 80)
print(f"""
The results above determine the next step:
- If D or E beats A by >0.05: use that config for the next submission.
- If gains are marginal (<0.05): the low-hanging fruit is exhausted for
  this feature set; move to PATH 1 from exp 08 (patient-level features).
- Combinations (B+D or B+E) can be tried next if individual gains compose.
""")

print("OK: targeted improvements complete.")
