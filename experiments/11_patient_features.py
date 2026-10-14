# %%
# ---------------------------------------------------------------------------
# Rôle     : features patient (agrégats X-only, sans fuite de target)
# Entrées  : data/X_train.csv, data/y_train.csv, data/X_test.csv
# Sorties  : hub + submissions/ (si amélioration confirmée)
# Exécuter : .venv/Scripts/python.exe experiments/11_patient_features.py
# ---------------------------------------------------------------------------
#
# Experiment 11: patient-level feature aggregation — corrected approach.
#
# Problem with original: PatientAggregator was fitted on train fold only, so
# validation patients (unseen by GroupKFold) received cohort-level fallbacks.
# This made the CV test meaningless — it measured fallback quality, not
# patient aggregates.
#
# Corrected approach:
#   - Aggregates (mean, median, count, std, slope, first/last) are computed
#     from ALL X visits of each patient BEFORE cross-validation, using only
#     X columns — target is never touched.
#   - These pre-computed aggregates are joined back to every row by patient_id.
#   - The CV then evaluates a standard tabular_pipeline on
#     (base features + patient aggregates), which is fair because in the
#     real challenge prediction is batch: all X visits are available at
#     inference time, so using X-only aggregates from the full patient history
#     is permitted.
#   - patient_id and Index are excluded from the final predictor matrix.
#
# Two variants, same GroupKFold(5) folds:
#   A. baseline : tabular_pipeline on base X columns only
#   B. +pat_agg : A + X-only patient aggregates (mean/median/count/std/slope)
#
# Submission only if B beats A by >= 0.01 RMSE.
#
# NOTE on rule compliance: this approach uses X features from validation visits
# to build patient aggregates. This is valid only if the challenge allows batch
# inference (all visits known at prediction time). If the challenge required
# row-by-row online prediction, this would be a data leak and must not be used.

import warnings

import numpy as np
import pandas as pd
import skore
from scipy import stats as sp_stats
from sklearn.metrics import root_mean_squared_error
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import clone
from skore import Project, login
from skrub import tabular_pipeline

from parkinson import PROJECT_ROOT
from parkinson.hub import load_skore_credentials

warnings.filterwarnings("ignore", category=FutureWarning)

DATA_DIR = PROJECT_ROOT / "data"

# ──────────────────────────────────────────────────────────────────────────────
# 1. Load data
# ──────────────────────────────────────────────────────────────────────────────
X_train = pd.read_csv(DATA_DIR / "X_train.csv")
y_train = pd.read_csv(DATA_DIR / "y_train.csv")
X_test  = pd.read_csv(DATA_DIR / "X_test.csv")

assert "target" not in X_train.columns, "target must not be in X_train"
assert "target" not in X_test.columns,  "target must not be in X_test"

visits = X_train.merge(y_train, on="Index")
assert len(visits) == len(X_train) == len(y_train), "Merge mismatch"

y      = visits["target"].reset_index(drop=True)
groups = visits["patient_id"].reset_index(drop=True)

print(f"Train: {len(visits)} rows, {groups.nunique()} patients")
print(f"Test : {len(X_test)} rows, {X_test['patient_id'].nunique()} patients")
print(f"Overlap train/test patients: "
      f"{len(set(X_train['patient_id']) & set(X_test['patient_id']))}")

FEATURE_COLS_BASE = [c for c in X_train.columns
                     if c not in ("Index", "patient_id")]
AGG_COLS = ["off", "on", "ledd"]
MIN_OBS_SLOPE = 3


# ──────────────────────────────────────────────────────────────────────────────
# 2. Patient aggregate computation (X-only, no target)
# ──────────────────────────────────────────────────────────────────────────────

def compute_patient_aggregates(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute per-patient aggregates from X columns only.

    Parameters
    ----------
    df : DataFrame containing patient_id, age, Index, and AGG_COLS.

    Returns
    -------
    DataFrame indexed by patient_id with one row per patient.
    Target is never passed to or used by this function.
    """
    assert "target" not in df.columns, "target must not be in df"

    records = []
    for pid, grp in df.groupby("patient_id"):
        row = {"patient_id": pid}
        for col in AGG_COLS:
            vals = grp[col].dropna()
            n = len(vals)
            row[f"{col}_mean_pat"]   = vals.mean()   if n > 0 else np.nan
            row[f"{col}_median_pat"] = vals.median() if n > 0 else np.nan
            row[f"{col}_count_pat"]  = n
            row[f"{col}_std_pat"]    = vals.std()    if n >= 2 else np.nan

        for col in ["off", "on"]:
            d = grp[["age", col]].dropna().sort_values("age")
            if len(d) >= MIN_OBS_SLOPE:
                slope, *_ = sp_stats.linregress(d["age"], d[col])
                row[f"{col}_slope_pat"] = slope
            else:
                row[f"{col}_slope_pat"] = np.nan
            d2 = grp.sort_values("Index")[col].dropna()
            row[f"{col}_first_pat"] = d2.iloc[0]  if len(d2) > 0 else np.nan
            row[f"{col}_last_pat"]  = d2.iloc[-1] if len(d2) > 0 else np.nan

        records.append(row)

    return pd.DataFrame(records).set_index("patient_id")


# Pre-compute from ALL training visits (X only) — every patient will be known
pat_agg_train = compute_patient_aggregates(X_train)

# Pre-compute from ALL test visits (X only) — used for final submission
pat_agg_test  = compute_patient_aggregates(X_test)

AGG_FEATURE_COLS = list(pat_agg_train.columns)
print(f"\nPatient aggregate columns ({len(AGG_FEATURE_COLS)}): {AGG_FEATURE_COLS}")


def join_patient_aggregates(X: pd.DataFrame,
                             pat_agg: pd.DataFrame) -> pd.DataFrame:
    """
    Join pre-computed patient aggregates into X by patient_id.
    Returns a DataFrame with patient_id and Index dropped (not predictors).
    Row order is preserved. No target column is ever present.
    """
    assert "target" not in X.columns, "target must not be in X"
    result = X.join(pat_agg, on="patient_id", how="left")
    # Verify row order preserved
    assert len(result) == len(X), "Row count changed after join"
    assert (result.index == X.index).all(), "Index changed after join"
    result = result.drop(columns=["patient_id", "Index"], errors="ignore")
    return result


# Build the two feature matrices (index preserved from visits)
visits_X = visits[list(X_train.columns)].copy()  # X columns only, no target

X_base = visits_X[FEATURE_COLS_BASE].copy()
X_agg  = join_patient_aggregates(visits_X, pat_agg_train)

print(f"\nX_base shape : {X_base.shape}")
print(f"X_agg  shape : {X_agg.shape}")
assert "target" not in X_base.columns, "target leaked into X_base"
assert "target" not in X_agg.columns,  "target leaked into X_agg"

# ──────────────────────────────────────────────────────────────────────────────
# 3. CV evaluation helper
# ──────────────────────────────────────────────────────────────────────────────
cv_splits = list(GroupKFold(n_splits=5).split(X_base, y, groups=groups))


def run_cv(name: str, pipeline, X: pd.DataFrame, y: pd.Series) -> tuple:
    """
    Run GroupKFold CV and return (mean_rmse, std_rmse, per_fold_rmses).
    Prints per-fold RMSE, mean, and std.
    """
    fold_rmses = []
    for i, (tr, val) in enumerate(cv_splits):
        clf = clone(pipeline)
        clf.fit(X.iloc[tr], y.iloc[tr])
        preds = clf.predict(X.iloc[val])
        rmse  = root_mean_squared_error(y.iloc[val], preds)
        fold_rmses.append(rmse)
        print(f"  {name}  fold {i+1}: RMSE={rmse:.4f}")

    mean_rmse = float(np.mean(fold_rmses))
    std_rmse  = float(np.std(fold_rmses))
    print(f"  {name}  => mean={mean_rmse:.4f}  std={std_rmse:.4f}\n")
    return mean_rmse, std_rmse, fold_rmses


# ──────────────────────────────────────────────────────────────────────────────
# 4. Run the two variants
# ──────────────────────────────────────────────────────────────────────────────
print()
print("=" * 80)
print("RUNNING CV — 2 variants, same GroupKFold(5) folds")
print("=" * 80)
print()

pipe_A = tabular_pipeline("regressor")
pipe_B = tabular_pipeline("regressor")

rmse_A, std_A, folds_A = run_cv("A. baseline", pipe_A, X_base, y)
rmse_B, std_B, folds_B = run_cv("B. +pat_agg", pipe_B, X_agg,  y)

# ──────────────────────────────────────────────────────────────────────────────
# 5. Comparison table
# ──────────────────────────────────────────────────────────────────────────────
print()
print("=" * 80)
print("COMPARISON TABLE")
print("=" * 80)
results = [
    ("A. baseline", rmse_A, std_A),
    ("B. +pat_agg", rmse_B, std_B),
]
print(f"{'Model':<25s}  {'RMSE':>8s}  {'std':>7s}  {'vs A':>8s}")
print("-" * 55)
for name, rmse, std in results:
    delta  = rmse - rmse_A
    marker = "  <-- BEST" if rmse == min(r for _, r, _ in results) else ""
    print(f"{name:<25s}  {rmse:>8.4f}  {std:>7.4f}  {delta:>+8.4f}{marker}")

gain = rmse_A - rmse_B
print()
print(f"Gain (A - B): {gain:+.4f}  "
      f"({'B better' if gain > 0 else 'A better or equal'})")

# ──────────────────────────────────────────────────────────────────────────────
# 6. Push CV report to Skore Hub (variant B — the submission model)
# ──────────────────────────────────────────────────────────────────────────────
# skore.evaluate receives the pre-materialised cv_splits list. sklearn's
# check_cv wraps it in _CVIterableWrapper whose .split() ignores X and y and
# replays the stored tuples verbatim — the GroupKFold(5) patient grouping is
# preserved exactly. pipe_B is unfitted; skore clones it per fold internally,
# identical to clone(pipe_B) in run_cv above.
hub_report = skore.evaluate(pipe_B, X_agg, y, splitter=cv_splits)

cfg = load_skore_credentials()
login(mode="hub")
project = Project(name="bobathon-esilv", mode="hub", workspace=cfg["workspace"])
project.put("11_patient_features", hub_report)
print("\nReport pushed: 11_patient_features")

# ──────────────────────────────────────────────────────────────────────────────
# 7. Submission — only if B beats A by at least 0.01 RMSE
# ──────────────────────────────────────────────────────────────────────────────
IMPROVEMENT_THRESHOLD = 0.01

if gain >= IMPROVEMENT_THRESHOLD:
    print()
    print(f"Gain >= {IMPROVEMENT_THRESHOLD:.2f}: generating submission...")

    # Refit B on full training set using pre-computed aggregates
    X_train_agg = join_patient_aggregates(X_train, pat_agg_train)
    X_test_agg  = join_patient_aggregates(X_test,  pat_agg_test)

    assert "target" not in X_train_agg.columns, "target in X_train_agg"
    assert "target" not in X_test_agg.columns,  "target in X_test_agg"

    final_pipe = tabular_pipeline("regressor")
    final_pipe.fit(X_train_agg, y)
    preds = final_pipe.predict(X_test_agg)

    assert len(preds) == len(X_test), f"Row mismatch: {len(preds)} vs {len(X_test)}"
    assert not np.isnan(preds).any(),  "NaN in predictions"
    assert not np.isinf(preds).any(),  "Inf in predictions"

    submission = X_test[["Index"]].copy()
    submission["target"] = preds
    assert list(submission.columns) == ["Index", "target"]
    assert submission["Index"].tolist() == X_test["Index"].tolist()

    out = PROJECT_ROOT / "submissions"
    out.mkdir(exist_ok=True)
    submission.to_csv(out / "11_patient_features.csv", index=False)
    print(f"Submission written: submissions/11_patient_features.csv  "
          f"({len(submission)} rows)")
else:
    print()
    print(f"Gain < {IMPROVEMENT_THRESHOLD:.2f} RMSE — no submission created.")

print()
print("OK: patient feature experiment complete.")
