# %%
# Experiment 12: interaction features between strong original features and
# patient-level aggregates from experiment 11.
#
# Strategy:
#   - Start from the experiment-11 feature set (base X cols + patient aggregates).
#   - Add carefully selected interaction terms: each strong original feature
#     (`off`, `on`, `ledd`, `age`, `age_at_diagnosis`) is multiplied by its own
#     patient-level aggregate (mean, slope where available) to capture how far
#     a visit deviates from a patient's personal baseline in absolute scale.
#   - These are ratio/deviation features that encode patient-specific context.
#   - No target is ever used to construct features.
#   - Same GroupKFold(5) splits as experiment 11 (splits re-derived identically).
#   - patient_id and Index are excluded from the predictor matrix.
#
# Comparison: experiment-11 feature set (B, RMSE ≈ 5.2019) vs +interactions (C).
# Submission only if C beats B by >= 0.01 RMSE.

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
# 1. Load data (identical to experiment 11)
# ──────────────────────────────────────────────────────────────────────────────
X_train = pd.read_csv(DATA_DIR / "X_train.csv")
y_train = pd.read_csv(DATA_DIR / "y_train.csv")
X_test  = pd.read_csv(DATA_DIR / "X_test.csv")

assert "target" not in X_train.columns
assert "target" not in X_test.columns

visits = X_train.merge(y_train, on="Index")
assert len(visits) == len(X_train) == len(y_train)

y      = visits["target"].reset_index(drop=True)
groups = visits["patient_id"].reset_index(drop=True)

print(f"Train: {len(visits)} rows, {groups.nunique()} patients")

FEATURE_COLS_BASE = [c for c in X_train.columns
                     if c not in ("Index", "patient_id")]
AGG_COLS = ["off", "on", "ledd"]
MIN_OBS_SLOPE = 3

# ──────────────────────────────────────────────────────────────────────────────
# 2. Patient aggregates (copied verbatim from experiment 11)
# ──────────────────────────────────────────────────────────────────────────────

def compute_patient_aggregates(df: pd.DataFrame) -> pd.DataFrame:
    """Per-patient X-only aggregates (mean, median, count, std, slope, first/last)."""
    assert "target" not in df.columns
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


pat_agg_train = compute_patient_aggregates(X_train)
pat_agg_test  = compute_patient_aggregates(X_test)

AGG_FEATURE_COLS = list(pat_agg_train.columns)
print(f"Patient aggregate columns ({len(AGG_FEATURE_COLS)}): {AGG_FEATURE_COLS}")


def join_patient_aggregates(X: pd.DataFrame,
                             pat_agg: pd.DataFrame) -> pd.DataFrame:
    """Join pre-computed aggregates and drop patient_id / Index."""
    assert "target" not in X.columns
    result = X.join(pat_agg, on="patient_id", how="left")
    assert len(result) == len(X)
    assert (result.index == X.index).all()
    result = result.drop(columns=["patient_id", "Index"], errors="ignore")
    return result


# ──────────────────────────────────────────────────────────────────────────────
# 3. Interaction feature construction
# ──────────────────────────────────────────────────────────────────────────────

def add_interaction_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add interactions between strong original features and patient aggregates.

    For each strong feature f and its patient mean m_f:
      - visit_minus_pat_mean : f - m_f  (absolute deviation from patient baseline)
      - visit_div_pat_mean   : f / m_f  (relative deviation; NaN-safe)
      - visit_x_pat_mean     : f * m_f  (scale interaction)

    Additionally:
      - age × off_slope_pat  : rate of off-worsening weighted by current age
      - age × on_slope_pat   : rate of on-change weighted by current age
      - age_at_diagnosis × off_mean_pat : onset severity proxy
      - age_at_diagnosis × ledd_mean_pat: cumulative medication burden proxy

    No target is ever used. patient_id and Index are assumed already dropped.
    """
    assert "target" not in df.columns
    out = df.copy()

    pairs = [
        ("off",  "off_mean_pat"),
        ("on",   "on_mean_pat"),
        ("ledd", "ledd_mean_pat"),
        ("age",  "off_mean_pat"),     # age vs patient off level
        ("age_at_diagnosis", "off_mean_pat"),
        ("age_at_diagnosis", "ledd_mean_pat"),
    ]

    for feat, agg in pairs:
        if feat not in out.columns or agg not in out.columns:
            continue
        f = out[feat]
        m = out[agg]
        stub = f"{feat}_x_{agg}"
        out[f"{stub}_diff"] = f - m
        denom = m.where(m.abs() > 1e-8)          # avoid divide-by-zero
        out[f"{stub}_ratio"] = f / denom
        out[f"{stub}_prod"]  = f * m

    # Slope-based interactions (available for off and on only)
    for col in ["off", "on"]:
        slope_col = f"{col}_slope_pat"
        if col in out.columns and slope_col in out.columns:
            out[f"age_x_{slope_col}"] = out["age"] * out[slope_col]

    assert "target" not in out.columns
    return out


# Build the experiment-11 feature matrix (B)
visits_X = visits[list(X_train.columns)].copy()  # X columns only
X_exp11  = join_patient_aggregates(visits_X, pat_agg_train)

# Build the experiment-12 feature matrix (C) by adding interactions
X_exp12  = add_interaction_features(X_exp11)

n_new = X_exp12.shape[1] - X_exp11.shape[1]
print(f"\nX_exp11 shape : {X_exp11.shape}")
print(f"X_exp12 shape : {X_exp12.shape}  (+{n_new} interaction features)")
print(f"New columns   : {[c for c in X_exp12.columns if c not in X_exp11.columns]}")

assert "target" not in X_exp11.columns
assert "target" not in X_exp12.columns
assert "patient_id" not in X_exp12.columns
assert "Index" not in X_exp12.columns

# ──────────────────────────────────────────────────────────────────────────────
# 4. Same GroupKFold(5) splits as experiment 11
# ──────────────────────────────────────────────────────────────────────────────
cv_splits = list(GroupKFold(n_splits=5).split(X_exp11, y, groups=groups))

pipe_B = tabular_pipeline("regressor")
pipe_C = tabular_pipeline("regressor")

# ──────────────────────────────────────────────────────────────────────────────
# 5. CV evaluation — identical folds for B and C
# ──────────────────────────────────────────────────────────────────────────────
print()
print("=" * 80)
print("RUNNING CV — experiment 11 (B) vs +interactions (C), same GroupKFold(5)")
print("=" * 80)
print()

EXP11_RMSE = 5.2019  # reported mean CV RMSE from experiment 11

fold_rmses_B = []
fold_rmses_C = []

for i, (tr, val) in enumerate(cv_splits):
    clf_B = clone(pipe_B)
    clf_B.fit(X_exp11.iloc[tr], y.iloc[tr])
    rmse_B_fold = root_mean_squared_error(y.iloc[val], clf_B.predict(X_exp11.iloc[val]))
    fold_rmses_B.append(rmse_B_fold)

    clf_C = clone(pipe_C)
    clf_C.fit(X_exp12.iloc[tr], y.iloc[tr])
    rmse_C_fold = root_mean_squared_error(y.iloc[val], clf_C.predict(X_exp12.iloc[val]))
    fold_rmses_C.append(rmse_C_fold)

    print(f"  fold {i+1}:  B={rmse_B_fold:.4f}  C={rmse_C_fold:.4f}"
          f"  diff={rmse_C_fold - rmse_B_fold:+.4f}")

mean_B = float(np.mean(fold_rmses_B))
std_B  = float(np.std(fold_rmses_B))
mean_C = float(np.mean(fold_rmses_C))
std_C  = float(np.std(fold_rmses_C))
gain   = mean_B - mean_C  # positive = C is better

# ──────────────────────────────────────────────────────────────────────────────
# 6. Comparison table
# ──────────────────────────────────────────────────────────────────────────────
print()
print("=" * 80)
print("COMPARISON TABLE")
print("=" * 80)
print(f"{'Model':<30s}  {'RMSE':>8s}  {'std':>7s}  {'vs B':>8s}")
print("-" * 60)
for label, rmse, std in [
    ("B. exp11 (pat_agg)",  mean_B, std_B),
    ("C. exp12 (+interact)", mean_C, std_C),
]:
    delta  = rmse - mean_B
    marker = "  <-- BEST" if rmse == min(mean_B, mean_C) else ""
    print(f"{label:<30s}  {rmse:>8.4f}  {std:>7.4f}  {delta:>+8.4f}{marker}")

print()
print(f"Reported exp11 RMSE : {EXP11_RMSE:.4f}")
print(f"This-run  exp11 RMSE: {mean_B:.4f}")
print(f"Gain (B - C)        : {gain:+.4f}  "
      f"({'C better' if gain > 0 else 'B better or equal'})")

# ──────────────────────────────────────────────────────────────────────────────
# 6. Push CV report to Skore Hub (variant C — the submission model)
# ──────────────────────────────────────────────────────────────────────────────
# skore.evaluate receives the pre-materialised cv_splits list. sklearn's
# check_cv wraps it in _CVIterableWrapper whose .split() ignores X and y and
# replays the stored tuples verbatim — the GroupKFold(5) patient grouping is
# preserved exactly. pipe_C is unfitted; skore clones it per fold internally,
# identical to clone(pipe_C) in the loop above.
hub_report = skore.evaluate(pipe_C, X_exp12, y, splitter=cv_splits)

cfg = load_skore_credentials()
login(mode="hub")
project = Project(name="bobathon-esilv", mode="hub", workspace=cfg["workspace"])
project.put("12_interaction_features", hub_report)
print("\nReport pushed: 12_interaction_features")

# ──────────────────────────────────────────────────────────────────────────────
# 7. Submission — only if C beats B by >= 0.01 RMSE
# ──────────────────────────────────────────────────────────────────────────────
IMPROVEMENT_THRESHOLD = 0.01

if gain >= IMPROVEMENT_THRESHOLD:
    print()
    print(f"Gain >= {IMPROVEMENT_THRESHOLD:.2f}: generating submission for experiment 12...")

    X_train_12 = add_interaction_features(
        join_patient_aggregates(X_train, pat_agg_train)
    )
    X_test_12  = add_interaction_features(
        join_patient_aggregates(X_test, pat_agg_test)
    )

    assert "target" not in X_train_12.columns
    assert "target" not in X_test_12.columns
    assert len(X_test_12) == len(X_test)

    final_pipe = tabular_pipeline("regressor")
    final_pipe.fit(X_train_12, y)
    preds = final_pipe.predict(X_test_12)

    assert len(preds) == len(X_test), f"Row mismatch: {len(preds)} vs {len(X_test)}"
    assert not np.isnan(preds).any(), "NaN in predictions"
    assert not np.isinf(preds).any(), "Inf in predictions"

    submission = X_test[["Index"]].copy()
    submission["target"] = preds
    assert list(submission.columns) == ["Index", "target"]
    assert submission["Index"].tolist() == X_test["Index"].tolist()

    out = PROJECT_ROOT / "submissions"
    out.mkdir(exist_ok=True)
    submission.to_csv(out / "12_interaction_features.csv", index=False)
    print(f"Submission written: submissions/12_interaction_features.csv  "
          f"({len(submission)} rows)")
else:
    print()
    print(f"Gain < {IMPROVEMENT_THRESHOLD:.2f} RMSE — no submission created.")

print()
print("OK: interaction feature experiment complete.")
