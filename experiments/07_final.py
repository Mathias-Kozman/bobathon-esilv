# %%
# Experiment 07: final model selection, full-train refit, and Kaggle submission.
#
# Model chosen: skrub tabular_pipeline("regressor")
#   = TableVectorizer + HistGradientBoostingRegressor
#
# Rationale: across all experiments evaluated with the same GroupKFold(5)
# by patient_id, skrub_tabular (exp 05) achieves the lowest CV RMSE (7.5263),
# beating HGBR on 8 numeric features alone (7.7520 in exp 04/06) by −2.9%.
# The gain comes from including all available columns (cohort, gene, rater_id)
# that carry cohort-level variance the numeric-only model cannot see.
#
# Summary of CV results (GroupKFold 5 folds, patient_id):
#   03 Ridge              RMSE 10.5965  ±0.267
#   04 HGBR numeric       RMSE  7.7520  ±0.155
#   05 skrub_tabular      RMSE  7.5263  ±0.138  <-- best
#   06 HGBR DataOps       RMSE  7.7520  ±0.155  (same as 04)

import pandas as pd
import numpy as np
import skore
from sklearn.base import clone
from sklearn.model_selection import GroupKFold
from skrub import tabular_pipeline

from parkinson import PROJECT_ROOT
from parkinson.hub import load_skore_credentials
from skore import Project, login

DATA_DIR = PROJECT_ROOT / "data"

# %% Load data
X_train = pd.read_csv(DATA_DIR / "X_train.csv")
y_train = pd.read_csv(DATA_DIR / "y_train.csv")
X_test = pd.read_csv(DATA_DIR / "X_test.csv")

# Merge and verify alignment
visits = X_train.merge(y_train, on="Index")
assert len(visits) == len(X_train) == len(y_train), (
    f"Alignment mismatch: merged={len(visits)}, "
    f"X_train={len(X_train)}, y_train={len(y_train)}"
)

y = visits["target"]

# Drop identifiers (Index, patient_id) and target — same as exp 05.
# No leakage: neither Index nor patient_id is a predictive feature.
X_full = visits.drop(columns=["Index", "patient_id", "target"])

print(f"Training set: {X_full.shape[0]} rows x {X_full.shape[1]} features")
print(f"Features: {list(X_full.columns)}")

# %% Confirm best model via CV (same folds as all previous experiments)
groups = visits["patient_id"]
cv_splits = list(GroupKFold(n_splits=5).split(X_full, y, groups=groups))

model = tabular_pipeline("regressor")
report = skore.evaluate(model, X_full, y, splitter=cv_splits)

per_fold_rmse = np.array([r.metrics.rmse() for r in report.reports_])
rmse_mean = per_fold_rmse.mean()
rmse_std = per_fold_rmse.std()

print(f"\nskrub_tabular CV RMSE per fold:")
for i, v in enumerate(per_fold_rmse):
    print(f"  fold {i}: {v:.4f}")
print(f"Mean : {rmse_mean:.4f}  std : {rmse_std:.4f}")
print(f"\nComparison:")
print(f"  03 Ridge numeric   : 10.5965")
print(f"  04/06 HGBR numeric :  7.7520")
print(f"  05/07 skrub_tabular:  {rmse_mean:.4f}  <-- best")

# %% Push evaluation report to Skore Hub
cfg = load_skore_credentials()
login(mode="hub")
project = Project(name="bobathon-esilv", mode="hub", workspace=cfg["workspace"])
project.put("07_final", report)
print("\nReport pushed: 07_final")

# %% Final refit on all training data
final_model = clone(model).fit(X_full, y)

# Align test columns to match training feature set
# (drop Index and patient_id; keep all other columns in the same order)
test_cols = [c for c in X_full.columns if c in X_test.columns]
X_test_aligned = X_test[test_cols]
assert list(X_test_aligned.columns) == list(X_full.columns), (
    f"Column mismatch:\n  test={list(X_test_aligned.columns)}\n"
    f"  train={list(X_full.columns)}"
)

# %% Predict and verify
preds = final_model.predict(X_test_aligned)

assert len(preds) == len(X_test), (
    f"Row count mismatch: {len(preds)} predictions vs {len(X_test)} test rows"
)
assert not pd.isnull(preds).any(), "NaN detected in predictions"
assert not np.isinf(preds).any(), "Infinite values detected in predictions"
assert (preds >= 0).all(), f"Unexpected negative predictions: min={preds.min():.2f}"

# %% Build submission CSV — same order as X_test
submission = X_test[["Index"]].copy()
submission["target"] = preds

assert list(submission.columns) == ["Index", "target"], (
    f"Wrong columns: {list(submission.columns)}"
)
assert submission["Index"].tolist() == X_test["Index"].tolist(), (
    "Index order differs from X_test"
)
assert submission["target"].isnull().sum() == 0, "NaN in final predictions"

# Verify against sample_submission if available
sample_path = DATA_DIR / "sample_submission.csv"
if sample_path.exists():
    sample = pd.read_csv(sample_path)
    assert len(submission) == len(sample), (
        f"Row count {len(submission)} != sample_submission {len(sample)}"
    )
    assert submission["Index"].tolist() == sample["Index"].tolist(), (
        "Index order differs from sample_submission"
    )
    print(f"\nIndex order matches sample_submission: OK")

out = PROJECT_ROOT / "submissions"
out.mkdir(exist_ok=True)
submission.to_csv(out / "07_final.csv", index=False)

print(f"\nSubmission written: submissions/07_final.csv")
print(f"  Rows   : {submission.shape[0]}")
print(f"  Columns: {list(submission.columns)}")
print(f"  NaN    : {submission['target'].isnull().sum()}")
print(f"  Inf    : {np.isinf(submission['target'].values).sum()}")
print(f"  Min    : {submission['target'].min():.4f}")
print(f"  Max    : {submission['target'].max():.4f}")
print(submission.head())
print("\nOK: all checks passed.")
