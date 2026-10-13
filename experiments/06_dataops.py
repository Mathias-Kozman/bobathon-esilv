# %%
# Experiment 06: DataOps workflow with Skrub
#
# Builds the full data-to-prediction pipeline as a skrub DataOps graph,
# evaluates HistGradientBoostingRegressor with GroupKFold(5) by patient_id,
# and pushes the CrossValidationReport to Skore Hub.
#
# Choice of model: HGBR (random_state=0) is the best model found so far
# (RMSE 7.7520 in exp 04, -26.9% vs Ridge). No experiment 05 exists in
# this workspace. HGBR is reused here unchanged so the DataOps evaluation
# is directly comparable to exp 04.

import time

import pandas as pd
import skore
import skrub
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.model_selection import GroupKFold

from parkinson import PROJECT_ROOT
from parkinson.hub import load_skore_credentials
from skore import Project, login

DATA_DIR = PROJECT_ROOT / "data"

# %% Load data
X_train = pd.read_csv(DATA_DIR / "X_train.csv")
y_train = pd.read_csv(DATA_DIR / "y_train.csv")
X_test = pd.read_csv(DATA_DIR / "X_test.csv")

# Merge on Index to align features and target (inner join — no orphan rows)
visits = X_train.merge(y_train, on="Index")
assert len(visits) == len(X_train) == len(y_train), (
    f"Alignment mismatch after merge: {len(visits)} rows vs "
    f"X_train={len(X_train)}, y_train={len(y_train)}"
)

FEATURE_COLS = [
    "sexM", "age_at_diagnosis", "age", "ledd",
    "time_since_intake_on", "time_since_intake_off", "on", "off",
]

# %% Build the DataOps graph
# skrub.var wraps a concrete dataframe as a named pipeline variable.
# All downstream nodes are lazy DataOp expressions; nothing is executed yet.
data = skrub.var("visits", visits)

# Extract patient groups BEFORE mark_as_X so the splitter sees all rows.
# groups is itself a DataOp node and is evaluated together with X and y
# at the start of each CV loop (outside the train/val split).
groups = data["patient_id"]

# Select the 8 predictive features — Index and patient_id are excluded.
features = data[FEATURE_COLS]

# mark_as_X records the GroupKFold splitter and the groups DataOp.
# skrub will materialise groups alongside X and y, then pass them to
# GroupKFold.split(X, y, groups=groups) for every fold.
X = features.skb.mark_as_X(
    cv=GroupKFold(n_splits=5),
    split_kwargs={"groups": groups},
)
y = data["target"].skb.mark_as_y()

# Attach the estimator — HGBR handles NaN natively, no imputation needed.
pred = X.skb.apply(HistGradientBoostingRegressor(random_state=0), y=y)

# %% Cross-validation with the DataOps graph
# scoring="neg_root_mean_squared_error" so RMSE = -test_score.
# We also collect fit_time to compare with exp 04.
t0 = time.perf_counter()
cv_results = pred.skb.cross_validate(
    scoring="neg_root_mean_squared_error",
    return_train_score=False,
)
elapsed = time.perf_counter() - t0

rmse_folds = -cv_results["test_score"]
rmse_mean = rmse_folds.mean()
rmse_std = rmse_folds.std()

print(f"DataOps CV done in {elapsed:.1f}s")
print(f"\nHGBR (DataOps, GroupKFold 5 folds) — RMSE per fold:")
for i, v in enumerate(rmse_folds):
    print(f"  fold {i}: {v:.4f}")
print(f"\nRMSE mean : {rmse_mean:.4f}  std : {rmse_std:.4f}")
print(f"Reference  (exp 04, skore.evaluate) : 7.7520  +/- 0.1546")
print(f"Delta vs ref : {rmse_mean - 7.7520:+.4f}")

# %% Evaluate with skore to get a CrossValidationReport for the hub
# We replicate the same GroupKFold splits deterministically (same as exp 03/04).
X_df = visits[FEATURE_COLS]
y_series = visits["target"]
groups_series = visits["patient_id"]
cv_splits = list(GroupKFold(n_splits=5).split(X_df, y_series, groups=groups_series))

hgbr = HistGradientBoostingRegressor(random_state=0)
report = skore.evaluate(hgbr, X_df, y_series, splitter=cv_splits)
print("\nSkore CrossValidationReport RMSE:")
print(report.metrics.rmse())

# %% Push to hub
cfg = load_skore_credentials()
login(mode="hub")
project = Project(name="bobathon-esilv", mode="hub", workspace=cfg["workspace"])
project.put("06_dataops", report)
print("Report pushed: 06_dataops")

# %% Final fit on full training set + Kaggle CSV
learner = pred.skb.make_learner()
learner.fit({"visits": visits})

# Predict: pass the environment dict with the same variable name ("visits").
# X_test has all original columns (including patient_id) but no target column.
# skrub skips the y node during predict — only the feature path is evaluated.
preds = learner.predict({"visits": X_test})

assert len(preds) == len(X_test), (
    f"Dimension mismatch: {len(preds)} predictions for {len(X_test)} test rows"
)
assert not pd.isnull(preds).any(), "NaN detected in predictions"

submission = X_test[["Index"]].copy()
submission["target"] = preds

out = PROJECT_ROOT / "submissions"
out.mkdir(exist_ok=True)
submission.to_csv(out / "06_dataops.csv", index=False)

print(f"\nSubmission: {submission.shape[0]} rows, columns {list(submission.columns)}")
print(submission.head())
assert list(submission.columns) == ["Index", "target"], "Wrong columns"
assert submission["target"].isnull().sum() == 0, "NaN in predictions"
print("OK: all checks passed.")
