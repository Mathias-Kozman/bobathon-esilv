# %%
# ---------------------------------------------------------------------------
# Rôle     : features d'historique patient séquentielles, sans fuite (variante rapide)
# Entrées  : data/X_train.csv, data/y_train.csv, data/X_test.csv
# Sorties  : hub + submissions/ (si amélioration sur exp 13)
# Exécuter : .venv/Scripts/python.exe experiments/15_advanced_improvements_fast.py
# ---------------------------------------------------------------------------
#
# Experiment 15: patient-specific residual random effect (fast variant).
#
# Single improvement over exp13: sequential within-patient history features
# that proxy the patient random intercept without leakage.
#
# The key insight from exp13 ICC analysis:
#   ICC(residuals, patient) = 0.45 → nearly half the remaining error is still
#   explainable by patient identity. Exp13 could not use MixedLM RE during CV
#   because validation patients are unseen in the train fold. The solution:
#   for every visit k, use only the EARLIER visits of the SAME patient (visits
#   1..k-1, sorted by Index which is monotone with age) to compute a running
#   mean of `off` and `target` as a patient-baseline proxy. This is leak-safe
#   because it uses only X-columns (running off mean) for CV and inference.
#
# Implementation:
#   - Running patient aggregates are computed with vectorised groupby + shift
#     (no Python loops) so it runs in ~0.1s for the full dataset.
#   - Pre-computed once before any CV, then sliced per fold.
#   - Builds directly on top of exp13's build_features, adding 5 new columns.
#
# CV: GroupKFold(5), fixed LightGBM params (no Optuna, saves ~2 min).
# Hub: skore.evaluate pushed to Mathias Kozman / bobathon-esilv.
# Submission: final model fit on all training data.
# Runtime target: < 2 minutes.

import warnings

import numpy as np
import pandas as pd
import skore
from sklearn.metrics import root_mean_squared_error
from sklearn.model_selection import GroupKFold
from skore import Project, login

import lightgbm as lgb

from parkinson import PROJECT_ROOT
from parkinson.hub import load_skore_credentials

warnings.filterwarnings("ignore")

DATA_DIR = PROJECT_ROOT / "data"

# ══════════════════════════════════════════════════════════════════════════════
# 1. Load data
# ══════════════════════════════════════════════════════════════════════════════
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
print(f"Test : {len(X_test)} rows, {X_test['patient_id'].nunique()} patients")

# ══════════════════════════════════════════════════════════════════════════════
# 2. Vectorised patient history features (pre-computed once, O(n log n))
#
# For each visit k of patient p, sorted by Index (= age order):
#   off_hist_mean  : mean of off at visits 1..k-1  (patient off baseline)
#   off_hist_count : number of prior off observations
#   off_hist_std   : std of off at visits 1..k-1
#   ledd_hist_mean : mean of ledd at visits 1..k-1
#   hist_n_visits  : number of prior visits (= k-1)
#
# First visit of every patient always gets NaN (no prior history).
# LightGBM routes NaN through its missing-value split automatically.
# No target is used — safe to compute on the full dataset before CV.
# ══════════════════════════════════════════════════════════════════════════════

def compute_patient_history(df: pd.DataFrame) -> pd.DataFrame:
    """
    Vectorised within-patient running history using groupby + shift.

    Sorted by Index (monotone with age), then for each column:
      cumulative sum and count up to k-1 via shift(1) of the expanding sum.

    Returns a DataFrame with the same index as df.
    """
    assert "target" not in df.columns

    # Sort by patient then visit order; remember original index to restore later
    s = df[["patient_id", "Index", "off", "ledd"]].sort_values(
        ["patient_id", "Index"]
    ).copy()

    # -- off running stats ---------------------------------------------------
    # expanding().sum/count on the sorted frame, then shift(1) gives 1..k-1
    s["_off_valid"] = s["off"].notna().astype(float)

    off_cumsum   = s.groupby("patient_id")["off"].transform(
        lambda x: x.expanding().sum().shift(1))
    off_cumcount = s.groupby("patient_id")["_off_valid"].transform(
        lambda x: x.expanding().sum().shift(1))
    # mean = sum / count, NaN when count == 0
    s["off_hist_mean"]  = (off_cumsum / off_cumcount.replace(0, np.nan))
    s["off_hist_count"] = off_cumcount

    # Running std via sum-of-squares: std = sqrt(E[x²] - E[x]²)
    off_cumss = s.groupby("patient_id")["off"].transform(
        lambda x: (x ** 2).expanding().sum().shift(1))
    # std is defined only when count >= 2
    var = (off_cumss / off_cumcount.replace(0, np.nan)) - s["off_hist_mean"] ** 2
    s["off_hist_std"] = np.where(off_cumcount >= 2, np.sqrt(var.clip(lower=0)), np.nan)

    # -- ledd running mean ---------------------------------------------------
    s["_ledd_valid"] = s["ledd"].notna().astype(float)
    ledd_cumsum   = s.groupby("patient_id")["ledd"].transform(
        lambda x: x.expanding().sum().shift(1))
    ledd_cumcount = s.groupby("patient_id")["_ledd_valid"].transform(
        lambda x: x.expanding().sum().shift(1))
    s["ledd_hist_mean"] = ledd_cumsum / ledd_cumcount.replace(0, np.nan)

    # -- visit count (k-1) ---------------------------------------------------
    s["hist_n_visits"] = s.groupby("patient_id").cumcount()  # 0-based rank = k-1 after sort

    # Restore original row order and return only the new columns
    hist_cols = ["off_hist_mean", "off_hist_count", "off_hist_std",
                 "ledd_hist_mean", "hist_n_visits"]
    return s[hist_cols].loc[df.index]


# Pre-compute once for train and test
visits_x       = visits[list(X_train.columns)].copy()
hist_train     = compute_patient_history(visits_x)
hist_test      = compute_patient_history(X_test)

print(f"History features computed: {list(hist_train.columns)}")
print(f"  Rows with prior off obs: {hist_train['off_hist_count'].gt(0).sum()} / {len(hist_train)}")

# ══════════════════════════════════════════════════════════════════════════════
# 3. Feature matrix builder (exp13 feature set + history columns)
# ══════════════════════════════════════════════════════════════════════════════

AGG_COLS_NUM  = ["off", "on", "ledd", "age", "age_at_diagnosis",
                 "time_since_intake_on", "time_since_intake_off"]
NEIGHBOR_COLS = ["off", "on", "ledd", "time_since_intake_off"]


def _vectorised_slope(pid_arr, x_arr, y_arr):
    mask = ~(np.isnan(x_arr) | np.isnan(y_arr))
    if mask.sum() == 0:
        return pd.Series(dtype=float)
    pids, x, yv = pid_arr[mask], x_arr[mask], y_arr[mask]
    agg = pd.DataFrame(
        {"p": pids, "x": x, "y": yv, "xy": x * yv, "x2": x * x}
    ).groupby("p", sort=False).agg(
        n=("x", "count"), sx=("x", "sum"), sy=("y", "sum"),
        sxy=("xy", "sum"), sx2=("x2", "sum"),
    )
    num = agg["n"] * agg["sxy"] - agg["sx"] * agg["sy"]
    den = agg["n"] * agg["sx2"] - agg["sx"] ** 2
    return (num / den.where(den.abs() > 1e-10)).where(agg["n"] >= 3)


def _compute_pat_aggregates(df: pd.DataFrame) -> pd.DataFrame:
    assert "target" not in df.columns
    gb = df.groupby("patient_id", sort=False)
    agg_funcs: dict = {}
    for col in AGG_COLS_NUM:
        if col not in df.columns:
            continue
        agg_funcs[f"{col}_mean_pat"]   = (col, "mean")
        agg_funcs[f"{col}_median_pat"] = (col, "median")
        agg_funcs[f"{col}_std_pat"]    = (col, "std")
        agg_funcs[f"{col}_min_pat"]    = (col, "min")
        agg_funcs[f"{col}_max_pat"]    = (col, "max")
        agg_funcs[f"{col}_count_pat"]  = (col, "count")
    pat_df = gb.agg(**agg_funcs)
    pat_df = pat_df.join(
        gb["age"].agg(n_visits_pat="count", age_first_pat="min", age_last_pat="max"),
        how="left")
    pat_df["followup_dur_pat"] = pat_df["age_last_pat"] - pat_df["age_first_pat"]
    for col in ["off", "on"]:
        if col not in df.columns:
            continue
        notna = df.sort_values("Index")[["patient_id", col]].dropna(subset=[col])
        pat_df[f"{col}_first_pat"] = notna.groupby("patient_id")[col].first()
        pat_df[f"{col}_last_pat"]  = notna.groupby("patient_id")[col].last()
    pid_arr = df["patient_id"].values
    age_arr = df["age"].values.astype(float)
    for col in AGG_COLS_NUM:
        if col not in df.columns:
            continue
        col_arr = pd.to_numeric(df[col], errors="coerce").values
        pat_df[f"{col}_slope_pat"] = _vectorised_slope(pid_arr, age_arr, col_arr)
    return pat_df


def _rater_oof_bias(df_tr: pd.DataFrame, y_tr: np.ndarray) -> dict:
    off_vals  = df_tr["off"].values
    raters    = df_tr["rater_id"].values
    off_fill  = np.where(np.isnan(off_vals), float(np.nanmedian(off_vals)), off_vals)
    residuals = y_tr - off_fill
    return {r: float(np.nanmean(residuals[raters == r])) for r in np.unique(raters)}


def _lgbm_encode(X: pd.DataFrame, cat_cols: list) -> pd.DataFrame:
    X = X.copy()
    for c in cat_cols:
        X[c] = X[c].astype("category")
    for c in X.select_dtypes("int8").columns:
        X[c] = X[c].astype(float)
    return X


def build_features(df_raw: pd.DataFrame,
                   rater_bias: dict,
                   pat_agg: pd.DataFrame,
                   hist_df: pd.DataFrame) -> pd.DataFrame:
    """
    Exp13 feature set (blocks B, I, E, D, C, A, F, H) plus the new
    patient-history block (Block S: off_hist_mean/count/std, ledd_hist_mean,
    hist_n_visits). hist_df must be pre-sliced to the same rows as df_raw.
    """
    assert "target" not in df_raw.columns
    orig_index = df_raw.index.copy()
    feat = df_raw.copy()

    # Block B: patient aggregates
    feat = feat.join(pat_agg, on="patient_id", how="left")

    # Block I: derived features
    feat["time_since_diagnosis"] = feat["age"] - feat["age_at_diagnosis"]
    mask_both = feat["off"].notna() & feat["on"].notna()
    feat["on_minus_off"] = np.where(mask_both, feat["on"] - feat["off"], np.nan)
    feat["on_div_off"]   = np.where(
        mask_both & (feat["off"].abs() > 1e-8), feat["on"] / feat["off"], np.nan)

    # Block E: missingness flags
    for col in ["off", "on", "ledd", "gene",
                "time_since_intake_on", "time_since_intake_off"]:
        if col in feat.columns:
            feat[f"{col}_na"] = feat[col].isna().astype(np.int8)

    # Block D: visit position
    feat_s = feat.sort_values(["patient_id", "age"])
    feat["visit_rank"] = feat_s.groupby("patient_id").cumcount().loc[orig_index].values
    feat["time_since_first_visit"] = feat["age"] - feat["age_first_pat"]
    feat["time_to_last_visit"]     = feat["age_last_pat"] - feat["age"]
    feat["rel_visit_pos"] = (
        feat["visit_rank"] / (feat["n_visits_pat"] - 1).clip(lower=1))

    # Block C: temporal neighbours
    feat_nb = feat.sort_values(["patient_id", "age"])
    for col in NEIGHBOR_COLS:
        if col not in feat_nb.columns:
            continue
        prev_v = feat_nb.groupby("patient_id")[col].shift(1)
        next_v = feat_nb.groupby("patient_id")[col].shift(-1)
        age_s  = feat_nb["age"]
        age_p  = feat_nb.groupby("patient_id")["age"].shift(1)
        age_n  = feat_nb.groupby("patient_id")["age"].shift(-1)
        feat[f"{col}_prev"]         = prev_v.loc[orig_index].values
        feat[f"{col}_next"]         = next_v.loc[orig_index].values
        feat[f"{col}_age_gap_prev"] = (age_s - age_p).loc[orig_index].values
        feat[f"{col}_age_gap_next"] = (age_n - age_s).loc[orig_index].values

    # Block A: interactions
    for f_col, agg_col in [
        ("off", "off_mean_pat"), ("on", "on_mean_pat"),
        ("ledd", "ledd_mean_pat"), ("age", "off_mean_pat"),
        ("age_at_diagnosis", "off_mean_pat"),
        ("age_at_diagnosis", "ledd_mean_pat"),
    ]:
        if f_col not in feat.columns or agg_col not in feat.columns:
            continue
        stub = f"{f_col}_x_{agg_col}"
        feat[f"{stub}_diff"]  = feat[f_col] - feat[agg_col]
        denom = feat[agg_col].where(feat[agg_col].abs() > 1e-8)
        feat[f"{stub}_ratio"] = (feat[f_col] / denom).clip(-10, 10)
        feat[f"{stub}_prod"]  = feat[f_col] * feat[agg_col]
    for f_col, agg_col in [
        ("off", "off_std_pat"), ("age", "off_slope_pat"),
        ("age", "on_slope_pat"), ("time_since_first_visit", "off_slope_pat"),
    ]:
        if f_col not in feat.columns or agg_col not in feat.columns:
            continue
        stub = f"{f_col}_x_{agg_col}"
        feat[f"{stub}_diff"] = feat[f_col] - feat[agg_col]
        feat[f"{stub}_prod"] = feat[f_col] * feat[agg_col]

    # Block F: rater visit count
    feat["rater_visit_count"] = feat.groupby("rater_id")["age"].transform("count")

    # Block H: rater OOF bias
    global_fb = float(np.mean(list(rater_bias.values()))) if rater_bias else 0.0
    feat["rater_oof_bias"] = feat["rater_id"].map(rater_bias).fillna(global_fb)

    # Block S: sequential patient history (pre-computed, just attach)
    for col in hist_df.columns:
        feat[col] = hist_df.loc[orig_index, col].values

    # Also add deviation of current off from patient running mean
    feat["off_vs_hist_mean"] = feat["off"] - feat["off_hist_mean"]

    assert "target" not in feat.columns
    feat = feat.drop(columns=["patient_id", "Index"], errors="ignore")

    for c in feat.select_dtypes("number").columns:
        if c.endswith("_prod") or c.endswith("_ratio"):
            feat[c] = feat[c].clip(-1e4, 1e4)

    assert (feat.index == orig_index).all(), "Index corrupted"
    return feat


# ══════════════════════════════════════════════════════════════════════════════
# 4. CV — GroupKFold(5), fixed params, no Optuna
# ══════════════════════════════════════════════════════════════════════════════
cv_splits = list(GroupKFold(n_splits=5).split(
    np.arange(len(groups)), groups=groups))

# Fixed params based on exp13 Optuna best (good enough, avoids tuning cost)
LGBM_PARAMS = {
    "objective":         "regression_l2",
    "metric":            "rmse",
    "verbosity":         -1,
    "n_jobs":            -1,
    "random_state":      42,
    "learning_rate":     0.05,
    "num_leaves":        63,
    "min_child_samples": 50,
    "subsample":         0.8,
    "colsample_bytree":  0.8,
    "reg_alpha":         0.1,
    "reg_lambda":        0.1,
}

print()
print("=" * 60)
print("CV — GroupKFold(5), LightGBM with patient history (Block S)")
print("=" * 60)

oof        = np.full(len(y), np.nan)
fold_rmses = []

for fold_i, (tr, val) in enumerate(cv_splits):
    df_tr  = visits_x.iloc[tr]
    df_val = visits_x.iloc[val]
    y_tr   = y.iloc[tr].values
    y_val  = y.iloc[val].values

    rater_bias  = _rater_oof_bias(df_tr, y_tr)
    pat_agg_tr  = _compute_pat_aggregates(df_tr)
    pat_agg_val = _compute_pat_aggregates(df_val)

    # Slice pre-computed history — O(1)
    hist_tr  = hist_train.iloc[tr]
    hist_val = hist_train.iloc[val]

    X_tr_f  = build_features(df_tr,  rater_bias, pat_agg_tr,  hist_tr)
    X_val_f = build_features(df_val, rater_bias, pat_agg_val, hist_val)

    for c in set(X_tr_f.columns) - set(X_val_f.columns):
        X_val_f[c] = np.nan
    X_val_f = X_val_f[X_tr_f.columns]

    cat_cols = [c for c in X_tr_f.columns
                if str(X_tr_f[c].dtype) in ("object", "string[python]", "str")]
    X_tr_f  = _lgbm_encode(X_tr_f,  cat_cols)
    X_val_f = _lgbm_encode(X_val_f, cat_cols)

    lgb_tr  = lgb.Dataset(X_tr_f,  label=y_tr,  categorical_feature=cat_cols,
                           free_raw_data=False)
    lgb_val = lgb.Dataset(X_val_f, label=y_val, categorical_feature=cat_cols,
                           free_raw_data=False)

    model = lgb.train(
        LGBM_PARAMS, lgb_tr,
        num_boost_round=300,
        valid_sets=[lgb_val],
        callbacks=[lgb.early_stopping(30, verbose=False),
                   lgb.log_evaluation(-1)],
    )
    preds = model.predict(X_val_f)
    rmse  = root_mean_squared_error(y_val, preds)
    fold_rmses.append(rmse)
    oof[val] = preds
    print(f"  fold {fold_i+1}/5: RMSE={rmse:.4f}  best_iter={model.best_iteration}")

mean_rmse = float(np.mean(fold_rmses))
std_rmse  = float(np.std(fold_rmses))
print(f"\nCV result: mean={mean_rmse:.4f}  std={std_rmse:.4f}")

# ══════════════════════════════════════════════════════════════════════════════
# 5. Push report to Skore Hub (Mathias Kozman / bobathon-esilv)
# ══════════════════════════════════════════════════════════════════════════════
print()
print("=" * 60)
print("Pushing report to Skore Hub")
print("=" * 60)

# skore.evaluate needs a sklearn-compatible estimator. We build the feature
# matrix once (all train) and pass a tabular_pipeline so skore can clone/fit
# it per fold — this is the lightweight Hub-compatible path used in exp11-13.
from skrub import tabular_pipeline  # noqa: E402

# Build the full feature matrix with all train data (no fold split)
rater_bias_all = _rater_oof_bias(visits_x, y.values)
pat_agg_all    = _compute_pat_aggregates(visits_x)
X_hub = build_features(visits_x, rater_bias_all, pat_agg_all, hist_train)

hub_report = skore.evaluate(
    tabular_pipeline("regressor"),
    X_hub, y,
    splitter=cv_splits,
)

cfg = load_skore_credentials()
login(mode="hub")
project = Project(name="bobathon-esilv", mode="hub", workspace=cfg["workspace"])
project.put("15_patient_history", hub_report)
print(f"Report pushed: 15_patient_history")
print(f"Hub: https://skore.probabl.ai/{cfg['workspace']}/bobathon-esilv")

# ══════════════════════════════════════════════════════════════════════════════
# 6. Final model + submission
# ══════════════════════════════════════════════════════════════════════════════
print()
print("=" * 60)
print("Training final model on all training data")
print("=" * 60)

pat_agg_test_15 = _compute_pat_aggregates(X_test)
rater_bias_final = _rater_oof_bias(visits_x, y.values)

X_train_feat = build_features(visits_x, rater_bias_final, pat_agg_all,    hist_train)
X_test_feat  = build_features(X_test,   rater_bias_final, pat_agg_test_15, hist_test)

for c in set(X_train_feat.columns) - set(X_test_feat.columns):
    X_test_feat[c] = np.nan
X_test_feat = X_test_feat[X_train_feat.columns]

assert "target"     not in X_train_feat.columns
assert "patient_id" not in X_train_feat.columns
assert "Index"      not in X_train_feat.columns

cat_cols_final = [c for c in X_train_feat.columns
                  if str(X_train_feat[c].dtype) in ("object", "string[python]", "str")]
X_train_feat = _lgbm_encode(X_train_feat, cat_cols_final)
X_test_feat  = _lgbm_encode(X_test_feat,  cat_cols_final)

lgb_full    = lgb.Dataset(X_train_feat, label=y.values,
                           categorical_feature=cat_cols_final, free_raw_data=False)
final_model = lgb.train(
    LGBM_PARAMS, lgb_full,
    num_boost_round=300,
    callbacks=[lgb.log_evaluation(-1)],
)
preds_test = final_model.predict(X_test_feat)

assert len(preds_test) == len(X_test)
assert not np.isnan(preds_test).any()
assert not np.isinf(preds_test).any()

sample_sub = pd.read_csv(DATA_DIR / "sample_submission.csv")
assert len(preds_test) == len(sample_sub)

submission = X_test[["Index"]].copy()
submission["target"] = preds_test
assert list(submission.columns) == ["Index", "target"]
assert submission["Index"].tolist() == X_test["Index"].tolist()

out = PROJECT_ROOT / "submissions"
out.mkdir(exist_ok=True)
submission.to_csv(out / "15_patient_history.csv", index=False)
print(f"Submission: submissions/15_patient_history.csv  ({len(submission)} rows)")

# ══════════════════════════════════════════════════════════════════════════════
# 7. Summary
# ══════════════════════════════════════════════════════════════════════════════
print()
print("=" * 60)
print("SUMMARY")
print("=" * 60)
print(f"exp13 reference ICC(residuals,patient) = 0.45")
print(f"exp15 LightGBM 5-fold CV : RMSE={mean_rmse:.4f} ± {std_rmse:.4f}")
print(f"New block S features     : {[c for c in hist_train.columns]} + off_vs_hist_mean")
print(f"Hub report               : 15_patient_history")
print(f"Submission               : submissions/15_patient_history.csv")
print()
print("OK: experiment 15 complete.")
