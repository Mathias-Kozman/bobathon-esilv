# %%
# ---------------------------------------------------------------------------
# Rôle     : correction du biais de régression vers la moyenne (isotonic, Huber, quantile)
# Entrées  : data/X_train.csv, data/y_train.csv, data/X_test.csv
# Sorties  : hub + submissions/ (pour le variant gagnant)
# Exécuter : .venv/Scripts/python.exe experiments/16_bias_correction.py
# ---------------------------------------------------------------------------
#
# Experiment 16: regression-to-mean bias correction.
#
# Builds on exp15 (patient history Block S, RMSE=4.12).
# Three variants, all evaluated with the same GroupKFold(5) folds:
#
#   A. Baseline  : exp15 model, objective=regression_l2 (MSE / RMSE)
#   B. +Isotonic : A's OOF predictions recalibrated with IsotonicRegression
#                  fitted on training OOF only. Applied to test at inference.
#   C. Huber     : same params, objective=regression_l1 (MAE / Huber-like)
#   D. Quantile  : same params, objective=quantile, alpha=0.5 (median)
#
# Calibration rule:
#   IsotonicRegression is fitted on the full OOF stack (all 5 folds
#   concatenated) — training targets only, never validation or test targets.
#   Applied to final test predictions after fitting on all training data.
#
# Winner selection: lowest OOF RMSE among A/B/C/D. Submission written only
# for the winner. Hub report pushed to Mathias Kozman / bobathon-esilv.
#
# Runtime target: < 2 minutes (5 variants × 5 folds × ~3s/fold = ~75s total).

import warnings

import numpy as np
import pandas as pd
import skore
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import root_mean_squared_error
from sklearn.model_selection import GroupKFold
from skore import Project, login
from skrub import tabular_pipeline

import lightgbm as lgb

from parkinson import PROJECT_ROOT
from parkinson.hub import load_skore_credentials

warnings.filterwarnings("ignore")

DATA_DIR = PROJECT_ROOT / "data"

# ══════════════════════════════════════════════════════════════════════════════
# 1. Load data (identical to exp15)
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

# ══════════════════════════════════════════════════════════════════════════════
# 2. Patient history + feature helpers (copied from exp15, unchanged)
# ══════════════════════════════════════════════════════════════════════════════

def compute_patient_history(df: pd.DataFrame) -> pd.DataFrame:
    """Vectorised within-patient running history (exp15 Block S)."""
    assert "target" not in df.columns
    s = df[["patient_id", "Index", "off", "ledd"]].sort_values(
        ["patient_id", "Index"]).copy()
    s["_off_valid"]  = s["off"].notna().astype(float)
    s["_ledd_valid"] = s["ledd"].notna().astype(float)
    off_cumsum   = s.groupby("patient_id")["off"].transform(lambda x: x.expanding().sum().shift(1))
    off_cumcount = s.groupby("patient_id")["_off_valid"].transform(lambda x: x.expanding().sum().shift(1))
    s["off_hist_mean"]  = off_cumsum / off_cumcount.replace(0, np.nan)
    s["off_hist_count"] = off_cumcount
    off_cumss = s.groupby("patient_id")["off"].transform(lambda x: (x**2).expanding().sum().shift(1))
    var = (off_cumss / off_cumcount.replace(0, np.nan)) - s["off_hist_mean"] ** 2
    s["off_hist_std"] = np.where(off_cumcount >= 2, np.sqrt(var.clip(lower=0)), np.nan)
    ledd_cumsum   = s.groupby("patient_id")["ledd"].transform(lambda x: x.expanding().sum().shift(1))
    ledd_cumcount = s.groupby("patient_id")["_ledd_valid"].transform(lambda x: x.expanding().sum().shift(1))
    s["ledd_hist_mean"] = ledd_cumsum / ledd_cumcount.replace(0, np.nan)
    s["hist_n_visits"]  = s.groupby("patient_id").cumcount()
    hist_cols = ["off_hist_mean", "off_hist_count", "off_hist_std",
                 "ledd_hist_mean", "hist_n_visits"]
    return s[hist_cols].loc[df.index]


AGG_COLS_NUM  = ["off", "on", "ledd", "age", "age_at_diagnosis",
                 "time_since_intake_on", "time_since_intake_off"]
NEIGHBOR_COLS = ["off", "on", "ledd", "time_since_intake_off"]


def _vectorised_slope(pid_arr, x_arr, y_arr):
    mask = ~(np.isnan(x_arr) | np.isnan(y_arr))
    if mask.sum() == 0:
        return pd.Series(dtype=float)
    pids, x, yv = pid_arr[mask], x_arr[mask], y_arr[mask]
    agg = pd.DataFrame({"p": pids, "x": x, "y": yv, "xy": x*yv, "x2": x*x}
                       ).groupby("p", sort=False).agg(
        n=("x","count"), sx=("x","sum"), sy=("y","sum"), sxy=("xy","sum"), sx2=("x2","sum"))
    num = agg["n"]*agg["sxy"] - agg["sx"]*agg["sy"]
    den = agg["n"]*agg["sx2"] - agg["sx"]**2
    return (num / den.where(den.abs() > 1e-10)).where(agg["n"] >= 3)


def _compute_pat_aggregates(df: pd.DataFrame) -> pd.DataFrame:
    assert "target" not in df.columns
    gb = df.groupby("patient_id", sort=False)
    agg_funcs: dict = {}
    for col in AGG_COLS_NUM:
        if col not in df.columns:
            continue
        for fn in ["mean", "median", "std", "min", "max", "count"]:
            agg_funcs[f"{col}_{fn}_pat"] = (col, fn)
    pat_df = gb.agg(**agg_funcs)
    pat_df = pat_df.join(gb["age"].agg(n_visits_pat="count", age_first_pat="min",
                                        age_last_pat="max"), how="left")
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
        pat_df[f"{col}_slope_pat"] = _vectorised_slope(
            pid_arr, age_arr, pd.to_numeric(df[col], errors="coerce").values)
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


def build_features(df_raw: pd.DataFrame, rater_bias: dict,
                   pat_agg: pd.DataFrame, hist_df: pd.DataFrame) -> pd.DataFrame:
    """Exp15 full feature set (blocks B,I,E,D,C,A,F,H,S)."""
    assert "target" not in df_raw.columns
    orig_index = df_raw.index.copy()
    feat = df_raw.copy()
    feat = feat.join(pat_agg, on="patient_id", how="left")
    feat["time_since_diagnosis"] = feat["age"] - feat["age_at_diagnosis"]
    mask_both = feat["off"].notna() & feat["on"].notna()
    feat["on_minus_off"] = np.where(mask_both, feat["on"] - feat["off"], np.nan)
    feat["on_div_off"]   = np.where(mask_both & (feat["off"].abs() > 1e-8),
                                    feat["on"] / feat["off"], np.nan)
    for col in ["off","on","ledd","gene","time_since_intake_on","time_since_intake_off"]:
        if col in feat.columns:
            feat[f"{col}_na"] = feat[col].isna().astype(np.int8)
    feat_s = feat.sort_values(["patient_id", "age"])
    feat["visit_rank"] = feat_s.groupby("patient_id").cumcount().loc[orig_index].values
    feat["time_since_first_visit"] = feat["age"] - feat["age_first_pat"]
    feat["time_to_last_visit"]     = feat["age_last_pat"] - feat["age"]
    feat["rel_visit_pos"] = feat["visit_rank"] / (feat["n_visits_pat"] - 1).clip(lower=1)
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
    for f_col, agg_col in [("off","off_mean_pat"),("on","on_mean_pat"),("ledd","ledd_mean_pat"),
                            ("age","off_mean_pat"),("age_at_diagnosis","off_mean_pat"),
                            ("age_at_diagnosis","ledd_mean_pat")]:
        if f_col not in feat.columns or agg_col not in feat.columns:
            continue
        stub = f"{f_col}_x_{agg_col}"
        feat[f"{stub}_diff"]  = feat[f_col] - feat[agg_col]
        feat[f"{stub}_ratio"] = (feat[f_col] / feat[agg_col].where(feat[agg_col].abs()>1e-8)).clip(-10,10)
        feat[f"{stub}_prod"]  = feat[f_col] * feat[agg_col]
    for f_col, agg_col in [("off","off_std_pat"),("age","off_slope_pat"),
                            ("age","on_slope_pat"),("time_since_first_visit","off_slope_pat")]:
        if f_col not in feat.columns or agg_col not in feat.columns:
            continue
        stub = f"{f_col}_x_{agg_col}"
        feat[f"{stub}_diff"] = feat[f_col] - feat[agg_col]
        feat[f"{stub}_prod"] = feat[f_col] * feat[agg_col]
    feat["rater_visit_count"] = feat.groupby("rater_id")["age"].transform("count")
    global_fb = float(np.mean(list(rater_bias.values()))) if rater_bias else 0.0
    feat["rater_oof_bias"] = feat["rater_id"].map(rater_bias).fillna(global_fb)
    for col in hist_df.columns:
        feat[col] = hist_df.loc[orig_index, col].values
    feat["off_vs_hist_mean"] = feat["off"] - feat["off_hist_mean"]
    assert "target" not in feat.columns
    feat = feat.drop(columns=["patient_id", "Index"], errors="ignore")
    for c in feat.select_dtypes("number").columns:
        if c.endswith("_prod") or c.endswith("_ratio"):
            feat[c] = feat[c].clip(-1e4, 1e4)
    assert (feat.index == orig_index).all()
    return feat


# ══════════════════════════════════════════════════════════════════════════════
# 3. Pre-compute history + aggregates once
# ══════════════════════════════════════════════════════════════════════════════
visits_x   = visits[list(X_train.columns)].copy()
hist_train = compute_patient_history(visits_x)
hist_test  = compute_patient_history(X_test)
print("History features pre-computed.")

cv_splits = list(GroupKFold(n_splits=5).split(np.arange(len(groups)), groups=groups))

# ══════════════════════════════════════════════════════════════════════════════
# 4. Generic CV runner — returns OOF predictions for any objective
# ══════════════════════════════════════════════════════════════════════════════

BASE_PARAMS = {
    "verbosity": -1, "n_jobs": -1, "random_state": 42,
    "learning_rate": 0.05, "num_leaves": 63, "min_child_samples": 50,
    "subsample": 0.8, "colsample_bytree": 0.8, "reg_alpha": 0.1, "reg_lambda": 0.1,
}


def run_cv(objective: str, extra: dict | None = None) -> tuple[float, float, np.ndarray]:
    """Run GroupKFold(5) and return (mean_rmse, std_rmse, oof_predictions)."""
    params = {**BASE_PARAMS, "objective": objective, "metric": "rmse",
              **(extra or {})}
    oof        = np.full(len(y), np.nan)
    fold_rmses = []
    for fold_i, (tr, val) in enumerate(cv_splits):
        df_tr  = visits_x.iloc[tr];  df_val = visits_x.iloc[val]
        y_tr   = y.iloc[tr].values;  y_val  = y.iloc[val].values
        rb     = _rater_oof_bias(df_tr, y_tr)
        pa_tr  = _compute_pat_aggregates(df_tr)
        pa_val = _compute_pat_aggregates(df_val)
        X_tr   = build_features(df_tr,  rb, pa_tr,  hist_train.iloc[tr])
        X_val  = build_features(df_val, rb, pa_val, hist_train.iloc[val])
        for c in set(X_tr.columns) - set(X_val.columns):
            X_val[c] = np.nan
        X_val = X_val[X_tr.columns]
        cats  = [c for c in X_tr.columns
                 if str(X_tr[c].dtype) in ("object","string[python]","str")]
        X_tr  = _lgbm_encode(X_tr,  cats)
        X_val = _lgbm_encode(X_val, cats)
        ds_tr  = lgb.Dataset(X_tr,  label=y_tr,  categorical_feature=cats, free_raw_data=False)
        ds_val = lgb.Dataset(X_val, label=y_val, categorical_feature=cats, free_raw_data=False)
        mdl = lgb.train(params, ds_tr, num_boost_round=300,
                        valid_sets=[ds_val],
                        callbacks=[lgb.early_stopping(30, verbose=False),
                                   lgb.log_evaluation(-1)])
        p = mdl.predict(X_val)
        fold_rmses.append(root_mean_squared_error(y_val, p))
        oof[val] = p
        print(f"    fold {fold_i+1}/5: RMSE={fold_rmses[-1]:.4f}")
    return float(np.mean(fold_rmses)), float(np.std(fold_rmses)), oof


# ══════════════════════════════════════════════════════════════════════════════
# 5. Variant A — baseline (regression_l2, same as exp15)
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "="*60)
print("Variant A — baseline (regression_l2)")
print("="*60)
mean_A, std_A, oof_A = run_cv("regression_l2")
print(f"  => mean={mean_A:.4f}  std={std_A:.4f}")

# ══════════════════════════════════════════════════════════════════════════════
# 6. Variant B — isotonic recalibration on A's OOF
#
# IsotonicRegression is fitted on OOF predictions vs training targets only.
# It learns a monotone correction curve for the regression-to-mean bias
# (model over-predicts mild cases, under-predicts severe cases).
# Applied post-hoc to final test predictions.
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "="*60)
print("Variant B — A + isotonic recalibration on OOF")
print("="*60)

valid = ~np.isnan(oof_A)
iso   = IsotonicRegression(out_of_bounds="clip")
iso.fit(oof_A[valid], y.values[valid])

oof_B     = oof_A.copy()
oof_B[valid] = iso.predict(oof_A[valid])
mean_B    = root_mean_squared_error(y.values[valid], oof_B[valid])
std_B     = 0.0  # single-pass calibration, no per-fold std
print(f"  OOF RMSE before calibration : {mean_A:.4f}")
print(f"  OOF RMSE after  calibration : {mean_B:.4f}")
print(f"  Gain from isotonic          : {mean_A - mean_B:+.4f}")

# ══════════════════════════════════════════════════════════════════════════════
# 7. Variant C — Huber / regression_l1 (MAE objective)
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "="*60)
print("Variant C — Huber (regression_l1 / MAE objective)")
print("="*60)
mean_C, std_C, oof_C = run_cv("regression_l1")
print(f"  => mean={mean_C:.4f}  std={std_C:.4f}")

# ══════════════════════════════════════════════════════════════════════════════
# 8. Variant D — quantile regression (median, alpha=0.5)
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "="*60)
print("Variant D — quantile regression (alpha=0.5, median)")
print("="*60)
mean_D, std_D, oof_D = run_cv("quantile", {"alpha": 0.5})
print(f"  => mean={mean_D:.4f}  std={std_D:.4f}")

# ══════════════════════════════════════════════════════════════════════════════
# 9. Comparison table + winner selection (validation results only)
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "="*60)
print("COMPARISON TABLE")
print("="*60)
results = [
    ("A. regression_l2 (baseline)",  mean_A, std_A),
    ("B. +isotonic calibration",     mean_B, std_B),
    ("C. Huber / regression_l1",     mean_C, std_C),
    ("D. quantile alpha=0.5",        mean_D, std_D),
]
best_rmse = min(r[1] for r in results)
print(f"{'Variant':<35s}  {'OOF RMSE':>9s}  {'std':>7s}  {'vs A':>8s}")
print("-" * 65)
for name, rmse, std in results:
    delta  = rmse - mean_A
    marker = "  <-- BEST" if abs(rmse - best_rmse) < 1e-9 else ""
    print(f"{name:<35s}  {rmse:>9.4f}  {std:>7.4f}  {delta:>+8.4f}{marker}")

winner_name, winner_rmse, _ = min(results, key=lambda r: r[1])
winner_oof = {"A": oof_A, "B": oof_B, "C": oof_C, "D": oof_D}[winner_name[0]]
print(f"\nWinner: {winner_name}  RMSE={winner_rmse:.4f}")

# ══════════════════════════════════════════════════════════════════════════════
# 10. Push report to Skore Hub
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "="*60)
print("Pushing report to Skore Hub (Mathias Kozman)")
print("="*60)

rater_bias_all = _rater_oof_bias(visits_x, y.values)
pat_agg_all    = _compute_pat_aggregates(visits_x)
X_hub = build_features(visits_x, rater_bias_all, pat_agg_all, hist_train)

hub_report = skore.evaluate(
    tabular_pipeline("regressor"), X_hub, y, splitter=cv_splits)

cfg = load_skore_credentials()
login(mode="hub")
project = Project(name="bobathon-esilv", mode="hub", workspace=cfg["workspace"])
project.put("16_bias_correction", hub_report)
print("Report pushed: 16_bias_correction")
print(f"Hub: https://skore.probabl.ai/{cfg['workspace']}/bobathon-esilv")

# ══════════════════════════════════════════════════════════════════════════════
# 11. Final model + submission (winner objective + isotonic if B won)
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "="*60)
print(f"Training final model ({winner_name})")
print("="*60)

win_obj    = {"A": "regression_l2", "B": "regression_l2",
              "C": "regression_l1", "D": "quantile"}[winner_name[0]]
win_extra  = {"D": {"alpha": 0.5}}.get(winner_name[0], {})
win_params = {**BASE_PARAMS, "objective": win_obj, "metric": "rmse", **win_extra}

pat_agg_test_16 = _compute_pat_aggregates(X_test)
X_train_feat = build_features(visits_x, rater_bias_all, pat_agg_all, hist_train)
X_test_feat  = build_features(X_test, rater_bias_all, pat_agg_test_16, hist_test)

for c in set(X_train_feat.columns) - set(X_test_feat.columns):
    X_test_feat[c] = np.nan
X_test_feat = X_test_feat[X_train_feat.columns]

assert "target"     not in X_train_feat.columns
assert "patient_id" not in X_train_feat.columns
assert "Index"      not in X_train_feat.columns

cats_final   = [c for c in X_train_feat.columns
                if str(X_train_feat[c].dtype) in ("object","string[python]","str")]
X_train_feat = _lgbm_encode(X_train_feat, cats_final)
X_test_feat  = _lgbm_encode(X_test_feat,  cats_final)

ds_full     = lgb.Dataset(X_train_feat, label=y.values,
                           categorical_feature=cats_final, free_raw_data=False)
final_model = lgb.train(win_params, ds_full, num_boost_round=300,
                         callbacks=[lgb.log_evaluation(-1)])
preds_test  = final_model.predict(X_test_feat)

# Apply isotonic calibration only if variant B won
if winner_name[0] == "B":
    print("Applying isotonic calibration to test predictions...")
    preds_test = iso.predict(preds_test)

assert not np.isnan(preds_test).any()
assert not np.isinf(preds_test).any()
assert len(preds_test) == len(pd.read_csv(DATA_DIR / "sample_submission.csv"))

submission = X_test[["Index"]].copy()
submission["target"] = preds_test
assert list(submission.columns) == ["Index", "target"]

out = PROJECT_ROOT / "submissions"
out.mkdir(exist_ok=True)
submission.to_csv(out / "16_bias_correction.csv", index=False)
print(f"Submission: submissions/16_bias_correction.csv  ({len(submission)} rows)")

# ══════════════════════════════════════════════════════════════════════════════
# 12. Summary
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "="*60)
print("FINAL SUMMARY")
print("="*60)
print(f"exp15 baseline (regression_l2) : RMSE={mean_A:.4f}")
print(f"B  +isotonic calibration       : RMSE={mean_B:.4f}  ({mean_A-mean_B:+.4f})")
print(f"C  Huber / regression_l1       : RMSE={mean_C:.4f}  ({mean_A-mean_C:+.4f})")
print(f"D  quantile alpha=0.5          : RMSE={mean_D:.4f}  ({mean_A-mean_D:+.4f})")
print(f"\nWinner: {winner_name}  RMSE={winner_rmse:.4f}")
print(f"Hub report : 16_bias_correction")
print(f"Submission : submissions/16_bias_correction.csv")
print()
print("OK: experiment 16 complete.")
