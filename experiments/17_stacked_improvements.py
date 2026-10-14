# %%
# Experiment 17: cumulative improvements — all methods stacked.
#
# Builds on exp16 (RMSE=3.911). Three additional improvements:
#
#   NEW A — 5-seed final model ensemble  (LOW IMPACT)
#     Train LightGBM 5 times with seeds 0-4, average test predictions.
#
#   NEW B — Target clipping [0, 132]  (TRIVIAL)
#     MDS-UPDRS is physically bounded. Clip final predictions.
#
#   NEW C — Levodopa decay proxy  (MEDIUM IMPACT)
#     exp(-time_since_intake_off / 2.0) when present, NaN otherwise.
#     Converts drug timing into a blood-concentration proxy matching
#     the generative model described in CONTEXT.md.
#
# NOTE: within-fold OOF patient target encoding was removed. It caused
#   train/val distribution mismatch: enc=patient mean on train rows,
#   enc=cohort mean (~37.5) on val rows (unseen patients). LightGBM
#   built heavy splits on the feature that collapsed at validation,
#   raising RMSE from 3.9 to 11.8. Patient baseline is already captured
#   leak-safely by Block S (off_hist_mean from exp15).
#
# Carried from exp15 (unchanged):
#   Block S — sequential within-patient history (off_hist_mean etc.)
#
# Carried from exp16 (unchanged):
#   Isotonic recalibration fitted on training OOF only.
#
# Hub push: tabular_pipeline CrossValidationReport on the full engineered
#   feature matrix — same pattern as exp11-13. Real metrics visible on Hub.
# CV: GroupKFold(5), LightGBM 800 rounds + early stopping.

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
# 2. Pre-compute: patient history (exp15 Block S) — done once, O(n log n)
# ══════════════════════════════════════════════════════════════════════════════

def compute_patient_history(df: pd.DataFrame) -> pd.DataFrame:
    """Vectorised within-patient running history (exp15 Block S, unchanged)."""
    assert "target" not in df.columns
    s = df[["patient_id", "Index", "off", "ledd"]].sort_values(
        ["patient_id", "Index"]).copy()
    s["_off_valid"]  = s["off"].notna().astype(float)
    s["_ledd_valid"] = s["ledd"].notna().astype(float)
    off_cs  = s.groupby("patient_id")["off"].transform(
        lambda x: x.expanding().sum().shift(1))
    off_cc  = s.groupby("patient_id")["_off_valid"].transform(
        lambda x: x.expanding().sum().shift(1))
    s["off_hist_mean"]  = off_cs / off_cc.replace(0, np.nan)
    s["off_hist_count"] = off_cc
    off_css = s.groupby("patient_id")["off"].transform(
        lambda x: (x**2).expanding().sum().shift(1))
    var = (off_css / off_cc.replace(0, np.nan)) - s["off_hist_mean"] ** 2
    s["off_hist_std"]   = np.where(off_cc >= 2, np.sqrt(var.clip(lower=0)), np.nan)
    ledd_cs = s.groupby("patient_id")["ledd"].transform(
        lambda x: x.expanding().sum().shift(1))
    ledd_cc = s.groupby("patient_id")["_ledd_valid"].transform(
        lambda x: x.expanding().sum().shift(1))
    s["ledd_hist_mean"] = ledd_cs / ledd_cc.replace(0, np.nan)
    s["hist_n_visits"]  = s.groupby("patient_id").cumcount()
    hist_cols = ["off_hist_mean", "off_hist_count", "off_hist_std",
                 "ledd_hist_mean", "hist_n_visits"]
    return s[hist_cols].loc[df.index]


visits_x   = visits[list(X_train.columns)].copy()
hist_train = compute_patient_history(visits_x)
hist_test  = compute_patient_history(X_test)
print("Patient history pre-computed.")

# ══════════════════════════════════════════════════════════════════════════════
# 3. Feature helpers
# ══════════════════════════════════════════════════════════════════════════════

AGG_COLS_NUM  = ["off", "on", "ledd", "age", "age_at_diagnosis",
                 "time_since_intake_on", "time_since_intake_off"]
NEIGHBOR_COLS = ["off", "on", "ledd", "time_since_intake_off"]
LEVO_T_HALF   = 2.0   # levodopa half-life in hours (Block D new in exp17)
TARGET_MIN    = 0.0   # MDS-UPDRS physical bounds
TARGET_MAX    = 132.0


def _vectorised_slope(pid_arr, x_arr, y_arr):
    mask = ~(np.isnan(x_arr) | np.isnan(y_arr))
    if mask.sum() == 0:
        return pd.Series(dtype=float)
    pids, x, yv = pid_arr[mask], x_arr[mask], y_arr[mask]
    agg = pd.DataFrame({"p": pids, "x": x, "y": yv, "xy": x*yv, "x2": x*x}
                       ).groupby("p", sort=False).agg(
        n=("x","count"), sx=("x","sum"), sy=("y","sum"),
        sxy=("xy","sum"), sx2=("x2","sum"))
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


def build_features(df_raw: pd.DataFrame,
                   rater_bias: dict,
                   pat_agg: pd.DataFrame,
                   hist_df: pd.DataFrame) -> pd.DataFrame:
    """
    Full feature matrix: exp15+16 blocks (B,I,E,D_pos,C,A,F,H,S) plus
    levo_decay_off (NEW exp17).
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

    # Block D_pos: visit position
    feat_s = feat.sort_values(["patient_id", "age"])
    feat["visit_rank"] = feat_s.groupby("patient_id").cumcount().loc[orig_index].values
    feat["time_since_first_visit"] = feat["age"] - feat["age_first_pat"]
    feat["time_to_last_visit"]     = feat["age_last_pat"] - feat["age"]
    feat["rel_visit_pos"] = feat["visit_rank"] / (feat["n_visits_pat"] - 1).clip(lower=1)

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
        ("off","off_mean_pat"), ("on","on_mean_pat"), ("ledd","ledd_mean_pat"),
        ("age","off_mean_pat"), ("age_at_diagnosis","off_mean_pat"),
        ("age_at_diagnosis","ledd_mean_pat"),
    ]:
        if f_col not in feat.columns or agg_col not in feat.columns:
            continue
        stub = f"{f_col}_x_{agg_col}"
        feat[f"{stub}_diff"]  = feat[f_col] - feat[agg_col]
        denom = feat[agg_col].where(feat[agg_col].abs() > 1e-8)
        feat[f"{stub}_ratio"] = (feat[f_col] / denom).clip(-10, 10)
        feat[f"{stub}_prod"]  = feat[f_col] * feat[agg_col]
    for f_col, agg_col in [
        ("off","off_std_pat"), ("age","off_slope_pat"),
        ("age","on_slope_pat"), ("time_since_first_visit","off_slope_pat"),
    ]:
        if f_col not in feat.columns or agg_col not in feat.columns:
            continue
        stub = f"{f_col}_x_{agg_col}"
        feat[f"{stub}_diff"] = feat[f_col] - feat[agg_col]
        feat[f"{stub}_prod"] = feat[f_col] * feat[agg_col]

    # Block F: rater visit count
    feat["rater_visit_count"] = feat.groupby("rater_id")["age"].transform("count")

    # Block H: rater OOF bias (exp15/16)
    global_fb = float(np.mean(list(rater_bias.values()))) if rater_bias else 0.0
    feat["rater_oof_bias"] = feat["rater_id"].map(rater_bias).fillna(global_fb)

    # Block S: sequential patient history (exp15)
    for col in hist_df.columns:
        feat[col] = hist_df.loc[orig_index, col].values
    feat["off_vs_hist_mean"] = feat["off"] - feat["off_hist_mean"]

    # NEW exp17 — levodopa decay proxy
    t_off = pd.to_numeric(
        feat.get("time_since_intake_off", pd.Series(np.nan, index=feat.index)),
        errors="coerce")
    feat["levo_decay_off"] = np.where(
        t_off.notna(), np.exp(-t_off / LEVO_T_HALF), np.nan)

    assert "target" not in feat.columns
    feat = feat.drop(columns=["patient_id", "Index"], errors="ignore")
    for c in feat.select_dtypes("number").columns:
        if c.endswith("_prod") or c.endswith("_ratio"):
            feat[c] = feat[c].clip(-1e4, 1e4)
    assert (feat.index == orig_index).all()
    return feat


# ══════════════════════════════════════════════════════════════════════════════
# 5. CV — GroupKFold(5), fixed params
# ══════════════════════════════════════════════════════════════════════════════
cv_splits = list(GroupKFold(n_splits=5).split(
    np.arange(len(groups)), groups=groups))

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
print("CV — GroupKFold(5), all improvements stacked")
print("=" * 60)

oof        = np.full(len(y), np.nan)
fold_rmses = []

for fold_i, (tr, val) in enumerate(cv_splits):
    df_tr  = visits_x.iloc[tr]
    df_val = visits_x.iloc[val]
    y_tr   = y.iloc[tr].values
    y_val  = y.iloc[val].values

    # Fold-specific computations
    rater_bias  = _rater_oof_bias(df_tr, y_tr)
    pat_agg_tr  = _compute_pat_aggregates(df_tr)
    pat_agg_val = _compute_pat_aggregates(df_val)

    X_tr_f  = build_features(df_tr,  rater_bias, pat_agg_tr,  hist_train.iloc[tr])
    X_val_f = build_features(df_val, rater_bias, pat_agg_val, hist_train.iloc[val])

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
        num_boost_round=800,
        valid_sets=[lgb_val],
        callbacks=[lgb.early_stopping(50, verbose=False),
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
# 6. Isotonic recalibration on OOF (exp16, unchanged)
# ══════════════════════════════════════════════════════════════════════════════
valid      = ~np.isnan(oof)
iso        = IsotonicRegression(out_of_bounds="clip")
iso.fit(oof[valid], y.values[valid])
oof_iso    = oof.copy()
oof_iso[valid] = iso.predict(oof[valid])
rmse_iso   = root_mean_squared_error(y.values[valid], oof_iso[valid])
use_iso    = rmse_iso < mean_rmse - 0.002
print(f"\nIsotonic: raw={mean_rmse:.4f}  calibrated={rmse_iso:.4f}  "
      f"({'applied' if use_iso else 'skipped'})")

# ══════════════════════════════════════════════════════════════════════════════
# 7. Hub push — tabular_pipeline on full engineered feature matrix
#
# Same pattern as exp11-13: skore.evaluate receives the pre-built feature
# matrix and cv_splits, clones tabular_pipeline per fold, and produces a
# real CrossValidationReport with correct RMSE metrics on the Hub.
# ══════════════════════════════════════════════════════════════════════════════
print()
print("=" * 60)
print("Pushing report to Skore Hub (Mathias Kozman)")
print("=" * 60)

rater_bias_hub = _rater_oof_bias(visits_x, y.values)
pat_agg_hub    = _compute_pat_aggregates(visits_x)
X_hub = build_features(visits_x, rater_bias_hub, pat_agg_hub, hist_train)

hub_report = skore.evaluate(
    tabular_pipeline("regressor"),
    X_hub, y,
    splitter=cv_splits,
)

cfg = load_skore_credentials()
login(mode="hub")
project = Project(name="bobathon-esilv", mode="hub", workspace=cfg["workspace"])
project.put("17_stacked_improvements", hub_report)
print("Report pushed: 17_stacked_improvements")
print(f"Hub: https://skore.probabl.ai/{cfg['workspace']}/bobathon-esilv")

# ══════════════════════════════════════════════════════════════════════════════
# 8. Final model — 5-seed ensemble (NEW exp17)
# ══════════════════════════════════════════════════════════════════════════════
print()
print("=" * 60)
print("Training final model (5-seed ensemble)")
print("=" * 60)

# Build full-train and test feature matrices
rater_bias_all  = rater_bias_hub   # reuse already computed
pat_agg_all     = pat_agg_hub      # reuse already computed
pat_agg_test_17 = _compute_pat_aggregates(X_test)

X_train_feat = build_features(visits_x, rater_bias_all, pat_agg_all, hist_train)
X_test_feat  = build_features(X_test,   rater_bias_all, pat_agg_test_17, hist_test)

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

# 5 seeds, average predictions
seed_preds = []
for seed in range(5):
    seed_params = {**LGBM_PARAMS, "random_state": seed}
    ds_full = lgb.Dataset(X_train_feat, label=y.values,
                           categorical_feature=cat_cols_final, free_raw_data=False)
    mdl = lgb.train(seed_params, ds_full,
                    num_boost_round=800,
                    callbacks=[lgb.log_evaluation(-1)])
    seed_preds.append(mdl.predict(X_test_feat))
    print(f"  seed {seed} trained")

preds_test = np.mean(seed_preds, axis=0)

# Isotonic recalibration if it helped on OOF
if use_iso:
    preds_test = iso.predict(preds_test)

# Target clipping to physical bounds (NEW exp17)
preds_test = np.clip(preds_test, TARGET_MIN, TARGET_MAX)

assert not np.isnan(preds_test).any()
assert not np.isinf(preds_test).any()
assert len(preds_test) == len(pd.read_csv(DATA_DIR / "sample_submission.csv"))

# ══════════════════════════════════════════════════════════════════════════════
# 9. Submission file
# ══════════════════════════════════════════════════════════════════════════════
submission = X_test[["Index"]].copy()
submission["target"] = preds_test
assert list(submission.columns) == ["Index", "target"]
assert submission["Index"].tolist() == X_test["Index"].tolist()

out = PROJECT_ROOT / "submissions"
out.mkdir(exist_ok=True)
submission.to_csv(out / "17_stacked_improvements.csv", index=False)
print(f"\nSubmission: submissions/17_stacked_improvements.csv  ({len(submission)} rows)")

# ══════════════════════════════════════════════════════════════════════════════
# 10. Summary
# ══════════════════════════════════════════════════════════════════════════════
print()
print("=" * 60)
print("FINAL SUMMARY")
print("=" * 60)
print(f"exp16 reference              : RMSE=3.911")
print(f"exp17 CV RMSE (raw)          : {mean_rmse:.4f} ± {std_rmse:.4f}")
print(f"exp17 CV RMSE (+ isotonic)   : {rmse_iso:.4f}")
print()
print("Methods stacked (cumulative):")
print("  exp13  LightGBM + exp12 feature set")
print("  exp15  + patient sequential history (Block S)")
print("  exp16  + isotonic recalibration on OOF")
print("  exp17  + levodopa decay proxy (new)")
print("  exp17  + 5-seed ensemble (new)")
print("  exp17  + target clipping [0, 132] (new)")
print()
print(f"Hub report : 17_stacked_improvements")
print(f"Submission : submissions/17_stacked_improvements.csv")
print()
print("OK: experiment 17 complete.")
