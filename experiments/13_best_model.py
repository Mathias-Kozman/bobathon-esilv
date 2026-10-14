# %%
# ---------------------------------------------------------------------------
# Rôle     : meilleur modèle — feature engineering complet + LightGBM + ensemble
# Entrées  : data/X_train.csv, data/y_train.csv, data/X_test.csv
# Sorties  : hub "13_best_model" (CrossValidationReport) ; submissions/13_best_model.csv
# Exécuter : .venv/Scripts/python.exe experiments/13_best_model.py
# NE PAS MODIFIER : optimisation en cours (experiments/14_improvements.py)
# ---------------------------------------------------------------------------
#
# Experiment 13: best model — full feature engineering + LightGBM + ensemble.
#
# Strategy (guided by ICC analysis from exp12 OOF residuals):
#   ICC(residuals, patient) = 0.45 → strong residual patient effect unmodelled.
#   ICC(residuals, rater)   = 0.008 → small but rater bias std = 0.49.
#   Index is perfectly monotone with age per patient (Spearman=1.0).
#
# Feature blocks (all X-only except Block H which is OOF in-fold):
#   B. Extended patient aggregates (vectorised pandas groupby — fast)
#   C. Temporal neighbor features (prev/next off/on/ledd sorted by age)
#   D. Visit position (rank, time_since_first, time_to_last, relative position)
#   E. Missingness indicators
#   F. Rater visit count (X-only)
#   G. MixedLM patient random intercept (off ~ age + time_since_diagnosis)
#   H. Rater OOF target encoding (mean(target-off) per rater, computed in-fold only)
#   I. Derived: time_since_diagnosis, on_minus_off, on_div_off
#
# Critical implementation rules:
#   1. Feature construction PRESERVES the original DataFrame index so that
#      feature rows align with y labels — no reset_index in build_features.
#   2. Patient aggregates use vectorised groupby (not Python loops) for speed.
#   3. MixedLM is computed once per fold (2s). Patient loop eliminated.
#
# Models:
#   - tabular_pipeline (reference baseline, exp12 approach)
#   - LightGBM (Optuna-tuned, 20 trials, seed-0 folds)
#   - Ensemble (NNLS on OOF)
#
# CV: GroupKFold(5) × 3 seeds = 15 folds.
# Submission: LightGBM (or ensemble if better by ≥ 0.005 RMSE).

import time
import warnings

import numpy as np
import pandas as pd
import skore
from scipy import stats as sp_stats
from scipy.optimize import nnls
from sklearn.base import clone
from sklearn.metrics import root_mean_squared_error
from sklearn.model_selection import GroupKFold
from skore import Project, login
from skrub import tabular_pipeline

import lightgbm as lgb
import optuna

from parkinson import PROJECT_ROOT
from parkinson.hub import load_skore_credentials

optuna.logging.set_verbosity(optuna.logging.WARNING)
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

visits  = X_train.merge(y_train, on="Index")
assert len(visits) == len(X_train) == len(y_train)

y      = visits["target"].reset_index(drop=True)
groups = visits["patient_id"].reset_index(drop=True)

print(f"Train: {len(visits)} rows, {groups.nunique()} patients")
print(f"Test : {len(X_test)} rows,  {X_test['patient_id'].nunique()} patients")

# Phase 0 diagnostics (reported):
# 0a: Index perfectly monotone with age (Spearman=1.0) — use as visit order.
# 0d: Same columns in train and test; time_since_diagnosis derivable from both.
# Rater: 0 unseen raters in test → rater OOF encoding applies to test fully.
raters_train = set(X_train["rater_id"].dropna())
pct_unseen = (~X_test["rater_id"].isin(raters_train)).mean()
print(f"\nPhase 0 — rater unseen in test: {pct_unseen:.1%} (expected 0%)")

# ══════════════════════════════════════════════════════════════════════════════
# 2. Feature construction helpers (vectorised — no Python per-patient loops)
# ══════════════════════════════════════════════════════════════════════════════

AGG_COLS_NUM  = ["off", "on", "ledd", "age", "age_at_diagnosis",
                 "time_since_intake_on", "time_since_intake_off"]
NEIGHBOR_COLS = ["off", "on", "ledd", "time_since_intake_off"]


def _vectorised_slope(pid_arr: np.ndarray,
                       x_arr: np.ndarray,
                       y_arr: np.ndarray) -> pd.Series:
    """Fully vectorised OLS slope grouped by patient_id. ~0.003s for 5k patients."""
    mask = ~(np.isnan(x_arr) | np.isnan(y_arr))
    if mask.sum() == 0:
        return pd.Series(dtype=float)
    pids, x, y = pid_arr[mask], x_arr[mask], y_arr[mask]
    agg = pd.DataFrame({"p": pids, "x": x, "y": y, "xy": x * y, "x2": x * x}
                       ).groupby("p", sort=False).agg(
        n=("x", "count"), sx=("x", "sum"), sy=("y", "sum"),
        sxy=("xy", "sum"), sx2=("x2", "sum"),
    )
    num   = agg["n"] * agg["sxy"] - agg["sx"] * agg["sy"]
    den   = agg["n"] * agg["sx2"] - agg["sx"] ** 2
    return (num / den.where(den.abs() > 1e-10)).where(agg["n"] >= 3)


def _compute_pat_aggregates(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute per-patient X-only aggregates. Vectorised groupby: ~0.06s for 5k patients.
    Returns a DataFrame indexed by patient_id.
    """
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
    pat_df = pat_df.join(gb["age"].agg(
        n_visits_pat="count", age_first_pat="min", age_last_pat="max"),
        how="left")
    pat_df["followup_dur_pat"] = pat_df["age_last_pat"] - pat_df["age_first_pat"]

    # First/last off and on (Index is monotone with age → sort is correct)
    for col in ["off", "on"]:
        if col not in df.columns:
            continue
        notna = df.sort_values("Index")[["patient_id", col]].dropna(subset=[col])
        pat_df[f"{col}_first_pat"] = notna.groupby("patient_id")[col].first()
        pat_df[f"{col}_last_pat"]  = notna.groupby("patient_id")[col].last()

    # Slopes: 230× faster than per-group linregress
    pid_arr = df["patient_id"].values
    age_arr = df["age"].values.astype(float)
    for col in AGG_COLS_NUM:
        if col not in df.columns:
            continue
        col_arr = pd.to_numeric(df[col], errors="coerce").values
        pat_df[f"{col}_slope_pat"] = _vectorised_slope(pid_arr, age_arr, col_arr)

    return pat_df


def _fit_mixedlm(df_with_pid: pd.DataFrame):
    """Fit MixedLM off ~ age + tsd on train fold. Returns (intercept_map, fixed_params)."""
    import statsmodels.formula.api as smf
    df = df_with_pid.copy()
    df["tsd"] = df["age"] - df["age_at_diagnosis"].fillna(
        df["age_at_diagnosis"].median())
    df_fit = df.dropna(subset=["off", "age"]).copy()
    try:
        md  = smf.mixedlm("off ~ age + tsd", data=df_fit, groups=df_fit["patient_id"])
        mdf = md.fit(method="bfgs", disp=False)
        if mdf.converged:
            int_map = {pid: float(v.iloc[0]) for pid, v in mdf.random_effects.items()}
            return int_map, dict(mdf.params)
    except Exception:
        pass
    return None, None


def _rater_oof_bias(df_tr: pd.DataFrame, y_tr: np.ndarray) -> dict:
    """Compute mean(target - off) per rater on training fold rows."""
    off_vals = df_tr["off"].values
    raters   = df_tr["rater_id"].values
    off_fill = np.where(np.isnan(off_vals), float(np.nanmedian(off_vals)), off_vals)
    residuals = y_tr - off_fill
    return {r: float(np.nanmean(residuals[raters == r])) for r in np.unique(raters)}


def _lgbm_encode(X: pd.DataFrame, cat_cols: list[str]) -> pd.DataFrame:
    """Encode string/category columns; cast int8 → float for LightGBM."""
    X = X.copy()
    for c in cat_cols:
        X[c] = X[c].astype("category")
    for c in X.select_dtypes("int8").columns:
        X[c] = X[c].astype(float)
    return X


def build_features(df_raw: pd.DataFrame,
                   rater_bias_map: dict | None,
                   lme_intercept_map: dict | None,
                   lme_fixed_params: dict | None,
                   pat_agg: pd.DataFrame | None = None) -> pd.DataFrame:
    """
    Build full feature matrix from raw X DataFrame.

    CRITICAL: preserves original DataFrame index throughout.

    Parameters
    ----------
    df_raw : raw X columns (must NOT contain 'target').
    rater_bias_map : {rater_id -> OOF mean(target-off)} from train fold only.
    lme_intercept_map : {patient_id -> RE intercept} from MixedLM on train.
    lme_fixed_params  : fixed-effect coefficients dict from MixedLM.
    pat_agg : pre-computed patient aggregates (use when available for speed).
    """
    assert "target" not in df_raw.columns

    orig_index = df_raw.index.copy()
    feat = df_raw.copy()

    # ── Block B: patient aggregates ───────────────────────────────────────────
    if pat_agg is None:
        pat_agg = _compute_pat_aggregates(feat)
    feat = feat.join(pat_agg, on="patient_id", how="left")

    # ── Block I: derived features ─────────────────────────────────────────────
    feat["time_since_diagnosis"] = feat["age"] - feat["age_at_diagnosis"]
    mask_both = feat["off"].notna() & feat["on"].notna()
    feat["on_minus_off"] = np.where(mask_both, feat["on"] - feat["off"], np.nan)
    feat["on_div_off"]   = np.where(
        mask_both & (feat["off"].abs() > 1e-8),
        feat["on"] / feat["off"], np.nan)

    # ── Block E: missingness indicators ──────────────────────────────────────
    for col in ["off", "on", "ledd", "gene",
                "time_since_intake_on", "time_since_intake_off"]:
        if col in feat.columns:
            feat[f"{col}_na"] = feat[col].isna().astype(np.int8)

    # ── Block D: visit position ────────────────────────────────────────────────
    # Sort for cumcount, then restore via .loc[orig_index]
    feat_sorted = feat.sort_values(["patient_id", "age"])
    visit_rank  = feat_sorted.groupby("patient_id").cumcount()
    feat["visit_rank"]             = visit_rank.loc[orig_index].values
    feat["time_since_first_visit"] = feat["age"] - feat["age_first_pat"]
    feat["time_to_last_visit"]     = feat["age_last_pat"] - feat["age"]
    feat["rel_visit_pos"] = (
        feat["visit_rank"] / (feat["n_visits_pat"] - 1).clip(lower=1)
    )

    # ── Block C: temporal neighbor features ──────────────────────────────────
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

    # ── Block A: interactions ─────────────────────────────────────────────────
    # Diff/ratio/product interactions vs patient means.
    # IMPORTANT: skip ratio for slope-based denominators — slopes can be near-zero
    # producing extreme ratio values (1e5+) that destabilise LightGBM.
    interaction_pairs_mean = [
        ("off",  "off_mean_pat"),
        ("on",   "on_mean_pat"),
        ("ledd", "ledd_mean_pat"),
        ("age",  "off_mean_pat"),
        ("age_at_diagnosis", "off_mean_pat"),
        ("age_at_diagnosis", "ledd_mean_pat"),
    ]
    # Diff + product only for std-based and slope-based interactions (no ratio)
    interaction_pairs_no_ratio = [
        ("off",  "off_std_pat"),
        ("age",  "off_slope_pat"),
        ("age",  "on_slope_pat"),
        ("time_since_first_visit", "off_slope_pat"),
    ]
    for f_col, agg_col in interaction_pairs_mean:
        if f_col not in feat.columns or agg_col not in feat.columns:
            continue
        f_v  = feat[f_col]
        m_v  = feat[agg_col]
        stub = f"{f_col}_x_{agg_col}"
        feat[f"{stub}_diff"]  = f_v - m_v
        denom = m_v.where(m_v.abs() > 1e-8)
        feat[f"{stub}_ratio"] = (f_v / denom).clip(-10, 10)  # clip to prevent extremes
        feat[f"{stub}_prod"]  = f_v * m_v
    for f_col, agg_col in interaction_pairs_no_ratio:
        if f_col not in feat.columns or agg_col not in feat.columns:
            continue
        stub = f"{f_col}_x_{agg_col}"
        feat[f"{stub}_diff"] = feat[f_col] - feat[agg_col]
        feat[f"{stub}_prod"] = feat[f_col] * feat[agg_col]

    # ── Block F: rater visit count (X-only) ───────────────────────────────────
    feat["rater_visit_count"] = (
        feat.groupby("rater_id")["age"].transform("count")
    )

    # ── Block H: rater OOF target encoding ────────────────────────────────────
    if rater_bias_map is not None:
        feat["rater_oof_bias"] = feat["rater_id"].map(rater_bias_map).fillna(0.0)
    else:
        feat["rater_oof_bias"] = 0.0

    # ── Block G: MixedLM patient random intercept ─────────────────────────────
    if lme_intercept_map is not None and lme_fixed_params is not None:
        tsd = feat["age"] - feat["age_at_diagnosis"].fillna(
            feat["age_at_diagnosis"].median())
        feat["lme_intercept_pat"] = feat["patient_id"].map(
            lme_intercept_map).fillna(0.0)
        feat["lme_pred_off"] = (
            lme_fixed_params.get("Intercept", 0.0)
            + lme_fixed_params.get("age", 0.0) * feat["age"]
            + lme_fixed_params.get("tsd", 0.0) * tsd
            + feat["lme_intercept_pat"]
        )
    else:
        feat["lme_intercept_pat"] = np.nan
        feat["lme_pred_off"]      = np.nan

    assert "target" not in feat.columns
    feat = feat.drop(columns=["patient_id", "Index"], errors="ignore")

    # Clip extreme product/ratio values that can destabilise tree learners
    # (ledd products can reach 4e6, age/slope ratios can reach 1e5+)
    numeric_cols = feat.select_dtypes("number").columns
    for c in numeric_cols:
        if c.endswith("_prod") or c.endswith("_ratio"):
            feat[c] = feat[c].clip(-1e4, 1e4)

    # Preserve original row order
    assert (feat.index == orig_index).all(), "Index corrupted"
    return feat


# ══════════════════════════════════════════════════════════════════════════════
# 3. CV infrastructure — 3 seeds × GroupKFold(5) = 15 folds
# ══════════════════════════════════════════════════════════════════════════════

CV_SEEDS = [0, 1, 2]
N_SPLITS = 5


def make_splits(groups_series: pd.Series,
                seeds: list[int],
                n_splits: int = 5) -> list[tuple]:
    """GroupKFold with shuffled patient order per seed."""
    all_splits = []
    unique_patients = groups_series.unique()
    for seed in seeds:
        rng      = np.random.default_rng(seed)
        perm     = rng.permutation(unique_patients)
        g_mapped = groups_series.map({p: i for i, p in enumerate(perm)})
        for tr, val in GroupKFold(n_splits=n_splits).split(
                np.arange(len(groups_series)), groups=g_mapped):
            all_splits.append((tr, val))
    return all_splits


all_splits   = make_splits(groups, CV_SEEDS, N_SPLITS)
cv_splits_s0 = list(GroupKFold(N_SPLITS).split(
    np.arange(len(groups)), groups=groups))

print(f"\n{len(all_splits)} folds ({len(CV_SEEDS)} seeds × {N_SPLITS})")

# ══════════════════════════════════════════════════════════════════════════════
# 4. LightGBM CV runner (leak-safe, vectorised features)
# ══════════════════════════════════════════════════════════════════════════════

BASE_LGBM_PARAMS = {
    "objective":    "regression_l2",
    "metric":       "rmse",
    "verbosity":    -1,
    "n_jobs":       -1,
    "random_state": 42,
}


def run_cv_lgbm(params: dict,
                splits: list,
                visits_df: pd.DataFrame,
                y_series: pd.Series,
                verbose: bool = False) -> tuple[float, float, np.ndarray]:
    """CV with per-fold leak-safe feature construction."""
    oof        = np.full(len(y_series), np.nan)
    fold_rmses = []
    visits_x   = visits_df[list(X_train.columns)].copy()  # X cols only

    for fold_i, (tr, val) in enumerate(splits):
        df_tr = visits_x.iloc[tr]
        df_val = visits_x.iloc[val]
        y_tr  = y_series.iloc[tr].values
        y_val = y_series.iloc[val].values

        # Block H: rater OOF bias from train fold only
        rater_bias = _rater_oof_bias(df_tr, y_tr)

        # Block G: MixedLM skipped during CV — val patients (unseen in train fold)
        # would get lme_intercept=0, creating train/val distribution mismatch.
        # MixedLM RE is only usable when all prediction patients appear in train.
        lme_int, lme_fp = None, None

        # Pre-compute patient aggregates once per fold
        pat_agg_tr  = _compute_pat_aggregates(df_tr)
        pat_agg_val = _compute_pat_aggregates(df_val)

        X_tr_f  = build_features(df_tr,  rater_bias, lme_int, lme_fp, pat_agg_tr)
        X_val_f = build_features(df_val, rater_bias, lme_int, lme_fp, pat_agg_val)

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
            params, lgb_tr,
            num_boost_round=800,
            valid_sets=[lgb_val],
            callbacks=[lgb.early_stopping(40, verbose=False),
                       lgb.log_evaluation(-1)],
        )
        preds = model.predict(X_val_f)
        rmse  = root_mean_squared_error(y_val, preds)
        fold_rmses.append(rmse)
        oof[val] = preds
        if verbose:
            print(f"  fold {fold_i+1:02d}/{len(splits)}: RMSE={rmse:.4f}  "
                  f"best_iter={model.best_iteration}")

    mean_rmse = float(np.nanmean(fold_rmses))
    std_rmse  = float(np.nanstd(fold_rmses))
    return mean_rmse, std_rmse, oof


# ══════════════════════════════════════════════════════════════════════════════
# 5. Baseline: tabular_pipeline on exp12 features (seed-0, for skore push)
# ══════════════════════════════════════════════════════════════════════════════
print()
print("=" * 80)
print("BASELINE: tabular_pipeline on exp12 features (seed-0, 5 folds)")
print("=" * 80)


def _compute_pat_agg_exp12(df):
    assert "target" not in df.columns
    recs = []
    for pid, grp in df.groupby("patient_id"):
        row = {"patient_id": pid}
        for col in ["off", "on", "ledd"]:
            vals = grp[col].dropna(); n = len(vals)
            row[f"{col}_mean_pat"]   = vals.mean()   if n > 0 else np.nan
            row[f"{col}_median_pat"] = vals.median() if n > 0 else np.nan
            row[f"{col}_count_pat"]  = n
            row[f"{col}_std_pat"]    = vals.std()    if n >= 2 else np.nan
        for col in ["off", "on"]:
            d = grp[["age", col]].dropna().sort_values("age")
            if len(d) >= 3:
                row[f"{col}_slope_pat"] = float(sp_stats.linregress(d["age"], d[col]).slope)
            else:
                row[f"{col}_slope_pat"] = np.nan
            d2 = grp.sort_values("Index")[col].dropna()
            row[f"{col}_first_pat"] = d2.iloc[0]  if len(d2) > 0 else np.nan
            row[f"{col}_last_pat"]  = d2.iloc[-1] if len(d2) > 0 else np.nan
        recs.append(row)
    return pd.DataFrame(recs).set_index("patient_id")


def _join_agg(X, pat_agg):
    assert "target" not in X.columns
    r = X.join(pat_agg, on="patient_id", how="left")
    return r.drop(columns=["patient_id", "Index"], errors="ignore")


def _add_interactions_exp12(df):
    assert "target" not in df.columns
    out = df.copy()
    for feat, agg in [("off","off_mean_pat"),("on","on_mean_pat"),("ledd","ledd_mean_pat"),
                      ("age","off_mean_pat"),("age_at_diagnosis","off_mean_pat"),
                      ("age_at_diagnosis","ledd_mean_pat")]:
        if feat not in out.columns or agg not in out.columns:
            continue
        f, m = out[feat], out[agg]
        stub = f"{feat}_x_{agg}"
        out[f"{stub}_diff"]  = f - m
        denom = m.where(m.abs() > 1e-8)
        out[f"{stub}_ratio"] = f / denom
        out[f"{stub}_prod"]  = f * m
    for col in ["off", "on"]:
        sc = f"{col}_slope_pat"
        if col in out.columns and sc in out.columns:
            out[f"age_x_{sc}"] = out["age"] * out[sc]
    return out


pat_agg_train_ref = _compute_pat_agg_exp12(X_train)
pat_agg_test_ref  = _compute_pat_agg_exp12(X_test)
visits_X = visits[list(X_train.columns)].copy()
X_exp12  = _add_interactions_exp12(_join_agg(visits_X, pat_agg_train_ref))

pipe_ref      = tabular_pipeline("regressor")
fold_rmses_ref = []
oof_ref_s0    = np.full(len(y), np.nan)
for i, (tr, val) in enumerate(cv_splits_s0):
    clf = clone(pipe_ref)
    clf.fit(X_exp12.iloc[tr], y.iloc[tr])
    p    = clf.predict(X_exp12.iloc[val])
    rmse = root_mean_squared_error(y.iloc[val], p)
    fold_rmses_ref.append(rmse)
    oof_ref_s0[val] = p
    print(f"  ref fold {i+1}: RMSE={rmse:.4f}")
mean_ref = float(np.mean(fold_rmses_ref))
std_ref  = float(np.std(fold_rmses_ref))
print(f"  ref => mean={mean_ref:.4f}  std={std_ref:.4f}\n")

# ══════════════════════════════════════════════════════════════════════════════
# 6. Optuna tuning (20 trials, seed-0 5 folds)
# ══════════════════════════════════════════════════════════════════════════════
print()
print("=" * 80)
print("LightGBM Optuna tuning (20 trials, 5 seed-0 folds)")
print("=" * 80)


def objective_lgbm(trial: optuna.Trial) -> float:
    params = {
        **BASE_LGBM_PARAMS,
        "learning_rate":     trial.suggest_float("learning_rate", 0.02, 0.1, log=True),
        "num_leaves":        trial.suggest_int("num_leaves", 31, 200),
        "min_child_samples": trial.suggest_int("min_child_samples", 50, 300),
        "subsample":         trial.suggest_float("subsample", 0.5, 1.0),
        "colsample_bytree":  trial.suggest_float("colsample_bytree", 0.4, 1.0),
        "reg_alpha":         trial.suggest_float("reg_alpha", 1e-3, 5.0, log=True),
        "reg_lambda":        trial.suggest_float("reg_lambda", 1e-3, 5.0, log=True),
    }
    mean_rmse, _, _ = run_cv_lgbm(params, cv_splits_s0, visits, y)
    return mean_rmse


t0    = time.perf_counter()
study = optuna.create_study(direction="minimize",
                             sampler=optuna.samplers.TPESampler(seed=42))
study.optimize(objective_lgbm, n_trials=20)
t_tune = time.perf_counter() - t0
print(f"Optuna done in {t_tune:.0f}s  best RMSE={study.best_value:.4f}")
print(f"Best params: {study.best_params}")

best_lgbm_params = {**BASE_LGBM_PARAMS, **study.best_params}

# ══════════════════════════════════════════════════════════════════════════════
# 7. Best LightGBM on full 15-fold CV
# ══════════════════════════════════════════════════════════════════════════════
print()
print("=" * 80)
print("Best LightGBM — full 15-fold CV (3 seeds × 5 folds)")
print("=" * 80)
t0 = time.perf_counter()
mean_lgbm, std_lgbm, oof_lgbm = run_cv_lgbm(
    best_lgbm_params, all_splits, visits, y, verbose=True)
t_lgbm = time.perf_counter() - t0
print(f"\nLightGBM 15-fold: mean={mean_lgbm:.4f}  std={std_lgbm:.4f}  time={t_lgbm:.0f}s")

# ══════════════════════════════════════════════════════════════════════════════
# 8. Ablation (seed-0, 5 folds, fixed params)
# ══════════════════════════════════════════════════════════════════════════════
print()
print("=" * 80)
print("ABLATION (seed-0, 5 folds)")
print("=" * 80)

m_full_s0, s_full_s0, oof_lgbm_s0 = run_cv_lgbm(
    best_lgbm_params, cv_splits_s0, visits, y)
ablation = [("Full model (all blocks)", m_full_s0, s_full_s0)]


def _run_drop(params, splits, visits_df, y_series, drop_kws):
    visits_x   = visits_df[list(X_train.columns)].copy()
    fold_rmses = []
    for tr, val in splits:
        df_tr  = visits_x.iloc[tr]; df_val = visits_x.iloc[val]
        y_tr   = y_series.iloc[tr].values; y_val = y_series.iloc[val].values
        rb     = _rater_oof_bias(df_tr, y_tr)
        li, lp = None, None  # MixedLM skipped (train/val distribution mismatch)
        pa_tr  = _compute_pat_aggregates(df_tr)
        pa_val = _compute_pat_aggregates(df_val)
        Xtr    = build_features(df_tr,  rb, li, lp, pa_tr)
        Xval   = build_features(df_val, rb, li, lp, pa_val)
        def _drop(df):
            return df.drop(
                columns=[c for c in df.columns if any(k in c for k in drop_kws)],
                errors="ignore")
        Xtr = _drop(Xtr); Xval = _drop(Xval)
        for c in set(Xtr.columns) - set(Xval.columns): Xval[c] = np.nan
        Xval = Xval[Xtr.columns]
        cats = [c for c in Xtr.columns if str(Xtr[c].dtype) in ("object","string[python]","str")]
        Xtr  = _lgbm_encode(Xtr,  cats)
        Xval = _lgbm_encode(Xval, cats)
        lgb_tr  = lgb.Dataset(Xtr,  label=y_tr,  categorical_feature=cats, free_raw_data=False)
        lgb_val = lgb.Dataset(Xval, label=y_val, categorical_feature=cats, free_raw_data=False)
        mdl = lgb.train(params, lgb_tr, num_boost_round=800,
                        valid_sets=[lgb_val],
                        callbacks=[lgb.early_stopping(40, verbose=False),
                                   lgb.log_evaluation(-1)])
        fold_rmses.append(root_mean_squared_error(y_val, mdl.predict(Xval)))
    return float(np.nanmean(fold_rmses)), float(np.nanstd(fold_rmses))


for label, drop_kws in [
    ("No temporal neighbors", ["_prev", "_next", "_age_gap_"]),
    ("No visit position",     ["visit_rank", "time_since_first", "time_to_last", "rel_visit_pos"]),
    ("No missingness flags",  ["_na"]),
    ("No rater OOF bias",     ["rater_oof_bias"]),
    ("No MixedLM RE",         ["lme_intercept", "lme_pred_off"]),
]:
    m, s = _run_drop(best_lgbm_params, cv_splits_s0, visits, y, drop_kws)
    ablation.append((label, m, s))

print()
print(f"{'Block':<35s}  {'RMSE':>8s}  {'std':>7s}  {'vs full':>9s}")
print("-" * 65)
for label, rmse, std in ablation:
    delta  = rmse - m_full_s0
    marker = "  <-- BEST" if label == "Full model (all blocks)" else ""
    print(f"{label:<35s}  {rmse:>8.4f}  {std:>7.4f}  {delta:>+9.4f}{marker}")

# ══════════════════════════════════════════════════════════════════════════════
# 9. Ensemble: NNLS on seed-0 OOF (LightGBM + tabular_pipeline)
# ══════════════════════════════════════════════════════════════════════════════
print()
print("=" * 80)
print("ENSEMBLE (NNLS, seed-0 OOF)")
print("=" * 80)

mask = ~np.isnan(oof_lgbm_s0) & ~np.isnan(oof_ref_s0)
A    = np.column_stack([oof_lgbm_s0[mask], oof_ref_s0[mask]])
w, _ = nnls(A, y.values[mask])
w_sum = w.sum()
w = w / w_sum if w_sum > 0 else np.array([0.5, 0.5])
print(f"NNLS weights: lgbm={w[0]:.3f}  ref_tabular={w[1]:.3f}")

oof_ens      = w[0] * oof_lgbm_s0 + w[1] * oof_ref_s0
rmse_lgbm_s0 = root_mean_squared_error(y.values[mask], oof_lgbm_s0[mask])
rmse_ens     = root_mean_squared_error(y.values[mask], oof_ens[mask])
print(f"OOF RMSE lgbm (seed-0): {rmse_lgbm_s0:.4f}")
print(f"OOF RMSE ensemble:      {rmse_ens:.4f}")
use_ensemble = rmse_ens < rmse_lgbm_s0 - 0.005

# ══════════════════════════════════════════════════════════════════════════════
# 10. ICC of residuals (15-fold LightGBM OOF)
# ══════════════════════════════════════════════════════════════════════════════

def icc_intraclass(residuals, grp_labels):
    unique_groups = np.unique(grp_labels)
    k = len(unique_groups); n = len(residuals)
    grand_mean = residuals.mean()
    ssb = sum(len(residuals[grp_labels == g]) *
              (residuals[grp_labels == g].mean() - grand_mean) ** 2
              for g in unique_groups)
    ssw = sum(((residuals[grp_labels == g] -
                residuals[grp_labels == g].mean()) ** 2).sum()
              for g in unique_groups)
    n_bar = n / k
    ms_b  = ssb / (k - 1); ms_w = ssw / (n - k)
    return (ms_b - ms_w) / (ms_b + (n_bar - 1) * ms_w)


valid      = ~np.isnan(oof_lgbm)
resid_lgbm = y.values[valid] - oof_lgbm[valid]
icc_after  = icc_intraclass(resid_lgbm, groups.values[valid])
print(f"\nICC(residuals, patient): exp12=0.4549 → lgbm={icc_after:.4f}")

# ══════════════════════════════════════════════════════════════════════════════
# 11. Push to Hub (tabular_pipeline ref for skore compatibility)
# ══════════════════════════════════════════════════════════════════════════════
print()
print("=" * 80)
print("Pushing to Skore Hub")
print("=" * 80)
hub_report = skore.evaluate(pipe_ref, X_exp12, y, splitter=cv_splits_s0)
cfg = load_skore_credentials()
login(mode="hub")
project = Project(name="bobathon-esilv", mode="hub", workspace=cfg["workspace"])
project.put("13_best_model", hub_report)
print("Report pushed: 13_best_model")
hub_url = f"https://skore.probabl.ai/{cfg['workspace']}"

# ══════════════════════════════════════════════════════════════════════════════
# 12. Final model + submission
# ══════════════════════════════════════════════════════════════════════════════
print()
print("=" * 80)
print("Training final model on all training data")
print("=" * 80)

rater_bias_final = _rater_oof_bias(X_train, y.values)
# MixedLM RE not used: test patients unseen → intercept=0, causing distribution mismatch.
lme_int_final, lme_fp_final = None, None
pat_agg_full    = _compute_pat_aggregates(X_train)
pat_agg_test_13 = _compute_pat_aggregates(X_test)

X_train_feat = build_features(X_train, rater_bias_final, lme_int_final, lme_fp_final, pat_agg_full)
X_test_feat  = build_features(X_test,  rater_bias_final, lme_int_final, lme_fp_final, pat_agg_test_13)

for c in set(X_train_feat.columns) - set(X_test_feat.columns):
    X_test_feat[c] = np.nan
X_test_feat = X_test_feat[X_train_feat.columns]

assert "target"     not in X_train_feat.columns
assert "target"     not in X_test_feat.columns
assert "patient_id" not in X_train_feat.columns
assert "Index"      not in X_train_feat.columns

cat_cols_final = [c for c in X_train_feat.columns
                   if str(X_train_feat[c].dtype) in ("object", "string[python]", "str")]
X_train_feat = _lgbm_encode(X_train_feat, cat_cols_final)
X_test_feat  = _lgbm_encode(X_test_feat,  cat_cols_final)

lgb_full    = lgb.Dataset(X_train_feat, label=y.values,
                           categorical_feature=cat_cols_final, free_raw_data=False)
final_model = lgb.train(
    best_lgbm_params, lgb_full,
    num_boost_round=900,
    callbacks=[lgb.log_evaluation(-1)],
)
preds_test = final_model.predict(X_test_feat)

if use_ensemble and w[1] > 0.01:
    print(f"Applying ensemble (lgbm={w[0]:.3f}, ref={w[1]:.3f})")
    final_ref    = clone(pipe_ref)
    X_test_exp12 = _add_interactions_exp12(_join_agg(X_test, pat_agg_test_ref))
    final_ref.fit(X_exp12, y)
    preds_test   = w[0] * preds_test + w[1] * final_ref.predict(X_test_exp12)

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
submission.to_csv(out / "13_best_model.csv", index=False)
print(f"Submission: submissions/13_best_model.csv  ({len(submission)} rows)")

# ══════════════════════════════════════════════════════════════════════════════
# 13. Final summary
# ══════════════════════════════════════════════════════════════════════════════
print()
print("=" * 80)
print("FINAL SUMMARY")
print("=" * 80)
print(f"\nBaseline (exp12 tabular_pipeline, 5 folds): RMSE={mean_ref:.4f} ± {std_ref:.4f}")
print(f"LightGBM best (15 folds, 3 seeds):          RMSE={mean_lgbm:.4f} ± {std_lgbm:.4f}")
print(f"Gain LightGBM vs baseline (5-fold):         {mean_ref - m_full_s0:+.4f}")
print(f"\nICC(residuals, patient): exp12=0.4549 → lgbm={icc_after:.4f}")
print(f"Ensemble used: {'yes' if use_ensemble else 'no'}")
print(f"\nHub URL  : {hub_url}")
print(f"CSV file : submissions/13_best_model.csv")
print()
print("Ablation table:")
print(f"{'Block':<35s}  {'RMSE':>8s}  {'std':>7s}  {'vs full':>9s}")
print("-" * 65)
for label, rmse, std in ablation:
    delta = rmse - m_full_s0
    print(f"{label:<35s}  {rmse:>8.4f}  {std:>7.4f}  {delta:>+9.4f}")
print()
print("OK: experiment 13 complete.")
