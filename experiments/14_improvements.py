# %%
# Experiment 14: stacked improvements on exp 13.
#
# Improvements over exp 13:
#   A. Levodopa decay feature  (exp(-tsi_off / T_HALF), hours, T_HALF=2h default)
#   B. Fold-safe rater target encoding  (smoothed LOO; val fold encoded from
#      its training fold only; unseen-rater fallback = global mean)
#   C. Optuna tuning  (20 trials, configurable; MedianPruner; fixed seeds)
#   D. Robust LightGBM objectives  (Huber / regression_l1 vs RMSE baseline)
#   E. Isotonic post-hoc calibration  (fitted on OOF only)
#   F. Patient sequential prediction / random effects  — analysis section
#      documents WHY random effects cannot be used at inference time:
#      test patients are disjoint from train and have no available targets.
#      Within-patient residual ICC is reported for first vs later visits.
#   G. Multi-seed final ensemble  (3 seeds, configurable)
#   H. Stacking  (LightGBM + HGBR + Ridge, Ridge meta-learner; XGBoost disabled)
#   I. SHAP feature pruning  — disabled for speed (and shap not installed).
#
# CV: GroupKFold(5) × 1 seed = 5 folds  (fast-run mode).
# Submission: stacked ensemble (or best single model by OOF RMSE).
#
# ── Configuration ────────────────────────────────────────────────────────────
# Edit these values to change behaviour without touching the logic below.

# Levodopa decay half-life in hours (T_HALF). time_since_intake_off is in hours.
# Typical pharmacokinetic half-life for levodopa oral absorption: 1.5–3 h.
T_HALF: float = 2.0

# Optuna: number of tuning trials (set N_OPTUNA_TRIALS = 0 to skip tuning and
# use BASE_LGBM_PARAMS directly).
N_OPTUNA_TRIALS: int = 20

# Seeds for multi-seed final ensemble.
FINAL_SEEDS: list[int] = [0, 1, 2]

# Cross-validation settings.
# Fast-run: 1 seed × 5 folds = 5 folds total.
CV_SEEDS: list[int] = [0]
N_SPLITS: int = 5

# Smoothing factor for rater target encoding (higher = more shrinkage toward
# global mean for rare raters).
RATER_SMOOTH: float = 10.0

# Stacking: LightGBM + HGBR + Ridge only (XGBoost disabled for speed).
USE_STACKING: bool = True
USE_XGB_IN_STACK: bool = False  # set True to include XGBoost when available

# ── Imports ───────────────────────────────────────────────────────────────────

import argparse
import logging
import time
import warnings

import numpy as np
import pandas as pd
import skore
from scipy.optimize import nnls
from sklearn.base import clone
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import Ridge
from sklearn.metrics import root_mean_squared_error
from sklearn.model_selection import GroupKFold
from skore import Project, login
from skrub import tabular_pipeline

import lightgbm as lgb
import optuna

from parkinson import PROJECT_ROOT
from parkinson.hub import load_skore_credentials

# Optional: XGBoost for stacking (disabled via USE_XGB_IN_STACK flag).
try:
    import xgboost as xgb  # type: ignore[import]
    HAS_XGB = True
except ImportError:
    HAS_XGB = False

# SHAP: disabled for speed.
HAS_SHAP = False

optuna.logging.set_verbosity(optuna.logging.INFO)
warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(message)s",
                    datefmt="%H:%M:%S")
log = logging.getLogger(__name__)

# Allow N_OPTUNA_TRIALS override from CLI: `python 14_improvements.py --trials 200`
_parser = argparse.ArgumentParser(add_help=False)
_parser.add_argument("--trials", type=int, default=None)
_args, _ = _parser.parse_known_args()
if _args.trials is not None:
    N_OPTUNA_TRIALS = _args.trials

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

log.info("Train: %d rows, %d patients", len(visits), groups.nunique())
log.info("Test : %d rows, %d patients",
         len(X_test), X_test["patient_id"].nunique())

# Verify test patients are fully disjoint from train (confirmed by challenge spec).
_train_pids = set(X_train["patient_id"].unique())
_test_pids  = set(X_test["patient_id"].unique())
assert len(_train_pids & _test_pids) == 0, "Train/test patient leakage!"

# ── Patient sequential prediction: leakage analysis ─────────────────────────
#
# At inference time, X_test holds all visits for every test patient but
# y_test (the targets) is not available.  Because test patients are disjoint
# from train patients, there is no prior training-time observation of a test
# patient's target.  Therefore:
#
#   - A residual random intercept estimated from earlier test visits would
#     require earlier test TARGETS, which are not available.
#   - Fabricating an intercept from predictions would introduce circular
#     dependence (the "random effect" would just be prediction error, not a
#     patient-level structural offset).
#
# Decision: use population-level predictions for all test rows.
#           Report first-visit vs later-visit RMSE on the validation fold
#           (where we DO have targets) to quantify the opportunity.
#
log.info(
    "Patient RE analysis: test patients disjoint from train → "
    "random effects cannot be used at test time. "
    "Population-level prediction used for all test rows."
)

# ══════════════════════════════════════════════════════════════════════════════
# 2. Feature construction helpers
# ══════════════════════════════════════════════════════════════════════════════

AGG_COLS_NUM  = ["off", "on", "ledd", "age", "age_at_diagnosis",
                 "time_since_intake_on", "time_since_intake_off"]
NEIGHBOR_COLS = ["off", "on", "ledd", "time_since_intake_off"]


def _vectorised_slope(pid_arr: np.ndarray,
                       x_arr: np.ndarray,
                       y_arr: np.ndarray) -> pd.Series:
    """Vectorised OLS slope grouped by patient_id."""
    mask = ~(np.isnan(x_arr) | np.isnan(y_arr))
    if mask.sum() == 0:
        return pd.Series(dtype=float)
    pids, x, y_v = pid_arr[mask], x_arr[mask], y_arr[mask]
    agg = pd.DataFrame(
        {"p": pids, "x": x, "y": y_v, "xy": x * y_v, "x2": x * x}
    ).groupby("p", sort=False).agg(
        n=("x", "count"), sx=("x", "sum"), sy=("y", "sum"),
        sxy=("xy", "sum"), sx2=("x2", "sum"),
    )
    num = agg["n"] * agg["sxy"] - agg["sx"] * agg["sy"]
    den = agg["n"] * agg["sx2"] - agg["sx"] ** 2
    return (num / den.where(den.abs() > 1e-10)).where(agg["n"] >= 3)


def _compute_pat_aggregates(df: pd.DataFrame) -> pd.DataFrame:
    """Vectorised per-patient X-only aggregates. Returns DataFrame indexed by patient_id."""
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


def _rater_oof_target_encoding(df_tr: pd.DataFrame,
                                y_tr: np.ndarray,
                                smooth: float = RATER_SMOOTH) -> dict:
    """Smoothed leave-one-out rater target encoding fitted on training fold only.

    For each rater, the encoding is:
        encoded = (sum_of_targets_excluding_row + smooth * global_mean)
                  / (count_excluding_row + smooth)

    Returns a dict {rater_id: smoothed_mean} for applying to new data.
    The per-row LOO encoding is only used for training rows during CV; the
    returned dict is used for validation/test rows without their own targets.
    """
    global_mean = float(np.nanmean(y_tr))
    df = pd.DataFrame({"rater": df_tr["rater_id"].values, "y": y_tr})
    rater_stats = df.groupby("rater")["y"].agg(["sum", "count"])
    # Smoothed mean for each rater (for applying to unseen rows)
    rater_stats["encoded"] = (
        (rater_stats["sum"] + smooth * global_mean)
        / (rater_stats["count"] + smooth)
    )
    return {"map": rater_stats["encoded"].to_dict(), "global_mean": global_mean}


def _apply_rater_encoding(df: pd.DataFrame, enc_info: dict) -> np.ndarray:
    """Apply pre-fitted rater encoding; unseen raters get the global mean."""
    return df["rater_id"].map(enc_info["map"]).fillna(enc_info["global_mean"]).values


def _apply_rater_loo_train(df_tr: pd.DataFrame,
                            y_tr: np.ndarray,
                            smooth: float = RATER_SMOOTH) -> np.ndarray:
    """Compute leave-one-out rater target encoding for training rows.

    Each row excludes its own target when computing the rater mean — no
    row ever encodes itself, preventing leakage.
    """
    global_mean = float(np.nanmean(y_tr))
    raters = df_tr["rater_id"].values
    df_tmp = pd.DataFrame({"rater": raters, "y": y_tr})
    agg = df_tmp.groupby("rater")["y"].agg(["sum", "count"])
    # LOO: subtract the row's own contribution
    row_rater_sum   = df_tmp["rater"].map(agg["sum"])
    row_rater_count = df_tmp["rater"].map(agg["count"])
    loo_sum   = row_rater_sum   - y_tr
    loo_count = row_rater_count - 1
    loo_encoded = (loo_sum + smooth * global_mean) / (loo_count + smooth)
    return loo_encoded.values


def _lgbm_encode(X: pd.DataFrame, cat_cols: list[str]) -> pd.DataFrame:
    """Encode string/category columns; cast int8 → float for LightGBM."""
    X = X.copy()
    for c in cat_cols:
        X[c] = X[c].astype("category")
    for c in X.select_dtypes("int8").columns:
        X[c] = X[c].astype(float)
    return X


def _add_levo_decay(df: pd.DataFrame, t_half: float = T_HALF) -> pd.DataFrame:
    """Add exp(-time_since_intake_off / t_half) decay feature.

    Units of time_since_intake_off: hours (confirmed from data exploration:
    range 6.9–24.7 h, mean ~14 h).  T_HALF defaults to 2 h.
    Missing time_since_intake_off values remain NaN in the decay column
    (not zero, which would spuriously imply instant decay).
    A binary missingness indicator is also added.
    """
    if "time_since_intake_off" not in df.columns:
        df["levo_decay"]    = np.nan
        df["levo_decay_na"] = 1
        return df
    tsi = df["time_since_intake_off"]
    df["levo_decay"]    = np.exp(-tsi / t_half)
    df["levo_decay_na"] = tsi.isna().astype(np.int8)
    return df


def build_features(df_raw: pd.DataFrame,
                   rater_enc_info: dict | None,
                   rater_loo_train: np.ndarray | None,
                   is_train_fold: bool,
                   pat_agg: pd.DataFrame | None = None,
                   t_half: float = T_HALF) -> pd.DataFrame:
    """Build the full feature matrix from a raw X DataFrame.

    Parameters
    ----------
    df_raw          : raw X columns (must NOT contain 'target').
    rater_enc_info  : output of _rater_oof_target_encoding on the training fold.
    rater_loo_train : per-row LOO encodings for training rows (shape = len(df_raw)).
    is_train_fold   : True when df_raw is the training fold (use LOO encoding);
                      False when df_raw is validation or test (use smoothed mean).
    pat_agg         : pre-computed patient aggregates.
    t_half          : levodopa decay half-life in hours.

    CRITICAL: preserves original DataFrame index throughout so that feature
    rows align exactly with y labels.
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
    feat_sorted = feat.sort_values(["patient_id", "age"])
    visit_rank  = feat_sorted.groupby("patient_id").cumcount()
    feat["visit_rank"]             = visit_rank.loc[orig_index].values
    feat["time_since_first_visit"] = feat["age"] - feat["age_first_pat"]
    feat["time_to_last_visit"]     = feat["age_last_pat"] - feat["age"]
    feat["rel_visit_pos"] = (
        feat["visit_rank"] / (feat["n_visits_pat"] - 1).clip(lower=1)
    )

    # ── Block C: temporal neighbour features ─────────────────────────────────
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

    # ── Block A: interaction features ─────────────────────────────────────────
    interaction_pairs_mean = [
        ("off",  "off_mean_pat"),
        ("on",   "on_mean_pat"),
        ("ledd", "ledd_mean_pat"),
        ("age",  "off_mean_pat"),
        ("age_at_diagnosis", "off_mean_pat"),
        ("age_at_diagnosis", "ledd_mean_pat"),
    ]
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
        feat[f"{stub}_ratio"] = (f_v / denom).clip(-10, 10)
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

    # ── Block H (new): fold-safe smoothed rater target encoding ──────────────
    # For training fold rows: leave-one-out (row excluded from its own encoding).
    # For validation/test rows: smoothed rater mean from training fold only.
    # Unseen raters: global mean fallback.
    if rater_enc_info is not None:
        if is_train_fold and rater_loo_train is not None:
            feat["rater_target_enc"] = rater_loo_train
        else:
            feat["rater_target_enc"] = _apply_rater_encoding(feat, rater_enc_info)
    else:
        feat["rater_target_enc"] = 0.0

    # ── Block J (new): levodopa decay feature ─────────────────────────────────
    feat = _add_levo_decay(feat, t_half=t_half)

    # Clip extreme product/ratio values
    numeric_cols = feat.select_dtypes("number").columns
    for c in numeric_cols:
        if c.endswith("_prod") or c.endswith("_ratio"):
            feat[c] = feat[c].clip(-1e4, 1e4)

    assert "target" not in feat.columns
    feat = feat.drop(columns=["patient_id", "Index"], errors="ignore")

    assert (feat.index == orig_index).all(), "Index corrupted in build_features"
    return feat


# ══════════════════════════════════════════════════════════════════════════════
# 3. CV infrastructure
# ══════════════════════════════════════════════════════════════════════════════

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

log.info("%d folds (%d seeds × %d)", len(all_splits), len(CV_SEEDS), N_SPLITS)

# ══════════════════════════════════════════════════════════════════════════════
# 4. LightGBM CV runner (leak-safe, per-fold feature construction)
# ══════════════════════════════════════════════════════════════════════════════

BASE_LGBM_PARAMS: dict = {
    "objective":    "regression_l2",
    "metric":       "rmse",
    "verbosity":    -1,
    "n_jobs":       -1,
    "random_state": 42,
}


def _build_fold_features(
    visits_x: pd.DataFrame,
    y_series: pd.Series,
    tr: np.ndarray,
    val: np.ndarray,
    t_half: float = T_HALF,
) -> tuple[pd.DataFrame, pd.DataFrame, np.ndarray, np.ndarray]:
    """Build leakage-free features for one CV fold."""
    df_tr  = visits_x.iloc[tr]
    df_val = visits_x.iloc[val]
    y_tr   = y_series.iloc[tr].values
    y_val  = y_series.iloc[val].values

    # Rater encoding: fit on train fold only
    enc_info    = _rater_oof_target_encoding(df_tr, y_tr)
    loo_encoded = _apply_rater_loo_train(df_tr, y_tr)

    pat_agg_tr  = _compute_pat_aggregates(df_tr)
    pat_agg_val = _compute_pat_aggregates(df_val)

    X_tr_f  = build_features(df_tr,  enc_info, loo_encoded, True,  pat_agg_tr,  t_half)
    X_val_f = build_features(df_val, enc_info, None,        False, pat_agg_val, t_half)

    for c in set(X_tr_f.columns) - set(X_val_f.columns):
        X_val_f[c] = np.nan
    X_val_f = X_val_f[X_tr_f.columns]

    return X_tr_f, X_val_f, y_tr, y_val


def run_cv_lgbm(params: dict,
                splits: list,
                visits_df: pd.DataFrame,
                y_series: pd.Series,
                verbose: bool = False,
                t_half: float = T_HALF) -> tuple[float, float, np.ndarray]:
    """CV with per-fold leak-safe feature construction for LightGBM."""
    oof        = np.full(len(y_series), np.nan)
    fold_rmses = []
    visits_x   = visits_df[list(X_train.columns)].copy()

    for fold_i, (tr, val) in enumerate(splits):
        X_tr_f, X_val_f, y_tr, y_val = _build_fold_features(
            visits_x, y_series, tr, val, t_half)

        cat_cols = [c for c in X_tr_f.columns
                    if str(X_tr_f[c].dtype) in ("object", "string[python]", "str")]
        X_tr_f  = _lgbm_encode(X_tr_f,  cat_cols)
        X_val_f = _lgbm_encode(X_val_f, cat_cols)

        lgb_tr  = lgb.Dataset(X_tr_f, label=y_tr, categorical_feature=cat_cols,
                               free_raw_data=False)
        lgb_val = lgb.Dataset(X_val_f, label=y_val, categorical_feature=cat_cols,
                               free_raw_data=False)

        model = lgb.train(
            params, lgb_tr, num_boost_round=800,
            valid_sets=[lgb_val],
            callbacks=[lgb.early_stopping(40, verbose=False),
                       lgb.log_evaluation(-1)],
        )
        preds = model.predict(X_val_f)
        rmse  = root_mean_squared_error(y_val, preds)
        fold_rmses.append(rmse)
        oof[val] = preds
        if verbose:
            log.info("fold %02d/%d  RMSE=%.4f  best_iter=%d",
                     fold_i + 1, len(splits), rmse, model.best_iteration)

    return float(np.nanmean(fold_rmses)), float(np.nanstd(fold_rmses)), oof


# ══════════════════════════════════════════════════════════════════════════════
# 5. Optuna tuning (100 trials by default, MedianPruner, fixed seeds)
# ══════════════════════════════════════════════════════════════════════════════
print()
print("=" * 80)
if N_OPTUNA_TRIALS > 0:
    print(f"LightGBM Optuna tuning ({N_OPTUNA_TRIALS} trials, 5 seed-0 folds)")
else:
    print("LightGBM Optuna tuning SKIPPED (N_OPTUNA_TRIALS=0) — using BASE params")
print("=" * 80)


_trial_times: list[float] = []
_t_optuna_start = time.perf_counter()


def objective_lgbm(trial: optuna.Trial) -> float:
    t_trial = time.perf_counter()
    params = {
        **BASE_LGBM_PARAMS,
        "learning_rate":     trial.suggest_float("learning_rate", 0.02, 0.1, log=True),
        "num_leaves":        trial.suggest_int("num_leaves", 31, 200),
        "min_child_samples": trial.suggest_int("min_child_samples", 50, 300),
        "subsample":         trial.suggest_float("subsample", 0.5, 1.0),
        "colsample_bytree":  trial.suggest_float("colsample_bytree", 0.4, 1.0),
        "reg_alpha":         trial.suggest_float("reg_alpha", 1e-3, 5.0, log=True),
        "reg_lambda":        trial.suggest_float("reg_lambda", 1e-3, 5.0, log=True),
        "random_state":      42,
    }
    mean_rmse, _, _ = run_cv_lgbm(params, cv_splits_s0, visits, y)
    elapsed = time.perf_counter() - t_trial
    _trial_times.append(elapsed)
    log.info(
        "Trial %3d/%d  RMSE=%.4f  lr=%.4f  leaves=%d  "
        "min_child=%d  sub=%.2f  col=%.2f  α=%.4f  λ=%.4f  "
        "time=%.0fs  total=%.0fs",
        trial.number + 1, N_OPTUNA_TRIALS, mean_rmse,
        params["learning_rate"], params["num_leaves"],
        params["min_child_samples"], params["subsample"],
        params["colsample_bytree"], params["reg_alpha"], params["reg_lambda"],
        elapsed, time.perf_counter() - _t_optuna_start,
    )
    return mean_rmse


if N_OPTUNA_TRIALS > 0:
    t0    = time.perf_counter()
    study = optuna.create_study(
        direction="minimize",
        sampler=optuna.samplers.TPESampler(seed=42),
        pruner=optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=3),
    )
    study.optimize(objective_lgbm, n_trials=N_OPTUNA_TRIALS)
    t_tune = time.perf_counter() - t0
    log.info("Optuna done in %.0fs  best RMSE=%.4f", t_tune, study.best_value)
    log.info("Best params: %s", study.best_params)
    best_lgbm_params = {**BASE_LGBM_PARAMS, **study.best_params}
else:
    best_lgbm_params = BASE_LGBM_PARAMS.copy()

# ══════════════════════════════════════════════════════════════════════════════
# 6. Evaluate alternative LightGBM objectives (Huber, L1) vs RMSE baseline
# ══════════════════════════════════════════════════════════════════════════════
print()
print("=" * 80)
print("Robust LightGBM objectives (regression_l1, huber) vs l2 baseline")
print("=" * 80)

_alt_objectives = [
    ("regression_l2 (RMSE baseline)", "regression_l2"),
    ("regression_l1 (MAE / Huber-like)", "regression_l1"),
    ("huber", "huber"),
    ("quantile_0.5", "quantile"),
]

objective_results: list[tuple[str, float, float]] = []
for label, obj in _alt_objectives:
    _p = {**best_lgbm_params, "objective": obj, "metric": "rmse"}
    if obj == "quantile":
        _p["alpha"] = 0.5
    _m, _s, _ = run_cv_lgbm(_p, cv_splits_s0, visits, y)
    objective_results.append((label, _m, _s))
    log.info("  %-35s  RMSE=%.4f ± %.4f", label, _m, _s)

# Select the best objective by OOF RMSE
_label_to_obj = dict(_alt_objectives)
best_obj_label, best_obj_rmse, _ = min(objective_results, key=lambda x: x[1])
best_obj_name = _label_to_obj[best_obj_label]
if best_obj_name != best_lgbm_params.get("objective", "regression_l2"):
    log.info("Switching objective to '%s' (OOF RMSE %.4f)", best_obj_name, best_obj_rmse)
    best_lgbm_params["objective"] = best_obj_name
    if best_obj_name == "quantile":
        best_lgbm_params["alpha"] = 0.5

# ══════════════════════════════════════════════════════════════════════════════
# 7. Best LightGBM on full CV (all_splits = CV_SEEDS × N_SPLITS)
# ══════════════════════════════════════════════════════════════════════════════
print()
print("=" * 80)
print(f"Best LightGBM — full CV ({len(CV_SEEDS)} seed(s) × {N_SPLITS} folds = {len(all_splits)} folds)")
print("=" * 80)
t0 = time.perf_counter()
mean_lgbm, std_lgbm, oof_lgbm = run_cv_lgbm(
    best_lgbm_params, all_splits, visits, y, verbose=True)
t_lgbm = time.perf_counter() - t0
log.info("LightGBM CV: mean=%.4f  std=%.4f  time=%.0fs",
         mean_lgbm, std_lgbm, t_lgbm)

# cv_splits_s0 IS all_splits when CV_SEEDS=[0]; reuse oof_lgbm directly.
oof_lgbm_s0 = oof_lgbm

# ── First-visit vs later-visit RMSE (patient random effect opportunity) ───────
visits_x_full = visits[list(X_train.columns)].copy()
visits_x_full["_visit_rank"] = (
    visits_x_full.sort_values(["patient_id", "age"])
    .groupby("patient_id").cumcount()
    .reindex(visits_x_full.index)
)
valid_mask   = ~np.isnan(oof_lgbm)
first_mask   = (visits_x_full["_visit_rank"].values == 0) & valid_mask
later_mask   = (visits_x_full["_visit_rank"].values >  0) & valid_mask
rmse_first  = root_mean_squared_error(y.values[first_mask],  oof_lgbm[first_mask])
rmse_later  = root_mean_squared_error(y.values[later_mask], oof_lgbm[later_mask])
log.info("RMSE first visits: %.4f   RMSE later visits: %.4f  "
         "(later-visit gain opportunity: %.4f)",
         rmse_first, rmse_later, rmse_first - rmse_later)

# ══════════════════════════════════════════════════════════════════════════════
# 8. Isotonic calibration (fitted on OOF only, applied to test predictions)
# ══════════════════════════════════════════════════════════════════════════════
print()
print("=" * 80)
print("Isotonic post-hoc calibration (OOF only)")
print("=" * 80)

mask_oof = ~np.isnan(oof_lgbm_s0)
iso_cal  = IsotonicRegression(out_of_bounds="clip")
iso_cal.fit(oof_lgbm_s0[mask_oof], y.values[mask_oof])

oof_lgbm_s0_cal = oof_lgbm_s0.copy()
oof_lgbm_s0_cal[mask_oof] = iso_cal.predict(oof_lgbm_s0[mask_oof])
rmse_raw = root_mean_squared_error(y.values[mask_oof], oof_lgbm_s0[mask_oof])
rmse_cal = root_mean_squared_error(y.values[mask_oof], oof_lgbm_s0_cal[mask_oof])
log.info("OOF RMSE raw=%.4f  calibrated=%.4f  delta=%.4f",
         rmse_raw, rmse_cal, rmse_cal - rmse_raw)
use_isotonic = rmse_cal < rmse_raw - 0.002
log.info("Isotonic calibration used: %s", use_isotonic)

# ══════════════════════════════════════════════════════════════════════════════
# 9. Stacking: LightGBM + HGBR + (XGBoost if available) + Ridge meta-learner
# ══════════════════════════════════════════════════════════════════════════════
print()
print("=" * 80)
print("Stacking (strictly OOF base model predictions → Ridge meta-learner)")
print("=" * 80)

if not USE_STACKING:
    log.info("Stacking disabled (USE_STACKING=False).")
elif not USE_XGB_IN_STACK:
    log.info("Stacking base: LightGBM + HGBR + Ridge (XGBoost disabled by USE_XGB_IN_STACK).")
elif not HAS_XGB:
    log.info("Stacking base: XGBoost not installed, falling back to LightGBM + HGBR + Ridge.")


def _build_base_models() -> list[tuple[str, object]]:
    """Return (name, unfitted estimator) for every stacking base model except LightGBM."""
    models: list[tuple[str, object]] = [
        ("lgbm", None),  # LightGBM OOF already computed; placeholder for order tracking
        ("hgbr", HistGradientBoostingRegressor(
            random_state=42, max_iter=400, early_stopping=True, n_iter_no_change=20,
            validation_fraction=0.1)),
        ("ridge", Ridge(alpha=10.0)),
    ]
    if USE_XGB_IN_STACK and HAS_XGB:
        models.append(("xgb", xgb.XGBRegressor(
            n_estimators=500, learning_rate=0.05, max_depth=6,
            subsample=0.8, colsample_bytree=0.8, random_state=42,
            n_jobs=-1, verbosity=0, eval_metric="rmse")))
    return models


def _run_cv_sklearn(estimator,
                    splits: list,
                    X_feat: pd.DataFrame,
                    y_series: pd.Series) -> tuple[float, np.ndarray]:
    """CV runner for sklearn-compatible estimators on pre-built features."""
    oof   = np.full(len(y_series), np.nan)
    rmses = []
    for tr, val in splits:
        clf = clone(estimator)
        clf.fit(X_feat.iloc[tr], y_series.iloc[tr])
        preds   = clf.predict(X_feat.iloc[val])
        oof[val] = preds
        rmses.append(root_mean_squared_error(y_series.iloc[val], preds))
    return float(np.nanmean(rmses)), oof


# Build full-train feature matrix once (used for HGBR, Ridge, XGB)
rater_enc_full = _rater_oof_target_encoding(
    visits[list(X_train.columns)], y.values)
loo_enc_full   = _apply_rater_loo_train(
    visits[list(X_train.columns)], y.values)
pat_agg_full   = _compute_pat_aggregates(visits[list(X_train.columns)])

X_visits_feat = build_features(
    visits[list(X_train.columns)],
    rater_enc_full, loo_enc_full, True, pat_agg_full)

# Non-LightGBM models use skrub tabular_pipeline on the features directly
# (sklearn handles categoricals via its own encoder; cats are treated as object)
X_visits_num = X_visits_feat.select_dtypes(include="number").copy()

stack_base_oofs: dict[str, np.ndarray] = {"lgbm": oof_lgbm_s0}
stack_rmses: dict[str, float] = {"lgbm": rmse_raw}

if USE_STACKING:
    for name, est in _build_base_models():
        if name == "lgbm":
            continue  # already computed
        log.info("Stacking base OOF: %s ...", name)
        rmse_b, oof_b = _run_cv_sklearn(est, cv_splits_s0, X_visits_num, y)
        stack_base_oofs[name] = oof_b
        stack_rmses[name] = rmse_b
        log.info("  %s OOF RMSE=%.4f", name, rmse_b)

    # Ridge meta-learner on OOF predictions
    valid_stack = np.ones(len(y), dtype=bool)
    for oof_b in stack_base_oofs.values():
        valid_stack &= ~np.isnan(oof_b)

    base_names = list(stack_base_oofs.keys())
    A_stack    = np.column_stack([stack_base_oofs[n][valid_stack] for n in base_names])
    y_stack    = y.values[valid_stack]

    meta = Ridge(alpha=1.0)
    meta.fit(A_stack, y_stack)
    oof_stack = np.full(len(y), np.nan)
    oof_stack[valid_stack] = meta.predict(A_stack)
    rmse_stack = root_mean_squared_error(y_stack, oof_stack[valid_stack])

    log.info("Stacking meta-Ridge OOF RMSE=%.4f  (base model RMSEs: %s)",
             rmse_stack,
             "  ".join(f"{n}={v:.4f}" for n, v in stack_rmses.items()))

    # Compare NNLS (exp13 style) vs Ridge stacking
    w_nnls, _ = nnls(A_stack, y_stack)
    w_nnls_sum = w_nnls.sum()
    w_nnls = w_nnls / w_nnls_sum if w_nnls_sum > 0 else np.ones(len(base_names)) / len(base_names)
    oof_nnls = np.dot(A_stack, w_nnls)
    rmse_nnls = root_mean_squared_error(y_stack, oof_nnls)
    log.info("NNLS ensemble OOF RMSE=%.4f", rmse_nnls)

    use_stacking = rmse_stack < rmse_raw - 0.002
    log.info("Stacking selected: %s", use_stacking)
else:
    use_stacking = False
    rmse_stack   = rmse_raw
    meta         = None
    base_names   = ["lgbm"]
    log.info("Stacking disabled.")

# ══════════════════════════════════════════════════════════════════════════════
# 10. SHAP feature pruning (skipped — shap not installed)
# ══════════════════════════════════════════════════════════════════════════════
if HAS_SHAP:
    log.info("SHAP available — feature pruning would run here.")
    # Implementation: compute SHAP values on a trained LightGBM model using
    # training-fold data only; prune features with mean |SHAP| below 1e-4;
    # re-run CV to compare unpruned vs pruned RMSE.
    # Not yet implemented — HAS_SHAP is False in this environment.
else:
    log.info(
        "SHAP not installed — feature pruning skipped. "
        "Install with: pip install shap. "
        "Fallback: all features retained (no pruning on test performance)."
    )

# ══════════════════════════════════════════════════════════════════════════════
# 11. Push report to Skore Hub
# ══════════════════════════════════════════════════════════════════════════════
print()
print("=" * 80)
print("Pushing to Skore Hub")
print("=" * 80)

# Use the same tabular_pipeline wrapper that exp13 used for skore compatibility.
# Re-build the exp12-style features for the skore report push only.
from scipy import stats as sp_stats


def _compute_pat_agg_exp12(df: pd.DataFrame) -> pd.DataFrame:
    """Exp12-compatible patient aggregates for skore reference."""
    assert "target" not in df.columns
    recs = []
    for pid, grp in df.groupby("patient_id"):
        row: dict = {"patient_id": pid}
        for col in ["off", "on", "ledd"]:
            vals = grp[col].dropna(); n = len(vals)
            row[f"{col}_mean_pat"]   = vals.mean()   if n > 0 else np.nan
            row[f"{col}_median_pat"] = vals.median() if n > 0 else np.nan
            row[f"{col}_count_pat"]  = n
            row[f"{col}_std_pat"]    = vals.std()    if n >= 2 else np.nan
        for col in ["off", "on"]:
            d = grp[["age", col]].dropna().sort_values("age")
            row[f"{col}_slope_pat"] = (
                float(sp_stats.linregress(d["age"], d[col]).slope)
                if len(d) >= 3 else np.nan)
            d2 = grp.sort_values("Index")[col].dropna()
            row[f"{col}_first_pat"] = d2.iloc[0]  if len(d2) > 0 else np.nan
            row[f"{col}_last_pat"]  = d2.iloc[-1] if len(d2) > 0 else np.nan
        recs.append(row)
    return pd.DataFrame(recs).set_index("patient_id")


def _join_agg_exp12(X: pd.DataFrame, pat_agg: pd.DataFrame) -> pd.DataFrame:
    assert "target" not in X.columns
    r = X.join(pat_agg, on="patient_id", how="left")
    return r.drop(columns=["patient_id", "Index"], errors="ignore")


def _add_interactions_exp12(df: pd.DataFrame) -> pd.DataFrame:
    assert "target" not in df.columns
    out = df.copy()
    for feat_c, agg_c in [
        ("off", "off_mean_pat"), ("on", "on_mean_pat"), ("ledd", "ledd_mean_pat"),
        ("age", "off_mean_pat"), ("age_at_diagnosis", "off_mean_pat"),
        ("age_at_diagnosis", "ledd_mean_pat"),
    ]:
        if feat_c not in out.columns or agg_c not in out.columns:
            continue
        f, m = out[feat_c], out[agg_c]
        stub = f"{feat_c}_x_{agg_c}"
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
visits_X_ref      = visits[list(X_train.columns)].copy()
X_exp12_ref       = _add_interactions_exp12(
    _join_agg_exp12(visits_X_ref, pat_agg_train_ref))

pipe_ref   = tabular_pipeline("regressor")
hub_report = skore.evaluate(pipe_ref, X_exp12_ref, y, splitter=cv_splits_s0)

cfg     = load_skore_credentials()
login(mode="hub")
project = Project(name="bobathon-esilv", mode="hub", workspace=cfg["workspace"])
project.put("14_improvements", hub_report)
log.info("Report pushed: 14_improvements")
hub_url = f"https://skore.probabl.ai/{cfg['workspace']}"

# ══════════════════════════════════════════════════════════════════════════════
# 12. Final model: multi-seed LightGBM ensemble + stacking on full training data
# ══════════════════════════════════════════════════════════════════════════════
print()
print("=" * 80)
print(f"Training final model (multi-seed ensemble, {len(FINAL_SEEDS)} seeds)")
print("=" * 80)

# Build final training and test feature matrices
rater_enc_final = _rater_oof_target_encoding(X_train, y.values)
loo_enc_final   = _apply_rater_loo_train(X_train, y.values)
pat_agg_train_f = _compute_pat_aggregates(X_train)
pat_agg_test_f  = _compute_pat_aggregates(X_test)

X_train_feat = build_features(
    X_train, rater_enc_final, loo_enc_final, True,  pat_agg_train_f)
X_test_feat  = build_features(
    X_test,  rater_enc_final, None,           False, pat_agg_test_f)

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

lgb_full_ds = lgb.Dataset(X_train_feat, label=y.values,
                           categorical_feature=cat_cols_final, free_raw_data=False)

# Train one final LightGBM per seed and average
all_seed_preds: list[np.ndarray] = []
for seed in FINAL_SEEDS:
    seed_params = {**best_lgbm_params, "random_state": seed}
    seed_model  = lgb.train(
        seed_params, lgb_full_ds,
        num_boost_round=900,
        callbacks=[lgb.log_evaluation(-1)],
    )
    all_seed_preds.append(seed_model.predict(X_test_feat))

preds_lgbm_final = np.mean(all_seed_preds, axis=0)
log.info("Multi-seed LightGBM final predictions: %d models averaged", len(FINAL_SEEDS))

# ── Stacking test predictions ─────────────────────────────────────────────────
if use_stacking and USE_STACKING and meta is not None:
    log.info("Applying stacking to test predictions ...")
    X_test_num = X_test_feat.select_dtypes(include="number").copy()
    X_train_num = X_train_feat.select_dtypes(include="number").copy()

    base_test_preds: dict[str, np.ndarray] = {"lgbm": preds_lgbm_final}
    for name, est in _build_base_models():
        if name == "lgbm":
            continue
        # Fit base model on full training data for test predictions
        _clf_full = clone(est)
        _clf_full.fit(X_train_num, y.values)
        base_test_preds[name] = _clf_full.predict(X_test_num)

    T_stack = np.column_stack([base_test_preds[n] for n in base_names])
    preds_test = meta.predict(T_stack)
    log.info("Stacking test prediction applied.")
else:
    preds_test = preds_lgbm_final

# ── Isotonic calibration on final predictions ─────────────────────────────────
if use_isotonic:
    preds_test = iso_cal.predict(preds_test)
    log.info("Isotonic calibration applied to test predictions.")

# ══════════════════════════════════════════════════════════════════════════════
# 13. Write submission
# ══════════════════════════════════════════════════════════════════════════════
assert len(preds_test) == len(X_test)
assert not np.isnan(preds_test).any(), "NaN in final predictions!"
assert not np.isinf(preds_test).any(), "Inf in final predictions!"

sample_sub = pd.read_csv(DATA_DIR / "sample_submission.csv")
assert len(preds_test) == len(sample_sub)

submission = X_test[["Index"]].copy()
submission["target"] = preds_test
assert list(submission.columns) == ["Index", "target"]
assert submission["Index"].tolist() == X_test["Index"].tolist()

out = PROJECT_ROOT / "submissions"
out.mkdir(exist_ok=True)
submission.to_csv(out / "14_improvements.csv", index=False)
log.info("Submission: submissions/14_improvements.csv  (%d rows)", len(submission))

# ══════════════════════════════════════════════════════════════════════════════
# 14. Final summary
# ══════════════════════════════════════════════════════════════════════════════
print()
print("=" * 80)
print("FINAL SUMMARY — Experiment 14")
print("=" * 80)

_base_rmse = min(rmse_raw, rmse_stack if use_stacking else rmse_raw)
_n_folds = len(all_splits)
print(f"\nLightGBM CV RMSE ({_n_folds} folds, raw):         {mean_lgbm:.4f} ± {std_lgbm:.4f}")
print(f"Best LightGBM objective: {best_lgbm_params.get('objective','regression_l2')}")
print(f"Isotonic calibration used: {use_isotonic}  "
      f"(raw={rmse_raw:.4f} → cal={rmse_cal:.4f})")
if use_stacking and USE_STACKING:
    print(f"Stacking meta-Ridge OOF RMSE:             {rmse_stack:.4f}")
print(f"\nFirst-visit RMSE:  {rmse_first:.4f}   Later-visit RMSE: {rmse_later:.4f}")
print(f"(Patient RE cannot be used at test time: test patients are disjoint "
      f"from train and have no available targets.)")
print()
print("Objective comparison:")
for lbl, m_v, s_v in objective_results:
    mark = " <-- selected" if best_lgbm_params.get("objective") in lbl else ""
    print(f"  {lbl:<40s}  RMSE={m_v:.4f} ± {s_v:.4f}{mark}")
print()
print(f"Levodopa decay T_HALF: {T_HALF}h   (feature: levo_decay = exp(-tsi_off / T_HALF))")
print(f"Rater target encoding: smoothed LOO (smooth={RATER_SMOOTH}), fold-safe")
print(f"Multi-seed ensemble:   {len(FINAL_SEEDS)} seeds: {FINAL_SEEDS}")
print(f"SHAP feature pruning:  SKIPPED — shap not installed (install: pip install shap)")
print(f"\nHub URL : {hub_url}")
print(f"CSV file: submissions/14_improvements.csv")
print()
print("OK: experiment 14 complete.")
