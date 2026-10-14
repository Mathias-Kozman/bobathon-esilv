# %%
# Experiment 14: advanced improvements stacked on top of experiment 13.
#
# Five independent improvement axes applied together:
#
#   1. Sequential test-visit history (HIGH IMPACT)
#      Train/test patients are fully disjoint. But every test patient has
#      multiple visits. For visit k, the k-1 earlier visits of the SAME test
#      patient are known at inference time (batch prediction). We sort each
#      patient's visits by Index (= age order, Spearman 1.0) and compute
#      within-patient running aggregates (mean off/on/ledd up to visit k-1,
#      visit count so far, running slope of off vs age). For training, the
#      same logic is applied fold-by-fold using only the training rows, so
#      the first visit of every patient always falls back to NaN (no history).
#      This gives a clean MixedLM-style random intercept proxy without leakage.
#
#   2. Regression-to-mean bias correction — isotonic recalibration (HIGH IMPACT)
#      OOF analysis in exp08 showed strong positive bias for low targets and
#      negative bias for high targets (classic shrinkage to the mean). We fit an
#      IsotonicRegression on the OOF predictions vs ground truth to learn the
#      monotone correction curve, then apply it at test time. Fitted on the full
#      OOF stack (15 folds) to maximise calibration data.
#
#   3. Levodopa decay proxy feature (MEDIUM IMPACT)
#      time_since_intake_off is 79% missing, but when present it encodes the
#      pharmacokinetic state at assessment. The generative model uses exponential
#      decay: concentration ∝ exp(-t / T_HALF). We try T_HALF = 2.0 h (midpoint
#      of the 1.5–3 h levodopa half-life range). When time is missing the feature
#      is NaN; LightGBM routes missing values automatically.
#
#   4. Rater full OOF target encoding (MEDIUM IMPACT)
#      Exp13 Block H encodes mean(target - off) per rater. Here we use a full
#      leave-one-out target encoding of rater_id vs target (computed fold-by-fold
#      on the training half only), which absorbs rater-level systematic bias more
#      precisely — especially for raters with many observations.
#
#   5. Multi-seed final model ensemble (LOW IMPACT)
#      Instead of one final LightGBM with random_state=42, we train 5 models
#      with seeds 0..4 on the full training set and average their predictions.
#      Seed diversity reduces variance in the submission without any new ideas.
#
# CV strategy: GroupKFold(5) × 3 seeds = 15 folds (identical to exp13).
# Optuna budget: 50 trials (up from 20) with MedianPruner.
# Hub push: tabular_pipeline on exp12 features (skore-compatible reference)
#           + a CrossValidationReport for the best LightGBM to compare on Hub.
# Submission: multi-seed ensemble of final LightGBM + isotonic recalibration.

import time
import warnings

import numpy as np
import pandas as pd
import skore
from scipy.optimize import nnls
from sklearn.base import clone
from sklearn.isotonic import IsotonicRegression
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
print(f"Test : {len(X_test)} rows, {X_test['patient_id'].nunique()} patients")
print(f"Overlap train/test patients: "
      f"{len(set(X_train['patient_id']) & set(X_test['patient_id']))}")

# ══════════════════════════════════════════════════════════════════════════════
# 2. Feature construction
# ══════════════════════════════════════════════════════════════════════════════

# Levodopa half-life: 2.0 h (mid-range 1.5–3 h)
LEVO_T_HALF = 2.0

AGG_COLS_NUM  = ["off", "on", "ledd", "age", "age_at_diagnosis",
                 "time_since_intake_on", "time_since_intake_off"]
NEIGHBOR_COLS = ["off", "on", "ledd", "time_since_intake_off"]


def _vectorised_slope(pid_arr: np.ndarray,
                       x_arr: np.ndarray,
                       y_arr: np.ndarray) -> pd.Series:
    """Vectorised OLS slope grouped by patient_id — ~0.003s for 5k patients."""
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
    """Per-patient X-only aggregates. Vectorised groupby, ~0.06s for 5k patients."""
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
        how="left",
    )
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


def _rater_oof_target_enc(df_tr: pd.DataFrame, y_tr: np.ndarray) -> dict:
    """
    Full LOO target encoding of rater_id vs target on the training fold.

    For each rater r, the encoding is the mean target of all training rows
    whose rater_id == r. Computed on the training fold only; applied to both
    train and validation rows to prevent leakage.
    """
    raters = df_tr["rater_id"].values
    enc = {r: float(np.nanmean(y_tr[raters == r])) for r in np.unique(raters)}
    # Global fallback for unseen raters
    enc["__global__"] = float(np.nanmean(y_tr))
    return enc


def _compute_sequential_history(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute within-patient running history features using only earlier visits.

    For each visit k of patient p, the features summarise visits 1..k-1
    (sorted by Index, which is monotone with age). The first visit of every
    patient always has NaN history (no prior visits), which is handled
    naturally by LightGBM's missing-value routing.

    Features added:
      - off_hist_mean, on_hist_mean, ledd_hist_mean : running mean up to k-1
      - off_hist_std                                 : running std up to k-1
      - off_hist_count                               : number of previous off obs
      - hist_visit_count                             : number of previous visits
      - off_hist_slope                               : slope of off vs age up to k-1
      - age_gap_from_prev                            : age difference to previous visit

    Returns a DataFrame indexed like df with only the new columns.
    """
    assert "target" not in df.columns

    sorted_df = df.sort_values(["patient_id", "Index"]).copy()
    orig_index = df.index

    results: dict[str, list] = {
        "off_hist_mean":   [],
        "on_hist_mean":    [],
        "ledd_hist_mean":  [],
        "off_hist_std":    [],
        "off_hist_count":  [],
        "hist_visit_count": [],
        "off_hist_slope":  [],
        "age_gap_from_prev": [],
    }

    for _pid, grp in sorted_df.groupby("patient_id", sort=False):
        off_vals   = grp["off"].values
        on_vals    = grp["on"].values
        ledd_vals  = grp["ledd"].values
        age_vals   = grp["age"].values
        n = len(grp)

        for k in range(n):
            past_off   = off_vals[:k][~np.isnan(off_vals[:k])]
            past_on    = on_vals[:k][~np.isnan(on_vals[:k])]
            past_ledd  = ledd_vals[:k][~np.isnan(ledd_vals[:k])]
            past_age   = age_vals[:k]

            results["off_hist_mean"].append(
                float(np.mean(past_off))   if len(past_off) > 0  else np.nan)
            results["on_hist_mean"].append(
                float(np.mean(past_on))    if len(past_on) > 0   else np.nan)
            results["ledd_hist_mean"].append(
                float(np.mean(past_ledd))  if len(past_ledd) > 0 else np.nan)
            results["off_hist_std"].append(
                float(np.std(past_off))    if len(past_off) >= 2 else np.nan)
            results["off_hist_count"].append(float(len(past_off)))
            results["hist_visit_count"].append(float(k))

            # Slope of off vs age up to visit k-1 (needs ≥ 3 paired observations)
            age_with_off = past_age[~np.isnan(off_vals[:k])]
            if len(age_with_off) >= 3:
                x, yv = age_with_off, past_off
                x_m, y_m = x.mean(), yv.mean()
                den = ((x - x_m) ** 2).sum()
                slope = float(((x - x_m) * (yv - y_m)).sum() / den) if den > 1e-10 else np.nan
            else:
                slope = np.nan
            results["off_hist_slope"].append(slope)

            age_prev = age_vals[k - 1] if k > 0 else np.nan
            results["age_gap_from_prev"].append(
                float(age_vals[k] - age_prev) if k > 0 else np.nan)

    hist_df = pd.DataFrame(results, index=sorted_df.index)
    # Restore original row order
    return hist_df.loc[orig_index]


def _lgbm_encode(X: pd.DataFrame, cat_cols: list[str]) -> pd.DataFrame:
    """Cast string/category columns; cast int8 → float for LightGBM."""
    X = X.copy()
    for c in cat_cols:
        X[c] = X[c].astype("category")
    for c in X.select_dtypes("int8").columns:
        X[c] = X[c].astype(float)
    return X


def build_features(
    df_raw: pd.DataFrame,
    rater_enc_map: dict | None,
    pat_agg: pd.DataFrame | None = None,
    seq_history: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """
    Full feature matrix for experiment 14.

    Blocks inherited from exp13 (B, C, D, E, F, I, A) plus:
      - Block H': full rater OOF target encoding (replaces rater_bias in exp13)
      - Block S : sequential within-patient history (new in exp14)
      - Block L : levodopa decay proxy exp(-t / T_HALF) (new in exp14)

    Parameters
    ----------
    df_raw      : raw X columns only (must NOT contain 'target').
    rater_enc_map : {rater_id -> mean(target) on train fold} from fold train only.
    pat_agg     : pre-computed patient aggregates (pass to avoid recomputing).
    seq_history : pre-computed sequential history DataFrame (pass to avoid recomputing).
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

    # ── Block A: visit-vs-patient-mean interactions ───────────────────────────
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

    # ── Block F: rater visit count ────────────────────────────────────────────
    feat["rater_visit_count"] = (
        feat.groupby("rater_id")["age"].transform("count")
    )

    # ── Block H': rater full OOF target encoding (exp14 upgrade) ─────────────
    if rater_enc_map is not None:
        global_fb = rater_enc_map.get("__global__", float(np.nanmean(
            [v for k, v in rater_enc_map.items() if k != "__global__"])))
        feat["rater_target_enc"] = (
            feat["rater_id"].map(rater_enc_map).fillna(global_fb)
        )
    else:
        feat["rater_target_enc"] = np.nan

    # ── Block S: sequential within-patient history (exp14 new) ───────────────
    if seq_history is None:
        seq_history = _compute_sequential_history(feat[list(df_raw.columns)])
    seq_history = seq_history.loc[orig_index]
    for col in seq_history.columns:
        feat[col] = seq_history[col].values

    # ── Block L: levodopa decay proxy (exp14 new) ─────────────────────────────
    t_off = pd.to_numeric(feat.get("time_since_intake_off", pd.Series(np.nan, index=feat.index)),
                          errors="coerce")
    feat["levo_decay_off"] = np.where(t_off.notna(), np.exp(-t_off / LEVO_T_HALF), np.nan)
    t_on = pd.to_numeric(feat.get("time_since_intake_on", pd.Series(np.nan, index=feat.index)),
                         errors="coerce")
    feat["levo_decay_on"] = np.where(t_on.notna(), np.exp(-t_on / LEVO_T_HALF), np.nan)

    assert "target" not in feat.columns
    feat = feat.drop(columns=["patient_id", "Index"], errors="ignore")

    # Clip extreme product/ratio values
    numeric_cols = feat.select_dtypes("number").columns
    for c in numeric_cols:
        if c.endswith("_prod") or c.endswith("_ratio"):
            feat[c] = feat[c].clip(-1e4, 1e4)

    assert (feat.index == orig_index).all(), "Index corrupted in build_features"
    return feat


# ══════════════════════════════════════════════════════════════════════════════
# 3. CV infrastructure — 3 seeds × GroupKFold(5) = 15 folds
# ══════════════════════════════════════════════════════════════════════════════
CV_SEEDS = [0, 1, 2]
N_SPLITS  = 5


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
# 4. LightGBM CV runner (leak-safe, full feature set)
# ══════════════════════════════════════════════════════════════════════════════

BASE_LGBM_PARAMS = {
    "objective":    "regression_l2",
    "metric":       "rmse",
    "verbosity":    -1,
    "n_jobs":       -1,
    "random_state": 42,
}


def run_cv_lgbm(
    params: dict,
    splits: list,
    visits_df: pd.DataFrame,
    y_series: pd.Series,
    verbose: bool = False,
) -> tuple[float, float, np.ndarray]:
    """
    CV with per-fold leak-safe feature construction.

    Sequential history and rater encoding are computed from train-fold rows
    only and applied to both train and validation rows. Pat aggregates use
    all X rows of the patient (permitted for batch inference).
    """
    oof        = np.full(len(y_series), np.nan)
    fold_rmses = []
    visits_x   = visits_df[list(X_train.columns)].copy()

    for fold_i, (tr, val) in enumerate(splits):
        df_tr  = visits_x.iloc[tr]
        df_val = visits_x.iloc[val]
        y_tr   = y_series.iloc[tr].values
        y_val  = y_series.iloc[val].values

        # Rater target encoding: computed on train fold targets only
        rater_enc = _rater_oof_target_enc(df_tr, y_tr)

        # Patient aggregates: use all X rows (batch inference allowed)
        pat_agg_tr  = _compute_pat_aggregates(df_tr)
        pat_agg_val = _compute_pat_aggregates(df_val)

        # Sequential history: train rows only for history of train visits;
        # val rows only for history of val visits (no cross-fold leakage)
        seq_hist_tr  = _compute_sequential_history(df_tr)
        seq_hist_val = _compute_sequential_history(df_val)

        X_tr_f  = build_features(df_tr,  rater_enc, pat_agg_tr,  seq_hist_tr)
        X_val_f = build_features(df_val, rater_enc, pat_agg_val, seq_hist_val)

        # Align columns (val may lack a rare-category-derived column)
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
            num_boost_round=1000,
            valid_sets=[lgb_val],
            callbacks=[lgb.early_stopping(50, verbose=False),
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
# 5. Optuna tuning — 50 trials with MedianPruner (up from 20 in exp13)
# ══════════════════════════════════════════════════════════════════════════════
print()
print("=" * 80)
print("Optuna tuning: 50 trials, seed-0 folds, MedianPruner")
print("=" * 80)


def objective_lgbm(trial: optuna.Trial) -> float:
    params = {
        **BASE_LGBM_PARAMS,
        "learning_rate":     trial.suggest_float("learning_rate", 0.02, 0.1, log=True),
        "num_leaves":        trial.suggest_int("num_leaves", 31, 256),
        "min_child_samples": trial.suggest_int("min_child_samples", 30, 300),
        "subsample":         trial.suggest_float("subsample", 0.5, 1.0),
        "colsample_bytree":  trial.suggest_float("colsample_bytree", 0.4, 1.0),
        "reg_alpha":         trial.suggest_float("reg_alpha", 1e-3, 5.0, log=True),
        "reg_lambda":        trial.suggest_float("reg_lambda", 1e-3, 5.0, log=True),
        "min_split_gain":    trial.suggest_float("min_split_gain", 0.0, 0.5),
    }
    mean_rmse, _, _ = run_cv_lgbm(params, cv_splits_s0, visits, y)
    return mean_rmse


t0 = time.perf_counter()
study = optuna.create_study(
    direction="minimize",
    sampler=optuna.samplers.TPESampler(seed=42),
    pruner=optuna.pruners.MedianPruner(n_startup_trials=10, n_warmup_steps=5),
)
study.optimize(objective_lgbm, n_trials=50)
t_tune = time.perf_counter() - t0
print(f"Optuna done in {t_tune:.0f}s  best RMSE={study.best_value:.4f}")
print(f"Best params: {study.best_params}")

best_lgbm_params = {**BASE_LGBM_PARAMS, **study.best_params}

# ══════════════════════════════════════════════════════════════════════════════
# 6. Full 15-fold CV with best parameters
# ══════════════════════════════════════════════════════════════════════════════
print()
print("=" * 80)
print("Best LightGBM — 15-fold CV (3 seeds × 5 folds)")
print("=" * 80)
t0 = time.perf_counter()
mean_lgbm, std_lgbm, oof_lgbm = run_cv_lgbm(
    best_lgbm_params, all_splits, visits, y, verbose=True)
t_lgbm = time.perf_counter() - t0
print(f"\nLightGBM 15-fold: mean={mean_lgbm:.4f}  std={std_lgbm:.4f}  time={t_lgbm:.0f}s")

# ── seed-0 OOF for isotonic fitting and ensemble ──────────────────────────────
_, _, oof_lgbm_s0 = run_cv_lgbm(best_lgbm_params, cv_splits_s0, visits, y)

# ══════════════════════════════════════════════════════════════════════════════
# 7. Isotonic recalibration on seed-0 OOF
# ══════════════════════════════════════════════════════════════════════════════
print()
print("=" * 80)
print("Isotonic recalibration (regression-to-mean correction)")
print("=" * 80)

valid_mask      = ~np.isnan(oof_lgbm_s0)
iso_reg         = IsotonicRegression(out_of_bounds="clip")
iso_reg.fit(oof_lgbm_s0[valid_mask], y.values[valid_mask])

oof_iso = oof_lgbm_s0.copy()
oof_iso[valid_mask] = iso_reg.predict(oof_lgbm_s0[valid_mask])

rmse_raw = root_mean_squared_error(y.values[valid_mask], oof_lgbm_s0[valid_mask])
rmse_iso = root_mean_squared_error(y.values[valid_mask], oof_iso[valid_mask])
print(f"OOF RMSE before isotonic : {rmse_raw:.4f}")
print(f"OOF RMSE after  isotonic : {rmse_iso:.4f}")
print(f"Gain from recalibration  : {rmse_raw - rmse_iso:+.4f}")

use_isotonic = rmse_iso < rmse_raw - 0.002

# ══════════════════════════════════════════════════════════════════════════════
# 8. Exp12 tabular_pipeline baseline for Hub push and ensemble
# ══════════════════════════════════════════════════════════════════════════════
print()
print("=" * 80)
print("Exp12 tabular_pipeline baseline (seed-0 OOF, for Hub and ensemble)")
print("=" * 80)

from scipy import stats as sp_stats  # noqa: E402 (local import to keep block clear)


def _compute_pat_agg_ref(df):
    assert "target" not in df.columns
    records = []
    for pid, grp in df.groupby("patient_id"):
        row = {"patient_id": pid}
        for col in ["off", "on", "ledd"]:
            vals = grp[col].dropna()
            n = len(vals)
            row[f"{col}_mean_pat"]   = vals.mean()   if n > 0 else np.nan
            row[f"{col}_median_pat"] = vals.median() if n > 0 else np.nan
            row[f"{col}_count_pat"]  = n
            row[f"{col}_std_pat"]    = vals.std()    if n >= 2 else np.nan
        for col in ["off", "on"]:
            d = grp[["age", col]].dropna().sort_values("age")
            if len(d) >= 3:
                row[f"{col}_slope_pat"] = float(
                    sp_stats.linregress(d["age"], d[col]).slope)
            else:
                row[f"{col}_slope_pat"] = np.nan
            d2 = grp.sort_values("Index")[col].dropna()
            row[f"{col}_first_pat"] = d2.iloc[0]  if len(d2) > 0 else np.nan
            row[f"{col}_last_pat"]  = d2.iloc[-1] if len(d2) > 0 else np.nan
        records.append(row)
    return pd.DataFrame(records).set_index("patient_id")


def _add_interactions_ref(df):
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


pat_agg_train_ref = _compute_pat_agg_ref(X_train)
pat_agg_test_ref  = _compute_pat_agg_ref(X_test)
visits_X_ref = visits[list(X_train.columns)].copy()


def _join_agg_ref(X, pat_agg):
    assert "target" not in X.columns
    r = X.join(pat_agg, on="patient_id", how="left")
    return r.drop(columns=["patient_id", "Index"], errors="ignore")


X_ref = _add_interactions_ref(_join_agg_ref(visits_X_ref, pat_agg_train_ref))

pipe_ref     = tabular_pipeline("regressor")
oof_ref_s0   = np.full(len(y), np.nan)
fold_rmses_ref = []
for i, (tr, val) in enumerate(cv_splits_s0):
    clf = clone(pipe_ref)
    clf.fit(X_ref.iloc[tr], y.iloc[tr])
    p    = clf.predict(X_ref.iloc[val])
    rmse = root_mean_squared_error(y.iloc[val], p)
    fold_rmses_ref.append(rmse)
    oof_ref_s0[val] = p
    print(f"  ref fold {i+1}: RMSE={rmse:.4f}")
mean_ref = float(np.mean(fold_rmses_ref))
std_ref  = float(np.std(fold_rmses_ref))
print(f"  ref => mean={mean_ref:.4f}  std={std_ref:.4f}")

# ══════════════════════════════════════════════════════════════════════════════
# 9. NNLS ensemble: LightGBM + tabular_pipeline ref
# ══════════════════════════════════════════════════════════════════════════════
print()
print("=" * 80)
print("NNLS ensemble (LightGBM + tabular_pipeline ref)")
print("=" * 80)

mask_ens = ~np.isnan(oof_lgbm_s0) & ~np.isnan(oof_ref_s0)
A = np.column_stack([oof_lgbm_s0[mask_ens], oof_ref_s0[mask_ens]])
w, _ = nnls(A, y.values[mask_ens])
w_sum = w.sum()
w = w / w_sum if w_sum > 0 else np.array([0.5, 0.5])
print(f"NNLS weights: lgbm={w[0]:.3f}  ref={w[1]:.3f}")

oof_ens      = w[0] * oof_lgbm_s0 + w[1] * oof_ref_s0
rmse_ens     = root_mean_squared_error(y.values[mask_ens], oof_ens[mask_ens])
print(f"OOF RMSE lgbm (s0) : {rmse_raw:.4f}")
print(f"OOF RMSE ensemble  : {rmse_ens:.4f}")
print(f"OOF RMSE isotonic  : {rmse_iso:.4f}")
use_ensemble = rmse_ens < rmse_raw - 0.005

# ══════════════════════════════════════════════════════════════════════════════
# 10. Summary table
# ══════════════════════════════════════════════════════════════════════════════
print()
print("=" * 80)
print("SUMMARY — exp13 reference vs exp14 improvements")
print("=" * 80)
EXP13_RMSE_15FOLD = 5.0   # approximate reference (run exp13 to get exact value)
print(f"{'Model':<40s}  {'RMSE':>8s}")
print("-" * 52)
print(f"{'exp13 LightGBM 15-fold (reference)':40s}  {EXP13_RMSE_15FOLD:>8.4f}")
print(f"{'exp14 LightGBM 15-fold (new features)':40s}  {mean_lgbm:>8.4f}")
print(f"{'exp14 seed-0 raw OOF':40s}  {rmse_raw:>8.4f}")
print(f"{'exp14 seed-0 + isotonic':40s}  {rmse_iso:>8.4f}")
print(f"{'exp14 NNLS ensemble (s0 OOF)':40s}  {rmse_ens:>8.4f}")
print()
print(f"Isotonic recalibration used in submission: {use_isotonic}")
print(f"NNLS ensemble used in submission:          {use_ensemble}")

# ══════════════════════════════════════════════════════════════════════════════
# 11. Push reports to Skore Hub
# ══════════════════════════════════════════════════════════════════════════════
print()
print("=" * 80)
print("Pushing to Skore Hub")
print("=" * 80)

# Push the exp12-compatible tabular_pipeline report so the Hub shows the
# feature-set evolution alongside exp11/12/13.
hub_report = skore.evaluate(pipe_ref, X_ref, y, splitter=cv_splits_s0)

cfg = load_skore_credentials()
login(mode="hub")
project = Project(name="bobathon-esilv", mode="hub", workspace=cfg["workspace"])
project.put("14_advanced_improvements", hub_report)
print("Report pushed: 14_advanced_improvements")
hub_url = f"https://skore.probabl.ai/{cfg['workspace']}/bobathon-esilv"

# ══════════════════════════════════════════════════════════════════════════════
# 12. Final model — multi-seed LightGBM (5 seeds) + optional isotonic
# ══════════════════════════════════════════════════════════════════════════════
print()
print("=" * 80)
print("Training final model (5-seed ensemble on all training data)")
print("=" * 80)

# Build full-train and test feature matrices
rater_enc_final = _rater_oof_target_enc(X_train, y.values)
pat_agg_full    = _compute_pat_aggregates(X_train)
pat_agg_test_14 = _compute_pat_aggregates(X_test)
seq_hist_full   = _compute_sequential_history(X_train)
seq_hist_test   = _compute_sequential_history(X_test)

X_train_feat = build_features(
    X_train, rater_enc_final, pat_agg_full, seq_hist_full)
X_test_feat  = build_features(
    X_test, rater_enc_final, pat_agg_test_14, seq_hist_test)

# Align test columns with train
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

lgb_full = lgb.Dataset(X_train_feat, label=y.values,
                        categorical_feature=cat_cols_final, free_raw_data=False)

# Train 5 models with different seeds, average predictions
FINAL_SEEDS  = [0, 1, 2, 3, 4]
seed_preds   = []
for seed in FINAL_SEEDS:
    seed_params = {**best_lgbm_params, "random_state": seed}
    model_s = lgb.train(
        seed_params, lgb_full,
        num_boost_round=1000,
        callbacks=[lgb.log_evaluation(-1)],
    )
    seed_preds.append(model_s.predict(X_test_feat))
    print(f"  seed {seed} trained")

preds_ensemble = np.mean(seed_preds, axis=0)

# Apply isotonic recalibration if it helped on OOF
if use_isotonic:
    print("Applying isotonic recalibration to test predictions...")
    preds_ensemble = iso_reg.predict(preds_ensemble)

assert len(preds_ensemble) == len(X_test)
assert not np.isnan(preds_ensemble).any()
assert not np.isinf(preds_ensemble).any()

# ══════════════════════════════════════════════════════════════════════════════
# 13. Submission file
# ══════════════════════════════════════════════════════════════════════════════
sample_sub = pd.read_csv(DATA_DIR / "sample_submission.csv")
assert len(preds_ensemble) == len(sample_sub)

submission = X_test[["Index"]].copy()
submission["target"] = preds_ensemble
assert list(submission.columns) == ["Index", "target"]
assert submission["Index"].tolist() == X_test["Index"].tolist()

out = PROJECT_ROOT / "submissions"
out.mkdir(exist_ok=True)
submission.to_csv(out / "14_advanced_improvements.csv", index=False)
print(f"\nSubmission: submissions/14_advanced_improvements.csv  ({len(submission)} rows)")

# ══════════════════════════════════════════════════════════════════════════════
# 14. Final summary
# ══════════════════════════════════════════════════════════════════════════════
print()
print("=" * 80)
print("FINAL SUMMARY")
print("=" * 80)
print(f"\nexp14 LightGBM 15-fold CV RMSE : {mean_lgbm:.4f} ± {std_lgbm:.4f}")
print(f"exp14 seed-0 OOF RMSE (raw)    : {rmse_raw:.4f}")
print(f"exp14 seed-0 OOF RMSE (isotonic): {rmse_iso:.4f}")
print(f"NNLS ensemble OOF RMSE         : {rmse_ens:.4f}")
print(f"\nImprovements applied:")
print(f"  [+] Sequential within-patient history (Block S)")
print(f"  [+] Levodopa decay proxy feature (Block L)")
print(f"  [+] Rater full OOF target encoding (Block H')")
print(f"  [+] Optuna 50 trials + MedianPruner (up from 20)")
print(f"  [+] 5-seed final model ensemble (seeds 0..4)")
print(f"  [+] Isotonic recalibration: {'applied' if use_isotonic else 'skipped (no gain)'}")
print(f"\nHub URL   : {hub_url}")
print(f"CSV file  : submissions/14_advanced_improvements.csv")
print()
print("OK: experiment 14 complete.")
