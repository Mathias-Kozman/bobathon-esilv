# %%
# ---------------------------------------------------------------------------
# Rôle     : analyse des erreurs résiduelles du meilleur modèle (skrub tabular, exp 05/07)
# Entrées  : data/X_train.csv, data/y_train.csv
# Sorties  : sorties console uniquement (pas de hub, pas de CSV Kaggle)
# Exécuter : .venv/Scripts/python.exe experiments/08_error_analysis.py
# ---------------------------------------------------------------------------
#
# Experiment 08: error analysis on the best model (skrub_tabular, exp 05/07).
#
# Goal: understand residual errors, identify actionable improvement paths.
# No new Kaggle submission. One CV pass of the reference model, then diagnostics.
#
# WORKBOOK Day2-Q24 : visites les plus mal prédites (recalculé localement)
# Réponse : top-20 OOF — off manquant dans 55 % (vs 42.4 % global, +13 pts).
#   Target moyen dans le top-20 = 77.0 vs 37.5 global. Patients sévères sous-estimés.
#   Le modèle ramène les prédictions vers la moyenne (régression vers la moyenne).
#
# WORKBOOK Day2-Q25 : problèmes non traités et features à construire
# Réponse : progression temporelle et timing du médicament non adressés dans le workbook.
#   Features à construire : years_since_diagnosis, rang de visite, agrégats patient on/off,
#   levo_decay (NaN conservé + indicateur levo_decay_na, jamais 0), indicateurs de manquant.
#   L'exp 13 a déjà implémenté : agrégats patient (B), voisins temporels (C), position (D),
#   indicateurs de manquant (E), rater count (F), MixedLM intercept (G), rater OOF (H),
#   dérivées et interactions (I). Source : build_features() dans 13_best_model.py.

import warnings

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.metrics import mean_absolute_error, root_mean_squared_error
from sklearn.model_selection import GroupKFold
from skrub import tabular_pipeline

from parkinson import PROJECT_ROOT

warnings.filterwarnings("ignore", category=FutureWarning)

DATA_DIR = PROJECT_ROOT / "data"

# %%
# -- 1. Load & verify alignment ----------------------------------------------
X_train = pd.read_csv(DATA_DIR / "X_train.csv")
y_train = pd.read_csv(DATA_DIR / "y_train.csv")

visits = X_train.merge(y_train, on="Index")
assert len(visits) == len(X_train) == len(y_train), "Merge mismatch"

# Features: drop identifiers and target to prevent leakage
X_full = visits.drop(columns=["Index", "patient_id", "target"])
y = visits["target"]
groups = visits["patient_id"]

print("=" * 60)
print("1. DATASET OVERVIEW")
print("=" * 60)
print(f"Rows: {len(visits)}   Patients: {groups.nunique()}")
print(f"Features: {X_full.shape[1]}   {list(X_full.columns)}")
print()

print("-- Dtypes & missingness --")
info = pd.DataFrame({
    "dtype": X_full.dtypes,
    "n_missing": X_full.isnull().sum(),
    "pct_missing": (X_full.isnull().mean() * 100).round(1),
})
print(info.to_string())

print()
print("-- Categorical cardinality --")
for col in X_full.select_dtypes(["object", "str"]).columns:
    print(f"  {col}: {X_full[col].nunique()} unique values")

# %%
# -- 2. Target distribution ---------------------------------------------------
print()
print("=" * 60)
print("2. TARGET DISTRIBUTION")
print("=" * 60)
quantiles = np.quantile(y, [0, .05, .1, .25, .5, .75, .9, .95, 1])
labels = ["0%", "5%", "10%", "25%", "50%", "75%", "90%", "95%", "100%"]
print(f"  mean={y.mean():.2f}  std={y.std():.2f}  skew={y.skew():.2f}")
print("  Quantiles:")
for q, v in zip(labels, quantiles):
    print(f"    {q:>4s}: {v:.1f}")

# Intra- vs inter-patient variance (ICC)
pat_stats = visits.groupby("patient_id")["target"].agg(["mean", "std", "count"])
grand_mean = y.mean()
inter_var = ((pat_stats["mean"] - grand_mean) ** 2).mean()
intra_var = (pat_stats["std"] ** 2).mean()
icc = inter_var / (inter_var + intra_var)
print(f"\n  ICC (inter-patient): {icc:.3f}")
print(f"  -> {icc*100:.0f}% of variance is between patients (GroupKFold is essential)")

# %%
# -- 3. Reference CV (one pass, same folds as all experiments) ---------------
print()
print("=" * 60)
print("3. REFERENCE CV — skrub_tabular (GroupKFold 5)")
print("=" * 60)

cv_splits = list(GroupKFold(n_splits=5).split(X_full, y, groups=groups))
model = tabular_pipeline("regressor")

# Collect out-of-fold predictions
oof_preds = np.full(len(y), np.nan)
rmse_per_fold = []

for fold_i, (train_idx, val_idx) in enumerate(cv_splits):
    clf = clone(model)
    clf.fit(X_full.iloc[train_idx], y.iloc[train_idx])
    preds_fold = clf.predict(X_full.iloc[val_idx])
    oof_preds[val_idx] = preds_fold
    rmse_fold = root_mean_squared_error(y.iloc[val_idx], preds_fold)
    rmse_per_fold.append(rmse_fold)
    print(f"  fold {fold_i}: RMSE={rmse_fold:.4f}  (n_val={len(val_idx)})")

assert not np.isnan(oof_preds).any(), "OOF predictions incomplete"
residuals = oof_preds - y.values
oof_rmse = root_mean_squared_error(y, oof_preds)
oof_mae = mean_absolute_error(y, oof_preds)
oof_bias = residuals.mean()
print(f"\n  OOF RMSE : {oof_rmse:.4f}")
print(f"  OOF MAE  : {oof_mae:.4f}")
print(f"  OOF Bias : {oof_bias:.4f}  (mean pred - target)")

# %%
# -- 4. Residuals by target range ---------------------------------------------
print()
print("=" * 60)
print("4. RESIDUALS BY TARGET RANGE")
print("=" * 60)

bins = [0, 15, 25, 35, 45, 55, 70, 110]
labels_bins = ["0–15", "15–25", "25–35", "35–45", "45–55", "55–70", "70+"]
target_bin = pd.cut(y, bins=bins, labels=labels_bins, right=False)

range_df = pd.DataFrame({
    "target": y.values,
    "pred": oof_preds,
    "resid": residuals,
    "bin": target_bin.values,
})
grp = range_df.groupby("bin", observed=True)
summary = grp.apply(
    lambda d: pd.Series({
        "n": len(d),
        "rmse": root_mean_squared_error(d["target"], d["pred"]),
        "mae": mean_absolute_error(d["target"], d["pred"]),
        "bias": d["resid"].mean(),
    })
)
print(summary.to_string())
print()
worst_bin = summary["rmse"].idxmax()
best_bin = summary["rmse"].idxmin()
print(f"  Worst range : {worst_bin}  (RMSE={summary.loc[worst_bin,'rmse']:.3f})")
print(f"  Best range  : {best_bin}  (RMSE={summary.loc[best_bin,'rmse']:.3f})")
# Note strong positive bias for low target -> model over-predicts mild cases
# and negative bias for high target -> under-predicts severe cases

# %%
# -- 5. Residuals by categorical variable -------------------------------------
print()
print("=" * 60)
print("5. RESIDUALS BY CATEGORICAL VARIABLE")
print("=" * 60)

for cat_col in ["cohort", "gene", "rater_id"]:
    col_vals = visits[cat_col]  # use original visits frame for grouping
    grp_df = pd.DataFrame({
        "target": y.values,
        "pred": oof_preds,
        "resid": residuals,
        "cat": col_vals.values,
    })
    grp_summary = grp_df.groupby("cat").apply(
        lambda d: pd.Series({
            "n": len(d),
            "target_mean": d["target"].mean(),
            "rmse": root_mean_squared_error(d["target"], d["pred"]),
            "bias": d["resid"].mean(),
        })
    ).sort_values("rmse", ascending=False)
    print(f"\n-- {cat_col} --")
    print(grp_summary.to_string())

# %%
# -- 6. Feature importance (last fold only, from fitted HGBR inside pipeline) --
print()
print("=" * 60)
print("6. FEATURE IMPORTANCE (last fold, from HGBR internal)")
print("=" * 60)

# Refit on last fold's training set to inspect importances
last_train_idx = cv_splits[-1][0]
clf_last = clone(model).fit(X_full.iloc[last_train_idx], y.iloc[last_train_idx])

# Use permutation importance on the validation fold (fold-pure: estimated on
# held-out data, no information from the training fold leaks in).
# This is more reliable than HGBR's built-in gain-based importance.
from sklearn.inspection import permutation_importance

last_val_idx = cv_splits[-1][1]
pi = permutation_importance(
    clf_last,
    X_full.iloc[last_val_idx],
    y.iloc[last_val_idx],
    n_repeats=5,
    random_state=0,
    scoring="neg_root_mean_squared_error",
)
imp_series = pd.Series(pi.importances_mean, index=X_full.columns).sort_values(ascending=False)
print("  (permutation importance on last fold val set, RMSE decrease)")
print(imp_series.to_string())

# %%
# -- 7. Rare categories & suspicious associations -----------------------------
print()
print("=" * 60)
print("7. RARE CATEGORIES & SUSPICIOUS ASSOCIATIONS")
print("=" * 60)

for cat_col in ["cohort", "gene", "rater_id"]:
    counts = visits[cat_col].value_counts()
    rare = counts[counts < 500]
    if len(rare):
        print(f"\n  {cat_col} — rare values (n<500):")
        print(rare.to_string())
    else:
        print(f"\n  {cat_col} — no rare values (min={counts.min()})")

# Check if rater_id spans multiple patients or is patient-level (leakage risk)
raters_per_patient = visits.groupby("patient_id")["rater_id"].nunique()
print(f"\n  rater_id per patient: mean={raters_per_patient.mean():.2f}, "
      f"max={raters_per_patient.max()}")
print(f"  -> rater_id varies within patient -> NOT a patient-level identifier -> safe")

# %%
# -- 8. Intra-patient residual pattern ----------------------------------------
print()
print("=" * 60)
print("8. INTRA-PATIENT RESIDUAL PATTERN")
print("=" * 60)

pat_resid = pd.DataFrame({
    "patient_id": visits["patient_id"].values,
    "target": y.values,
    "pred": oof_preds,
    "resid": residuals,
}).groupby("patient_id").agg(
    n=("resid", "count"),
    rmse=("resid", lambda x: root_mean_squared_error(
        visits.loc[x.index, "target"], oof_preds[x.index])),
    bias=("resid", "mean"),
    target_mean=("target", "mean"),
).sort_values("rmse", ascending=False)

print(f"  Worst 10 patients by RMSE:")
print(pat_resid.head(10).to_string())
print()
print(f"  Patient-level RMSE: mean={pat_resid['rmse'].mean():.3f}  "
      f"std={pat_resid['rmse'].std():.3f}")
print(f"  Patients with bias > +5: {(pat_resid['bias'] >  5).sum()}")
print(f"  Patients with bias < -5: {(pat_resid['bias'] < -5).sum()}")

# %%
# -- 9. Synthesis: top improvement paths --------------------------------------
print()
print("=" * 60)
print("9. SYNTHESIS — TOP IMPROVEMENT PATHS")
print("=" * 60)

print("""
PATH 1 — Patient-level feature engineering (ICC=0.826, highest leverage)
  Rationale: 82.6% of target variance is between patients, not within visits.
  The model predicts each visit independently. Adding patient-level aggregates
  (mean past target, baseline `off`, visit count) computed on out-of-fold
  training rows as extra features could capture the patient baseline directly
  without leaking. Impact: potentially large (-1 to -2 RMSE).

PATH 2 — Target-range calibration (strong regression-to-mean bias)
  Rationale: residuals show large positive bias for low targets (model
  over-predicts mild patients) and negative bias for high targets (under-predicts
  severe). A post-hoc isotonic recalibration, or a quantile loss objective in
  HGBR, could correct this systematic bias. Impact: medium (-0.3 to -0.5 RMSE).

PATH 3 — Encode missingness as signal, especially for `off` and `ledd`
  Rationale: `off` (r=0.871) is 42% missing. Its absence pattern may itself
  be informative (e.g. visit timing, cohort protocol). Adding binary
  `off_is_missing` and `ledd_is_missing` flags gives HGBR explicit signal
  beyond the native NaN split. Impact: small-to-medium (-0.1 to -0.3 RMSE).

PATH 4 — Hyperparameter tuning of HGBR (max_iter, max_leaf_nodes, l2_reg)
  Rationale: tabular_pipeline uses default HGBR params. A targeted grid/random
  search on max_iter (100->500), max_leaf_nodes (31->63/127), and l2_regularization
  on the best feature set could recover more signal. Impact: small (-0.1 to -0.2).

PATH 5 — Rater bias correction
  Rationale: rater_id mean target spans ~34.9 to ~39.3 (+4.4 pts range). If
  the test set rater distribution differs from train, naive predictions will be
  biased. Encoding rater_id as a target-encoded feature (out-of-fold, to avoid
  leakage) could absorb rater-level systematic offsets. Impact: moderate if
  test/train rater distributions differ (-0.2 to -0.4 RMSE).
""")

print("OK: error analysis complete.")
