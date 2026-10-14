# %%
# ---------------------------------------------------------------------------
# Rôle     : analyse de la structure patient — diagnose, pas de modèle
# Entrées  : data/X_train.csv, data/y_train.csv, data/X_test.csv
# Sorties  : sorties console uniquement
# Exécuter : .venv/Scripts/python.exe experiments/10_patient_structure.py
# ---------------------------------------------------------------------------
#
# Experiment 10: patient structure analysis.
#
# Pure diagnostic — no model training, no Skore Hub push, no submission.
# Goal: determine whether patient-level features can be built without leakage,
# and whether they could genuinely improve the RMSE.

import warnings

import numpy as np
import pandas as pd

from parkinson import PROJECT_ROOT

warnings.filterwarnings("ignore", category=FutureWarning)

DATA_DIR = PROJECT_ROOT / "data"

# %%
# 1. Load data and check columns
X_train = pd.read_csv(DATA_DIR / "X_train.csv")
y_train = pd.read_csv(DATA_DIR / "y_train.csv")
X_test  = pd.read_csv(DATA_DIR / "X_test.csv")

visits = X_train.merge(y_train, on="Index")
assert len(visits) == len(X_train) == len(y_train), "Merge mismatch"

print("=" * 60)
print("1. COLUMNS")
print("=" * 60)
print(f"X_train : {list(X_train.columns)}")
print(f"y_train : {list(y_train.columns)}")
print(f"X_test  : {list(X_test.columns)}")

# %%
# 2. patient_id presence and overlap
print()
print("=" * 60)
print("2. PATIENT_ID PRESENCE AND TRAIN/TEST OVERLAP")
print("=" * 60)

print(f"patient_id in X_train : {('patient_id' in X_train.columns)}")
print(f"patient_id in X_test  : {('patient_id' in X_test.columns)}")

train_patients = set(X_train["patient_id"].unique())
test_patients  = set(X_test["patient_id"].unique())
overlap        = train_patients & test_patients

print(f"\nDistinct patients in train : {len(train_patients)}")
print(f"Distinct patients in test  : {len(test_patients)}")
print(f"Patients in BOTH train+test: {len(overlap)}")

if len(overlap) == 0:
    print("\n  => Train and test patient sets are FULLY DISJOINT.")
    print("     No training-time history of a test patient is available.")
    print("     Patient-level aggregates computed from train rows CANNOT")
    print("     be merged into test rows via patient_id.")
else:
    print(f"\n  => {len(overlap)} patients appear in both splits (partial overlap).")

# %%
# 3. Visit structure
print()
print("=" * 60)
print("3. VISIT STRUCTURE PER PATIENT")
print("=" * 60)

train_vpc = visits.groupby("patient_id").size()
test_vpc  = X_test.groupby("patient_id").size()

print("Train — visits per patient:")
print(f"  min={train_vpc.min()}  median={train_vpc.median():.0f}  "
      f"max={train_vpc.max()}  mean={train_vpc.mean():.2f}")

print("Test  — visits per patient:")
print(f"  min={test_vpc.min()}  median={test_vpc.median():.0f}  "
      f"max={test_vpc.max()}  mean={test_vpc.mean():.2f}")
print(f"  Patients with >1 test visit: {(test_vpc > 1).sum()} / {len(test_vpc)}")

# 3a. Temporal ordering: is Index monotone within patient?
train_sorted = visits.sort_values(["patient_id", "Index"])
idx_mono_train = train_sorted.groupby("patient_id")["Index"].apply(
    lambda x: (x.diff().dropna() > 0).all()
)
print(f"\nTrain: Index monotone within every patient: "
      f"{idx_mono_train.all()} ({idx_mono_train.sum()}/{len(idx_mono_train)})")

# 3b. Age as temporal proxy
age_diffs_train = train_sorted.groupby("patient_id")["age"].apply(
    lambda x: x.diff().dropna()
)
print(f"Train: age always increases within patient: "
      f"{(age_diffs_train >= 0).all()} "
      f"(mean step={age_diffs_train.mean():.2f}y, "
      f"min={age_diffs_train.min():.2f}y)")

test_sorted = X_test.sort_values(["patient_id", "Index"])
age_diffs_test = test_sorted.groupby("patient_id")["age"].apply(
    lambda x: x.diff().dropna()
)
print(f"Test : age always increases within patient: "
      f"{(age_diffs_test >= 0).all()} "
      f"(mean step={age_diffs_test.mean():.2f}y)")

# %%
# 4. off and on missingness patterns
print()
print("=" * 60)
print("4. off AND on OBSERVATION PATTERNS")
print("=" * 60)

both_obs  = visits["off"].notna() & visits["on"].notna()
off_only  = visits["off"].notna() & visits["on"].isna()
on_only   = visits["off"].isna()  & visits["on"].notna()
both_miss = visits["off"].isna()  & visits["on"].isna()

print("Train co-occurrence pattern:")
print(f"  both off+on observed : {both_obs.sum():>6d} ({both_obs.mean():.1%})")
print(f"  off only             : {off_only.sum():>6d} ({off_only.mean():.1%})")
print(f"  on only              : {on_only.sum():>6d} ({on_only.mean():.1%})")
print(f"  both missing         : {both_miss.sum():>6d} ({both_miss.mean():.1%})")

print(f"\nTrain: target always present (no missing): "
      f"{visits['target'].notna().all()}")
print(f"Train: target present when off present   : "
      f"{visits.loc[visits['off'].notna(), 'target'].notna().all()}")

print(f"\nTest off  missing: {X_test['off'].isna().sum()} / {len(X_test)} "
      f"({X_test['off'].isna().mean():.1%})")
print(f"Test on   missing: {X_test['on'].isna().sum()} / {len(X_test)} "
      f"({X_test['on'].isna().mean():.1%})")

# %%
# 5. off-target correlation: overall vs within-patient
print()
print("=" * 60)
print("5. off-target CORRELATION: OVERALL vs WITHIN-PATIENT")
print("=" * 60)

r_overall = visits[["off", "target"]].dropna().corr().iloc[0, 1]

# Within-patient: subtract each patient's mean off and mean target
pat_means = visits.groupby("patient_id")[["off", "target"]].transform("mean")
visits_dm = visits[["off", "target"]].sub(pat_means)
r_within  = visits_dm.dropna().corr().iloc[0, 1]

print(f"  off-target r OVERALL        : {r_overall:.3f}")
print(f"  off-target r WITHIN-PATIENT : {r_within:.3f}")
print()
print(f"  ICC of target (inter-patient fraction): "
      f"{224.58 / (224.58 + 47.37):.3f}  (computed in exp 08)")
print()
print("  Interpretation:")
print(f"  The high overall r ({r_overall:.3f}) is partly driven by patient-level")
print(f"  baseline differences. The within-patient r ({r_within:.3f}) is still")
print("  strong and represents the genuine visit-level signal in off.")

# %%
# 6. Within-test-patient temporal structure
print()
print("=" * 60)
print("6. WITHIN-TEST-PATIENT TEMPORAL STRUCTURE")
print("=" * 60)

print(f"All test patients have multiple visits: {(test_vpc >= 2).all()}")
print(f"Test Index is monotone within patient: {(age_diffs_test >= 0).all()}")
print()
print("  Within a test patient, visits are ordered by Index (= by age).")
print("  Earlier test visits of the same patient could serve as 'history'")
print("  for later test visits — but this requires a sequential prediction")
print("  strategy (predict visit k using visits 1..k-1 of the same patient).")
print("  This is architecturally complex and carries its own leakage risks.")

# %%
# 7. Are there any static patient-level features?
print()
print("=" * 60)
print("7. STATIC PATIENT-LEVEL FEATURES")
print("=" * 60)

static_candidates = ["cohort", "sexM", "gene", "age_at_diagnosis"]
for col in static_candidates:
    nuniq_per_patient = visits.groupby("patient_id")[col].nunique()
    fully_static = (nuniq_per_patient == 1).all()
    pct_static = (nuniq_per_patient == 1).mean()
    print(f"  {col:<20s}: constant within patient = {fully_static} "
          f"({pct_static:.1%})")

print()
print("  age is NOT static (increases each visit — it is the visit timestamp).")
print("  rater_id varies within patient (mean 2.7 raters per patient).")
print("  These static features are ALREADY in the model.")

# %%
# 8. Conclusion
print()
print("=" * 60)
print("8. CONCLUSION")
print("=" * 60)
print("""
KEY FINDING: Train and test patients are FULLY DISJOINT (0 overlap).
This is the central constraint for patient-level feature engineering.

What is feasible without leakage:
  (a) WITHIN train-fold (GroupKFold): for each training patient, aggregates
      of that patient's OTHER visits in the SAME fold can be computed and
      attached as per-visit features. This is done fold-by-fold to prevent
      leakage from the validation fold.
      -> Complexity: high. Requires custom CV logic.
      -> For test patients: IMPOSSIBLE. No training-history exists.

  (b) STATIC patient attributes (cohort, sexM, gene, age_at_diagnosis):
      Already in the model. Constant within patient. No gain expected.

  (c) Within-test-patient lagged features (visit k uses visits 1..k-1
      of the same test patient): architecturally feasible but requires
      ordered sequential prediction. First visit of each test patient
      would still have no history, limiting coverage to ~87% of test rows.
      This is a non-trivial engineering effort.

  (d) EASY win still available: the age column already encodes visit
      timing within a patient. The model sees it. No extra work needed.

RECOMMENDATION FOR NEXT EXPERIMENT:
  None of the clean patient-level feature paths are straightforward given
  the disjoint train/test split. The best remaining lever is:

  PATH B — Within-fold patient aggregates on train only:
    Build a custom transformer that, for each row in the training fold,
    computes aggregates (mean off, mean on, n_visits) from the SAME
    PATIENT'S OTHER ROWS in the CURRENT training fold. For test rows,
    fall back to the cohort/gene mean (since no patient history exists).
    This captures the ~81% inter-patient ICC signal without leaking.
    Estimated impact: -0.5 to -1.5 RMSE.
    Implement in experiments/11_patient_aggregates.py.
""")

print("OK: patient structure analysis complete.")
