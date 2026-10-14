# %% [markdown]
# # Experiment: 01_dummy — mean baseline
#
# **Date:** 2025-10-07
# **Goal:** Establish the RMSE floor using a DummyRegressor(strategy="mean").
#           Every future model must beat this number.
# **Result:** filled in after the run.
#
# ---------------------------------------------------------------------------
# Rôle           : baseline Dummy (prédit toujours la moyenne de train)
# Entrées        : data/X_train.csv, data/y_train.csv, data/X_test.csv
# Sorties        : hub "01_dummy" (CrossValidationReport), submissions/01_dummy.csv
# Exécuter       : .venv/Scripts/python.exe experiments/01_dummy.py
# ---------------------------------------------------------------------------
#
# WORKBOOK Day1-Q7 : RMSE dummy vs écart-type de target
# Réponse : RMSE dummy (split aléatoire 80/20) = 16.48 ; std(target) = 16.50.
#   Quasi-identiques car le Dummy prédit la moyenne du train : son erreur sur
#   le holdout est mathématiquement l'écart-type de target.
#
# WORKBOOK Day1-Q9 : seuil d'amélioration réelle
# Réponse : Dummy CV groupé (GroupKFold 5 folds) = 16.50 ± 0.31.
#   Un gain est réel s'il dépasse nettement 0.31 pt et reste cohérent fold par fold.

# %%
import pandas as pd
import skore
from sklearn.dummy import DummyRegressor

from parkinson import PROJECT_ROOT
from parkinson.hub import load_skore_credentials
from skore import Project, login

# %% [markdown]
# ## Load and merge training data

# %%
DATA_DIR = PROJECT_ROOT / "data"

X_train = pd.read_csv(DATA_DIR / "X_train.csv")
y_train = pd.read_csv(DATA_DIR / "y_train.csv")
X_test = pd.read_csv(DATA_DIR / "X_test.csv")

visits = X_train.merge(y_train, on="Index")

FEATURE_COLS = [
    "sexM",
    "age_at_diagnosis",
    "age",
    "ledd",
    "time_since_intake_on",
    "time_since_intake_off",
    "on",
    "off",
]

X = visits[FEATURE_COLS]
y = visits["target"]

# %% [markdown]
# ## Evaluate with skore
#
# `splitter=0.2` is the default random 20 % row holdout.
# The patient-ID leak across the split is intentional for this baseline —
# the dummy ignores every column anyway.

# %%
dummy = DummyRegressor(strategy="mean")
report = skore.evaluate(dummy, X, y)
report.metrics.rmse()

# %% [markdown]
# ## Push to Skore Hub

# %%
cfg = load_skore_credentials()
login(mode="hub")
project = Project(name="bobathon-esilv", mode="hub", workspace=cfg["workspace"])
project.put("01_dummy", report)

# %% [markdown]
# ## Kaggle submission
#
# Refit the dummy on **all** training rows (no CV split),
# predict the test visits, write submissions/01_dummy.csv.

# %%
submissions_dir = PROJECT_ROOT / "submissions"
submissions_dir.mkdir(exist_ok=True)

dummy_final = DummyRegressor(strategy="mean")
dummy_final.fit(X, y)
predictions = dummy_final.predict(X_test[FEATURE_COLS])

submission = X_test[["Index"]].copy()
submission["target"] = predictions
submission.to_csv(submissions_dir / "01_dummy.csv", index=False)

print(f"Submission written: {submissions_dir / '01_dummy.csv'} ({len(submission)} rows)")
