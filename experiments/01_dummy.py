# %% [markdown]
# # Experiment: 01_dummy — mean baseline
#
# **Date:** 2025-10-07
# **Goal:** Establish the RMSE floor using a DummyRegressor(strategy="mean").
#           Every future model must beat this number.
# **Result:** filled in after the run.

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
