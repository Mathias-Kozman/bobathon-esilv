# %%
# ---------------------------------------------------------------------------
# Rôle     : évaluation honnête par CV groupé par patient (GroupKFold 5 folds)
# Entrées  : data/X_train.csv, data/y_train.csv
# Sorties  : hub "03_dummy_cv" et "03_ridge_cv" (CrossValidationReports)
# Exécuter : .venv/Scripts/python.exe experiments/03_groupkfold.py
# ---------------------------------------------------------------------------
#
# WORKBOOK Day2-Q14 : split aléatoire vs GroupKFold
# Réponse : Dummy aléatoire 16.48 → CV groupé 16.50 (Δ=+0.02, insensible).
#   Ridge aléatoire 10.49 → CV groupé 10.60 (Δ=+0.11, léger — modèle linéaire
#   peu sensible à la fuite patient).
#
# WORKBOOK Day2-Q15 : quel estimé croire ?
# Réponse : le CV groupé, car il simule le split Kaggle (holdout par patient entier).
#   À comparer avec notre score public Kaggle.
import pandas as pd
import skore
from sklearn.dummy import DummyRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline

from parkinson import PROJECT_ROOT
from parkinson.hub import load_skore_credentials
from skore import Project, login

DATA_DIR = PROJECT_ROOT / "data"
X_train = pd.read_csv(DATA_DIR / "X_train.csv")
y_train = pd.read_csv(DATA_DIR / "y_train.csv")

visits = X_train.merge(y_train, on="Index")
FEATURE_COLS = ["sexM", "age_at_diagnosis", "age", "ledd",
                "time_since_intake_on", "time_since_intake_off", "on", "off"]
X = visits[FEATURE_COLS]
y = visits["target"]

# %% CV groupé par patient (pas de fuite patient entre train et validation)
groups = visits["patient_id"]
cv_splits = list(GroupKFold(n_splits=5).split(X, y, groups=groups))
print(f"Nombre de folds : {len(cv_splits)}")
for i, (tr, val) in enumerate(cv_splits):
    print(f"  fold {i}: train={len(tr)}, val={len(val)}")

# %% Évaluation Dummy avec CV groupé
dummy = DummyRegressor(strategy="mean")
dummy_report = skore.evaluate(dummy, X, y, splitter=cv_splits)
dummy_rmse_folds = dummy_report.metrics.rmse()
print("\nDummy CV groupé — RMSE par fold et moyen :")
print(dummy_rmse_folds)

# %% Évaluation Ridge alpha=1.0 avec CV groupé
ridge_pipe = make_pipeline(SimpleImputer(strategy="median"), Ridge(alpha=1.0))
ridge_report = skore.evaluate(ridge_pipe, X, y, splitter=cv_splits)
ridge_rmse_folds = ridge_report.metrics.rmse()
print("\nRidge CV groupé — RMSE par fold et moyen :")
print(ridge_rmse_folds)

# %% Comparaison avec le split aléatoire de 02_ridge
print("\n--- Comparaison split aléatoire (exp 02) vs CV groupé (exp 03) ---")
print(f"Dummy  : aléatoire=16.4779  CV groupé={dummy_rmse_folds}")
print(f"Ridge  : aléatoire=10.4877  CV groupé={ridge_rmse_folds}")

# %% Push sur le hub
cfg = load_skore_credentials()
login(mode="hub")
project = Project(name="bobathon-esilv", mode="hub", workspace=cfg["workspace"])
project.put("03_dummy_cv", dummy_report)
project.put("03_ridge_cv", ridge_report)
print("Reports poussés : 03_dummy_cv, 03_ridge_cv")
