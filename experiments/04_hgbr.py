# %%
# ---------------------------------------------------------------------------
# Rôle     : HistGradientBoostingRegressor — NaN conservés comme signal
# Entrées  : data/X_train.csv, data/y_train.csv, data/X_test.csv
# Sorties  : hub "04_hgbr", "04_ridge_cv_ref" ; submissions/04_hgbr.csv
# Exécuter : .venv/Scripts/python.exe experiments/04_hgbr.py
# ---------------------------------------------------------------------------
#
# WORKBOOK Day2-Q16 : HGBR bat-il Ridge sur chaque fold ?
# Réponse : oui, sur les 5/5 folds. RMSE moyen HGBR = 7.75 vs Ridge 10.60.
#   Le gain est robuste, pas un artefact de moyenne.
#
# WORKBOOK Day2-Q17 : temps de calcul (mesuré sur fold 0, 35 672 lignes train)
# Réponse : Ridge fit=0.24s, HGBR fit=11.8s → ratio ×50 en fit, ×7 en predict.
#   Le gain de −2.85 pts RMSE justifie ce coût en production par lots.
#
# WORKBOOK Day2-Q18 : importance par permutation vs coefficients Ridge
# Réponse : top features HGBR = off (7.33), on (5.11), age (2.26), age_at_diagnosis (1.84).
#   Ridge classe time_since_intake_on premier : artefact d'échelle (0–6 h vs 100–1000 mg/j).
#   Permutation importance : indépendante de l'échelle + exploite NaN + non-linéarités.
import time

import pandas as pd
import skore
from sklearn.ensemble import HistGradientBoostingRegressor
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
X_test = pd.read_csv(DATA_DIR / "X_test.csv")

visits = X_train.merge(y_train, on="Index")
FEATURE_COLS = ["sexM", "age_at_diagnosis", "age", "ledd",
                "time_since_intake_on", "time_since_intake_off", "on", "off"]
X = visits[FEATURE_COLS]
y = visits["target"]

# %% CV groupé — mêmes folds déterministes que l'expérience 03
groups = visits["patient_id"]
cv_splits = list(GroupKFold(n_splits=5).split(X, y, groups=groups))
print(f"Folds : {len(cv_splits)}")
for i, (tr, val) in enumerate(cv_splits):
    print(f"  fold {i}: train={len(tr)}, val={len(val)}")

# %% Évaluation Ridge (référence — sans imputation manuelle, HGBR gère le NaN)
# Ridge nécessite une imputation ; on garde SimpleImputer(median) comme en exp 03.
t0 = time.perf_counter()
ridge_pipe = make_pipeline(SimpleImputer(strategy="median"), Ridge(alpha=1.0))
ridge_report = skore.evaluate(ridge_pipe, X, y, splitter=cv_splits)
ridge_time = time.perf_counter() - t0
ridge_rmse = ridge_report.metrics.rmse()
print(f"\nRidge (référence) — RMSE par fold et moyen [{ridge_time:.1f}s] :")
print(ridge_rmse)

# %% Évaluation HistGradientBoostingRegressor
# HGBR gère nativement les NaN — pas d'imputation nécessaire.
t0 = time.perf_counter()
hgbr = HistGradientBoostingRegressor(random_state=0)
hgbr_report = skore.evaluate(hgbr, X, y, splitter=cv_splits)
hgbr_time = time.perf_counter() - t0
hgbr_rmse = hgbr_report.metrics.rmse()
print(f"\nHistGradientBoostingRegressor — RMSE par fold et moyen [{hgbr_time:.1f}s] :")
print(hgbr_rmse)

# %% Résumé comparatif
print("\n--- Comparaison Ridge vs HGBR (GroupKFold 5 folds) ---")
print(f"Ridge  RMSE moyen : {ridge_rmse}")
print(f"HGBR   RMSE moyen : {hgbr_rmse}")

# %% Push sur le hub
cfg = load_skore_credentials()
login(mode="hub")
project = Project(name="bobathon-esilv", mode="hub", workspace=cfg["workspace"])
# On pousse chaque CrossValidationReport séparément (ComparisonReport refusé par project.put)
project.put("04_hgbr", hgbr_report)
project.put("04_ridge_cv_ref", ridge_report)
print("Reports poussés : 04_hgbr, 04_ridge_cv_ref")

# %% Fit final sur tout le train + CSV Kaggle
final_hgbr = HistGradientBoostingRegressor(random_state=0)
final_hgbr.fit(X, y)

X_test_features = X_test[FEATURE_COLS]
preds = final_hgbr.predict(X_test_features)

assert len(preds) == len(X_test), (
    f"Dimension mismatch: {len(preds)} prédictions pour {len(X_test)} lignes de test"
)
assert not pd.isnull(preds).any(), "Prédictions manquantes détectées"

submission = X_test[["Index"]].copy()
submission["target"] = preds

out = PROJECT_ROOT / "submissions"
out.mkdir(exist_ok=True)
submission.to_csv(out / "04_hgbr.csv", index=False)

print(f"\nSubmission : {submission.shape[0]} lignes, colonnes {list(submission.columns)}")
print(submission.head())
assert list(submission.columns) == ["Index", "target"], "Colonnes incorrectes"
assert submission["target"].isnull().sum() == 0, "NaN dans les prédictions"
print("OK: Toutes les verifications passent.")
