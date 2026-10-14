# %%
# ---------------------------------------------------------------------------
# Rôle     : modèle linéaire Ridge, balayage d'alpha, comparaison au Dummy
# Entrées  : data/X_train.csv, data/y_train.csv, data/X_test.csv
# Sorties  : hub "02_ridge" (meilleur alpha), submissions/02_ridge.csv
# Exécuter : .venv/Scripts/python.exe experiments/02_ridge.py
# ---------------------------------------------------------------------------
#
# WORKBOOK Day2-Q10 : gain Ridge vs Dummy
# Réponse : Dummy CV = 16.50, Ridge CV (α=1) = 10.60. Gain = 35.8 %.
#
# WORKBOOK Day2-Q11 : coefficients Ridge
# Réponse : plus grand |coef| = time_since_intake_on (+1.70).
#   off (+0.59) et on (+0.48) sont positifs — attendu cliniquement.
#
# WORKBOOK Day2-Q12 : sans standardisation
# Réponse : artefact d'échelle — time_since_intake_on varie 0–6 h, ledd 100–1000 mg/j,
#   age 35–80 ans. Coefficients bruts non comparables entre variables d'échelles différentes.
#   Il faudrait standardiser pour lire l'importance relative.
#
# WORKBOOK Day2-Q13 : balayage alpha {0.1, 1, 10, 100}
# Réponse : RMSE identique (10.5965) sur toute la plage d'alpha.
#   Seulement 8 variables pour 44 590 lignes : quasi aucune variance à réguler.
import pandas as pd
import skore
from pathlib import Path
from sklearn.base import clone
from sklearn.dummy import DummyRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
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

# %% Balayage d'alpha (même split par défaut -> comparable)
alphas = [0.01, 0.1, 1.0, 10.0, 100.0, 1000.0]
results, reports, coefs = {}, {}, {}
for a in alphas:
    pipe = make_pipeline(SimpleImputer(strategy="median"), Ridge(alpha=a))
    rep = skore.evaluate(pipe, X, y)
    results[a] = rep.metrics.rmse()
    reports[a] = rep
    coefs[a] = pd.Series(
        clone(pipe).fit(X, y)[-1].coef_, index=FEATURE_COLS
    )

dummy_rmse = skore.evaluate(DummyRegressor(strategy="mean"), X, y).metrics.rmse()
print("Dummy :", dummy_rmse)
print(pd.Series(results, name="rmse"))
print(pd.DataFrame(coefs).round(3))   # pour la question "coefs entre petit/grand alpha"

best_alpha = min(results, key=results.get)
print("Meilleur alpha :", best_alpha)

# %% Push du meilleur report sur le hub
cfg = load_skore_credentials()
login(mode="hub")
project = Project(name="bobathon-esilv", mode="hub", workspace=cfg["workspace"])
project.put("02_ridge", reports[best_alpha])   # l'URL s'affiche dans la console

# %% Fit sur tout le train + CSV Kaggle
best_pipe = make_pipeline(SimpleImputer(strategy="median"), Ridge(alpha=best_alpha))
final = clone(best_pipe).fit(X, y)

submission = X_test[["Index"]].copy()
submission["target"] = final.predict(X_test[FEATURE_COLS])

out = PROJECT_ROOT / "submissions"
out.mkdir(exist_ok=True)
submission.to_csv(out / "02_ridge.csv", index=False)
print(submission.shape, submission.head())