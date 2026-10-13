# %%
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