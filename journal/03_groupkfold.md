# Experiment: 03_groupkfold — CV groupé par patient

## Question / hypothesis

Quand on remplace le split aléatoire par un CV groupé par patient, les RMSE
changent-ils significativement ? Le split aléatoire faisait-il fuiter l'information
patient vers l'ensemble de validation ?

## Motivation

L'expérience 02 a utilisé `splitter=0.2` (tirage aléatoire de lignes). Or le dataset
contient ~8 visites par patient : avec un split aléatoire, le même patient peut
figurer en train et en validation simultanément. Le modèle peut ainsi « voir » le
niveau de base du patient pendant l'entraînement, ce qui gonfle artificiellement le
score. GroupKFold garantit que chaque patient est soit entièrement en train, soit
entièrement en validation — c'est le seul estimateur non-biaisé ici.

## Method

- Mêmes données et features que 02 : `data/X_train.csv` + `data/y_train.csv`,
  merge sur `Index`, 8 colonnes (`sexM`, `age_at_diagnosis`, `age`, `ledd`,
  `time_since_intake_on`, `time_since_intake_off`, `on`, `off`).
- `groups = visits["patient_id"]`
- `cv_splits = list(GroupKFold(n_splits=5).split(X, y, groups=groups))`
  → 5 folds, ~8 917–8 921 lignes de validation chacun.
- Évaluer séparément :
  - `DummyRegressor(strategy="mean")` avec `skore.evaluate(..., splitter=cv_splits)`
  - `make_pipeline(SimpleImputer(strategy="median"), Ridge(alpha=1.0))` idem
- Push des deux CrossValidationReport sur le hub : clés `03_dummy_cv` et
  `03_ridge_cv`. Pas de ComparisonReport (refusé par `project.put`).
- `cv_splits` sera réutilisé dans toutes les expériences suivantes.

## Risks

- Avec 5 folds sur 5 576 patients, chaque fold de validation couvre ~1 115
  patients. La variance entre folds reste faible (std ≈ 0.27–0.34 RMSE) mais
  non nulle : de petites différences entre modèles peuvent encore être du bruit.
- GroupKFold ne stratifie pas sur la cible : les folds pourraient avoir des
  distributions légèrement différentes. Avec 5 576 patients c'est peu probable.
- Le dummy n'utilise pas les features, son RMSE en CV groupé reflète uniquement
  la variance de la cible entre patients — c'est le vrai floor.

## Results

**CV :** GroupKFold(n_splits=5), groupé par `patient_id`.

### Dummy (DummyRegressor mean)

| Fold | Train rows | Val rows | RMSE val |
|---|---|---|---|
| 0 | 35 672 | 8 918 | (agrégé) |
| 1 | 35 669 | 8 921 | (agrégé) |
| 2 | 35 673 | 8 917 | (agrégé) |
| 3 | 35 673 | 8 917 | (agrégé) |
| 4 | 35 673 | 8 917 | (agrégé) |

**RMSE moyen = 16.4992 (std = 0.343)**

### Ridge (SimpleImputer median + Ridge alpha=1.0)

**RMSE moyen = 10.5965 (std = 0.267)**

### Comparaison split aléatoire (exp 02) vs CV groupé (exp 03)

| Modèle | RMSE split aléatoire | RMSE CV groupé | Δ absolu |
|---|---|---|---|
| Dummy | 16.4779 | 16.4992 | +0.02 |
| Ridge | 10.4877 | 10.5965 | +0.11 |

Le dummy est quasi-inchangé (+0.02), ce qui est attendu — il ne tire pas
parti de l'information patient. Le Ridge augmente de +0.11 RMSE (+1.0 %) :
effet de fuite modéré mais réel. Le split aléatoire surestimait légèrement
les performances du Ridge.

## Questions workbook

**Quel modèle change le plus entre split aléatoire et CV groupé ?**

Le Ridge. Le dummy est insensible au type de split (il prédit toujours la
même constante), donc son RMSE ne varie que par la distribution de la cible
dans le fold. Le Ridge, lui, exploite des features corrélées au niveau de
base du patient (`off`, `age`, etc.) ; avec un split aléatoire, il peut
avoir « appris » ce niveau de base depuis d'autres visites du même patient,
ce qui réduit artificiellement son erreur. En CV groupé cette fuite disparaît
et le RMSE remonte légèrement.

**Pourquoi le CV groupé est-il plus fiable ?**

Parce qu'il simule la vraie condition de déploiement : prédire le score d'un
patient *que le modèle n'a jamais vu pendant l'entraînement*. Le split
aléatoire contamine la validation : des visites du même patient peuvent
figurer des deux côtés de la coupure, et le modèle peut indirectement
mémoriser le niveau de base individuel. Le CV groupé garantit que chaque
patient appartient entièrement soit au train soit à la validation — c'est
l'estimateur de généralisation non-biaisé pour des données longitudinales.

**Score Kaggle (soumission publique) :**

*(Non disponible — soumission pas encore faite.)*

## Report hub

- **Dummy CV groupé :** [03_dummy_cv](https://skore.probabl.ai/Mathias%20Kozman/bobathon-esilv/cross-validations/67962)
- **Ridge CV groupé :** [03_ridge_cv](https://skore.probabl.ai/Mathias%20Kozman/bobathon-esilv/cross-validations/68005)

## Status

- **State:** done
- **Approved by user:** 2026-10-13
- **Headline result:** Ridge RMSE 10.5965 (CV groupé, 5 folds) — dummy RMSE 16.4992
- **Implication for next iteration:** Le CV groupé est maintenant la référence pour toutes les expériences suivantes (`cv_splits`). La fuite patient dans le split aléatoire est faible (+0.11 RMSE) mais réelle. Prochain axe : tenter des modèles non-linéaires (Random Forest, Gradient Boosting) qui peuvent mieux exploiter `off` et gérer les valeurs manquantes.
