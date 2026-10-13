# Experiment: 02_ridge — Ridge regression alpha sweep

## Question / hypothesis

Can a Ridge regression on the eight available features beat the dummy RMSE of 16.48,
and does regularisation strength (alpha) meaningfully change the result on this dataset?

## Motivation

The EDA identified `off` (r = 0.871 with target) as the strongest predictor, plus
`age`, `age_at_diagnosis`, `time_since_intake_on`, and `time_since_intake_off` as
moderate signals. A linear model trained on all eight features should exploit these
correlations and substantially reduce RMSE. Sweeping alpha from 0.01 to 1000 answers
whether the feature set is already well-conditioned or whether shrinkage matters.

## Method

- Load `data/X_train.csv`, `data/y_train.csv`, merge on `Index`.
- Feature matrix `X` = `sexM`, `age_at_diagnosis`, `age`, `ledd`,
  `time_since_intake_on`, `time_since_intake_off`, `on`, `off`.
- `y` = `target`.
- Pipeline: `SimpleImputer(strategy="median")` → `Ridge(alpha=a)`.
- Evaluate each alpha with `skore.evaluate(pipe, X, y)` using the default
  `splitter=0.2` (same random 20 % row holdout as experiment 01 for comparability).
- Also evaluate `DummyRegressor(strategy="mean")` for the baseline.
- Alpha sweep: 0.01, 0.1, 1.0, 10, 100, 1000.
- Push best report to Skore Hub project `"bobathon-esilv"` with key `"02_ridge"`.
- Refit on the full training set; predict `X_test`; write
  `submissions/02_ridge.csv` with columns `Index, target`.

## Risks

- `splitter=0.2` is a random row holdout — the same patient can appear on both
  sides. This artificially flatters the Ridge (it may partially memorise patient
  baselines). RMSE will rise once we switch to GroupKFold in experiment 03.
- `off` is 42 % missing and `time_since_intake_off` is 79 % missing; median
  imputation is a rough substitute. Tree-based models or explicit missingness
  indicators may recover more signal.
- Near-collinearity between `age` and `age_at_diagnosis` (r = 0.94): Ridge
  handles this gracefully but the coefficient signs may be misleading.

## Results

**Split:** random 20 % row holdout (`splitter=0.2`), same as experiment 01.

| Model | RMSE | Δ vs dummy |
|---|---|---|
| Dummy (mean) | 16.4779 | — |
| Ridge alpha=1.0 | 10.4877 | −36.4 % |

**Alpha sweep** (all evaluated on the same default split):

| alpha | RMSE |
|---|---|
| 0.01 | ~10.4877 |
| 0.1 | ~10.4877 |
| 1.0 | 10.4877 |
| 10 | ~10.4877 |
| 100 | ~10.4877 |
| 1000 | 10.4881 |

Regularisation has negligible effect across five decades. The feature set is
well-conditioned and Ridge finds the same near-OLS solution for alpha ≤ 100.

**Top coefficients (alpha = 1.0):**

| Feature | Coefficient |
|---|---|
| time_since_intake_on | +1.70 |
| age | +1.07 |
| age_at_diagnosis | −0.96 |

Coefficients are quasi-identical across all alpha values, confirming that
shrinkage is not needed here.

**Best alpha selected:** 1.0

## Report hub

- **Skore Hub:** [02_ridge](https://skore.probabl.ai/Mathias%20Kozman/bobathon-esilv/estimators/67464)
- **Submission CSV:** `submissions/02_ridge.csv` (11 013 rows)

## Status

- **State:** done
- **Approved by user:** 2026-10-13
- **Headline result:** RMSE 10.4877 (random 20 % row holdout, `splitter=0.2`) — −36.4 % vs dummy
- **Implication for next iteration:** Alpha insensitivity suggests the data is well-conditioned for a linear model. The key risk is patient leakage in the random split; experiment 03 will re-evaluate with GroupKFold to get a leak-free estimate.
