# Experiment: 01_dummy — mean baseline

## Question / hypothesis

Does the global mean of `target` over all training visits beat
random chance, and what RMSE does it produce as the floor for all
future models?

## Motivation

A dummy mean model always predicts the same value (the training
mean) regardless of features. Its RMSE equals the test-set standard
deviation of the target when the holdout is a random row sample —
which is essentially what `splitter=0.2` gives. Any model that
fails to beat this floor has learnt nothing from its features.
Per the Day 1 lab guide, this experiment is mandatory before any
real modelling.

## Method

- Load `data/X_train.csv`, `data/y_train.csv`, merge on `Index`.
- Feature matrix `X` = the eight columns listed in the guide:
  `sexM`, `age_at_diagnosis`, `age`, `ledd`, `time_since_intake_on`,
  `time_since_intake_off`, `on`, `off`.
- `y` = `target`.
- Learner: `sklearn.dummy.DummyRegressor(strategy="mean")`.
- Evaluate with `skore.evaluate(dummy, X, y)` using the default
  `splitter=0.2` (random 20 % row holdout). No grouped split — the
  patient-ID leak is intentional here; it shows the floor under
  the same conditions a naive model would face.
- Push report to Skore Hub project `"bobathon-esilv"` with key
  `"01_dummy"`.
- Refit on the full training set; predict `X_test`; write
  `submissions/01_dummy.csv` with columns `Index, target`.

## Risks

- `splitter=0.2` is a random row holdout; the same patient may
  appear on both sides of the split. This inflates the apparent
  score for any patient-aware model but is irrelevant for a dummy
  (it predicts the same value everywhere). The risk lands in
  experiments 02+.
- `off` is ~42 % missing; its presence in `X` has no effect on
  the dummy but will be important once real models are trained.

## Status

- **State:** done
- **Approved by user:** 2025-10-07
- **Headline result:** RMSE 16.48 (random 20 % row holdout, `splitter=0.2`)
- **Implication for next iteration:** Any model using actual features must beat RMSE 16.48 to be useful. Ridge regression on the same features should close much of the gap given that `off` (r = 0.871 with target) is in the feature set.
