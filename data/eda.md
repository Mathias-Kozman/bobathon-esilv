<!--
Exploratory data analysis summary — written from the data/eda.py run.
Ground every claim in what the run showed; do not invent facts.
-->

# EDA: Parkinson's Disease OFF Motor Score

_Generated from `data/eda.py` on 2025-10-07._

## Dataset at a glance

- **Tables:** 2 (training set, test set)
- **Shape:** 44 590 × 13 features + 1 target (train), 11 013 × 13 features (test)
- **Target:** `target` — continuous true / debiased OFF MDS-UPDRS motor score (regression)
- **Rich reports:** [eda_train.html](eda_train.html) · [eda_test.html](eda_test.html)

| Split | Rows | Patients |
|-------|-----:|--------:|
| Train | 44 590 | 5 576 |
| Test  | 11 013 | 1 395 |

Average visits per patient (train): **~8.0** (44 590 ÷ 5 576).

---

## Per-column findings

| Column | dtype | Null % | N unique | Notes |
|--------|-------|-------:|--------:|-------|
| `Index` | Int64 | 0 % | 44 590 | Row index — **drop before modelling** (unique per row) |
| `patient_id` | String | 0 % | 5 576 | Group key — repeated per patient. Train/test are disjoint. |
| `cohort` | String | 0 % | 2 | Binary categorical (2 values) |
| `sexM` | Int64 | 0 % | 2 | Binary flag (0/1) |
| `gene` | String | **32.4 %** | 4 | High missingness; 4 categories when present |
| `age_at_diagnosis` | Float | **5.2 %** | 563 | Moderate missingness |
| `age` | Float | 0 % | 1 082 | Age at visit; always present |
| `ledd` | Float | **36.6 %** | 1 320 | Levodopa equivalent daily dose; heavily missing |
| `time_since_intake_on` | Float | **46.4 %** | 64 | Discrete-ish (64 values); missing nearly half of visits |
| `time_since_intake_off` | Float | **78.8 %** | 178 | Very heavily missing (only ~21 % of visits have it) |
| `rater_id` | String | 0 % | 50 | 50 raters; low unique ratio — likely group-level bias signal |
| `on` | Float | **29.6 %** | 83 | Measured ON motor score; present for ~70 % of visits |
| `off` | Float | **42.4 %** | 101 | Measured OFF motor score; present for ~58 % of visits |
| `target` | Float | 0 % | 867 | **Never missing in train** — the label |

Key observations:
- `Index` is a row identifier with unique_ratio = 1.0 — must be dropped.
- `time_since_intake_off` is the most missing feature (78.8 %) — treat as a strong signal of
  "OFF test not performed", not just random missingness.
- `ledd` (36.6 % missing) and `gene` (32.4 % missing) may be systematically missing by cohort
  (see Open questions).
- No datetime column detected by dtype; no explicit visit-date column exists.

---

## Target

- **Type:** continuous regression (MDS-UPDRS motor score, 0–132 theoretical range)
- **Observed range:** 0.0 – 109.5 (does not reach the theoretical 132 ceiling)
- **Mean:** 37.5 · **Median:** 37.3 · **Std:** 16.5
- **Quartiles:** Q1 = 25.6 · Q3 = 49.3 · IQR = 23.7
- **Distribution:** roughly symmetric / slightly right-skewed (mean ≈ median).
  The histogram shows a fairly bell-shaped distribution peaking in the 32–44 bin
  (10 703 observations). A long right tail trails to ~110; no values above 110.
- **No missing values** in train (null_pct = 0.0).

The target is not severely skewed, so a log or power transform is not mandatory but could
slightly help if residuals appear heteroscedastic.

---

## Structure

**No datetime columns** were detected (no `date`-dtype columns). There is no explicit
visit-date feature.

**Group / id-like columns** (by unique ratio):

| Column | Unique ratio |
|--------|------------:|
| `Index` | 1.000 (pure row id) |
| `patient_id` | 0.125 (group key — ~8 visits/patient on average) |
| `ledd` | 0.030 (numeric, moderate cardinality) |
| `rater_id` | 0.001 (50 raters; potential rater-effect confounder) |

`patient_id` has **unique_ratio = 0.125**: the same patient appears roughly 8 times.
The Kaggle holdout is by patient (train patients ≠ test patients), so rows from the
same patient must never span a train/validation split boundary.

---

## Associations

**Feature ↔ target associations (Pearson correlation, sorted):**

| Feature | Pearson r | Cramér V | Verdict |
|---------|----------:|---------:|---------|
| `off` | **0.871** | 0.406 | Strongest predictor — **potential near-leakage** (see below) |
| `on` | 0.669 | 0.368 | Strong predictor |
| `ledd` | 0.298 | 0.236 | Moderate |
| `age` | 0.310 | 0.114 | Moderate |
| `age_at_diagnosis` | 0.133 | 0.051 | Weak |
| `time_since_intake_on` | ~0.000 | 0.177 | Near-zero linear, non-linear signal only |
| `time_since_intake_off` | 0.008 | 0.092 | Minimal linear signal |
| `cohort` | n/a | 0.090 | Weak |
| `gene` | n/a | 0.052 | Weak |
| `rater_id` | n/a | 0.019 | Very weak |
| `sexM` | -0.001 | 0.023 | Negligible |

**Notable feature ↔ feature associations:**

| Pair | Pearson r | Notes |
|------|----------:|-------|
| `age_at_diagnosis` ↔ `age` | 0.942 | Strong co-linearity — one may be redundant; can derive `time_since_diagnosis = age - age_at_diagnosis` |
| `on` ↔ `off` | 0.872 | High correlation between the two clinical measurements |
| `off` ↔ `target` | 0.871 | Near-identical correlation magnitude — see leakage flag below |

**⚠ Leakage flag — `off`:**  
`off` correlates with `target` at r = 0.871. The measured `off` score and the debiased
`target` are conceptually related (both measure OFF-state severity), but the
correlation is very high. In the real competition, `off` is **measured** (biased, often
missing) while `target` is the debiased ground truth; the whole point of the task is to
recover `target` when `off` is absent or unreliable. Using `off` as-is risks the model
learning to pass it through rather than understanding the underlying biology. It should
be treated with care: its missingness pattern (~42 % missing) is itself informative, and
when it is present, the model must not "cheat" by relying on it as a near-copy of the label.

---

## Modelling implications

1. **Splitter: `GroupKFold` on `patient_id`** — same patient must not appear in both
   train and validation folds. The Kaggle holdout is patient-level. `StratifiedKFold` or
   random split would leak patient trajectories.

2. **No time dimension available** — no visit-date column means `TimeSeriesSplit` is not
   directly applicable with the current features. Temporal ordering would need to be
   reconstructed from domain knowledge if wanted.

3. **Regression metric: RMSE** — continuous target, symmetric distribution.
   Also track MAE for robustness. No need for stratification gates.

4. **Drop `Index`** before modelling — it is a unique row counter with no predictive value
   and would confuse tree models.

5. **`patient_id` is a group key, not a feature** — pass to the splitter's `groups`
   argument; drop from feature matrix (or keep as low-cardinality string if
   patient-level averaging is desired, but handle leakage carefully).

6. **Missingness as signal** — `time_since_intake_off` is missing 79 % of the time,
   which likely encodes "OFF test was not performed". Tree-based models handle NaN
   natively; for linear models, add a binary missingness indicator.

7. **`off` and `on` — use with care** — strong predictors but heavily missing and
   potentially near-leakage for `off`. Options: (a) include with a missingness indicator,
   (b) treat as optional imputation step, (c) train two models (with/without).

8. **`age_at_diagnosis` + `age` co-linearity** — derive `time_since_diagnosis = age - age_at_diagnosis`
   (meaningful clinically as years of disease progression) and potentially drop one of the originals.

9. **`rater_id`** — 50 raters; may introduce systematic scoring bias. Consider as a
   group/categorical feature or encode effects.

10. **Target not severely skewed** — a log transform is optional; try without first.

---

## Open questions

1. **Is `off` truly available at inference time?** If the task requires predicting `target`
   without ever observing `off`, including it may inflate train scores. Clarify the
   competition rules on feature availability.
2. **Why is `ledd` missing?** Is it systematically absent for one cohort (pre-treatment
   patients)? Cohort-conditional imputation may be needed.
3. **What are the two `cohort` values?** Are they known clinical populations with different
   disease progression rates?
4. **`gene` missingness (32 %)** — is it missing at random, or does absence indicate a
   negative genetic test result (i.e., missingness = signal)?
5. **Does visit ordering within a patient matter?** No explicit date, but if visits are
   sorted by `age`, a patient's trajectory could be reconstructed.
