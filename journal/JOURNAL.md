# JOURNAL

<!--
Durable index of every experiment in this workspace. Four sections,
in order: Status, Data understanding (EDA), History, Backlog - keep
them so the file stays quick to scan. Each journal/NN_short_name.md
design note pairs one-to-one with experiments/NN_short_name.py (same
stem).
-->

## Status

- **Project / dataset:** Parkinson's disease OFF motor score — regression (synthetic multi-cohort)
- **Goal:** Predict the debiased true OFF MDS-UPDRS motor score (`target`) for every patient visit
- **Last experiment:** n/a - no experiments run yet
- **Last result:** n/a

<!--
Workspace decisions: one-time project-setup choices. Record each when
it is made and treat it as fixed unless you deliberately change one
(e.g. switch pandas → polars), updating the recorded date. Reading
this block on later sessions avoids re-deciding what's already settled.
-->

- **Workspace decisions** (immutable unless the user pivots):
  - tabular library: pandas - recorded: 2025-10-07
  - env manager: pip+venv - recorded: 2025-10-07
  - agent feature: <installed> - recorded: <YYYY-MM-DD>
  - optional features: <name1, name2 | none> - recorded: <YYYY-MM-DD>
  - package name (`src/<pkg>/`): <pkg> - recorded: <YYYY-MM-DD>
  - skore mode: <local | hub | mlflow> - recorded: <YYYY-MM-DD>
  - skore hub workspace: <hub-workspace-name | n/a> - recorded: <YYYY-MM-DD>
  - skore mlflow tracking uri: <mlflow-tracking-uri | n/a> - recorded: <YYYY-MM-DD>
  - student prior: <beginner | some-sklearn | comfortable> - recorded: <YYYY-MM-DD>
  - CV splitter family: GroupKFold (on `patient_id`) - recorded: 2025-10-07

## Data understanding (EDA)

<!--
Short index entry - the full analysis lives in data/eda.md. If the
data exploration was skipped, keep just the Status: skipped line.
-->

- **Status:** done - 2025-10-07
- **Summary:** 44 590 train rows × 13 features, 5 576 patients (~8 visits each), 1 395 test
  patients (disjoint from train). Target is a continuous OFF motor score (mean 37.5, range 0–110,
  roughly symmetric). Key findings: `patient_id` repeats across rows → **GroupKFold** required;
  `off` (r = 0.871 with target) is the strongest predictor but potentially near-leaky and
  42 % missing; `time_since_intake_off` is 79 % missing (treat absence as signal); `age_at_diagnosis`
  and `age` are near-collinear (r = 0.94). No datetime column detected; no temporal splitter needed
  unless visit order is reconstructed from `age`.
- **Report:** [data/eda.md](../data/eda.md)

## History

<!--
One row per experiment, in chronological order. Newest at the bottom.
Status values: planned | approved | running | done | abandoned.
-->

| Stem | Intent (one line) | Status | Headline result | Design note |
|---|---|---|---|---|
| — | — | — | — | — |

## Backlog

<!--
Ideas not yet committed to a journal/NN_*.md design note. Each row
has a stable B<N> index so it can be picked by number ("go with B2").
-->

| # | Item | Source |
|---|---|---|
| — | — | — |
