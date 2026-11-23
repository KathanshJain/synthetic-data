# PROJECT_BRIEF.md — Machine Learning on Artificially Forged Data

I am a 3rd-year BTech AI/ML student. This file is the single source of truth for the
mini-project. Every step of the build reads it first.

## General rules (followed in every step)

- Python 3.11+ style. Libraries: pandas, numpy, scikit-learn, xgboost, imbalanced-learn,
  sdv, scipy, matplotlib, seaborn, streamlit, pyyaml, pytest.
- Simple functions, no complex class hierarchies, no clever tricks. Clear variable names.
- Every `.py` file starts with a docstring explaining in plain English what the file does
  and why it exists. Comments explain WHY, not just what.
- All settings come from `config.yaml` (no hard-coded numbers in code).
- **GOLDEN RULE:** the locked test set is used ONLY for the final model evaluation. It must
  never be used for cleaning decisions, fitting encoders, seed selection, training
  generators, or tuning. Its SHA-256 hash is verified before every load.
- Random seeds are fixed everywhere (numpy, random, torch, sklearn, xgboost) for
  reproducibility.

## Research question

How much real data does a generator need to produce synthetic tabular data that is
realistic, useful for training ML models, and does not copy real records?

## Dataset

Adult Census Income (OpenML "adult", version 2), binary target income >50K
(about 24% positive).

## Pipeline

1. Load + clean data (drop fnlwgt and education, missing values become the category
   "Missing", drop exact duplicate rows BEFORE dropping columns, add a row_id column).
2. EDA report on the development pool only.
3. Stratified split: 80% development pool, 20% locked test set + SHA-256 manifest.
4. Nested stratified seeds of size 100, 250, 500, 1000, 2000 from the dev pool (each
   smaller seed is contained in the larger one), repeated for random seeds 0, 1, 2.
5. Generators on each seed: CTGAN (SDV) with a fixed total number of training steps,
   Gaussian Copula (SDV), and SMOTE-NC (imbalanced-learn). Generators output 5000 rows.
6. Fidelity: KS (numerical), TVD + chi-square (categorical), correlation difference,
   category coverage — synthetic vs dev pool.
7. Privacy: Distance to Closest Record (DCR) vs a real-holdout baseline.
8. Utility: Random Forest and XGBoost trained on 5 arms (TRTR-full, TRTR-seed, SMOTE-NC,
   TSTR-CTGAN, TSTR-GC), all evaluated on the locked test set. Primary metric PR-AUC;
   also ROC-AUC, F1, precision, recall, accuracy.
9. Analysis: mean ± std over random seeds, plots vs seed size, apply criteria.yaml to
   find the minimum useful seed.
10. Streamlit app: About, Experiment Results, Try It Yourself (upload any CSV).

## Team

- Kathansh = data analysis track (data, EDA, seeds, fidelity, privacy).
- Aryan = GenAI & ML track (generators, models, experiment runner, app).
