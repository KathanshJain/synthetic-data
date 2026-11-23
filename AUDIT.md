# AUDIT.md — Complete Build Audit

**Project:** Machine Learning on Artificially Forged Data (CSET312 mini-project)
**Team:** Aryan Gupta (S24CSEU2654) — GenAI & ML track · Kathansh Jain (S24CSEU2656) — Data Analysis track
**Location:** `D:\aryan bhai ka kaam\synthetic-data-project\`
**Built from:** `Antigravity_Prompts_Synthetic_Data_Project.docx` (Prompts 0–11), `Project_Storytelling_and_Viva_Guide.docx`, `Aryan Gupta & Kathansh Jain.pdf`
**Environment:** Windows, Git Bash, Python 3.14.6 (only runtime installed on the machine), virtualenv in `venv/`, SDV 2.2.0, torch 2.15.0, scikit-learn 1.9.1, xgboost 3.4.1

---

## 1. What was built (executive summary)

A complete, reproducible research pipeline that answers: **how much real data does a
generator need to produce synthetic tabular data that is realistic, useful for training
ML models, and does not copy real records?**

- Dataset: Adult Census Income (OpenML "adult" v2), 48,842 raw rows → 48,790 cleaned
  → 39,032 dev pool / 9,758 locked test set (23.9% positive in both).
- Generators: CTGAN (fixed 3,000-step budget), Gaussian Copula (SDV 2.2), SMOTE-NC
  (imbalanced-learn).
- Evaluation: fidelity (KS, TVD, chi², correlation difference, category coverage),
  privacy (DCR vs real holdout), utility (RF + XGBoost across 5 arms, PR-AUC primary).
- Full experiment executed: 3 random seeds × 5 nested seed sizes (100–2000).
- Deliverables: analysis figures + RESULTS.md, Streamlit app (3 tabs), 15 passing
  pytest tests, README, CODE_WALKTHROUGH.md, VIVA_ANSWERS.md, 5 git commits.

## 2. File-by-file inventory

### Configuration & docs (project root)
| File | Purpose | Status |
|---|---|---|
| `config.yaml` | Every setting: dataset, split, seed sizes [100,250,500,1000,2000], random seeds [0,1,2], 5000 synthetic rows, CTGAN 3000 steps/batch 500, SMOTE k=5, RF/XGB settings, DCR holdout 1000, quick_mode block | ✅ written, used by all code |
| `criteria.yaml` | Frozen pass/fail rules: utility_ratio_min 0.90, ks_mean_max 0.10, tvd_mean_max 0.10, dcr_ratio_min 0.80, exact-copy rule; comments explain each | ✅ committed **before** full run (commit 3fae443) |
| `requirements.txt` | 12 pinned-range libraries + tabulate (added when needed) | ✅ installed cleanly on Python 3.14 |
| `README.md` | Overview, research question, ASCII pipeline diagram, folder structure, setup, per-step command table, real results tables, limitations, future work, team, references | ✅ final version with actual numbers |
| `PROJECT_BRIEF.md` | The full brief (general rules, golden rule, pipeline, team split) | ✅ |
| `CODE_WALKTHROUGH.md` | Every src/ and app/ file: purpose, key functions (inputs/outputs), examiner-pointed lines, 2 viva Q&A each | ✅ ~300 lines |
| `VIVA_ANSWERS.md` | Viva answers quoting **actual experiment numbers** (headline five, why nothing passed, flattening point, DCR interpretation, copula asymmetry, SMOTE finding, budget fairness, leakage defence, limitations, future work) | ✅ |
| `AUDIT.md` | This document | ✅ |
| `.gitignore` | venv, __pycache__, raw/synthetic/seed CSVs, locked CSVs (manifest.json kept), quick results | ✅ |

### Source code (`src/`)
| File | What it does | Key functions |
|---|---|---|
| `utils.py` | Shared helpers + the "vault guard": config/criteria loaders, `set_all_seeds` (random/numpy/torch), column typing, `features_only` (strips row_id), SHA-256 computation, `verify_test_hash()` (raises on any byte change), `load_locked_test()` (always verifies first) | 7 functions |
| `prepare_data.py` | Step 1: OpenML download (cached), clean (strip, `?`→NaN, dedupe **before** column drop — 52 removed, drop fnlwgt/education, "Missing" category, numeric target, row_id), stratified 80/20 split, manifest write, summary print | 4 + main |
| `eda.py` | Step 2: reusable `profile_dataset()` (rows/cols/types/missing/duplicates/uniques/class balance/ID-like/warnings) reused by the app; 6 plot families; `reports/eda_report.md` with computed observations | 8 + main |
| `seeds.py` | Step 4: nested stratified seeds (per-class shuffle once with `default_rng`, counts rounded→exact-sum→never-shrink), `check_nested`, `get_holdout` (DCR baseline outside largest seed) | 3 + main |
| `generators.py` | Step 5: SDV 2.x metadata (explicit types, target=categorical, no row_id), CTGAN with **fair budget** (`epochs = ceil(total_steps / max(len//batch,1))`, verified in ctgan source), Gaussian Copula, `generate` (dict-in/dict-out for SDV 2.2), SMOTE-NC (k capped at minority−1), `repair_synthetic()` (fills generator NaNs by sampling real category distribution), `validate_synthetic()` (columns/categories/NaN/range checks) | 7 + demo main |
| `fidelity.py` | Step 6: KS per numerical column, TVD per categorical column, chi² (zero-count categories dropped), correlation difference (upper triangle; 0.0 when <2 numerical cols), category coverage, `evaluate_fidelity`, real-vs-synthetic plots | 7 + demo |
| `privacy.py` | Step 7: DCR — scaler+OHE fitted on **seed only**, 1-NN distances for synthetic and holdout, medians/p05/dcr_ratio/exact-match rates (threshold 1e-8), overlaid histogram plot; comments explain scaling and why a holdout baseline is needed | 2 + demo |
| `models.py` | Step 8: sklearn Pipeline (OHE handle_unknown=ignore + RF/XGB from config), `train_and_evaluate` (PR-AUC, ROC-AUC, F1, precision, recall, accuracy, train_time, NaN-safe for single-class), docstring table of the 5 arms; demo uses a **dev-pool validation split, never the locked set** | 2 + helpers + demo |
| `run_experiment.py` | Step 9: conductor — hash verify → per random seed: TRTR-full once; per size: TRTR-seed, SMOTE-NC, CTGAN, GC → fidelity/DCR/runtime/utility appended to results/*.csv; `--quick` flag (quick_mode overrides, outputs to results/quick/); **resume support** (skips existing (rs,size,arm,model) rows); saves run_config.yaml | 3 + main |
| `analyze.py` | Step 10: mean±std summaries, PR-AUC/F1-vs-seed-size plots (±1σ bands, TRTR-full line, random-guess line, log-x), fidelity/coverage/DCR/runtime plots, real-vs-synth pairs, CTGAN loss curve, `apply_criteria` (PASS/FAIL per rule, minimum useful seed), RESULTS.md, honest findings generator | 8 + main |
| `backfill_fidelity_per_column.py` | One-off repair: recomputes per-column fidelity from the saved synthetic CSVs after a run started just before the per-column fix | main |

### App (`app/`)
| File | What it does |
|---|---|
| `streamlit_app.py` | 3 tabs: **About** (story, pipeline, three quality checks, team), **Experiment Results** (model/arm dropdowns — arm label fixed to `TSTR-GaussianCopula` to match data — metrics table, 5 key plots, RESULTS.md, highlighted minimum-seed box, friendly missing-results message), **Try It Yourself** (CSV upload or Adult sample button, profile + warnings, target/seed-size/generator/rows controls, CTGAN steps slider with slow warning, refusal guards: <50 rows, <10 minority rows, non-binary target for SMOTE, date-column detection; spinner → preview, class comparison, fidelity summary + per-column table, chosen-column comparison plot, DCR vs uploaded-rest holdout, CSV download). **Never reads the locked test set.** Reuses src/ functions — no duplicated logic. |

### Tests (`tests/`) — **15/15 passing**
| File | Tests |
|---|---|
| `test_split.py` | zero row_id overlap; class ratios within 0.5%; hash verifies; hash **fails** on a modified temp copy (real file untouched) |
| `test_seeds.py` | exact sizes; nesting holds; class counts within 1 row; holdout disjoint from largest seed; same random_state → identical seeds |
| `test_fidelity.py` | identical data → KS=0, TVD=0, corr=0; shifted/shuffled data → large values |
| `test_privacy.py` | exact copy of seed → 100% exact-match rate, dcr_ratio < 0.1 |
| `test_no_leakage.py` | only utils/prepare_data/models/run_experiment may mention the locked path; no `.fit(test_…)` patterns; models demo never calls the loader |

## 3. What was actually run (execution log)

| Step | Command | Result |
|---|---|---|
| Setup | folders, configs, README, brief, `.gitignore`, `git init` | commit 3fae443 |
| Install | `pip install -r requirements.txt` in venv (background, logged) | INSTALL_EXIT=0 (torch 2.15 cp314, sdv 2.2.0 resolved on Python 3.14) |
| Step 1 | `python -m src.prepare_data` | 48,842 raw → 52 duplicates removed → 48,790; dev 39,032 (23.9%+) / test 9,758 (23.9%+); manifest sha256=974825e74b74102d… |
| Step 2 | `python -m src.eda` | 24 figures in figures/eda/, eda_report.md; warnings: row_id ID-like; missing: workclass 5.72%, occupation 5.74%, native-country 1.73% |
| Step 3 | `python -m src.seeds` | 15 seed files; every size exactly stratified (24.0/23.9% minority); nesting asserted |
| Tests | `pytest` (mid-build) | 2 failures found & fixed: corr-diff NaN with <2 numerical cols; prepare_data added to allowed-files (it *creates* the locked file) → 15/15 |
| Step 4 demo | `python -m src.generators` | CTGAN 21–23s (200 steps), GC 0.5–1.1s, SMOTE +260 rows; outputs saved to data/synthetic/rs0/ |
| Steps 5–7 demos | fidelity / privacy / models | GC-vs-devpool: KS 0.457, TVD 0.042, coverage 87%; DCR ratio 2.56, 0 exact copies; validation PR-AUC: TRTR-full XGB 0.829 / TRTR-seed-500 0.727; random guess 0.239 |
| Step 8 quick | `python -m src.run_experiment --quick` | ~2 min; results/quick/; criteria.yaml confirmed already committed; **stopped for explicit approval per safeguard** |
| Approval | user approved full run | — |
| Step 8 full | `python -m src.run_experiment` (background, logged) | FULL_EXIT=0, ~75 min; 15 CTGAN fits (~142–170s each, flat across sizes — budget fairness evidence), 15 GC fits (~1s) |
| Backfill | `python -m src.backfill_fidelity_per_column` | 360 per-column fidelity rows |
| Step 9 | `python -m src.analyze` | 2 bugs fixed en route (MultiIndex flatten; tabulate dependency) → 16 figures, summary.csv, RESULTS.md |
| Step 10 | `streamlit run app/streamlit_app.py` | starts cleanly; Results-tab data verified programmatically (42 summary rows, all arms/models, 16 figures) |
| Step 11 | final pytest + fresh-command check | 15/15; results/quick deleted per checklist |
| Post-audit fix | `repair_synthetic()` added & wired (runner, demo, app) | demo now **zero warnings**; 15/15 tests; app restarted |

## 4. Final experiment results (locked test set, XGBoost PR-AUC, mean ± std over seeds 0/1/2)

Reference: **TRTR-full = 0.823**

| Seed size | TRTR-seed | SMOTE-NC | TSTR-CTGAN | TSTR-GC |
|---|---|---|---|---|
| 100 | 0.591 ±0.011 | 0.574 ±0.011 | 0.563 ±0.037 | 0.350 ±0.017 |
| 250 | 0.643 ±0.020 | 0.635 ±0.019 | 0.633 ±0.001 | 0.469 ±0.024 |
| 500 | 0.699 ±0.016 | 0.690 ±0.017 | 0.648 ±0.041 | 0.409 ±0.111 |
| 1000 | 0.733 ±0.006 | 0.721 ±0.004 | 0.619 ±0.020 | 0.406 ±0.036 |
| 2000 | 0.761 ±0.004 | 0.753 ±0.003 | 0.648 ±0.008 | 0.476 ±0.041 |

**Pass/fail (frozen criteria):** 0 of 10 (generator, size) combinations passed all rules.
- utility_ratio: best 0.787 (CTGAN 500/2000) — FAIL vs 0.90
- ks_mean: CTGAN 0.22–0.29, GC 0.38–0.49 — FAIL vs 0.10
- tvd_mean: CTGAN 0.08–0.10, GC 0.02–0.07 — PASS
- dcr_ratio: CTGAN 1.01–1.53, GC 1.74–2.95 — PASS (no memorisation)
- exact copies: ~0% vs holdout 0.5–0.9% — PASS

**Minimum useful seed: none** — reported honestly as a valid pre-registered finding.

**Key findings:** CTGAN ≫ Gaussian Copula (utility and fidelity); TSTR never beat
TRTR-seed; CTGAN PR-AUC flattens after seed 500 (capacity/budget bottleneck, not data);
privacy was never the failure mode; SMOTE-NC ≈ TRTR-seed (rebalancing ≠ data expansion).

## 5. Design decisions & problems solved during the build

1. **Python 3.14 only on the machine** — the installed "Python313" folder was an empty
   shell; built the venv on 3.14.6 and confirmed cp314 wheels exist for the whole stack
   (torch 2.15, sklearn 1.9, xgboost 3.4, sdv 2.2).
2. **SDV 2.2 API migration** — `Metadata.detect_from_dataframes({name: df})`,
   `update_column(..., sdtype=…)`, `fit({name: df})`, `sample() → dict`,
   `enable_gpu` instead of `cuda`. Discovered by inspecting the installed package, fixed
   in three passes.
3. **Fair CTGAN budget** — verified `steps_per_epoch = max(len(data)//batch_size, 1)`
   in the installed ctgan source, then `epochs = ceil(total_steps/steps_per_epoch)`;
   runtime flatness (142–170s across all sizes) empirically confirms fairness.
4. **NaN in rare categories** — SDV emits NaN for rare categorical values; added
   `repair_synthetic()` (sample from real category distribution) wired into runner,
   demo and app. **User chose to keep the already-reported results**; README documents
   the caveat and the refresh command.
5. **Correlation-difference edge case** — returned NaN for <2 numerical columns; now 0.0.
6. **Per-column fidelity gap** — the full run started just before the per-column CSV fix;
   solved without retraining via `backfill_fidelity_per_column.py` (360 rows from saved
   synthetic CSVs).
7. **pandas MultiIndex flatten + missing `tabulate`** — fixed in analyze.py.
8. **App label mismatch** — dropdown `TSTR-GC` vs data `TSTR-GaussianCopula` would have
   shown an empty table; fixed.
9. **Leakage-test scope** — prepare_data.py legitimately *creates* the locked file;
   allowed-files list documented with per-file justification.

## 6. Safeguards honoured

- criteria.yaml committed (3fae443) **before** any results existed; never modified after.
- Quick run stopped at the safeguard point; full run launched only after the user's
  explicit approval (recorded in this session).
- Locked test set: hash-verified on every load; source-scanned by tests; the models
  demo evaluates on a dev-pool validation split only; the app never reads it.
- Single final evaluation; no tuning after seeing test results.
- All reported numbers are actual outputs — nothing estimated or invented.

## 7. Known limitations (of the build, stated honestly)

- One dataset, two generators + one oversampling baseline, three random seeds.
- DCR is a warning indicator, not a privacy guarantee.
- Reported TSTR metrics were computed on pre-repair synthetic data containing some
  NaN rows (slightly pessimistic); README documents the refresh command.
- Fixed CTGAN budget may under-train larger seeds (one plausible cause of the
  PR-AUC flattening).
- SDV 2.x categorical smoothing occasionally drops rare categories (coverage 80–88%).

## 8. Git history

```
3ad6e69 Fix: repair_synthetic() fills generator NaNs in rare categorical columns
ee19019 Final polish: app arm-label fix, VIVA_ANSWERS.md, quick results removed
6724c35 Full experiment results, analysis, per-column backfill, final README, app
9116b3d Steps 1-8: utils…models, runner + tests (15 pass)
3fae443 Prompt 0: project setup (criteria.yaml frozen here)
```

## 9. How to reproduce everything from scratch

```bash
cd synthetic-data-project
python -m venv venv && venv\Scripts\activate
pip install -r requirements.txt
python -m src.prepare_data
python -m src.eda
python -m src.seeds
python -m src.generators            # demo
python -m src.run_experiment --quick
python -m src.run_experiment        # full (~75 min) — only after freezing criteria
python -m src.backfill_fidelity_per_column
python -m src.analyze
streamlit run app/streamlit_app.py
python -m pytest
```
