# CODE_WALKTHROUGH.md — every file explained for the viva

A plain-English guide to every source file: what it does, its key functions,
the lines an examiner might point at, and likely viva questions.

---

## src/utils.py — shared helpers and the "vault guard"

**What it does:** small functions every step needs: loading config, fixing
random seeds, splitting columns into numerical/categorical, and — most
importantly — the guarded loader for the locked test set.

**Key functions**
| Function | Input | Output |
|---|---|---|
| `load_config()` | config.yaml path | dict of all settings |
| `set_all_seeds(seed)` | an integer | seeds python/numpy/torch |
| `get_column_types(df, target)` | DataFrame | (numerical list, categorical list) |
| `features_only(df)` | DataFrame | DataFrame without row_id |
| `compute_file_sha256(path)` | file path | 64-char hash string |
| `verify_test_hash()` | nothing | raises if the locked file changed |
| `load_locked_test()` | nothing | the test DataFrame (hash verified first) |

**Lines an examiner might point at**
- `compute_file_sha256` reads the file in 64 KB chunks — hashing a big file
  without loading it all into memory.
- `verify_test_hash` compares the *recomputed* hash with the one stored at
  creation time; a single changed byte produces a totally different hash.

**Viva questions**
1. *Q: Is a SHA-256 hash encryption?* A: No. It is a one-way fingerprint for
   integrity checking. It does not hide the file and cannot prove nobody
   viewed it.
2. *Q: Why must row_id never reach a generator?* A: It is a unique label per
   row; a generator that learned it would just memorise IDs and it carries
   no real information.

---

## src/prepare_data.py — Step 1: clean data and lock the test set

**What it does:** downloads Adult from OpenML (once), cleans it, splits
80/20 into dev pool + locked test set, writes the SHA-256 manifest.

**Key functions**
| Function | Input | Output |
|---|---|---|
| `download_raw_data(config)` | config | raw DataFrame (+ data/raw/adult.csv) |
| `clean_data(df, config)` | raw DataFrame | cleaned DataFrame with row_id |
| `stratified_split(df, config)` | cleaned DataFrame | (dev_pool, test_set) |
| `write_manifest(test_set)` | test DataFrame | manifest.json with hash |

**Lines an examiner might point at**
- Duplicates are dropped **BEFORE** dropping columns. If we dropped fnlwgt
  first, two people differing only in fnlwgt would look identical and we
  would wrongly delete a real record.
- Missing values become the category `"Missing"` instead of deleted rows —
  this preserves the class ratio (deleting rows would bias it).
- `train_test_split(..., stratify=df[target])` keeps the class ratio in both
  parts.

**Viva questions**
1. *Q: Why is the split stratified?* A: So dev pool and test set both keep
   the ~24% positive rate; otherwise the test set could be unrepresentative.
2. *Q: Why keep missing values as a category?* A: In Adult, missingness is
   informative (e.g. no occupation often means unemployed), and dropping
   rows would shrink the minority class.

---

## src/eda.py — Step 2: exploratory data analysis

**What it does:** profiles the dev pool and produces all EDA figures and a
Markdown report. `profile_dataset()` is reused by the web app's upload
profiling.

**Key functions**
| Function | Input | Output |
|---|---|---|
| `profile_dataset(df, target)` | any DataFrame | dict: counts, types, missing, warnings |
| `plot_*` family | DataFrame + column names | PNG files in figures/eda/ |
| `write_eda_report(profile, path)` | profile dict | reports/eda_report.md |

**Lines an examiner might point at**
- ID-like detection: `df[c].nunique() > 0.95 * len(df)` — a column where
  almost every row differs is probably an ID, not a feature.
- Warnings are computed from data (e.g. minority class < 20 rows), not
  hard-coded.

**Viva questions**
1. *Q: What is class imbalance?* A: One class has far fewer rows (~24%
   >50K vs 76% ≤50K); it breaks accuracy and motivates PR-AUC and SMOTE.
2. *Q: What is skewness?* A: A distribution lopsided to one side —
   capital-gain is mostly 0 with a few huge values.

---

## src/seeds.py — Step 4: nested stratified seeds

**What it does:** builds the 100→2000 row samples and the DCR holdout.

**Key functions**
| Function | Input | Output |
|---|---|---|
| `build_nested_seeds(dev_pool, target, sizes, rs)` | dev pool | {size: DataFrame} |
| `check_nested(seeds)` | seeds dict | True if smaller ⊂ larger |
| `get_holdout(dev_pool, largest_seed, size, rs)` | dev pool | holdout DataFrame |

**Lines an examiner might point at**
- Each class's indices are shuffled **once**; each seed takes the *first k*
  of that fixed order — that is what makes seeds nested (Russian dolls).
- Counts are rounded, then adjusted to sum exactly to n and never shrink —
  nesting survives rounding.

**Viva questions**
1. *Q: Why nested seeds?* A: Like Russian dolls — the 100-row sample is
   inside the 250-row one, so differences come from sample size, not from
   which rows were picked.
2. *Q: Why stratify the seeds?* A: A random 100 rows might contain almost
   no >50K people; stratification keeps the class ratio in every seed.
|---|---|---|
| `check_nested` | seeds dict | boolean |
| `get_holdout` | dev pool, largest seed | rows outside the seed |

**Lines an examiner might point at**
- `rng.shuffle(cls_indices)` happens once per class; all seeds slice the
  same shuffled order.
- The holdout is sampled only from rows NOT in the largest seed.

**Viva questions**
1. *Q: What is the holdout used for?* A: The DCR baseline — genuine records
   the generator never saw, so "closeness to the seed" has a fair reference.
2. *Q: Why three random seeds?* A: One sample could be lucky/unlucky; three
   lets us report mean ± std.

---

## src/generators.py — Step 5: CTGAN, Gaussian Copula, SMOTE-NC

**What it does:** trains the three generators on a seed and samples rows.

**Key functions**
| Function | Input | Output |
|---|---|---|
| `make_metadata(df, ...)` | seed DataFrame | SDV metadata with explicit types |
| `train_ctgan(seed_df, config, rs)` | seed | (synthesizer, seconds, losses) |
| `train_gaussian_copula(...)` | seed | (synthesizer, seconds) |
| `generate(synthesizer, n)` | fitted model | (DataFrame, seconds) |
| `run_smote_nc(...)` | seed + config | (resampled DataFrame, n new rows) |
| `validate_synthetic(...)` | synthetic vs real | list of warnings |

**Lines an examiner might point at**
- `epochs = ceil(total_steps / steps_per_epoch)` — the fairness fix: CTGAN
  computes steps/epoch as `max(len(data)//batch_size, 1)`, so without this a
  100-row seed would get ~50× more epochs per "epoch" than a 2000-row seed.
- `k_neighbors = min(k, minority_count - 1)` — SMOTE cannot use more
  neighbours than exist in the minority class.
- Metadata sets the target as **categorical** — it is a label, not a number.

**Viva questions**
1. *Q: How does a GAN learn?* A: The generator invents rows; the
   discriminator guesses real vs fake; both improve until fakes look real.
2. *Q: Why fix total CTGAN steps?* A: Otherwise bigger seeds would train
   longer, confounding "more data" with "more training".

---

## src/fidelity.py — Step 6: does synthetic data look real?

**Key functions**
| Function | Input | Output |
|---|---|---|
| `ks_per_column` | real, synth, numerical cols | KS statistic + p per column |
| `tvd_per_column` | real, synth, categorical cols | TVD per column |
| `chi_square_per_column` | real, synth | chi² statistic + p per column |
| `correlation_difference` | real, synth | mean abs diff of correlations |
| `category_coverage` | real, synth | % real categories present |
| `evaluate_fidelity` | everything | (per-column table, summary dict) |

**Lines an examiner might point at**
- TVD = 0.5 × Σ|p_real − p_synth| over the union of categories; 0 means
  identical proportions.
- We decide with KS/TVD *statistics*, not p-values — p-values shrink toward
  0 as sample size grows even for trivial differences.

**Viva questions**
1. *Q: KS in one sentence?* A: The biggest vertical gap between the real and
   synthetic cumulative distribution curves of a numerical column.
2. *Q: Why category coverage?* A: A small seed may never contain rare
   categories (e.g. some native-country values), so the generator cannot
   produce them — coverage quantifies that loss.

---

## src/privacy.py — Step 7: DCR memorisation check

**Key functions**
| Function | Input | Output |
|---|---|---|
| `compute_dcr(seed, synth, holdout, ...)` | three DataFrames | medians, p05, ratio, exact-match rates |
| `plot_dcr(...)` | two distance arrays | overlaid histogram PNG |

**Lines an examiner might point at**
- Scaler + one-hot encoder are fitted on the **seed only** — fitting them on
  synthetic data would leak information into the evaluation.
- `dcr_ratio = synthetic median / holdout median`; near 1 is healthy, well
  below 1 warns of memorisation.
- Exact-match threshold 1e-8 catches literal copies.

**Viva questions**
1. *Q: Does DCR prove privacy?* A: No — it is a warning indicator. Real
   guarantees need differential privacy and a formal analysis.
2. *Q: Why a holdout baseline?* A: Real people can naturally share identical
   values; without a baseline, "close" would be meaningless.

---

## src/models.py — Step 8: classifiers and the five arms

**Key functions**
| Function | Input | Output |
|---|---|---|
| `build_pipeline(...)` | model name + columns + config | sklearn Pipeline |
| `train_and_evaluate(...)` | train df, test df | dict of metrics |

**Lines an examiner might point at**
- The Pipeline bundles OneHotEncoder + model, so the encoder is fitted on
  training data of that arm only — no evaluation leakage.
- `average_precision_score` = PR-AUC, our primary metric.
- Single-class training data returns NaN metrics instead of crashing.

**Viva questions**
1. *Q: RF vs XGBoost?* A: Random Forest trains many independent trees in
   parallel and votes; XGBoost builds trees sequentially, each correcting
   the previous ones' errors (boosting).
2. *Q: Why PR-AUC over accuracy?* A: With a 24% positive class, always
   predicting the majority gets 76% accuracy but finds nothing; PR-AUC
   measures how well the minority class is actually found.

---

## src/run_experiment.py — Step 9: the conductor

**What it does:** loops over random seeds and seed sizes; for each it trains
TRTR-full once, then per seed size: TRTR-seed, SMOTE-NC, CTGAN, Gaussian
Copula — measuring fidelity, privacy, runtime and utility; appends every
row to results/*.csv; supports `--quick` and resume (skips rows already in
metrics.csv).

**Lines an examiner might point at**
- `verify_test_hash()` runs before anything else.
- Quick-mode outputs go to `results/quick/` so they never mix with real ones.
- `run_config.yaml` is saved so the exact settings are recorded.

**Viva questions**
1. *Q: Why resume support?* A: A laptop/Colab disconnect must not lose hours
   of CTGAN training; finished rows are skipped on restart.
2. *Q: Why is the config saved?* A: Reproducibility — later readers can see
   exactly which settings produced these numbers.

---

## src/analyze.py — Step 10: from raw rows to the answer

**Key functions**
| Function | Input | Output |
|---|---|---|
| `build_summary(...)` | 4 CSVs | mean ± std tables |
| `plot_metric_vs_seed_size(...)` | summary | the main PR-AUC/F1 figures |
| `apply_criteria(...)` | summaries | PASS/FAIL table + minimum useful seed |
| `write_results_md(...)` | everything | results/RESULTS.md |

**Lines an examiner might point at**
- Criteria are applied as frozen; the file is never modified here.
- If nothing passes, that is reported as a finding, not hidden.

**Viva questions**
1. *Q: How do you read the main plot?* A: X-axis = seed size (log scale),
   Y-axis = PR-AUC; each line is an arm with a ±1 std band; the black
   horizontal line is the full-data reference; the dashed line is random
   guessing (~0.24).
2. *Q: What if no size passes?* A: It is still a valid result: none of the
   tested sizes met the pre-declared criteria on this dataset.

---

## frontend/web — the React web app (presentation layer)

The Streamlit demo app was removed; the React app in `frontend/web` is the
only UI. It is a Vite + React + TypeScript single-page app that talks to the
backend through the API contract in `src/api/client.ts` and
`src/api/types.ts` (with mock fixtures in `src/mocks/fixtures.ts` for
backend-free demos). Its sections map to the pipeline: story, data snapshot,
utility, fidelity/privacy studios, criteria, runtime, and the interactive
lab. All heavy lifting (profiling, generation, fidelity, DCR) still happens
in `src/` — the frontend only renders and requests.

**Viva questions**
1. *Q: Why a separate frontend?* A: It separates presentation from the
   computation pipeline; the same src/ functions serve both the batch
   experiment and interactive use, so numbers cannot drift between them.
2. *Q: What are the mock fixtures for?* A: They let the UI run without the
   backend (mock mode), which separates frontend development from backend
   availability and makes the API contract explicit.
