# Final evaluation speaking order

**Total planned time 11:30.** Use the main deck in order. Leave the three backup slides for questions.

| Starts | Slide | Speaker | Topic |
| --- | --- | --- | --- |
| 0:00 | 1 | Aryan | Machine Learning on Artificially Forged Data |
| 0:30 | 2 | Aryan | A team may receive only a small sample. |
| 1:05 | 3 | Kathansh | Useful synthetic data has to clear several checks. |
| 1:50 | 4 | Kathansh | Adult income is an imbalanced prediction task. |
| 2:40 | 5 | Kathansh | Nested seeds make the comparison fairer. |
| 3:40 | 6 | Aryan | The methods create rows in different ways. |
| 4:35 | 7 | Kathansh | The two tracks meet in one experiment. |
| 5:25 | 8 | Kathansh | Each metric asks a different question. |
| 6:15 | 9 | Aryan | Training on the real seed gave better utility. |
| 7:20 | 10 | Kathansh | CTGAN led on KS. Copula led on TVD. |
| 8:10 | 11 | Kathansh | The distance check raised no memorisation warning. |
| 9:00 | 12 | Aryan | None of the tested sizes passed. |
| 9:55 | 13 | Aryan | The web app makes the experiment explorable. |
| 10:40 | 14 | Kathansh | The result is specific to this experiment. |

## Source checks

- Utility chart source is `reports/report_facts.json`. The full-real XGBoost reference is **0.8228** (0.823 on the slide).
- The best archived utility ratio is **0.787** against **0.90**.
- The README’s KS range refers to CTGAN. Its archived values span **0.2197–0.2851** against **0.10**. Gaussian Copula ranges from **0.3798 to 0.4879**.
- The lowest reported mean DCR ratio is **1.0056**. The checkout contains aggregate ratios, so the claim is about reported means.
- Frozen outcome is **none passed**.
- The GitHub checkout contains React and FastAPI. The older project brief calls the UI Streamlit.
- The repository has no figures directory in this checkout. The class balance figure uses the tracked EDA counts.
- The web screenshot is the original repository image retrieved from Git LFS and matched to its SHA-256 object ID.
- Per-column fidelity is described from development EDA when the measured per-column CSV is absent.
- The reported TSTR results predate the missing-value repair. The README describes them as slightly pessimistic.

## Scripts and viva prompts

### 01. Machine Learning on Artificially Forged Data

**Aryan**

Good morning. We are Aryan Gupta and Kathansh Jain, and this is our mini-project on machine learning with generated tabular data. We wanted to know how far a small real sample can take us when we use it to train a data generator. We will explain how we made the comparison fair, what each measurement means, and what we actually found on the held-out Adult Census Income data.

**Likely viva question**  Why use the word forged in the title?

Here it means artificially generated tabular records. We use the records for a controlled machine-learning experiment.

### 02. A team may receive only a small sample.

**Aryan**

Imagine a bank with a large table of customer records. Its analysts may receive only a sample because the full table cannot be shared freely. Could they use that sample to generate extra rows and train a useful classifier? That is a motivating example. Our actual data are public census records. The distance checks in our study measure record similarity, and we do not infer that generated data can always be shared.

**Likely viva question**  Did you use any bank customer records?

We used the public Adult Census Income dataset. The bank scenario illustrates why a limited sample can matter.

### 03. Useful synthetic data has to clear several checks.

**Kathansh**

We changed the number of real training rows given to each generator and measured the output in three ways. Fidelity asks whether columns and relationships resemble real data. Utility asks whether a model trained on the generated rows can predict income on unseen real data. Privacy risk asks whether a generated row sits unusually close to the records used to train the generator. We required the checks to pass together.

**Likely viva question**  Why look beyond fidelity?

A generator can match individual column distributions while missing combinations that predict income. The utility test checks whether those patterns transfer.

### 04. Adult income is an imbalanced prediction task.

**Kathansh**

We used the public Adult Census Income dataset. It has 48,842 raw rows and 48,790 after exact deduplication. We excluded the survey weight and kept the numeric education representation. Missing categories were recorded explicitly and row IDs were excluded from modelling. About 24% of development rows are above the income cutoff, so the classes are uneven. That is why we chose average precision for our main utility comparison.

**Likely viva question**  Why drop the survey weight?

It is an estimate of how many people each census row represents. The experiment models individual predictors.

### 05. Nested seeds make the comparison fairer.

**Kathansh**

We used an approximately 80% development and 20% locked test split while keeping the income-class ratio close in both. The SHA-256 fingerprint verifies that the test file still has the same bytes before scoring. Inside development, every smaller seed is a subset of the next, with a similar class balance. Repeating the whole sampling procedure 3 times helps us see whether one sample ordering drove the result.

**Likely viva question**  Is the test file encrypted by SHA-256?

SHA-256 verifies the integrity of the file bytes. Anyone with read access can still see the CSV contents.

### 06. The methods create rows in different ways.

**Aryan**

CTGAN learns a table by asking a generator to make rows and a critic to score them. Conditioning gives rare categories more training attention. Gaussian Copula learns each column's distribution and an estimate of how columns vary together. SMOTE-NC has a different job. It uses nearest minority neighbours to make extra minority examples. We compare that balanced real-data arm with the models trained entirely on generated data.

**Likely viva question**  What is the role of SMOTE-NC in your comparison?

It is a class-imbalance baseline that keeps the original seed rows. It adds minority examples to the seed.

### 07. The two tracks meet in one experiment.

**Kathansh**

I worked on the data side, from cleaning and the split through nested seeds, distribution comparisons and the distance check. Aryan worked on the generators, classifiers, experiment runner and application. Both tracks use the same development data. Each arm fits its own preprocessing and classifier. We used Random Forest and XGBoost, and the final decision uses XGBoost average precision because a single, consistent evaluator makes the training-source comparison easier to read.

**Likely viva question**  Where do the locked rows enter the pipeline?

They enter only when the fitted classifiers make final evaluation predictions. Fitting, seed selection and generator diagnostics use development data.

### 08. Each metric asks a different question.

**Kathansh**

For numerical columns, KS is the largest vertical gap between the real and synthetic cumulative curves. TVD compares categorical shares. Correlation difference looks at relationships between numeric columns. DCR is the distance to the closest seed row after encoding and scaling. For utility, our PR-AUC is average precision. It adds precision weighted by the gain in recall. That emphasizes the ability to retrieve the smaller positive income class.

**Likely viva question**  Which direction is better for the metrics?

Lower KS, TVD and correlation difference mean a closer match. Higher average precision is better, while DCR needs a genuine holdout reference.

### 09. Training on the real seed gave better utility.

**Aryan**

On the horizontal axis is the number of real rows given to each seed-based arm. The vertical axis is XGBoost average precision. The dashed full-real reference is 0.823, and the best synthetic-only score was 0.648. The real seed beat both synthetic-only arms at every tested size. The README says the saved synthetic data predates rare-category missing-value repair, so these TSTR scores may be slightly pessimistic. We report the saved measurements and do not estimate the effect of the fix.

**Likely viva question**  What do TRTR and TSTR stand for?

TRTR means train on real data and test on real data. TSTR means train on synthetic data and test on real data.

### 10. CTGAN led on KS. Copula led on TVD.

**Kathansh**

The left chart compares numerical distributions. CTGAN's lowest mean KS is 0.220, against the frozen limit of 0.10. The right chart compares categorical shares. The lowest Copula TVD is 0.018. The two metrics look at different properties. An accurate category distribution can coexist with inaccurate numeric distributions or loss of combinations that help an income classifier. The plots show mean scores at each tested seed size.

**Likely viva question**  Why can Copula match category shares yet have lower utility?

Category shares describe columns one at a time. An income model can need feature combinations and numerical distributions that those shares miss.

### 11. The distance check raised no memorisation warning.

**Kathansh**

For every generated row, we measure the distance to its closest seed record. We also measure genuine holdout rows against that same seed. We divide the generated median distance by the holdout median. The smallest reported mean ratio is 1.006. Our copy-rate rule is another check on near duplicates. These tests check one kind of record similarity. They cannot tell us whether any specific privacy attack would succeed.

**Likely viva question**  Can these DCR results certify that you may share synthetic records?

DCR measures nearest-record similarity under a chosen representation. A privacy guarantee requires a separate formal mechanism and analysis.

### 12. None of the tested sizes passed.

**Aryan**

A useful seed had to clear every frozen rule, including utility, numerical and categorical fidelity, and both record-similarity checks. None of the sizes cleared all the checks. In this run, the blocking rules were utility and numerical KS. The best synthetic utility ratio was 0.787, while the required ratio was 0.90. We kept the thresholds declared before test scoring. This decision answers the question for the sizes and methods we tested and gives us a clear starting point for another experiment.

**Likely viva question**  Why keep the original thresholds when the candidates fall short?

Changing the threshold after scoring would let the test outcomes drive the rules. A future experiment can specify different thresholds in advance.

### 13. The web app makes the experiment explorable.

**Aryan**

The React front end presents the story, figures and an interactive CSV lab. FastAPI connects those views to saved experiment outputs and wraps the Python generator functions for lab jobs. The app can browse results without rerunning the full experiment during a demonstration. This interface is a separate layer from the research pipeline. The chart data in these slides come from the saved research outputs, so they can be checked independently of the app.

**Likely viva question**  Do you need to run the full experiment to show the web app?

The presentation can use the saved results and screenshots. A live lab needs the connected backend and appropriate input data.

### 14. The result is specific to this experiment.

**Kathansh**

Our project does not claim that synthetic data is automatically private or that it can replace real data in every setting. We evaluated a specific tabular dataset and a defined set of generators. Our contribution is a controlled comparison of how seed size affects statistical fidelity, downstream predictive utility, and a nearest-record memorisation indicator. We reported the results honestly. None of the tested sizes satisfied all the predefined criteria.

**Likely viva question**  What would you test next?

I would rerun the fixed pipeline on a separately defined evaluation protocol and vary the training budget as its own factor. I would also test record disclosure under a defined threat model.

### 15. The frozen rules use measurable quantities.

**Kathansh**

The criteria file specifies what must pass together. The utility ratio compares a synthetic-only classifier with the full-real reference on the same test. We average KS across numeric columns and TVD across categorical columns. The distance ratio uses medians, while the exact-copy rule compares near-zero distances with the holdout. Average precision is the summary of the precision-recall curve used in the utility ratio.

**Likely viva question**  Why use a holdout reference for copies?

Genuine records can share identical measured predictors. The holdout gives us a comparison rate for these natural matches.

### 16. EDA flagged skew and rare categories.

**Kathansh**

A single mean can hide a difficult feature. In Adult, capital gain and capital loss have many zeros and a small number of much larger values. Native country includes rare levels, and some work and occupation entries are missing in the real data. The available per-feature evidence comes from development EDA. We would need the archived per-column scores to rank synthetic differences. A feature-level check can help distinguish problems in rare groups from problems that affect the whole numerical distribution.

**Likely viva question**  Which features would you inspect first after a refresh?

I would compare the empirical curves for capital gain and loss, then compare rare-category shares. Those are clear places where small seeds can leave gaps.

### 17. CTGAN took longer under a fixed step budget.

**Aryan**

Mean fit time for CTGAN ranged from 141.7 to 169.8 seconds. Gaussian Copula took 0.8 to 1.3 seconds. The CTGAN budget was 3,000 iterations per fit, with a batch size of 500. The code converts that budget to a different number of epochs when the number of steps per epoch changes. This keeps compute more comparable when we vary the real seed size. The chart reports fitting time, with sampling stored separately.

**Likely viva question**  Why use steps per fit instead of epochs per fit?

Different seed sizes can require different numbers of batches per epoch. A fixed step budget keeps the number of training updates comparable.
