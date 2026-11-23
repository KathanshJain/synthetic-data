# VIVA_ANSWERS.md — answers with OUR actual numbers

Generic definitions are in `Project_Storytelling_and_Viva_Guide.docx`.
This file adds the answers that quote **our real experiment results**, so you
can defend the actual numbers an examiner sees on the slides.

## Our headline numbers (memorise these five)

| Fact | Number |
|---|---|
| Reference: TRTR-full XGBoost PR-AUC (locked test) | **0.823** |
| Best synthetic utility (TSTR-CTGAN, any size) | **0.648** (seed 500 and 2000) |
| Best small-real baseline (TRTR-seed 2000) | **0.761** |
| Random-guess PR-AUC | **0.239** |
| Minimum useful seed | **none passed** (best utility ratio 0.787 vs 0.90 required) |

## Q: What did your experiment actually find?

A: Three things. First, utility: CTGAN-trained models reached at best 79% of
the full-data reference (PR-AUC 0.648 vs 0.823), and training on the small
real seed itself (TRTR-seed, 0.761 at 2000 rows) always beat training on
synthetic data. Second, generator comparison: CTGAN clearly beat the Gaussian
Copula on both utility (0.65 vs 0.48 PR-AUC) and fidelity (mean KS 0.22 vs
0.40). Third, privacy was never the problem: every DCR ratio was at or above
1.0, meaning synthetic rows sat *farther* from the training seed than genuine
unseen records do, and the exact-copy rate was ~0.

## Q: Why did no seed size pass? Isn't that a failure?

A: No — it is a valid, pre-registered finding. We froze the thresholds in
criteria.yaml and committed them to Git *before* the full run, so we could not
move the goalposts. The failing rules were utility (best ratio 0.787 < 0.90)
and KS (0.22–0.29 > 0.10). The honest conclusion is that, on Adult with a
fixed 3000-step CTGAN budget, no seed between 100 and 2000 rows produces
synthetic data good enough to replace real data — you need more rows or a
better generator. Reporting this honestly is exactly what the project brief
asked for.

## Q: Why is CTGAN's PR-AUC flat after seed 500?

A: TSTR-CTGAN was 0.563 at 100 rows, 0.633 at 250, 0.648 at 500 — and then
stayed flat (0.619 at 1000, 0.648 at 2000). The generator's capacity, not the
data volume, becomes the bottleneck: with a fixed 3000-step budget, more data
does not translate into a better learned distribution. This is the
"flattening point" on our main plot.

## Q: Your DCR ratio is above 1 — is that good?

A: Yes. DCR ratio = median synthetic distance to the seed ÷ median *genuine
holdout* distance to the seed. A ratio near or above 1 means synthetic rows
are about as far from real records as genuinely new people are — no
memorisation signal. CTGAN averaged 1.22, Gaussian Copula 2.43 (the copula is
smoother, so its samples sit even farther out). Exact-copy rates were ~0%
against a holdout baseline of 0.5–0.9%.

## Q: Why does the Gaussian Copula have better TVD but worse utility?

A: Interesting asymmetry. The copula nails marginal category proportions
(TVD 0.02–0.07, better than CTGAN's 0.08–0.10) but badly distorts numerical
columns (KS ~0.40 vs CTGAN's 0.22) and, more importantly, the *joint*
structure — its correlation difference and its PR-AUC (0.35–0.48) show it
does not preserve the feature combinations that predict income. Marginal
fidelity does not imply joint fidelity.

## Q: What did SMOTE-NC show?

A: SMOTE-NC tracked TRTR-seed closely (0.753 vs 0.761 at 2000 rows) — it
merely rebalances the classes, so with RF/XGBoost (which handle imbalance
reasonably) it adds little. It is a class-imbalance tool, not a data
expansion tool, and our results confirm that distinction.

## Q: What was your CTGAN training budget and why?

A: 3000 total gradient steps for every seed size. We verified in the
installed ctgan source that steps per epoch = max(len(data) // batch_size,
1); without our correction, a 100-row seed would train ~20× more epochs than
a 2000-row seed for the same epoch count, confounding data size with training
duration. Runtime was nearly constant: ~142–170 s per fit across all sizes —
which is itself evidence the budget was fair.

## Q: How did you guarantee the test set was never misused?

A: Four layers: (1) a stratified 80/20 split with the test file stored
separately; (2) a SHA-256 manifest verified before every load — the code
refuses to run if the file changed by one byte; (3) a source-scanning pytest
(test_no_leakage.py) that only allows utils/prepare_data/models/run_experiment
to even mention the locked path and forbids fitting anything on test data;
(4) process discipline — criteria and model settings were frozen and
committed before the single final evaluation.

## Q: What are the limitations you'd state if pressed?

A: One public dataset; two generators plus an oversampling baseline; three
random seeds; DCR is a warning indicator, not a privacy guarantee; CTGAN's
fixed budget may under-train larger seeds; and SDV 2.x emitted NaNs for rare
categories (workclass, occupation, native-country), which our validation
flagged — a known limitation of categorical smoothing, not a bug in our
pipeline.

## Q: What would you do next?

A: More datasets, TVAE and tabular diffusion models, membership-inference
attacks instead of DCR alone, differential privacy, and a larger CTGAN budget
arm to separate "not enough data" from "not enough training".
