# EDA Report (development pool only)

- Rows: **39032**, Columns: **14**
- Numerical columns: age, education-num, capital-gain, capital-loss, hours-per-week
- Categorical columns: workclass, marital-status, occupation, relationship, race, sex, native-country
- Duplicate rows: 0
- Class balance: 0: 29687 (76.06%), 1: 9345 (23.94%)
- Missing values: workclass (5.72%), occupation (5.74%), native-country (1.73%)

## Warnings
- Column 'row_id' looks like an ID (almost all values unique) — it should not be used as a model feature.

## Observations
- The dataset is imbalanced: the smaller class is only 23.94% of rows. This is why we use PR-AUC instead of accuracy later.
- Missing values exist in workclass, occupation, native-country; we keep them as an explicit 'Missing' category instead of deleting rows, to preserve the class ratio.
- capital-gain and capital-loss are extremely skewed: most people have 0, a few have very large values. Generators find such columns hard.
- native-country is dominated by 'United States'; rare countries are categories a small seed may not even contain (coverage risk).
- age and hours-per-week are roughly bell-shaped but with spikes (e.g. full-time 40 hours/week).
- ID-like columns detected: row_id — excluded from modelling.