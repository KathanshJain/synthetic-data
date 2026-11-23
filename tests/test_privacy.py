"""
tests/test_privacy.py — sanity check for the DCR privacy measure.

Why this test exists: the DCR check is only meaningful if it screams when
the generator copies data. So we feed it an EXACT COPY of the seed and
demand a 100% exact-match rate.
"""

import numpy as np
import pandas as pd

from src.privacy import compute_dcr


def test_exact_copy_has_100_percent_exact_match_rate():
    rng = np.random.default_rng(0)
    seed = pd.DataFrame({
        "row_id": range(100),
        "age": rng.integers(20, 60, size=100),
        "hours": rng.integers(20, 60, size=100),
        "sex": rng.choice(["M", "F"], size=100),
    })
    holdout = seed.copy()
    holdout["age"] = holdout["age"] + 30  # clearly different genuine rows
    holdout["row_id"] = range(1000, 1100)

    copy_of_seed = seed.drop(columns=["row_id"]).copy()

    results = compute_dcr(seed, copy_of_seed, holdout,
                          numerical_cols=["age", "hours"],
                          categorical_cols=["sex"])
    assert results["synth_exact_match_rate"] == 1.0
    # a copy sits far closer to the seed than genuine holdout rows do
    assert results["dcr_ratio"] < 0.1
