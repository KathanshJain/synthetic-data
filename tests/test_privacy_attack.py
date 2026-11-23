"""
tests/test_privacy_attack.py — sanity checks for the membership-inference
extension (EXTENSION: not part of the frozen criteria).

Why these tests exist: the attack AUC is only meaningful if it behaves
correctly on data we control. A generator that copied its seed must score
near 1.0 (every seed record is trivially findable in the synthetic data);
a generator that produced unrelated rows must score near 0.5 (no
separation between seed and holdout records).
"""

import numpy as np
import pandas as pd

from src.privacy_attack import membership_inference_auc


def make_data(n_rows, seed=0, offset=0):
    rng = np.random.default_rng(seed)
    return pd.DataFrame({
        "row_id": range(n_rows),
        "age": rng.integers(20, 60, size=n_rows) + offset,
        "hours": rng.integers(20, 60, size=n_rows) + offset,
        "sex": rng.choice(["M", "F"], size=n_rows),
    })


def test_copied_seed_scores_near_one():
    """If synthetic data IS the seed, every seed record is trivially found."""
    seed = make_data(100)
    # offset the holdout into a disjoint value range so no coincidental ties
    holdout = make_data(100, seed=99, offset=100)
    holdout["row_id"] = range(1000, 1100)
    results = membership_inference_auc(
        seed, seed.drop(columns=["row_id"]), holdout,
        numerical_cols=["age", "hours"], categorical_cols=["sex"])
    assert results["attack_auc"] > 0.99


def test_unrelated_synthetic_scores_near_half():
    """Unrelated synthetic rows give no separation: AUC near 0.5."""
    seed = make_data(100)
    holdout = make_data(100, seed=99)
    holdout["row_id"] = range(1000, 1100)
    rng = np.random.default_rng(5)
    unrelated = pd.DataFrame({
        "age": rng.integers(20, 60, size=500),
        "hours": rng.integers(20, 60, size=500),
        "sex": rng.choice(["M", "F"], size=500),
    })
    results = membership_inference_auc(
        seed, unrelated, holdout,
        numerical_cols=["age", "hours"], categorical_cols=["sex"])
    assert 0.4 < results["attack_auc"] < 0.6
