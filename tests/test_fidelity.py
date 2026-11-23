"""
tests/test_fidelity.py — sanity checks for the fidelity measures.

Why these tests exist: before trusting KS/TVD numbers on real synthetic
data, we prove the measures behave correctly on data we control:
identical data must score 0, wildly different data must score high.
"""

import numpy as np
import pandas as pd

from src.fidelity import evaluate_fidelity


def make_data(n_rows, shift=0.0, category_shuffle=False):
    """A tiny table with one numerical and one categorical column."""
    rng = np.random.default_rng(0)
    df = pd.DataFrame({
        "age": rng.normal(40 + shift, 10, size=n_rows),
        "workclass": rng.choice(["A", "B", "C"], size=n_rows,
                                p=[0.5, 0.3, 0.2]),
    })
    if category_shuffle:
        df["workclass"] = df["workclass"].map(
            {"A": "C", "B": "B", "C": "A"})  # completely different proportions
    return df


def test_identical_data_gives_zero():
    """KS and TVD must be 0 when synthetic data IS the real data."""
    real = make_data(1000)
    per_column, summary = evaluate_fidelity(
        real, real.copy(), ["age"], ["workclass"])
    assert summary["ks_mean"] == 0.0
    assert summary["tvd_mean"] == 0.0
    assert summary["corr_diff"] == 0.0


def test_different_data_gives_large_values():
    """Shifted numbers + shuffled categories must give clearly large scores."""
    real = make_data(1000)
    synth = make_data(1000, shift=50, category_shuffle=True)
    per_column, summary = evaluate_fidelity(
        real, synth, ["age"], ["workclass"])
    assert summary["ks_mean"] > 0.5
    assert summary["tvd_mean"] > 0.3
