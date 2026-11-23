"""
tests/test_seeds.py — tests for Step 4 (nested stratified seeds).

Why these tests exist: the whole sample-size comparison is only fair if the
seeds really are nested, really have the right sizes, and really keep the
class proportions. These tests prove each of those claims.
"""

import pandas as pd
import pytest

from src.seeds import build_nested_seeds, check_nested, get_holdout
from src.utils import PROJECT_ROOT, load_config


@pytest.fixture(scope="module")
def setup():
    config = load_config()
    dev_path = PROJECT_ROOT / "data" / "processed" / "dev_pool.csv"
    if not dev_path.exists():
        from src.prepare_data import main as prepare_main
        prepare_main()
    dev_pool = pd.read_csv(dev_path)
    return config, dev_pool


def test_exact_sizes(setup):
    """Each seed must contain exactly the requested number of rows."""
    config, dev_pool = setup
    seeds = build_nested_seeds(dev_pool, "income",
                               config["seeds"]["seed_sizes"], 0)
    for size, seed_df in seeds.items():
        assert len(seed_df) == size


def test_nesting_holds(setup):
    """Smaller seeds must be subsets of larger seeds."""
    config, dev_pool = setup
    seeds = build_nested_seeds(dev_pool, "income",
                               config["seeds"]["seed_sizes"], 0)
    assert check_nested(seeds)


def test_class_ratio_within_one_row(setup):
    """Per-class counts must be within 1 row of the exact proportion."""
    config, dev_pool = setup
    target = "income"
    seeds = build_nested_seeds(dev_pool, target,
                               config["seeds"]["seed_sizes"], 0)
    overall_positive = dev_pool[target].mean()
    for size, seed_df in seeds.items():
        expected_positive = size * overall_positive
        actual_positive = seed_df[target].sum()
        assert abs(actual_positive - expected_positive) <= 1


def test_holdout_no_overlap_with_largest_seed(setup):
    """Holdout rows must not appear in the largest seed."""
    config, dev_pool = setup
    seeds = build_nested_seeds(dev_pool, "income",
                               config["seeds"]["seed_sizes"], 0)
    largest_seed = seeds[max(seeds)]
    holdout = get_holdout(dev_pool, largest_seed, 100, 0)
    assert len(set(holdout["row_id"]) & set(largest_seed["row_id"])) == 0


def test_same_random_state_gives_identical_seeds(setup):
    """Reproducibility: the same random_state must rebuild the same seeds."""
    config, dev_pool = setup
    seeds_a = build_nested_seeds(dev_pool, "income",
                                 config["seeds"]["seed_sizes"], 7)
    seeds_b = build_nested_seeds(dev_pool, "income",
                                 config["seeds"]["seed_sizes"], 7)
    for size in seeds_a:
        assert list(seeds_a[size]["row_id"]) == list(seeds_b[size]["row_id"])
