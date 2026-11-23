"""
tests/test_split.py — tests for Step 1 (data preparation and the locked vault).

Why these tests exist: the whole experiment trusts that the locked test set
is (a) disjoint from the development pool and (b) unchanged since creation.
If either broke, every final metric would be quietly wrong.
"""

import shutil

import pandas as pd
import pytest

from src.utils import (
    LOCKED_TEST_PATH,
    MANIFEST_PATH,
    PROJECT_ROOT,
    load_config,
    verify_test_hash,
)


@pytest.fixture(scope="module")
def prepared():
    """Make sure the data has been prepared once for this test module."""
    config = load_config()
    dev_path = PROJECT_ROOT / "data" / "processed" / "dev_pool.csv"
    if not dev_path.exists() or not LOCKED_TEST_PATH.exists():
        from src.prepare_data import main as prepare_main
        prepare_main()
    dev_pool = pd.read_csv(dev_path)
    test_set = pd.read_csv(LOCKED_TEST_PATH)
    return config, dev_pool, test_set


def test_no_row_id_overlap(prepared):
    """Dev pool and locked test set must share zero row_ids."""
    config, dev_pool, test_set = prepared
    overlap = set(dev_pool["row_id"]) & set(test_set["row_id"])
    assert len(overlap) == 0


def test_class_ratio_within_half_percent(prepared):
    """Both splits must keep nearly the same positive-class ratio."""
    config, dev_pool, test_set = prepared
    target = config["dataset"]["target"]
    dev_ratio = dev_pool[target].mean()
    test_ratio = test_set[target].mean()
    assert abs(dev_ratio - test_ratio) <= 0.005


def test_verify_hash_passes():
    """The real locked file must match its manifest hash."""
    verify_test_hash()  # raises if broken — no assert needed


def test_verify_hash_detects_modification(prepared, tmp_path):
    """
    If even one byte of the test file changes, verify_test_hash must fail.
    We test this on a TEMPORARY COPY — the real locked file is never touched.
    """
    config, dev_pool, test_set = prepared

    # temporarily point the utils paths at our sandbox copies
    import src.utils as utils
    real_test_path = utils.LOCKED_TEST_PATH
    real_manifest_path = utils.MANIFEST_PATH
    try:
        fake_test = tmp_path / "test_locked.csv"
        fake_manifest = tmp_path / "manifest.json"
        test_set.to_csv(fake_test, index=False)

        # a manifest that matches the untouched copy -> should pass
        from src.utils import compute_file_sha256
        import json
        fake_manifest.write_text(json.dumps({"sha256": compute_file_sha256(fake_test)}))
        utils.LOCKED_TEST_PATH = fake_test
        utils.MANIFEST_PATH = fake_manifest
        verify_test_hash()  # OK

        # now modify one value in the copy -> must raise
        tampered = test_set.copy()
        tampered.loc[tampered.index[0], "age"] = tampered["age"].iloc[0] + 1
        tampered.to_csv(fake_test, index=False)
        with pytest.raises(ValueError):
            verify_test_hash()
    finally:
        # always restore the real paths
        utils.LOCKED_TEST_PATH = real_test_path
        utils.MANIFEST_PATH = real_manifest_path
