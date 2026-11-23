"""
utils.py — small helper functions used by every other file in the project.

Why this file exists: several steps need the same things (loading config,
fixing random seeds, knowing which columns are numerical vs categorical, and
safely loading the locked test set). Putting them in one place means every
step behaves the same way and the GOLDEN RULE (verify the test-set hash
before loading it) is enforced in exactly one place.
"""

import hashlib
import json
import random
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

# The project root is the folder that contains src/, data/, config.yaml etc.
# We compute it from this file's location so the code works from any
# working directory (and inside Streamlit too).
PROJECT_ROOT = Path(__file__).resolve().parent.parent

CONFIG_PATH = PROJECT_ROOT / "config.yaml"
CRITERIA_PATH = PROJECT_ROOT / "criteria.yaml"
LOCKED_TEST_PATH = PROJECT_ROOT / "data" / "locked" / "test_locked.csv"
MANIFEST_PATH = PROJECT_ROOT / "data" / "locked" / "manifest.json"


def load_config(path=CONFIG_PATH):
    """Read config.yaml and return it as a dictionary."""
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_criteria(path=CRITERIA_PATH):
    """Read criteria.yaml (the pass/fail rules) and return it as a dictionary."""
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def set_all_seeds(seed):
    """
    Fix the random seed of every library we use, so that re-running the
    project gives the same numbers. 'seed' is an integer like 0, 1 or 2.
    """
    random.seed(seed)          # python's built-in random module
    np.random.seed(seed)       # numpy
    try:                       # torch is optional (only used inside CTGAN)
        import torch
        torch.manual_seed(seed)
    except ImportError:
        pass


def get_column_types(df, target):
    """
    Split the columns of a DataFrame into numerical and categorical lists.

    The target column and row_id are never returned — row_id is just a label
    we added for traceability, and generators/models must never use it.
    """
    numerical_cols = []
    categorical_cols = []
    for column in df.columns:
        if column in (target, "row_id"):
            continue
        if pd.api.types.is_numeric_dtype(df[column]):
            numerical_cols.append(column)
        else:
            categorical_cols.append(column)
    return numerical_cols, categorical_cols


def features_only(df):
    """Drop row_id from a DataFrame. Generators and models must never see it."""
    return df.drop(columns=["row_id"], errors="ignore")


def compute_file_sha256(path):
    """Compute the SHA-256 hash (a unique fingerprint) of a file on disk."""
    sha256 = hashlib.sha256()
    with open(path, "rb") as f:
        # read the file in chunks so we do not load the whole file in memory
        for chunk in iter(lambda: f.read(65536), b""):
            sha256.update(chunk)
    return sha256.hexdigest()


def verify_test_hash():
    """
    Re-compute the SHA-256 hash of the locked test file and compare it with
    the hash stored in manifest.json (written when the file was created).

    If the file changed by even one byte, this raises a clear error. This is
    our tamper-detection lock: it proves the file is unchanged, which is how
    we can trust later results were computed on the same locked test set.
    """
    if not LOCKED_TEST_PATH.exists():
        raise FileNotFoundError(
            f"Locked test set not found at {LOCKED_TEST_PATH}. "
            "Run 'python -m src.prepare_data' first."
        )
    if not MANIFEST_PATH.exists():
        raise FileNotFoundError(
            f"Manifest not found at {MANIFEST_PATH}. "
            "Run 'python -m src.prepare_data' first."
        )
    with open(MANIFEST_PATH, "r", encoding="utf-8") as f:
        manifest = json.load(f)
    actual_hash = compute_file_sha256(LOCKED_TEST_PATH)
    if actual_hash != manifest["sha256"]:
        raise ValueError(
            "LOCKED TEST SET HASH MISMATCH! The file data/locked/test_locked.csv "
            "has changed since it was created. Do not trust any results computed "
            "on it. Restore the original file or re-run prepare_data.py."
        )


def load_locked_test():
    """
    The ONLY approved way to load the locked test set.

    It always verifies the hash first, so nobody can accidentally evaluate on
    a modified (or silently replaced) test file.
    """
    verify_test_hash()
    return pd.read_csv(LOCKED_TEST_PATH)
