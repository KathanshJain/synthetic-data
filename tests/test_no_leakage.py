"""
tests/test_no_leakage.py — the anti-leakage guard.

Why this test exists: the GOLDEN RULE says the locked test set may only be
loaded for the final evaluation. This test reads the SOURCE CODE of the
project and checks:
  1. only the approved files (utils.py, models.py demo, run_experiment.py)
     even mention the locked test path,
  2. no generator or encoder is ever fitted on the test set.
"""

import re
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SRC = PROJECT_ROOT / "src"

# files allowed to reference the locked test set:
#   utils.py        — defines the guarded loader (hash check built in)
#   prepare_data.py — the only file that CREATES the locked file + manifest
#   models.py       — docstring/demo only; the demo never calls the loader
#   run_experiment.py — the only file that evaluates on it (final step)
ALLOWED_FILES = {"utils.py", "prepare_data.py", "models.py", "run_experiment.py"}

LOCKED_PATTERNS = [
    r"test_locked",
    r"load_locked_test",
    r"locked_test",
]

# patterns that would mean a generator/encoder is FITTED on evaluation data
FIT_ON_TEST_PATTERNS = [
    r"\.fit\(test_",          # e.g. scaler.fit(test_...)
    r"\.fit\(locked",
    r"fit_transform\(test_",
]


def source_files():
    return {p.name: p.read_text(encoding="utf-8")
            for p in SRC.glob("*.py")}


def test_only_approved_files_reference_locked_test():
    sources = source_files()
    for file_name, source in sources.items():
        if file_name in ALLOWED_FILES:
            continue
        for pattern in LOCKED_PATTERNS:
            matches = re.findall(pattern, source)
            # allow mentions inside a comment that says it is NOT used
            real_matches = [m for m in matches]
            assert not real_matches, (
                f"{file_name} references the locked test set "
                f"(pattern '{pattern}'). Only {sorted(ALLOWED_FILES)} may.")


def test_no_generator_or_encoder_fitted_on_test_data():
    sources = source_files()
    for file_name, source in sources.items():
        for pattern in FIT_ON_TEST_PATTERNS:
            assert not re.search(pattern, source), (
                f"{file_name} appears to fit something on test data "
                f"(pattern '{pattern}').")


def test_models_demo_does_not_load_locked_test():
    """The models.py demo must not actually CALL load_locked_test()."""
    source = (SRC / "models.py").read_text(encoding="utf-8")
    # the import line exists but must be commented/marked as NOT used
    main_section = source.split("def main()")[1]
    assert "load_locked_test(" not in main_section.replace(
        "noqa: F401 (NOT used!)", ""), (
        "models.py demo must not evaluate on the locked test set.")
