"""
Run the pytest suite and cache the summary to results/checks.json.

The API's /health endpoint reads this file instead of running tests inside
a request — tests take seconds and must never run inside a web request.
Run it after changing code:  python -m scripts.run_checks
"""

import json
import subprocess
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RESULTS = PROJECT_ROOT / "results"


def main() -> int:
    print("Running the test suite (this can take a minute)...")
    targets = ["tests/"]
    api_tests = PROJECT_ROOT / "tests" / "api"
    if api_tests.exists():
        targets.append("tests/api/")
    proc = subprocess.run(
        [sys.executable, "-m", "pytest", *targets,
         "-q", "--tb=no", "-p", "no:cacheprovider"],
        cwd=PROJECT_ROOT, capture_output=True, text=True)
    output = proc.stdout + proc.stderr
    tail = output.strip().splitlines()[-1] if output.strip() else ""

    # pytest -q ends with a line like "17 passed in 2.15s" or
    # "2 failed, 15 passed in 3.00s" — parse counts with regexes.
    import re
    m_pass = re.search(r"(\d+) passed", tail)
    m_fail = re.search(r"(\d+) failed", tail)
    m_error = re.search(r"(\d+) error", tail)
    tests_passing = int(m_pass.group(1)) if m_pass else 0
    failed = int(m_fail.group(1)) if m_fail else 0
    errors = int(m_error.group(1)) if m_error else 0
    tests_total = tests_passing + failed + errors
    status = "pass" if proc.returncode == 0 and failed == 0 and errors == 0 else "fail"

    RESULTS.mkdir(exist_ok=True)
    payload = {
        "status": status,
        "tests_passing": max(tests_passing, 0),
        "tests_total": tests_total,
        "generated": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "summary_line": tail,
    }
    (RESULTS / "checks.json").write_text(json.dumps(payload, indent=2),
                                         encoding="utf-8")
    print(f"{tail}  ->  results/checks.json ({status})")
    return proc.returncode


if __name__ == "__main__":
    sys.exit(main())
