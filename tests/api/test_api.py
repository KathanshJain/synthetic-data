"""
API contract tests: every endpoint returns the exact shape the frontend
expects, errors use the {error:{code,message,hint}} envelope, and the
locked test set is never touched. Uses FastAPI's TestClient (in-process,
no server needed).
"""

import io
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from api.main import app

client = TestClient(app)


def test_health_shape():
    r = client.get("/api/health")
    assert r.status_code == 200
    body = r.json()
    assert set(body) == {"status", "version", "tests_passing", "tests_total"}
    assert body["tests_passing"] == body["tests_total"]


def test_overview_shape_and_locked_flag():
    r = client.get("/api/overview")
    assert r.status_code == 200
    body = r.json()
    for key in ("rows_raw", "dev_rows", "test_rows", "positive_rate_dev",
                "features", "locked"):
        assert key in body
    assert body["test_rows"] == 9758          # from manifest.json only
    assert body["locked"]["verified"] is True  # hash check ran live
    assert isinstance(body["locked"]["sha256"], str)


def test_seeds_shape():
    r = client.get("/api/seeds")
    assert r.status_code == 200
    body = r.json()
    assert body["sizes"] == [100, 250, 500, 1000, 2000]
    assert body["nested_ok"] is True
    assert set(body["class_counts"]["100"]) == {"majority", "minority"}


def test_summary_unknown_model_is_friendly_404():
    r = client.get("/api/results/summary", params={"model": "bert"})
    assert r.status_code == 404
    err = r.json()["error"]
    assert set(err) == {"code", "message", "hint"}


def test_criteria_matches_old_verdict():
    r = client.get("/api/results/criteria")
    assert r.status_code == 200
    body = r.json()
    assert body["min_useful_seed"] is None      # the honest old-result verdict
    assert not any(e["all_passed"] for e in body["evaluations"])
    best = max(e["values"]["utility_ratio"] for e in body["evaluations"])
    assert abs(best - 0.787) < 0.01


def test_loss_curve_friendly_404_when_absent():
    # loss files exist for the stored runs; a size that was never trained
    # must return the friendly envelope, not a crash
    r = client.get("/api/results/loss-curve",
                   params={"seed_size": 999, "random_seed": 0})
    assert r.status_code == 404
    assert "code" in r.json()["error"]


@pytest.mark.parametrize("path", [
    "/api/results/privacy/histogram?generator=CTGAN&seed_size=2000&random_seed=0",
    "/api/compare/distribution?column=age&generator=CTGAN&seed_size=2000&random_seed=0",
    "/api/compare/correlation?generator=CTGAN&seed_size=2000&random_seed=0",
    "/api/compare/projection?generator=CTGAN&seed_size=2000&random_seed=0",
])
def test_compare_shapes(path):
    r = client.get(path)
    assert r.status_code == 200
    body = r.json()
    if "privacy/histogram" in path:
        assert len(body["bins"]) == len(body["synthetic"]) == len(body["holdout"])
    elif "distribution" in path:
        assert len(body["real"]) == len(body["synthetic"]) > 0
    if "correlation" in path:
        n = len(body["columns"])
        assert len(body["diff"]) == n and len(body["diff"][0]) == n
    if "projection" in path:
        assert len(body["real"]) <= 800 and len(body["synthetic"]) <= 800
        assert len(body["explained_variance"]) == 2


def test_dataset_distribution_shape():
    r = client.get("/api/dataset/distribution", params={"column": "age"})
    assert r.status_code == 200
    body = r.json()
    assert body["type"] == "histogram"
    assert len(body["counts"]) > 0
    r = client.get("/api/dataset/distribution", params={"column": "nope"})
    assert r.status_code == 404
    assert "code" in r.json()["error"]


def _small_csv() -> bytes:
    """120 rows, binary target (~30% positive), 2 numerical + 1 categorical."""
    rng = np.random.default_rng(7)
    n = 120
    label = (rng.random(n) < 0.3).astype(int)
    df = pd.DataFrame({
        "age": rng.integers(20, 60, n),
        "hours": rng.integers(10, 60, n),
        "region": rng.choice(["a", "b", "c"], n),
        "diagnosis": label,
    })
    buf = io.StringIO()
    df.to_csv(buf, index=False)
    return buf.getvalue().encode()


def test_lab_upload_and_gaussian_copula_end_to_end():
    up = client.post("/api/lab/upload",
                     files={"file": ("tiny.csv", _small_csv(), "text/csv")})
    assert up.status_code == 200, up.text
    profile = up.json()
    assert profile["profile"]["rows"] == 120
    assert "diagnosis" in profile["profile"]["columns"]

    run = client.post("/api/lab/run", json={
        "dataset_id": profile["dataset_id"],
        "target": "diagnosis",
        "seed_size": 100,
        "generator": "GaussianCopula",
        "synthetic_rows": 500,
        "ctgan_steps": 200,
    })
    assert run.status_code == 200, run.text
    job_id = run.json()["job_id"]

    # poll until the job finishes (Gaussian Copula trains in seconds)
    import time
    for _ in range(120):
        job = client.get(f"/api/lab/jobs/{job_id}").json()
        if job["status"] in ("completed", "failed"):
            break
        time.sleep(0.5)
    assert job["status"] == "completed", job.get("error")
    result = job["result"]
    assert set(result) == {"class_balance", "fidelity", "privacy", "per_column"}
    assert set(result["fidelity"]) == {"ks_mean", "tvd_mean", "corr_diff",
                                       "coverage_mean"}
    assert "dcr_ratio" in result["privacy"]
    assert "age" in result["per_column"]

    dl = client.get(f"/api/lab/jobs/{job_id}/download")
    assert dl.status_code == 200
    assert dl.headers["content-type"].startswith("text/csv")


def test_lab_rejects_unknown_dataset_and_bad_target():
    r = client.post("/api/lab/run", json={
        "dataset_id": "nope", "target": "x", "seed_size": 100,
        "generator": "CTGAN", "synthetic_rows": 500, "ctgan_steps": 200})
    assert r.status_code == 404
    assert r.json()["error"]["code"] == "unknown_dataset"
