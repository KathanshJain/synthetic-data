"""
Health + overview + seeds + distribution routers.

/health reads the cached pytest summary produced by scripts/run_checks.py
(NOTHING runs tests inside the request — that would make the endpoint slow
and fragile).
"""

import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from fastapi import APIRouter, Query

from api.errors import APIError
from api.schemas import Distribution, Health, Overview, Seeds
from api.services import results_service
from api.settings import API_SETTINGS

router = APIRouter()


@router.get("/health", response_model=Health)
def health():
    """Liveness + the cached test summary from scripts/run_checks.py."""
    checks = results_service.get_json(
        PROJECT_ROOT / "results" / "checks.json") or {}
    return {
        "status": "ok" if checks.get("status") == "pass" else "degraded",
        "version": API_SETTINGS["version"],
        "tests_passing": int(checks.get("tests_passing", 0)),
        "tests_total": int(checks.get("tests_total", 0)),
    }


@router.get("/overview", response_model=Overview)
def overview():
    """Dataset story + locked-test-set status (hash check, never its rows)."""
    return results_service.get_overview()


@router.get("/seeds", response_model=Seeds)
def seeds():
    """Seed sizes, random seeds, nested-split proof and class counts."""
    return results_service.get_seeds()


@router.get("/dataset/distribution", response_model=Distribution)
def distribution(column: str = Query(..., description="Column name")):
    """Real-data-only distribution of one column (for the data snapshot)."""
    try:
        return results_service.get_distribution(column)
    except KeyError:
        raise APIError(
            "unknown_column",
            f"Column '{column}' does not exist in the dataset.",
            "Pick a column name listed in /api/overview features.",
            status_code=404,
        )
