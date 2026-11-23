"""
Compare router: real-vs-synthetic distribution, correlation and PCA
projection for the Fidelity Studio (all served from stored synthetic CSVs).
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from fastapi import APIRouter, Query

from api.errors import APIError
from api.schemas import Comparison, Correlation, Projection
from api.services import compare_service
from api.settings import API_SETTINGS

router = APIRouter()


def _not_found(detail: str) -> APIError:
    return APIError("run_not_found", detail,
                    "Check /api/results/fidelity for available "
                    "(generator, seed_size) pairs and /api/seeds for "
                    "random_seed values.",
                    status_code=404)


@router.get("/compare/distribution", response_model=Comparison)
def compare_distribution(column: str = Query(...),
                         generator: str = Query(...),
                         seed_size: int = Query(...),
                         random_seed: int = Query(...)):
    """Shared-bin / shared-category real vs synthetic counts for a column."""
    try:
        return compare_service.compare_distribution(column, generator,
                                                    seed_size, random_seed)
    except KeyError as exc:
        raise _not_found(f"Unknown column or run: {exc.args[0]}")


@router.get("/compare/correlation", response_model=Correlation)
def compare_correlation(generator: str = Query(...),
                        seed_size: int = Query(...),
                        random_seed: int = Query(...)):
    """Pearson matrices (real vs synthetic) + absolute difference."""
    try:
        return compare_service.compare_correlation(generator, seed_size,
                                                   random_seed)
    except KeyError as exc:
        raise _not_found(f"Unknown run: {exc.args[0]}")


@router.get("/compare/projection", response_model=Projection)
def compare_projection(generator: str = Query(...),
                       seed_size: int = Query(...),
                       random_seed: int = Query(...)):
    """PCA(2) fitted on real only, ≤ max_projection_points per side."""
    try:
        return compare_service.compare_projection(
            generator, seed_size, random_seed,
            max_points=API_SETTINGS["max_projection_points"])
    except KeyError as exc:
        raise _not_found(f"Unknown run: {exc.args[0]}")
