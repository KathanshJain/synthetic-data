"""
Results router: serves the OLD (v1) experiment artefacts — summary,
reference, fidelity, privacy (+ histogram), runtime, loss curve, criteria.

Every number comes from results/*.csv or reports/report_facts.json; nothing
is computed ad hoc here. The loss curve only exists after a --refresh run,
so it answers with a friendly 404 until then.
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from fastapi import APIRouter, Query

from api.errors import APIError
from api.schemas import (Criteria, DcrHistogram, Fidelity, LossCurve,
                         PrivacyItem, Reference, RuntimeItem, SummaryItem)

from api.services import results_service

router = APIRouter()


@router.get("/results/summary", response_model=list[SummaryItem])
def summary(model: str = Query(..., description="xgboost or random_forest")):
    """Per-arm summary for one model (metric means/stds across runs)."""
    if model not in ("xgboost", "random_forest"):
        raise APIError("unknown_model", f"Unknown model '{model}'.",
                       "Use model=xgboost or model=random_forest.",
                       status_code=404)
    try:
        return results_service.get_summary(model)
    except FileNotFoundError:
        raise APIError("results_missing",
                       "results/summary.csv is missing on the server.",
                       "Run the experiment first (python -m src.run_experiment).",
                       status_code=503)


@router.get("/results/reference", response_model=Reference)
def reference():
    """TRTR-full reference metrics per model + random-guess PR-AUC."""
    return results_service.get_reference()


@router.get("/results/fidelity", response_model=Fidelity)
def fidelity():
    """Fidelity summary (KS/TVD/chi2/correlation/coverage) + per-column."""
    return results_service.get_fidelity()


@router.get("/results/privacy", response_model=list[PrivacyItem])
def privacy():
    """DCR summary averaged over random seeds."""
    return results_service.get_privacy()


@router.get("/results/privacy/histogram", response_model=DcrHistogram)
def privacy_histogram(generator: str = Query(...),
                      seed_size: int = Query(...),
                      random_seed: int = Query(...)):
    """DCR distance histogram for one (generator, size, seed) run."""
    try:
        return results_service.get_dcr_histogram(generator, seed_size,
                                                 random_seed)
    except KeyError:
        raise APIError(
            "histogram_not_found",
            f"No DCR histogram for {generator} seed_size={seed_size} "
            f"random_seed={random_seed}.",
            "Check /api/results/privacy for available (generator, size) "
            "pairs and use a random_seed from /api/seeds.",
            status_code=404,
        )


@router.get("/results/runtime", response_model=list[RuntimeItem])
def runtime():
    """Generator training seconds by generator and seed size."""
    return results_service.get_runtime()


@router.get("/results/loss-curve", response_model=LossCurve)
def loss_curve(seed_size: int = Query(...), random_seed: int = Query(...)):
    """CTGAN loss curve for one run (exists only after a --refresh run)."""
    try:
        return results_service.get_loss_curve(seed_size, random_seed)
    except FileNotFoundError:
        raise APIError(
            "loss_curve_unavailable",
            "No CTGAN loss curve is stored for this run. The original "
            "experiment predates loss saving; run --refresh to generate it.",
            "The UI shows a friendly 'unavailable' state — this is expected "
            "until someone re-runs the experiment with --refresh.",
            status_code=404,
        )


@router.get("/results/criteria", response_model=Criteria)
def criteria():
    """Frozen criteria rules + the pass/fail table (computed by src/)."""
    return results_service.get_criteria()
