"""
Lab router: upload → profile → run → job polling/SSE → download.

Guards (per the contract):
- rate limits per IP on upload/run (slowapi);
- ONE CTGAN trains at a time (others queue; the job log says so);
- uploads live in a temp dir with TTL cleanup, never inside the project;
- the locked test set is never involved in any lab path.
"""

import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from fastapi import APIRouter, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse, StreamingResponse
from slowapi import Limiter
from slowapi.util import get_remote_address

from api.errors import APIError
from api.jobs import MANAGER
from api.schemas import Job, LabRunRequest, Upload
from api.services import lab_service

router = APIRouter()
limiter = Limiter(key_func=get_remote_address)

VALID_GENERATORS = ("CTGAN", "GaussianCopula", "SMOTE-NC")


@router.post("/lab/upload", response_model=Upload)
@limiter.limit("10/minute")
async def upload_dataset(request: Request, file: UploadFile):
    """Profile an uploaded CSV and store it for lab runs (TTL applies)."""
    content = await file.read()
    try:
        dataset_id, df = lab_service.save_upload(content, file.filename or "")
    except ValueError as exc:
        raise APIError("bad_upload", str(exc),
                       "Upload a CSV of at least 50 rows under the size "
                       "limit; include a binary target column.")
    profile = lab_service.profile_for_contract(df)
    return {"dataset_id": dataset_id, "profile": profile}


@router.post("/lab/run")
@limiter.limit("10/minute")
async def run_lab(request: Request, body: LabRunRequest):
    """Start a lab job; returns {job_id} immediately (202-style contract)."""
    if body.generator not in VALID_GENERATORS:
        raise APIError("unknown_generator",
                       f"Unknown generator '{body.generator}'.",
                       f"Use one of: {', '.join(VALID_GENERATORS)}.",
                       status_code=422)
    try:
        # fail fast if the dataset_id is unknown (before starting the job)
        lab_service.load_upload(body.dataset_id)
    except FileNotFoundError:
        raise APIError("unknown_dataset",
                       f"Unknown dataset_id '{body.dataset_id}'.",
                       "Upload the CSV again with /api/lab/upload first.",
                       status_code=404)
    job = MANAGER.create()
    params = body.model_dump()
    MANAGER.submit(job, lambda j: lab_service.run_lab_job(j, params),
                   use_ctgan_slot=(body.generator == "CTGAN"))
    job.update(status="running", log="Job accepted.")
    return {"job_id": job.id}


@router.get("/lab/jobs/{job_id}", response_model=Job)
def job_status(job_id: str):
    """Poll a lab job's status, stage, progress, log and result."""
    job = MANAGER.get(job_id)
    if job is None:
        raise APIError("job_not_found", f"Unknown job_id '{job_id}'.",
                       "Jobs expire after the TTL; start a new run.",
                       status_code=404)
    return job.snapshot()


@router.get("/lab/jobs/{job_id}/events")
def job_events(job_id: str):
    """Server-Sent Events stream of job snapshots until the job finishes."""
    job = MANAGER.get(job_id)
    if job is None:
        raise APIError("job_not_found", f"Unknown job_id '{job_id}'.",
                       "Jobs expire after the TTL; start a new run.",
                       status_code=404)

    def stream():
        import json as _json
        last_line_count = -1
        while True:
            snap = job.snapshot()
            # send the full snapshot; the frontend re-renders from it
            yield f"data: {_json.dumps(snap)}\n\n"
            if snap["status"] in ("completed", "failed"):
                break
            if len(snap["log"]) == last_line_count and snap["status"] == "queued":
                # nothing changed and still queued — keep the stream alive
                pass
            last_line_count = len(snap["log"])
            time.sleep(1.0)

    return StreamingResponse(stream(), media_type="text/event-stream",
                             headers={"Cache-Control": "no-cache"})


@router.get("/lab/jobs/{job_id}/download")
def job_download(job_id: str):
    """Download the synthetic CSV produced by a completed job."""
    job = MANAGER.get(job_id)
    if job is None:
        raise APIError("job_not_found", f"Unknown job_id '{job_id}'.",
                       "Jobs expire after the TTL; start a new run.",
                       status_code=404)
    path = lab_service.JOBS_DIR / job_id / "synthetic.csv"
    if not path.exists():
        raise APIError("download_unavailable",
                       "No synthetic CSV is available for this job yet.",
                       "Download appears once the job completes.",
                       status_code=404)
    return FileResponse(path, media_type="text/csv",
                        filename=f"synthetic_{job_id}.csv")
