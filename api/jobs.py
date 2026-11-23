"""
In-memory job manager for the interactive lab.

Design decisions:
- ONE CTGAN at a time: a global semaphore serialises CTGAN training (it is
  the heavy step); GaussianCopula/SMOTE jobs are cheap and run concurrently.
- Jobs live in memory + a temp directory; a cleanup thread deletes job
  directories older than the configured TTL.
- Progress stages: validate → seed → train → sample → fidelity → privacy →
  done, so the frontend can show a meaningful progress bar.
"""

import threading
import time
import uuid


class Job:
    """One lab run: status, stage, progress, log lines, result, error."""

    def __init__(self, job_id: str):
        self.id = job_id
        self.status = "queued"      # queued|running|completed|failed
        self.stage = "queued"
        self.progress = 0
        self.log: list = []
        self.result: dict | None = None
        self.error: str | None = None
        self.created = time.time()
        self.updated = time.time()
        self._lock = threading.Lock()

    def update(self, stage: str | None = None, progress: int | None = None,
               status: str | None = None, log: str | None = None,
               error: str | None = None, result: dict | None = None):
        with self._lock:
            if stage is not None:
                self.stage = stage
            if progress is not None:
                self.progress = max(self.progress, progress)
            if status is not None:
                self.status = status
            if log is not None:
                stamp = time.strftime("%H:%M:%S")
                self.log.append(f"[{stamp}] {log}")
            if error is not None:
                self.error = error
            if result is not None:
                self.result = result
            self.updated = time.time()

    def snapshot(self) -> dict:
        with self._lock:
            return {
                "status": self.status,
                "stage": self.stage,
                "progress": self.progress,
                "log": list(self.log),
                "result": self.result,
                "error": self.error,
            }


class JobManager:
    """Creates jobs, runs them on worker threads, cleans up old ones."""

    def __init__(self, ttl_minutes: int = 120, max_workers: int = 4):
        self.jobs: dict[str, Job] = {}
        self.ttl_seconds = ttl_minutes * 60
        self.ctgan_semaphore = threading.Semaphore(1)  # ONE CTGAN at a time
        self._lock = threading.Lock()
        self._cleanup_thread = threading.Thread(target=self._cleanup_loop,
                                                daemon=True)
        self._cleanup_thread.start()

    def create(self) -> Job:
        job = Job(uuid.uuid4().hex[:12])
        with self._lock:
            self.jobs[job.id] = job
        return job

    def get(self, job_id: str) -> Job | None:
        return self.jobs.get(job_id)

    def submit(self, job: Job, fn, use_ctgan_slot: bool):
        """Run fn(job) on a worker thread, optionally holding the CTGAN slot."""

        def runner():
            try:
                if use_ctgan_slot:
                    job.update(log="Waiting for a free CTGAN slot "
                                   "(one CTGAN trains at a time).")
                    with self.ctgan_semaphore:
                        fn(job)
                else:
                    fn(job)
            except Exception as exc:  # noqa: BLE001 - job errors go to the client
                job.update(status="failed", error=str(exc),
                           log=f"FAILED: {exc}")

        thread = threading.Thread(target=runner, daemon=True)
        thread.start()

    def _cleanup_loop(self):
        while True:
            time.sleep(60)
            cutoff = time.time() - self.ttl_seconds
            with self._lock:
                for job_id in list(self.jobs):
                    if self.jobs[job_id].updated < cutoff:
                        del self.jobs[job_id]


MANAGER = JobManager(ttl_minutes=120)
