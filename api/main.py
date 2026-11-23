"""
FastAPI application: serves the OLD (v1) results to the React frontend and
runs the interactive lab.

- every route is mounted TWICE: under /api (the contract's base path) and
  at the root, so both http://host:8000/api/health and /health work —
  the frontend's VITE_API_URL may or may not include the /api prefix;
- CORS is open only to the configured Vite origins;
- if frontend/web/dist exists (production build) it is served at / so one
  process can host the whole app;
- every error uses the {"error": {code, message, hint}} envelope.

GOLDEN RULE honoured here: nothing under api/ ever loads the locked test
set; the only interaction is verify_test_hash() → verified true/false.
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from api.errors import install_error_handlers
from api.routers import compare, health, lab, results
from api.settings import API_SETTINGS

app = FastAPI(
    title="Forged Data API",
    description="Serves the Machine Learning on Artificially Forged Data "
                "study: results, comparisons and the interactive lab.",
    version=API_SETTINGS["version"],
    docs_url="/docs",
)

install_error_handlers(app)

app.add_middleware(
    CORSMiddleware,
    allow_origins=API_SETTINGS["cors_origins"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Routers at BOTH /api/... and /... (contract base path + convenience).
for prefix in ("/api", ""):
    app.include_router(health.router, prefix=prefix)
    app.include_router(results.router, prefix=prefix)
    app.include_router(compare.router, prefix=prefix)
    app.include_router(lab.router, prefix=prefix)

# Production: serve the built frontend if it exists. Registered LAST so API
# routes win; anything else falls through to the SPA bundle.
DIST = PROJECT_ROOT / "frontend" / "web" / "dist"
if DIST.exists():
    app.mount("/", StaticFiles(directory=str(DIST), html=True),
              name="frontend")
