"""
Settings for the FastAPI service, read once at startup from config.yaml.

The `api:` block in config.yaml is the single place every limit lives
(upload size, rate limits, job TTL, histogram bins) so tuning the service
never means touching code. Defaults are safe if the block is missing.
"""

from pathlib import Path

import yaml

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def load_api_settings() -> dict:
    """Read the api: block from config.yaml with sane defaults."""
    config_path = PROJECT_ROOT / "config.yaml"
    defaults = {
        "version": "1.0.0",
        "cors_origins": [
            "http://localhost:5173",
            "http://localhost:4173",
            "http://127.0.0.1:5173",
        ],
        "max_upload_mb": 20,
        "max_upload_rows": 200000,
        "ctgan_steps_max": 3000,
        "job_ttl_minutes": 120,
        "rate_limit": "60/minute",
        "lab_rate_limit": "10/minute",
        "max_projection_points": 800,
        "histogram_bins": 40,
        "comparison_bins": 30,
    }
    if config_path.exists():
        with open(config_path, "r", encoding="utf-8") as handle:
            full_config = yaml.safe_load(handle) or {}
        defaults.update(full_config.get("api", {}) or {})
    return defaults


API_SETTINGS = load_api_settings()
