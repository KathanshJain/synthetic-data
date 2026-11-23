"""
Tiny mtime-based cache for CSV/JSON artefacts.

Results files change only when the experiment is re-run, so caching by
(mtime, size) is safe and keeps the API fast without any staleness risk:
the key includes the file's modification time, so a refreshed file is
re-read automatically.
"""

import json
from pathlib import Path

_cache: dict = {}


def get(path: Path, loader):
    """Return the cached parse of `path`, re-reading it if it changed."""
    try:
        stat = path.stat()
        key = (str(path), stat.st_mtime_ns, stat.st_size)
    except OSError:
        return None
    hit = _cache.get(key[0])
    if hit and hit[0] == key[1:]:
        return hit[1]
    try:
        value = loader(path)
    except Exception:
        return None
    _cache[key[0]] = (key[1:], value)
    return value


def get_json(path: Path):
    """Cached JSON parse (report_facts.json, manifest.json, checks.json)."""
    return get(path, lambda p: json.loads(p.read_text(encoding="utf-8")))


def get_frame(path: Path):
    """Cached DataFrame parse (every results/*.csv)."""
    import pandas as pd
    return get(path, lambda p: pd.read_csv(p))
