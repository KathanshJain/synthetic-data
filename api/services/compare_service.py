"""
compare_service: real-vs-synthetic comparisons for the Fidelity Studio.

- /compare/distribution: shared bins (numerical) or shared categories
  (categorical) so the two counts are directly comparable.
- /compare/correlation: Pearson matrices for real and synthetic, plus the
  absolute difference matrix the heatmap renders.
- /compare/projection: PCA(2) fitted on REAL data only, then applied to
  synthetic — the projection must not learn from synthetic records.
  Points are subsampled to a fixed cap per side for a responsive payload.
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA

from src.utils import get_column_types, load_config

from .cache import get_frame
from .results_service import DATA, GENERATORS, _dev_pool

SYNTH_DIRS = {"CTGAN": "ctgan", "GaussianCopula": "gc"}


def _synthetic_frame(generator: str, seed_size: int, random_seed: int):
    """Load one synthetic CSV; raise KeyError if that run does not exist."""
    prefix = SYNTH_DIRS.get(generator)
    if prefix is None:
        raise KeyError(generator)
    path = DATA / "synthetic" / f"rs{random_seed}" / f"{prefix}_{seed_size}.csv"
    frame = get_frame(path)
    if frame is None:
        raise KeyError(f"{generator}/{seed_size}/rs{random_seed}")
    return frame


def compare_distribution(column: str, generator: str, seed_size: int,
                         random_seed: int) -> dict:
    real = _dev_pool()
    synth = _synthetic_frame(generator, seed_size, random_seed)
    for frame in (real, synth):
        if column not in frame.columns:
            raise KeyError(column)
    if pd.api.types.is_numeric_dtype(real[column]):
        low = float(min(real[column].min(), synth[column].min()))
        high = float(max(real[column].max(), synth[column].max()))
        if high <= low:
            high = low + 1e-9
        edges = np.linspace(low, high, 31)
        real_counts, _ = np.histogram(real[column], bins=edges)
        synth_counts, _ = np.histogram(synth[column], bins=edges)
        return {"type": "histogram",
                "bins": [round(float(e), 4) for e in edges],
                "real": [int(v) for v in real_counts],
                "synthetic": [int(v) for v in synth_counts]}
    categories = sorted(set(real[column].astype(str)) |
                        set(synth[column].astype(str)))
    real_counts = real[column].astype(str).value_counts().reindex(categories,
                                                                  fill_value=0)
    synth_counts = synth[column].astype(str).value_counts().reindex(categories,
                                                                    fill_value=0)
    return {"type": "categories",
            "categories": categories,
            "real": [int(v) for v in real_counts],
            "synthetic": [int(v) for v in synth_counts]}


def compare_correlation(generator: str, seed_size: int, random_seed: int) -> dict:
    config = load_config()
    real = _dev_pool()
    synth = _synthetic_frame(generator, seed_size, random_seed)
    numerical, _ = get_column_types(real, config["dataset"]["target"])
    columns = [c for c in numerical if c in synth.columns]
    real_corr = real[columns].corr(method="pearson")
    synth_corr = synth[columns].corr(method="pearson")
    diff = (real_corr - synth_corr).abs()
    return {
        "columns": columns,
        "real": [[round(float(v), 4) for v in row] for row in real_corr.values],
        "synthetic": [[round(float(v), 4) for v in row] for row in synth_corr.values],
        "diff": [[round(float(v), 4) for v in row] for row in diff.values],
    }


def compare_projection(generator: str, seed_size: int, random_seed: int,
                       max_points: int = 800) -> dict:
    """PCA fitted on real only; ≤ max_points per side; fixed subsample seed."""
    config = load_config()
    real = _dev_pool()
    synth = _synthetic_frame(generator, seed_size, random_seed)
    numerical, _ = get_column_types(real, config["dataset"]["target"])
    columns = [c for c in numerical if c in synth.columns]

    real_x = real[columns].dropna()
    synth_x = synth[columns].dropna()
    if len(real_x) > max_points:
        real_x = real_x.sample(max_points, random_state=0)
    if len(synth_x) > max_points:
        synth_x = synth_x.sample(max_points, random_state=0)

    pca = PCA(n_components=2, random_state=0)
    real_proj = pca.fit_transform(real_x)          # fitted on REAL only
    synth_proj = pca.transform(synth_x)            # projected with real params
    return {
        "real": [[round(float(x), 4), round(float(y), 4)]
                 for x, y in real_proj],
        "synthetic": [[round(float(x), 4), round(float(y), 4)]
                      for x, y in synth_proj],
        "explained_variance": [round(float(v), 4)
                               for v in pca.explained_variance_ratio_],
    }
