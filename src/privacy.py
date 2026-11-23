"""
privacy.py — Step 7: Distance to Closest Record (DCR) memorisation check.

Why this file exists: we must show that the generator did not simply COPY
rows of its training seed. DCR measures, for every synthetic row, the
distance to the closest row of the seed. If the generator memorised, many
synthetic rows would sit (almost) on top of real seed rows.

The catch: "close" must be compared against a fair baseline. Real people can
be naturally near-identical on these columns (two 40-hour/week married
office workers). So we compare against a HOLDOUT of genuine rows the
generator never saw. If synthetic rows are much closer to the seed than
genuine new records are, that is a memorisation warning.

IMPORTANT: DCR is a WARNING INDICATOR, not a privacy guarantee. It cannot
prove that no information leaked; formal guarantees need methods like
differential privacy.

Scaling note: we standardise numerical columns BEFORE computing distances,
otherwise capital-gain (values in the hundreds of thousands) would totally
dominate age (values under 100) and the distance would measure nothing else.
The scaler and the one-hot encoder are fitted on the SEED ONLY — never on
synthetic data (that would leak) and never on the locked test set.
"""

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.neighbors import NearestNeighbors
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

EXACT_MATCH_THRESHOLD = 1e-8  # distances below this count as "exact copy"


def compute_dcr(seed_df, synth_df, holdout_df, numerical_cols, categorical_cols):
    """
    Compute DCR of synthetic rows and of holdout rows to the seed.

    Returns a dict with medians, 5th percentiles, the dcr_ratio
    (synthetic median / holdout median) and exact-match rates for both.
    """
    seed_features = seed_df.drop(columns=["row_id"], errors="ignore")
    synth_features = synth_df.drop(columns=["row_id"], errors="ignore")
    holdout_features = holdout_df.drop(columns=["row_id"], errors="ignore")

    # fit preprocessing on the SEED ONLY (fitting on synthetic would leak)
    preprocessor = ColumnTransformer([
        ("numerical", StandardScaler(), numerical_cols),
        ("categorical", OneHotEncoder(handle_unknown="ignore",
                                      sparse_output=False), categorical_cols),
    ])
    seed_transformed = preprocessor.fit_transform(seed_features)
    synth_transformed = preprocessor.transform(synth_features)
    holdout_transformed = preprocessor.transform(holdout_features)

    # nearest-neighbour search over the seed rows
    neighbours = NearestNeighbors(n_neighbors=1)
    neighbours.fit(seed_transformed)

    synth_distances, _ = neighbours.kneighbors(synth_transformed)
    holdout_distances, _ = neighbours.kneighbors(holdout_transformed)
    synth_distances = synth_distances.flatten()
    holdout_distances = holdout_distances.flatten()

    synth_median = float(np.median(synth_distances))
    holdout_median = float(np.median(holdout_distances))

    return {
        "synth_median_dcr": round(synth_median, 4),
        "synth_p05_dcr": round(float(np.percentile(synth_distances, 5)), 4),
        "holdout_median_dcr": round(holdout_median, 4),
        "holdout_p05_dcr": round(float(np.percentile(holdout_distances, 5)), 4),
        "dcr_ratio": round(synth_median / max(holdout_median, 1e-12), 4),
        "synth_exact_match_rate": round(
            float((synth_distances < EXACT_MATCH_THRESHOLD).mean()), 4),
        "holdout_exact_match_rate": round(
            float((holdout_distances < EXACT_MATCH_THRESHOLD).mean()), 4),
    }


def plot_dcr(synth_dcr, holdout_dcr, save_path):
    """Two overlaid histograms: how far synthetic vs genuine rows sit from the seed."""
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(synth_dcr, bins=40, alpha=0.6, label="synthetic rows", density=True)
    ax.hist(holdout_dcr, bins=40, alpha=0.6, label="holdout rows (real)",
            density=True)
    ax.set_xlabel("Distance to closest seed row")
    ax.set_ylabel("Density")
    ax.set_title("DCR: synthetic vs genuine holdout baseline")
    ax.legend()
    fig.tight_layout()
    fig.savefig(save_path)
    plt.close(fig)


def main():
    """Demo on the Step 4 outputs for seed 500 with a 1000-row holdout."""
    from src.seeds import get_holdout
    from src.utils import PROJECT_ROOT, get_column_types, load_config

    config = load_config()
    target = config["dataset"]["target"]
    dev_pool = pd.read_csv(PROJECT_ROOT / "data" / "processed" / "dev_pool.csv")
    seed_df = pd.read_csv(PROJECT_ROOT / "data" / "seeds" / "rs0" / "seed_500.csv")
    synth_df = pd.read_csv(PROJECT_ROOT / "data" / "synthetic" / "rs0" / "gc_500.csv")

    holdout_size = config["dcr"]["holdout_size"]
    holdout_df = get_holdout(dev_pool, seed_df, holdout_size, 0)
    numerical_cols, categorical_cols = get_column_types(seed_df, target)

    results = compute_dcr(seed_df, synth_df, holdout_df,
                          numerical_cols, categorical_cols)
    for key, value in results.items():
        print(f"{key}: {value}")

    # recompute raw distances for the plot
    figures_dir = PROJECT_ROOT / "figures" / "examples"
    figures_dir.mkdir(parents=True, exist_ok=True)
    preprocessor = ColumnTransformer([
        ("numerical", StandardScaler(), numerical_cols),
        ("categorical", OneHotEncoder(handle_unknown="ignore",
                                      sparse_output=False), categorical_cols),
    ])
    seed_t = preprocessor.fit_transform(seed_df.drop(columns=["row_id"],
                                                 errors="ignore"))
    neighbours = NearestNeighbors(n_neighbors=1).fit(seed_t)
    synth_t = preprocessor.transform(synth_df.drop(columns=["row_id"],
                                                   errors="ignore"))
    holdout_t = preprocessor.transform(holdout_df.drop(columns=["row_id"],
                                                      errors="ignore"))
    synth_distances = neighbours.kneighbors(synth_t)[0].flatten()
    holdout_distances = neighbours.kneighbors(holdout_t)[0].flatten()
    plot_dcr(synth_distances, holdout_distances,
             figures_dir / "example_dcr.png")
    print(f"DCR plot saved to {figures_dir / 'example_dcr.png'}")


if __name__ == "__main__":
    main()
