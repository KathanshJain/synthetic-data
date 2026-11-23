"""
backfill_fidelity_per_column.py — one-off repair script.

Why this exists: the full experiment was launched just before
run_experiment.py was fixed to write PER-COLUMN fidelity values into
results/fidelity_per_column.csv (the running process only wrote summary
rows there). The synthetic CSVs themselves were saved correctly, so we can
recompute the per-column table from disk without retraining anything.

Run AFTER the full experiment:  python -m src.backfill_fidelity_per_column
"""

import pandas as pd

from src.fidelity import evaluate_fidelity
from src.utils import PROJECT_ROOT, get_column_types, load_config

RESULTS_DIR = PROJECT_ROOT / "results"


def main():
    config = load_config()
    target = config["dataset"]["target"]
    dev_pool = pd.read_csv(PROJECT_ROOT / "data" / "processed" / "dev_pool.csv")
    numerical_cols, categorical_cols = get_column_types(dev_pool, target)

    rows = []
    for random_seed in config["seeds"]["random_seeds"]:
        for seed_size in config["seeds"]["seed_sizes"]:
            for prefix, generator in [("ctgan", "CTGAN"),
                                      ("gc", "GaussianCopula")]:
                synth_path = (PROJECT_ROOT / "data" / "synthetic" /
                              f"rs{random_seed}" / f"{prefix}_{seed_size}.csv")
                if not synth_path.exists():
                    print(f"missing {synth_path} — skipping")
                    continue
                synth_df = pd.read_csv(synth_path)
                per_column, _ = evaluate_fidelity(
                    dev_pool, synth_df, numerical_cols, categorical_cols)
                for column_name, column_row in per_column.iterrows():
                    rows.append({
                        "random_seed": random_seed,
                        "seed_size": seed_size,
                        "generator": generator,
                        "column": column_name,
                        **column_row.to_dict(),
                    })

    table = pd.DataFrame(rows)
    table.to_csv(RESULTS_DIR / "fidelity_per_column.csv", index=False)
    print(f"Wrote {len(table)} per-column rows to "
          f"results/fidelity_per_column.csv")


if __name__ == "__main__":
    main()
