"""
run_experiment.py — Step 9: run the whole pipeline end to end.

Why this file exists: this is the "conductor". For every random seed and
every seed size it trains the generators, evaluates fidelity, privacy and
utility, and appends one row per measurement to CSV files in results/.
It supports --quick (a fast smoke test using quick_mode settings) and can
RESUME an interrupted run: any (random_seed, seed_size, arm, model) row that
already exists in metrics.csv is skipped.

SAFETY: the locked test set is loaded here (hash verified first) because the
5 arms are finally evaluated on it. Quick-mode outputs go to results/quick/
so they never mix with the real results.
"""

import argparse
import shutil

import pandas as pd
from sklearn.model_selection import train_test_split

from src.fidelity import evaluate_fidelity
from src.generators import (
    generate,
    repair_synthetic,
    run_smote_nc,
    train_ctgan,
    train_gaussian_copula,
    validate_synthetic,
)
from src.models import train_and_evaluate
from src.privacy import compute_dcr
from src.seeds import build_nested_seeds, get_holdout
from src.utils import (
    PROJECT_ROOT,
    get_column_types,
    load_config,
    load_locked_test,
    set_all_seeds,
    verify_test_hash,
)

RESULTS_DIR = PROJECT_ROOT / "results"


def append_row(csv_path, row):
    """Append one result row to a CSV file (creating it if needed)."""
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    existing = pd.read_csv(csv_path) if csv_path.exists() else pd.DataFrame()
    combined = pd.concat([existing, pd.DataFrame([row])], ignore_index=True)
    combined.to_csv(csv_path, index=False)


def already_done(metrics_path, random_seed, seed_size, arm, model):
    """Resume support: check whether a result row already exists."""
    if not metrics_path.exists():
        return False
    metrics = pd.read_csv(metrics_path)
    mask = ((metrics["random_seed"] == random_seed) &
            (metrics["seed_size"] == seed_size) &
            (metrics["arm"] == arm) &
            (metrics["model"] == model))
    return mask.any()


def run_experiment(quick=False):
    config = load_config()

    # ---- quick mode overrides (a fast smoke test of the whole pipeline) ----
    if quick:
        quick_config = dict(config)
        quick_config["seeds"] = {
            "seed_sizes": config["quick_mode"]["seed_sizes"],
            "random_seeds": config["quick_mode"]["random_seeds"],
        }
        quick_config["generators"] = dict(config["generators"])
        quick_config["generators"]["synthetic_rows"] = \
            config["quick_mode"]["synthetic_rows"]
        quick_config["generators"]["ctgan"] = {
            "total_training_steps":
                config["quick_mode"]["ctgan_total_training_steps"],
            "batch_size": config["generators"]["ctgan"]["batch_size"],
            "use_gpu": config["generators"]["ctgan"]["use_gpu"],
        }
        config = quick_config
        results_dir = RESULTS_DIR / "quick"
    else:
        results_dir = RESULTS_DIR
    results_dir.mkdir(parents=True, exist_ok=True)

    # save the exact config used, for reproducibility
    import yaml
    with open(results_dir / "run_config.yaml", "w") as f:
        yaml.dump(config, f)

    metrics_path = results_dir / "metrics.csv"
    fidelity_path = results_dir / "fidelity.csv"
    fidelity_per_column_path = results_dir / "fidelity_per_column.csv"
    privacy_path = results_dir / "privacy.csv"
    runtime_path = results_dir / "runtime.csv"

    # ---- 1. load data (hash verified) --------------------------------------
    verify_test_hash()
    test_set = load_locked_test()
    dev_pool = pd.read_csv(PROJECT_ROOT / "data" / "processed" / "dev_pool.csv")
    target = config["dataset"]["target"]
    numerical_cols, categorical_cols = get_column_types(dev_pool, target)

    total_runs = (len(config["seeds"]["random_seeds"]) *
                  len(config["seeds"]["seed_sizes"]))
    run_number = 0

    for random_seed in config["seeds"]["random_seeds"]:
        # ---- a. TRTR-full once per random seed -----------------------------
        set_all_seeds(random_seed)
        run_number += 1
        print(f"\n[RUN {run_number}/{total_runs}] rs={random_seed} "
              f"-> TRTR-full (RF + XGB)")
        for model_name in ["random_forest", "xgboost"]:
            if already_done(metrics_path, random_seed, 0, "TRTR-full",
                            model_name):
                print(f"    {model_name} already done — skipping.")
                continue
            metrics = train_and_evaluate(
                dev_pool, test_set, model_name, target, config, random_seed)
            append_row(metrics_path, {
                "random_seed": random_seed, "seed_size": 0,
                "arm": "TRTR-full", "model": model_name, **metrics})

        # ---- b. nested seeds + DCR holdout ---------------------------------
        seeds = build_nested_seeds(dev_pool, target,
                                   config["seeds"]["seed_sizes"], random_seed)
        largest_seed = seeds[max(seeds)]
        holdout = get_holdout(dev_pool, largest_seed,
                              config["dcr"]["holdout_size"], random_seed)

        for seed_size in config["seeds"]["seed_sizes"]:
            seed_df = seeds[seed_size]
            print(f"\n[RUN {run_number}/{total_runs}] rs={random_seed} "
                  f"n={seed_size}")
            run_number += 1

            # ---- TRTR-seed ----
            for model_name in ["random_forest", "xgboost"]:
                if already_done(metrics_path, random_seed, seed_size,
                                "TRTR-seed", model_name):
                    continue
                metrics = train_and_evaluate(
                    seed_df, test_set, model_name, target, config, random_seed)
                append_row(metrics_path, {
                    "random_seed": random_seed, "seed_size": seed_size,
                    "arm": "TRTR-seed", "model": model_name, **metrics})

            # ---- SMOTE-NC ----
            if not already_done(metrics_path, random_seed, seed_size,
                                "SMOTE-NC", "xgboost"):
                smote_df, n_new = run_smote_nc(
                    seed_df, numerical_cols, categorical_cols, target,
                    config["generators"]["smote"]["k_neighbors"], random_seed)
                print(f"    SMOTE-NC added {n_new} minority rows")
                for model_name in ["random_forest", "xgboost"]:
                    metrics = train_and_evaluate(
                        smote_df, test_set, model_name, target, config,
                        random_seed)
                    append_row(metrics_path, {
                        "random_seed": random_seed, "seed_size": seed_size,
                        "arm": "SMOTE-NC", "model": model_name, **metrics})

            # ---- CTGAN and Gaussian Copula (same steps for each) ------------
            for generator_name, train_function in [
                    ("CTGAN", train_ctgan),
                    ("GaussianCopula", train_gaussian_copula)]:
                arm = f"TSTR-{generator_name}"
                print(f"    {generator_name} training...")
                if generator_name == "CTGAN":
                    synthesizer, fit_seconds, _ = train_ctgan(
                        seed_df, config, random_seed)
                else:
                    synthesizer, fit_seconds = train_gaussian_copula(
                        seed_df, config, random_seed)

                synth_df, sample_seconds = generate(
                    synthesizer, config["generators"]["synthetic_rows"])
                # fill NaNs the generator left in rare categorical columns
                synth_df = repair_synthetic(synth_df, seed_df,
                                            numerical_cols, categorical_cols)

                # save the synthetic data
                synth_dir = PROJECT_ROOT / "data" / "synthetic" / f"rs{random_seed}"
                synth_dir.mkdir(parents=True, exist_ok=True)
                file_prefix = "ctgan" if generator_name == "CTGAN" else "gc"
                synth_df.to_csv(
                    synth_dir / f"{file_prefix}_{seed_size}.csv", index=False)

                # validate
                warnings = validate_synthetic(
                    synth_df, dev_pool, numerical_cols, categorical_cols)
                for warning in warnings:
                    print(f"    WARNING: {warning}")

                # fidelity vs dev pool
                per_column_table, fidelity_summary = evaluate_fidelity(
                    dev_pool, synth_df, numerical_cols, categorical_cols)
                append_row(fidelity_path, {
                    "random_seed": random_seed, "seed_size": seed_size,
                    "generator": generator_name, **fidelity_summary})
                # per-column values get one row per column
                for column_name, column_row in per_column_table.iterrows():
                    append_row(fidelity_per_column_path, {
                        "random_seed": random_seed, "seed_size": seed_size,
                        "generator": generator_name,
                        "column": column_name, **column_row.to_dict()})

                # privacy (DCR)
                dcr = compute_dcr(seed_df, synth_df, holdout,
                                  numerical_cols, categorical_cols)
                append_row(privacy_path, {
                    "random_seed": random_seed, "seed_size": seed_size,
                    "generator": generator_name, **dcr})

                # runtime
                append_row(runtime_path, {
                    "random_seed": random_seed, "seed_size": seed_size,
                    "generator": generator_name,
                    "fit_seconds": round(fit_seconds, 1),
                    "sample_seconds": round(sample_seconds, 1)})

                # utility: TSTR arms
                for model_name in ["random_forest", "xgboost"]:
                    if already_done(metrics_path, random_seed, seed_size,
                                    arm, model_name):
                        continue
                    metrics = train_and_evaluate(
                        synth_df, test_set, model_name, target, config,
                        random_seed)
                    append_row(metrics_path, {
                        "random_seed": random_seed, "seed_size": seed_size,
                        "arm": arm, "model": model_name, **metrics})
                print(f"    {generator_name} done.")

    print(f"\nExperiment {'(QUICK) ' if quick else ''}finished. "
          f"Results in {results_dir}")


def main():
    parser = argparse.ArgumentParser(
        description="Run the full synthetic-data experiment.")
    parser.add_argument("--quick", action="store_true",
                        help="fast smoke test using quick_mode settings")
    args = parser.parse_args()
    run_experiment(quick=args.quick)


if __name__ == "__main__":
    main()
