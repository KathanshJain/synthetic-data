"""
generators.py — Step 5: the three data generators.

Why this file exists: this is the heart of the project. Given a small real
seed, we create artificial rows with three different approaches:

  * CTGAN         — a tabular GAN. A generator network invents rows; a
                    discriminator network tries to spot the fakes. They
                    compete until the fakes look real. CTGAN adds tricks for
                    tables: mode-specific normalisation for multimodal
                    numerical columns and conditional sampling so rare
                    categories still get training attention.
  * Gaussian Copula — a statistical model: it fits each column's marginal
                    distribution plus a dependence structure between columns,
                    then samples new combinations. Fast and interpretable.
  * SMOTE-NC      — NOT a full generator. It oversamples the minority class
                    by interpolating between nearby minority rows. It is our
                    classical class-imbalance baseline.

FAIR TRAINING BUDGET: CTGAN's epochs depend on dataset size
(steps_per_epoch = max(len(data) // batch_size, 1)), so a small seed would
get MORE epochs for the same epoch count. To compare seed sizes fairly we
fix the TOTAL number of gradient steps and compute
epochs = ceil(total_steps / steps_per_epoch) for each seed size.
"""

import math
import time

import numpy as np
import pandas as pd

from src.utils import PROJECT_ROOT, get_column_types, load_config


def make_metadata(df, numerical_cols, categorical_cols, target):
    """
    Build SDV metadata and EXPLICITLY set column types, so SDV does not have
    to guess (a wrong guess would silently distort the synthetic data).
    The target is treated as categorical because it is a class label.
    row_id is excluded — generators must never see it.
    """
    from sdv.metadata import Metadata

    # SDV 2.x: detection takes a dict of {table_name: DataFrame}
    metadata = Metadata.detect_from_dataframes(
        {"data": df.drop(columns=["row_id"])})
    table_metadata = metadata.tables["data"]
    for column in numerical_cols:
        table_metadata.update_column(column, sdtype="numerical")
    for column in categorical_cols + [target]:
        table_metadata.update_column(column, sdtype="categorical")
    return metadata


def train_ctgan(seed_df, config, random_state):
    """
    Train CTGAN with a FAIR, size-independent training budget.

    Prints rows, steps-per-epoch and epochs so the budget is transparent.
    Returns (synthesizer, training_seconds, losses_or_None).
    """
    from sdv.single_table import CTGANSynthesizer

    generator_config = config["generators"]["ctgan"]
    total_steps = generator_config["total_training_steps"]
    batch_size = generator_config["batch_size"]

    # CTGAN counts steps per epoch as max(len(data) // batch_size, 1).
    # We verified this in the installed ctgan source (CTGAN.fit).
    steps_per_epoch = max(len(seed_df) // batch_size, 1)
    epochs = math.ceil(total_steps / steps_per_epoch)
    print(f"    CTGAN: rows={len(seed_df)} steps/epoch={steps_per_epoch} "
          f"epochs={epochs} (total {total_steps} steps)")

    synthesizer = CTGANSynthesizer(
        make_metadata(seed_df, *get_column_types_for_metadata(seed_df)),
        batch_size=batch_size,
        epochs=epochs,
        enable_gpu=config["generators"]["ctgan"]["use_gpu"],
        verbose=False,
    )
    start_time = time.time()
    # SDV 2.x fits on a dict {table_name: DataFrame}; our table is "data"
    synthesizer.fit({"data": seed_df.drop(columns=["row_id"])})
    training_seconds = time.time() - start_time

    # loss values, if the underlying CTGAN exposed them
    losses = None
    loss_values = getattr(synthesizer, "loss_values", None)
    if loss_values is not None:
        losses = loss_values
        losses_dir = PROJECT_ROOT / "results" / "ctgan_losses"
        losses_dir.mkdir(parents=True, exist_ok=True)
        losses.to_csv(losses_dir /
                      f"ctgan_losses_{int(time.time())}.csv", index=False)
    return synthesizer, training_seconds, losses


def get_column_types_for_metadata(seed_df):
    """Numerical, categorical and target column lists for metadata."""
    config = load_config()
    target = config["dataset"]["target"]
    numerical_cols, categorical_cols = get_column_types(seed_df, target)
    return numerical_cols, categorical_cols, target


def train_gaussian_copula(seed_df, config, random_state):
    """Train a Gaussian Copula synthesizer. Returns (synthesizer, seconds)."""
    from sdv.single_table import GaussianCopulaSynthesizer

    synthesizer = GaussianCopulaSynthesizer(
        make_metadata(seed_df, *get_column_types_for_metadata(seed_df)))
    start_time = time.time()
    synthesizer.fit({"data": seed_df.drop(columns=["row_id"])})
    return synthesizer, time.time() - start_time


def generate(synthesizer, n_rows):
    """Sample n_rows from a fitted synthesizer. Returns (DataFrame, seconds)."""
    start_time = time.time()
    sampled = synthesizer.sample(n_rows)
    # SDV 2.x returns a dict {table_name: DataFrame}; take our single table
    synthetic_df = sampled["data"] if isinstance(sampled, dict) else sampled
    return synthetic_df, time.time() - start_time


def run_smote_nc(seed_df, numerical_cols, categorical_cols, target,
                 k_neighbors, random_state):
    """
    SMOTE-NC oversampling. Returns (resampled_df, n_new_rows).

    k_neighbors must be SMALLER than the minority-class count; for a tiny
    seed with e.g. 24 minority rows, k=5 still works, but we cap it anyway:
    k = min(configured k, minority_count - 1).
    """
    from imblearn.over_sampling import SMOTENC

    minority_count = seed_df[target].value_counts().min()
    k_neighbors = min(k_neighbors, minority_count - 1)
    if k_neighbors < 1:
        raise ValueError("Minority class too small for SMOTE-NC.")

    # SMOTENC needs categorical columns referenced by POSITION
    categorical_positions = [
        i for i, c in enumerate(numerical_cols + categorical_cols)
        if c in categorical_cols]
    feature_columns = numerical_cols + categorical_cols
    X = seed_df[feature_columns]
    y = seed_df[target]

    smote = SMOTENC(
        categorical_features=categorical_positions,
        k_neighbors=k_neighbors,
        random_state=random_state,
    )
    X_resampled, y_resampled = smote.fit_resample(X, y)
    resampled = X_resampled.copy()
    resampled[target] = y_resampled
    n_new_rows = len(resampled) - len(seed_df)
    return resampled, n_new_rows


def validate_synthetic(synth_df, real_df, numerical_cols, categorical_cols):
    """
    Basic sanity checks on synthetic data. Returns a list of warnings:
      * columns must match the real columns,
      * categorical values must come from the real categories,
      * numbers must be inside a plausible range (e.g. no negative age).
    """
    warnings = []
    real_columns = set(real_df.columns) - {"row_id"}
    if set(synth_df.columns) != real_columns:
        warnings.append("Column mismatch between synthetic and real data.")

    for column in categorical_cols:
        if column in synth_df.columns:
            if synth_df[column].isna().any():
                warnings.append(
                    f"Column '{column}' has missing values in the synthetic "
                    "output (the generator failed to sample a category).")
            extra = set(synth_df[column].dropna().unique()) - \
                set(real_df[column].dropna().unique())
            if extra:
                warnings.append(
                    f"Column '{column}' has invented categories: "
                    f"{sorted(extra)[:5]}")

    for column in numerical_cols:
        if column in synth_df.columns:
            real_min, real_max = real_df[column].min(), real_df[column].max()
            if synth_df[column].min() < real_min - 3 * real_df[column].std() \
                    or synth_df[column].max() > real_max + 3 * real_df[column].std():
                warnings.append(
                    f"Column '{column}' has values outside a plausible range "
                    f"[{real_min}, {real_max}].")
    return warnings


def main():
    """
    Demo: train all three generators on seed 500 (quick-mode CTGAN budget),
    generate rows, and show the results + validation warnings.
    """
    config = load_config()
    target = config["dataset"]["target"]
    seed_df = pd.read_csv(PROJECT_ROOT / "data" / "seeds" / "rs0" / "seed_500.csv")
    numerical_cols, categorical_cols = get_column_types(seed_df, target)

    quick_config = dict(config)
    quick_config["generators"] = dict(config["generators"])
    quick_config["generators"]["ctgan"] = {
        "total_training_steps": config["quick_mode"]["ctgan_total_training_steps"],
        "batch_size": config["generators"]["ctgan"]["batch_size"],
        "use_gpu": config["generators"]["ctgan"]["use_gpu"],
    }

    print("Training CTGAN (quick budget)...")
    ctgan_synth, ctgan_seconds, _ = train_ctgan(seed_df, quick_config, 0)
    ctgan_rows, ctgan_sample_seconds = generate(
        ctgan_synth, config["quick_mode"]["synthetic_rows"])
    print(f"CTGAN done in {ctgan_seconds:.1f}s, "
          f"sampled {len(ctgan_rows)} rows in {ctgan_sample_seconds:.1f}s")

    print("Training Gaussian Copula...")
    gc_synth, gc_seconds = train_gaussian_copula(seed_df, config, 0)
    gc_rows, _ = generate(gc_synth, config["quick_mode"]["synthetic_rows"])
    print(f"Gaussian Copula done in {gc_seconds:.1f}s")

    # save the demo outputs so the fidelity/privacy demos can reuse them
    synth_dir = PROJECT_ROOT / "data" / "synthetic" / "rs0"
    synth_dir.mkdir(parents=True, exist_ok=True)
    ctgan_rows.to_csv(synth_dir / "ctgan_500.csv", index=False)
    gc_rows.to_csv(synth_dir / "gc_500.csv", index=False)

    print("Running SMOTE-NC...")
    smote_rows, n_new = run_smote_nc(
        seed_df, numerical_cols, categorical_cols, target,
        config["generators"]["smote"]["k_neighbors"], 0)
    print(f"SMOTE-NC added {n_new} minority rows")

    for name, rows in [("CTGAN", ctgan_rows), ("GaussianCopula", gc_rows),
                       ("SMOTE-NC (resampled seed)", smote_rows)]:
        print(f"\n--- {name}: first 5 rows ---")
        print(rows.head().to_string())
        positive_pct = 100 * rows[target].mean()
        print(f"positive class: {positive_pct:.1f}%")

    print("\nValidation warnings (CTGAN):")
    for warning in validate_synthetic(ctgan_rows, seed_df,
                                      numerical_cols, categorical_cols):
        print(f"  - {warning}")


if __name__ == "__main__":
    main()
