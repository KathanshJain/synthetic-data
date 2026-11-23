"""
prepare_data.py — Step 1 of the pipeline: download, clean and split the data.

Why this file exists: everything downstream depends on a clean dataset and a
honestly held-out test set. This file
  1. downloads the Adult Census Income dataset from OpenML (once),
  2. cleans it (duplicates, missing values, useless columns),
  3. splits it 80/20 into a development pool and a LOCKED test set,
  4. writes a SHA-256 manifest so any later change to the test file is detected.

The locked test set is the "vault": it is only opened for the final evaluation.
"""

import json
from datetime import datetime

import pandas as pd
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split

from src.utils import (
    MANIFEST_PATH,
    PROJECT_ROOT,
    compute_file_sha256,
    load_config,
)


def download_raw_data(config):
    """
    Download the Adult dataset from OpenML and save it as CSV.
    If data/raw/adult.csv already exists, skip the download (saves time and
    makes the project work offline after the first run).
    """
    raw_path = PROJECT_ROOT / "data" / "raw" / "adult.csv"
    if raw_path.exists():
        print(f"Raw data already exists at {raw_path} — skipping download.")
        return pd.read_csv(raw_path)

    dataset_config = config["dataset"]
    print(f"Downloading OpenML dataset '{dataset_config['name']}' version "
          f"{dataset_config['version']}...")
    data = fetch_openml(
        name=dataset_config["name"],
        version=dataset_config["version"],
        as_frame=True,
        parser="auto",
    )
    df = data.frame
    raw_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(raw_path, index=False)
    print(f"Saved raw data to {raw_path} ({len(df)} rows).")
    return df


def clean_data(df, config):
    """
    Clean the raw Adult data. The order of operations matters:

    1. Strip spaces and turn '?' into NaN  (Adult uses '?' for unknown).
    2. Drop EXACT duplicate rows BEFORE dropping columns — if we dropped
       columns first, two people who differ only in fnlwgt would suddenly
       look identical and we would wrongly delete a real record.
    3. THEN drop the configured columns (fnlwgt, education).
    4. Fill missing categorical values with the category "Missing" — Adult's
       missingness is informative (unemployed people have no occupation),
       and keeping the rows preserves the class ratio.
    5. Create the numeric target column and a row_id for traceability.
    """
    df = df.copy()

    # 1. tidy text columns: remove stray spaces, treat '?' as missing
    for column in df.columns:
        if df[column].dtype == object:
            df[column] = df[column].astype(str).str.strip()
            df[column] = df[column].replace("?", pd.NA)

    # 2. drop exact duplicate rows (BEFORE dropping columns)
    rows_before = len(df)
    df = df.drop_duplicates()
    duplicates_removed = rows_before - len(df)

    # 3. drop the configured columns
    df = df.drop(columns=config["dataset"]["drop_columns"])

    # 4. missing categorical values become their own category
    for column in df.columns:
        if df[column].dtype == object:
            df[column] = df[column].fillna("Missing")

    # 5. numeric target + row_id
    target = config["dataset"]["target"]
    positive_label = config["dataset"]["positive_label"]
    # the raw target column in OpenML is called 'class'
    df[target] = (df["class"] == positive_label).astype(int)
    df = df.drop(columns=["class"])
    df.insert(0, "row_id", range(len(df)))

    print(f"Cleaning done: removed {duplicates_removed} duplicate rows, "
          f"{len(df)} rows remain.")
    return df


def stratified_split(df, config):
    """
    Split into 80% development pool and 20% locked test set.
    'stratified' means both parts keep the same income class ratio, so the
    test set is a fair picture of the population.
    """
    target = config["dataset"]["target"]
    dev_pool, test_set = train_test_split(
        df,
        test_size=config["split"]["test_size"],
        random_state=config["split"]["random_state"],
        stratify=df[target],
    )
    return dev_pool, test_set


def write_manifest(test_set):
    """
    Save the SHA-256 hash of the locked test file into manifest.json.
    From now on, any code that loads the test set verifies this hash first.
    """
    LOCKED_DIR = PROJECT_ROOT / "data" / "locked"
    LOCKED_DIR.mkdir(parents=True, exist_ok=True)
    test_path = LOCKED_DIR / "test_locked.csv"
    test_set.to_csv(test_path, index=False)

    manifest = {
        "sha256": compute_file_sha256(test_path),
        "n_rows": int(len(test_set)),
        "class_counts": {str(k): int(v) for k, v in
                         test_set["income"].value_counts().items()},
        "created": datetime.now().isoformat(),
    }
    with open(MANIFEST_PATH, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    print(f"Manifest written to {MANIFEST_PATH} "
          f"(sha256={manifest['sha256'][:16]}...)")


def main():
    config = load_config()
    raw_df = download_raw_data(config)
    clean_df = clean_data(raw_df, config)
    dev_pool, test_set = stratified_split(clean_df, config)

    processed_dir = PROJECT_ROOT / "data" / "processed"
    processed_dir.mkdir(parents=True, exist_ok=True)
    dev_pool.to_csv(processed_dir / "dev_pool.csv", index=False)
    write_manifest(test_set)

    # summary table
    target = config["dataset"]["target"]
    print("\n=== Data preparation summary ===")
    print(f"Total cleaned rows:   {len(clean_df)}")
    print(f"Development pool:     {len(dev_pool)} rows "
          f"({dev_pool[target].mean():.1%} positive)")
    print(f"Locked test set:      {len(test_set)} rows "
          f"({test_set[target].mean():.1%} positive)")


if __name__ == "__main__":
    main()
