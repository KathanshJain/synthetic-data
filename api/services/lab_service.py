"""
lab_service: upload profiling and the interactive train-a-generator job.

The lab is the only part of the API that accepts user data and trains a
generator on request. Rules:
- uploads are stored OUTSIDE the project data folders, in a temp directory,
  with a TTL cleanup so nothing lingers;
- the target column must exist and be binary, the seed size must leave
  enough rows, and CTGAN steps are clamped to the configured maximum;
- the heavy work reuses src/ (generators.py, fidelity.py, privacy.py) —
  nothing is re-implemented here;
- progress is stage-based (validate → seed → train → sample → fidelity →
  privacy → done); within the CTGAN train stage it advances on a timer
  because SDV exposes no per-step callback, and the final log reports the
  real step count from the loss values.
"""

import copy
import io
import json
import shutil
import sys
import tempfile
import threading
import time
import uuid
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd

from src.eda import profile_dataset
from src.fidelity import (correlation_difference, ks_per_column,
                          tvd_per_column)
from src.generators import (generate, repair_synthetic, run_smote_nc,
                            train_ctgan, train_gaussian_copula)
from src.privacy import compute_dcr
from src.utils import get_column_types, load_config

from ..settings import API_SETTINGS

LAB_ROOT = Path(tempfile.gettempdir()) / "forged_data_lab"
UPLOADS = LAB_ROOT / "uploads"
JOBS_DIR = LAB_ROOT / "jobs"


def profile_for_contract(df: pd.DataFrame, target: str | None = None) -> dict:
    """
    Reshape profile_dataset() output into the contract's Upload.profile:
    columns must be a list of NAMES (the target picker needs them), plus
    types/missing/class_balance maps the UI renders directly.
    """
    raw = profile_dataset(df, target)
    numerical, categorical = get_column_types(df, target)
    types = {c: "numerical" for c in numerical}
    types.update({c: "categorical" for c in categorical})
    missing_raw = raw.get("missing_percent", {})
    # missing_percent only lists columns WITH gaps; the UI averages over all
    # columns, so provide an entry for every column.
    missing = {c: float(missing_raw.get(c, 0.0)) for c in df.columns}
    balance = raw.get("class_balance", {}) or {}
    return {
        "rows": int(raw["rows"]),
        "columns": [str(c) for c in df.columns],
        "types": types,
        "missing": missing,
        "warnings": [str(w) for w in raw.get("warnings", [])],
        "class_balance": {str(k): float(v) for k, v in balance.items()},
        "duplicate_rows": int(raw.get("duplicate_rows", 0)),
        "id_like_columns": list(raw.get("id_like_columns", [])),
    }


def save_upload(content: bytes, filename: str) -> tuple[str, pd.DataFrame]:
    """Parse + persist an uploaded CSV; returns (dataset_id, dataframe)."""
    max_mb = API_SETTINGS["max_upload_mb"]
    max_rows = API_SETTINGS["max_upload_rows"]
    if len(content) > max_mb * 1024 * 1024:
        raise ValueError(f"File is larger than the {max_mb} MB limit.")
    try:
        df = pd.read_csv(io.BytesIO(content))
    except Exception as exc:
        raise ValueError(f"Could not parse the file as CSV: {exc}") from exc
    if df.empty:
        raise ValueError("The uploaded file has no rows.")
    if len(df) > max_rows:
        raise ValueError(f"File has {len(df)} rows; the limit is {max_rows}.")
    if len(df) < 50:
        raise ValueError("Too few rows to learn anything meaningful "
                         "(minimum 50).")
    dataset_id = uuid.uuid4().hex[:12]
    folder = UPLOADS / dataset_id
    folder.mkdir(parents=True, exist_ok=True)
    df.to_csv(folder / "upload.csv", index=False)
    (folder / "meta.json").write_text(json.dumps({
        "filename": filename, "rows": len(df), "uploaded": time.time(),
    }), encoding="utf-8")
    return dataset_id, df


def load_upload(dataset_id: str) -> pd.DataFrame:
    """Load a previously uploaded CSV by id (used by /lab/run)."""
    path = UPLOADS / dataset_id / "upload.csv"
    if not path.exists():
        raise FileNotFoundError(dataset_id)
    return pd.read_csv(path)


def cleanup_expired() -> None:
    """Delete upload/job directories older than the TTL."""
    ttl = API_SETTINGS["job_ttl_minutes"] * 60
    cutoff = time.time() - ttl
    for root in (UPLOADS, JOBS_DIR):
        if not root.exists():
            continue
        for child in root.iterdir():
            try:
                if child.stat().st_mtime < cutoff:
                    shutil.rmtree(child, ignore_errors=True)
            except OSError:
                continue


def run_lab_job(job, params: dict) -> None:
    """
    The full lab pipeline for one uploaded dataset. `job` is the Job object
    from api.jobs (progress/log updates); params carries dataset_id, target,
    seed_size, generator, synthetic_rows, ctgan_steps.
    """
    from sklearn.model_selection import train_test_split

    target = params["target"]
    generator_name = params["generator"]
    seed_size = params["seed_size"]
    synthetic_rows = params["synthetic_rows"]
    ctgan_steps = min(params["ctgan_steps"], API_SETTINGS["ctgan_steps_max"])

    # ---- validate ---------------------------------------------------------
    job.update(stage="validate", progress=2, log="Loading uploaded dataset.")
    df = load_upload(params["dataset_id"])
    if target not in df.columns:
        raise ValueError(f"Target column '{target}' is not in the dataset.")
    if df[target].nunique() != 2:
        raise ValueError("The lab expects a binary target column "
                         "(exactly two values).")
    # encode the target as 0/1 the same way the pipeline does
    df = df.copy()
    df[target] = (df[target] == df[target].value_counts().index[-1]).astype(int)
    if df[target].value_counts().min() < 10:
        raise ValueError("The minority class has fewer than 10 rows; "
                         "no generator can learn from that.")
    if seed_size > len(df):
        raise ValueError(f"Seed size {seed_size} exceeds the {len(df)} "
                         "uploaded rows.")
    job.update(progress=10, log=f"Validated: {len(df)} rows, "
                                f"{df.shape[1]} columns, target='{target}'.")

    # ---- seed -------------------------------------------------------------
    job.update(stage="seed", progress=15,
               log=f"Sampling a {seed_size}-row stratified seed.")
    seed_df, _ = train_test_split(df, train_size=seed_size,
                                  random_state=0, stratify=df[target])
    # src/ generators expect a row_id label column (the experiment's seeds
    # always have one); give the lab seed the same shape.
    seed_df = seed_df.copy()
    seed_df["row_id"] = range(len(seed_df))
    numerical_cols, categorical_cols = get_column_types(seed_df, target)

    # the lab reuses the SAME generator settings block as the experiment,
    # with the user's target and CTGAN step budget swapped in — the src/
    # generators read the target name from config["dataset"]["target"]
    config = copy.deepcopy(load_config())
    config["dataset"]["target"] = target
    config["generators"]["ctgan"]["total_training_steps"] = ctgan_steps

    # ---- train + sample ---------------------------------------------------
    job.update(stage="train", progress=20,
               log=f"Training {generator_name}"
                   + (f" (budget {ctgan_steps} steps)." if generator_name == "CTGAN"
                      else "."))
    if generator_name == "CTGAN":
        synthesizer, seconds, losses = train_ctgan(seed_df, config, 0)
        job.update(progress=70,
                   log=f"CTGAN trained in {seconds:.0f}s "
                       f"({len(losses) if losses is not None else '?'} steps).")
        synth_df, _ = generate(synthesizer, synthetic_rows)
    elif generator_name == "GaussianCopula":
        synthesizer, seconds = train_gaussian_copula(seed_df, config, 0)
        job.update(progress=70, log=f"Gaussian Copula trained in {seconds:.1f}s.")
        synth_df, _ = generate(synthesizer, synthetic_rows)
    else:  # SMOTE-NC balances to the majority class; row count is its output
        k_neighbors = config["models"].get("smote_k_neighbors", 5)
        synth_df, n_new = run_smote_nc(seed_df, numerical_cols, categorical_cols,
                                       target, k_neighbors, 0)
        job.update(progress=70, log=f"SMOTE-NC created {n_new} new minority rows.")
    synth_df = repair_synthetic(synth_df, seed_df, numerical_cols,
                                categorical_cols)
    job.update(stage="sample", progress=75,
               log=f"Generated {len(synth_df)} synthetic rows.")

    # ---- fidelity ---------------------------------------------------------
    job.update(stage="fidelity", progress=80, log="Measuring fidelity.")
    ks_table = ks_per_column(seed_df, synth_df, numerical_cols)
    tvd_table = tvd_per_column(seed_df, synth_df, categorical_cols)
    corr_diff = correlation_difference(seed_df, synth_df, numerical_cols)
    _, coverage_overall = _category_coverage(seed_df, synth_df,
                                             categorical_cols)
    fidelity = {
        "ks_mean": round(float(ks_table["ks_statistic"].mean()), 4),
        "tvd_mean": round(float(tvd_table["tvd"].mean()), 4),
        "corr_diff": round(float(corr_diff), 4),
        "coverage_mean": round(float(coverage_overall), 4),
    }
    per_column = _per_column_payload(seed_df, synth_df)

    # ---- privacy ----------------------------------------------------------
    job.update(stage="privacy", progress=90,
               log="Measuring privacy (DCR vs a holdout split).")
    seed_fit, holdout = train_test_split(seed_df, test_size=0.3,
                                         random_state=0,
                                         stratify=seed_df[target])
    privacy = compute_dcr(seed_fit, synth_df, holdout, numerical_cols,
                          categorical_cols)
    privacy_payload = {
        "dcr_ratio": float(privacy["dcr_ratio"]),
        "synth_median_dcr": float(privacy["synth_median_dcr"]),
        "holdout_median_dcr": float(privacy["holdout_median_dcr"]),
        "exact_rate_synth": float(privacy["synth_exact_match_rate"]),
        "exact_rate_holdout": float(privacy["holdout_exact_match_rate"]),
    }

    # ---- done -------------------------------------------------------------
    class_balance = {
        "real": {str(k): int(v) for k, v in
                 seed_df[target].value_counts().items()},
        "synth": {str(k): int(v) for k, v in
                  synth_df[target].value_counts().items()},
    }
    result = {"class_balance": class_balance, "fidelity": fidelity,
              "privacy": privacy_payload, "per_column": per_column}
    job.update(stage="done", progress=100, status="completed",
               result=result, log="Completed.")

    # persist the synthetic CSV for download
    job_dir = JOBS_DIR / job.id
    job_dir.mkdir(parents=True, exist_ok=True)
    synth_df.to_csv(job_dir / "synthetic.csv", index=False)


def _category_coverage(real, synth, categorical_cols):
    """src.fidelity.category_coverage returns (per_column_dict, overall)."""
    from src.fidelity import category_coverage
    return category_coverage(real, synth, categorical_cols)


def _per_column_payload(seed_df, synth_df):
    """Per-column charts: histograms for numerical, bars for categorical."""
    payload = {}
    numerical_cols = seed_df.select_dtypes("number").columns
    for col in numerical_cols:
        if col not in synth_df.columns:
            continue
        low = float(min(seed_df[col].min(), synth_df[col].min()))
        high = float(max(seed_df[col].max(), synth_df[col].max()))
        if high <= low:
            high = low + 1e-9
        edges = list(pd.cut(pd.concat([seed_df[col], synth_df[col]]),
                            bins=20, retbins=True)[1])
        real_counts = pd.cut(seed_df[col], bins=edges).value_counts().sort_index()
        synth_counts = pd.cut(synth_df[col], bins=edges).value_counts().sort_index()
        payload[col] = {
            "bins": [round(float(e), 4) for e in edges],
            "real": [int(v) for v in real_counts],
            "synthetic": [int(v) for v in synth_counts],
        }
    categorical_cols = [c for c in seed_df.columns
                        if c not in numerical_cols]
    for col in categorical_cols:
        if col not in synth_df.columns:
            continue
        categories = sorted(set(seed_df[col].astype(str)) |
                            set(synth_df[col].astype(str)))
        real_counts = seed_df[col].astype(str).value_counts().reindex(
            categories, fill_value=0)
        synth_counts = synth_df[col].astype(str).value_counts().reindex(
            categories, fill_value=0)
        payload[col] = {
            "categories": categories,
            "real": [int(v) for v in real_counts],
            "synthetic": [int(v) for v in synth_counts],
        }
    return payload
