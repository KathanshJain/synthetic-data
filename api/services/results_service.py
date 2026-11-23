"""
results_service: serves every read-only number from the OLD (v1) results.

Rules enforced here:
- numbers come ONLY from results/*.csv, reports/report_facts.json and
  manifest.json — never invented, never estimated;
- the locked test set is NEVER loaded; its counts come from manifest.json
  and its hash status from verify_test_hash() (verified true/false only);
- heavy computation is never done ad hoc — artefacts are exported by
  src/analyze.py and served from disk.
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd

from src.analyze import apply_criteria, criteria_rules
from src.utils import (PROJECT_ROOT as ROOT, load_config, load_criteria,
                       verify_test_hash)

from .cache import get_frame, get_json

RESULTS = ROOT / "results"
REPORTS = ROOT / "reports"
DATA = ROOT / "data"

METRICS = ["pr_auc", "roc_auc", "f1", "precision", "recall", "accuracy"]
GENERATORS = {"CTGAN": "ctgan", "GaussianCopula": "gc"}


def _facts():
    return get_json(REPORTS / "report_facts.json") or {}


def _manifest():
    return get_json(DATA / "locked" / "manifest.json") or {}


def _config():
    return load_config()


def _criteria():
    return load_criteria()


def get_overview() -> dict:
    """Overview card: counts from report_facts + manifest, hash check live."""
    facts = _facts()
    ds = facts.get("dataset", {})
    manifest = _manifest()
    # The only allowed interaction with the locked set: verify_test_hash()
    # returns None on success and RAISES on tampering — so verified=True
    # means "the hash matched", never "I read the file".
    try:
        verify_test_hash()
        verified = True
    except Exception:
        verified = False
    return {
        "rows_raw": ds.get("rows_raw", 0),
        "duplicates_removed": ds.get("duplicates_removed", 0),
        "rows_clean": ds.get("rows_clean", 0),
        "dev_rows": ds.get("dev_rows", 0),
        "test_rows": ds.get("test_rows", 0),
        "positive_rate_dev": ds.get("positive_rate_dev", 0.0),
        "positive_rate_test": ds.get("positive_rate_test", 0.0),
        "n_features": ds.get("n_features", 0),
        "n_numerical": ds.get("n_numerical", 0),
        "n_categorical": ds.get("n_categorical", 0),
        "features": ds.get("features", []),
        "locked": {
            "sha256": manifest.get("sha256", ""),
            "verified": verified,
            "created": manifest.get("created", ""),
            "n_rows": manifest.get("n_rows", 0),
        },
    }


def get_distribution(column: str) -> dict:
    """Real-data-only distribution of one column from the dev pool."""
    df = _dev_pool()
    if column not in df.columns:
        raise KeyError(column)
    if pd.api.types.is_numeric_dtype(df[column]):
        counts, edges = pd.cut(df[column], bins=30, retbins=True,
                               duplicates="drop")
        grouped = df[column].groupby(counts, observed=True).size()
        return {"type": "histogram",
                "bins": [round(float(e), 4) for e in edges],
                "counts": [int(v) for v in grouped.tolist()]}
    vc = df[column].astype(str).value_counts()
    return {"type": "categories",
            "categories": vc.index.tolist(),
            "counts": [int(v) for v in vc.tolist()]}


def get_seeds() -> dict:
    """Seed sizes, random seeds, nested-split check and per-size class counts."""
    config = _config()
    from src.seeds import build_nested_seeds, check_nested
    from src.utils import get_column_types

    dev = _dev_pool()
    target = config["dataset"]["target"]
    sizes = config["seeds"]["seed_sizes"]
    random_seeds = config["seeds"]["random_seeds"]
    seeds = build_nested_seeds(dev, target, sizes, random_seeds[0])
    nested_ok = check_nested(seeds)
    class_counts = {}
    for size in sizes:
        seed_df = seeds[size]
        counts = seed_df[target].value_counts()
        class_counts[str(size)] = {
            "majority": int(counts.get(0, 0)),
            "minority": int(counts.get(1, 0)),
        }
    return {"sizes": sizes, "random_seeds": random_seeds,
            "nested_ok": bool(nested_ok), "class_counts": class_counts}


def get_summary(model: str) -> list:
    """Per-arm summary rows for one model, from results/summary.csv."""
    frame = get_frame(RESULTS / "summary.csv")
    if frame is None:
        raise FileNotFoundError("summary.csv")
    metrics = get_frame(RESULTS / "metrics.csv")
    rows = []
    for _, row in frame[frame["model"] == model].iterrows():
        mask = ((metrics["model"] == model) &
                (metrics["arm"] == row["arm"]) &
                (metrics["seed_size"] == row["seed_size"]))
        rows.append({
            "arm": row["arm"],
            "seed_size": int(row["seed_size"]),
            "metric_means": {m: float(row[f"{m}_mean"]) for m in METRICS},
            "metric_stds": {m: float(row[f"{m}_std"]) for m in METRICS},
            "n_runs": int(mask.sum()),
        })
    return rows


def get_reference() -> dict:
    """TRTR-full reference metrics per model + random-guess PR-AUC."""
    frame = get_frame(RESULTS / "summary.csv")
    if frame is None:
        raise FileNotFoundError("summary.csv")
    trtr = frame[frame["arm"] == "TRTR-full"]
    trtr_full = {}
    for _, row in trtr.iterrows():
        trtr_full[row["model"]] = {m: float(row[f"{m}_mean"]) for m in METRICS}
    facts = _facts()
    return {
        "trtr_full": trtr_full,
        "random_guess_pr_auc": facts.get("dataset", {}).get("positive_rate_dev", 0.239),
    }


def get_fidelity() -> dict:
    """Fidelity summary (with chi2 from the export) + per-column long table."""
    fid = get_frame(RESULTS / "fidelity.csv")
    chi2 = get_frame(RESULTS / "fidelity_chi2.csv")
    per_col = get_frame(RESULTS / "fidelity_per_column.csv")
    if fid is None:
        raise FileNotFoundError("fidelity.csv")

    overall = []
    for (gen, size), group in fid.groupby(["generator", "seed_size"]):
        row = group.mean(numeric_only=True)
        stds = group.std(numeric_only=True).fillna(0.0)
        chi2_mean = 0.0
        if chi2 is not None:
            sel = chi2[(chi2["generator"] == gen) &
                       (chi2["seed_size"] == size)]
            if len(sel):
                chi2_mean = float(sel["chi2_mean"].mean())
        overall.append({
            "generator": gen,
            "seed_size": int(size),
            "ks_mean": float(row["ks_mean"]),
            "tvd_mean": float(row["tvd_mean"]),
            "chi2_mean": chi2_mean,
            "corr_diff": float(row["corr_diff"]),
            "coverage": float(row["coverage_mean"]),
            "stds": {k: float(stds.get(k, 0.0))
                     for k in ["ks_mean", "tvd_mean", "corr_diff", "coverage_mean"]},
        })

    per_column = []
    if per_col is not None:
        grouped = (per_col.groupby(["generator", "seed_size", "column"])
                   [["ks_statistic", "tvd", "coverage_pct"]].mean().reset_index())
        for _, row in grouped.iterrows():
            for metric, value in [("ks_statistic", row["ks_statistic"]),
                                  ("tvd", row["tvd"]),
                                  ("coverage_pct", row["coverage_pct"])]:
                per_column.append({
                    "generator": row["generator"],
                    "seed_size": int(row["seed_size"]),
                    "column": row["column"],
                    "metric": metric,
                    "value": float(value),
                })
    return {"overall": overall, "per_column": per_column}


def get_privacy() -> list:
    """DCR summary averaged over random seeds, renamed to contract fields."""
    frame = get_frame(RESULTS / "privacy.csv")
    if frame is None:
        raise FileNotFoundError("privacy.csv")
    rows = []
    for (gen, size), group in frame.groupby(["generator", "seed_size"]):
        mean = group.mean(numeric_only=True)
        rows.append({
            "generator": gen,
            "seed_size": int(size),
            "dcr_median_synth": float(mean["synth_median_dcr"]),
            "dcr_median_holdout": float(mean["holdout_median_dcr"]),
            "dcr_ratio": float(mean["dcr_ratio"]),
            "exact_rate_synth": float(mean["synth_exact_match_rate"]),
            "exact_rate_holdout": float(mean["holdout_exact_match_rate"]),
            "p05_synth": float(mean["synth_p05_dcr"]),
            "p05_holdout": float(mean["holdout_p05_dcr"]),
        })
    return rows


def get_dcr_histogram(generator: str, seed_size: int, random_seed: int) -> dict:
    """DCR histogram for one run, from the exported bins."""
    frame = get_frame(RESULTS / "dcr_histograms.csv")
    if frame is None:
        raise FileNotFoundError("dcr_histograms.csv")
    sel = frame[(frame["generator"] == generator) &
                (frame["seed_size"] == seed_size) &
                (frame["random_seed"] == random_seed)]
    if not len(sel):
        raise KeyError(f"{generator}/{seed_size}/rs{random_seed}")
    sel = sel.sort_values("bin_left")
    # bins are x-axis LABELS for the frontend chart: one per bucket (the
    # midpoint), so len(bins) == len(counts) exactly.
    bins = [round((float(l) + float(r)) / 2, 4)
            for l, r in zip(sel["bin_left"], sel["bin_right"])]
    return {"bins": bins,
            "synthetic": [int(v) for v in sel["synthetic_count"]],
            "holdout": [int(v) for v in sel["holdout_count"]]}


def get_runtime() -> list:
    """Generator training/sampling seconds grouped by generator and size."""
    frame = get_frame(RESULTS / "runtime.csv")
    if frame is None:
        raise FileNotFoundError("runtime.csv")
    rows = []
    for (gen, size), group in frame.groupby(["generator", "seed_size"]):
        rows.append({
            "generator": gen,
            "seed_size": int(size),
            "seconds_mean": float(group["fit_seconds"].mean()),
            "seconds_std": float(group["fit_seconds"].std() or 0.0),
        })
    return rows


def get_loss_curve(seed_size: int, random_seed: int) -> dict:
    """CTGAN loss curve from the exported CSV (only exists after a refresh)."""
    path = RESULTS / "ctgan_losses" / f"ctgan_losses_rs{random_seed}_n{seed_size}.csv"
    if not path.exists():
        raise FileNotFoundError(path.name)
    frame = get_frame(path)
    if frame is None:
        raise FileNotFoundError(path.name)
    step_col = "Epoch" if "Epoch" in frame.columns else "step"
    return {
        "step": [int(v) for v in frame[step_col]],
        "generator_loss": [float(v) for v in frame["Generator Loss"]],
        "discriminator_loss": [float(v) for v in frame["Discriminator Loss"]],
    }


def get_criteria() -> dict:
    """Frozen criteria rules + the pass/fail table computed by apply_criteria."""
    config = _config()
    rules_src = criteria_rules(config)
    rules = [{"id": r["id"], "label": r["label"],
              "threshold": float(r["threshold"]), "direction": r["direction"]}
             for r in rules_src]

    fid = get_frame(RESULTS / "fidelity.csv")
    priv = get_frame(RESULTS / "privacy.csv")
    summary = get_frame(RESULTS / "summary.csv")
    # apply_criteria expects frames averaged over random seeds, with
    # <column>_mean names (exactly what the report used).
    fid_avg = (fid.groupby(["generator", "seed_size"], as_index=False)
               .mean(numeric_only=True)
               .rename(columns={"ks_mean": "ks_mean_mean",
                                "tvd_mean": "tvd_mean_mean",
                                "corr_diff": "corr_diff_mean",
                                "coverage_mean": "coverage_mean_mean"}))
    priv_avg = (priv.groupby(["generator", "seed_size"], as_index=False)
                .mean(numeric_only=True)
                .rename(columns={"dcr_ratio": "dcr_ratio_mean",
                                 "synth_exact_match_rate":
                                     "synth_exact_match_rate_mean",
                                 "holdout_exact_match_rate":
                                     "holdout_exact_match_rate_mean"}))
    table, min_useful_seed = apply_criteria(summary, fid_avg, priv_avg)

    rule_ids = [r["id"] for r in rules]
    # apply_criteria appends its PASS/FAIL columns in rule order — map by position.
    pass_cols = [c for c in table.columns if c not in
                 {"generator", "seed_size", "utility_ratio", "ks_mean",
                  "tvd_mean", "dcr_ratio", "ALL_PASS"}]
    evaluations = []
    for _, row in table.iterrows():
        passed = {}
        for i, col in enumerate(pass_cols):
            passed[rule_ids[i] if i < len(rule_ids) else col] = (
                str(row[col]) == "PASS")
        evaluations.append({
            "generator": row["generator"],
            "seed_size": int(row["seed_size"]),
            "values": {
                "utility_ratio": float(row["utility_ratio"]),
                "ks_mean": float(row["ks_mean"]),
                "tvd_mean": float(row["tvd_mean"]),
                "dcr_ratio": float(row["dcr_ratio"]),
            },
            "passed": passed,
            "all_passed": bool(row["ALL_PASS"]),
        })
    explanation = (
        "Every seed size is judged against the frozen criteria.yaml decided "
        "BEFORE any synthetic data was generated. A size is 'useful' only if "
        "it passes every rule at once; the honest answer may be 'none'."
    )
    return {"rules": rules, "evaluations": evaluations,
            "min_useful_seed":
                int(min_useful_seed) if min_useful_seed is not None else None,
            "explanation": explanation}


def _dev_pool():
    """Load the dev pool (the only data the API may read directly)."""
    config = _config()
    cache_path = DATA / "processed" / "dev_pool.csv"
    frame = get_frame(cache_path)
    if frame is None:
        from src.prepare_data import load_and_clean
        frame = load_and_clean(ROOT / "data" / "raw" / "adult.csv",
                               config["dataset"]["target"])
    return frame
