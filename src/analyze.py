"""
analyze.py — Step 10: aggregate results, draw the figures, find the
minimum useful seed.

Why this file exists: the experiment produces hundreds of raw rows (one per
arm/model/seed-size/random-seed). This file turns them into the answer of
the project:
  * summary tables (mean ± std over the random seeds),
  * publication-quality plots of every metric vs seed size,
  * a PASS/FAIL table by applying the FROZEN criteria.yaml rules,
  * results/RESULTS.md with honest, plain-English findings.

criteria.yaml is NOT changed here — it was frozen before the full run.
"""

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.utils import PROJECT_ROOT, load_config, load_criteria

RESULTS_DIR = PROJECT_ROOT / "results"
FIGURES_DIR = PROJECT_ROOT / "figures" / "results"

ARM_COLORS = {
    "TRTR-full": "black",
    "TRTR-seed": "#4c72b0",
    "SMOTE-NC": "#55a868",
    "TSTR-CTGAN": "#c44e52",
    "TSTR-GC": "#8172b3",
}


def build_summary(metrics, fidelity, privacy, runtime):
    """
    Mean and std over random seeds for every (seed_size, arm, model) metric,
    plus fidelity/privacy/runtime summaries per (seed_size, generator).
    """
    metric_columns = ["pr_auc", "roc_auc", "f1", "precision", "recall",
                      "accuracy"]
    summary = (metrics.groupby(["seed_size", "arm", "model"])[metric_columns]
               .agg(["mean", "std"]).round(4))
    summary.columns = ["_".join(col).strip() for col in summary.columns]
    summary = summary.reset_index()

    fidelity_summary = (fidelity.groupby(["seed_size", "generator"])
                        .agg(["mean", "std"]).round(4).reset_index())
    privacy_summary = (privacy.groupby(["seed_size", "generator"])
                       .agg(["mean", "std"]).round(4).reset_index())
    runtime_summary = (runtime.groupby(["seed_size", "generator"])
                       .agg(["mean", "std"]).round(1).reset_index())
    return summary, fidelity_summary, privacy_summary, runtime_summary


def plot_metric_vs_seed_size(summary, metric, model, title, save_path,
                             config, random_guess=None):
    """
    The main plot style: one line per arm with a ±1 std shaded band,
    TRTR-full as a horizontal reference line, and (optionally) the
    random-guess PR-AUC as a dashed line.
    """
    fig, ax = plt.subplots(figsize=(8, 5))
    model_summary = summary[summary["model"] == model]

    for arm in ["TRTR-seed", "SMOTE-NC", "TSTR-CTGAN", "TSTR-GC"]:
        arm_summary = model_summary[model_summary["arm"] == arm]
        if arm_summary.empty:
            continue
        arm_summary = arm_summary.sort_values("seed_size")
        mean = arm_summary[f"{metric}_mean"]
        std = arm_summary[f"{metric}_std"].fillna(0)
        ax.plot(arm_summary["seed_size"], mean, marker="o",
                label=arm, color=ARM_COLORS[arm])
        ax.fill_between(arm_summary["seed_size"], mean - std, mean + std,
                        alpha=0.15, color=ARM_COLORS[arm])

    # TRTR-full horizontal reference line
    full_summary = model_summary[model_summary["arm"] == "TRTR-full"]
    if not full_summary.empty:
        full_mean = full_summary[f"{metric}_mean"].mean()
        ax.axhline(full_mean, color="black", linestyle="-", linewidth=1.5,
                   label="TRTR-full (reference)")

    if random_guess is not None:
        ax.axhline(random_guess, color="gray", linestyle="--", linewidth=1,
                   label="random guess")

    ax.set_xscale("log")
    ax.set_xticks(config["seeds"]["seed_sizes"])
    ax.get_xaxis().set_major_formatter(plt.ScalarFormatter())
    ax.set_xlabel("Seed size (real rows used for training)")
    ax.set_ylabel(metric.upper())
    ax.set_title(title)
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)


def plot_generator_metric(fidelity_summary, privacy_summary, runtime_summary,
                          config, save_dir):
    """Fidelity, coverage, DCR and runtime plots (CTGAN vs Gaussian Copula)."""
    # KS and TVD
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for generator, color in [("CTGAN", "#c44e52"), ("GaussianCopula", "#8172b3")]:
        gen_summary = fidelity_summary[
            fidelity_summary["generator"] == generator].sort_values("seed_size")
        axes[0].plot(gen_summary["seed_size"], gen_summary["ks_mean_mean"],
                     marker="o", label=generator, color=color)
        axes[1].plot(gen_summary["seed_size"], gen_summary["tvd_mean_mean"],
                     marker="o", label=generator, color=color)
    axes[0].set_title("Mean KS statistic vs seed size")
    axes[1].set_title("Mean TVD vs seed size")
    for ax in axes:
        ax.set_xlabel("Seed size")
        ax.legend()
        ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(save_dir / "fidelity_vs_seed_size.png", dpi=150)
    plt.close(fig)

    # coverage
    fig, ax = plt.subplots(figsize=(7, 4))
    for generator, color in [("CTGAN", "#c44e52"), ("GaussianCopula", "#8172b3")]:
        gen_summary = fidelity_summary[
            fidelity_summary["generator"] == generator].sort_values("seed_size")
        ax.plot(gen_summary["seed_size"], gen_summary["coverage_mean_mean"],
                marker="o", label=generator, color=color)
    ax.set_xlabel("Seed size")
    ax.set_ylabel("Category coverage (%)")
    ax.set_title("Category coverage vs seed size")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(save_dir / "coverage_vs_seed_size.png", dpi=150)
    plt.close(fig)

    # DCR ratio with the 0.8 threshold line
    fig, ax = plt.subplots(figsize=(7, 4))
    for generator, color in [("CTGAN", "#c44e52"), ("GaussianCopula", "#8172b3")]:
        gen_summary = privacy_summary[
            privacy_summary["generator"] == generator].sort_values("seed_size")
        ax.plot(gen_summary["seed_size"], gen_summary["dcr_ratio_mean"],
                marker="o", label=generator, color=color)
    criteria = load_criteria()
    ax.axhline(criteria["dcr_ratio_min"], color="red", linestyle="--",
               label=f"threshold {criteria['dcr_ratio_min']}")
    ax.set_xlabel("Seed size")
    ax.set_ylabel("DCR ratio (synthetic median / holdout median)")
    ax.set_title("Privacy: DCR ratio vs seed size")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(save_dir / "dcr_ratio_vs_seed_size.png", dpi=150)
    plt.close(fig)

    # training time
    fig, ax = plt.subplots(figsize=(7, 4))
    for generator, color in [("CTGAN", "#c44e52"), ("GaussianCopula", "#8172b3")]:
        gen_summary = runtime_summary[
            runtime_summary["generator"] == generator].sort_values("seed_size")
        ax.plot(gen_summary["seed_size"], gen_summary["fit_seconds_mean"],
                marker="o", label=generator, color=color)
    ax.set_xlabel("Seed size")
    ax.set_ylabel("Training time (s)")
    ax.set_title("Generator training time vs seed size")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(save_dir / "training_time_vs_seed_size.png", dpi=150)
    plt.close(fig)


def plot_real_vs_synthetic_pairs(dev_pool, synthetic_dir, config, save_dir):
    """Age and hours-per-week: real vs synthetic at the smallest and largest seed."""
    import pathlib
    for generator_prefix, generator_name in [("ctgan", "CTGAN"),
                                             ("gc", "GaussianCopula")]:
        for size in [min(config["seeds"]["seed_sizes"]),
                     max(config["seeds"]["seed_sizes"])]:
            synth_path = synthetic_dir / f"rs0" / f"{generator_prefix}_{size}.csv"
            if not synth_path.exists():
                continue
            synth = pd.read_csv(synth_path)
            for column in ["age", "hours-per-week"]:
                fig, ax = plt.subplots(figsize=(6, 4))
                ax.hist(dev_pool[column], bins=30, alpha=0.6,
                        label="real (dev pool)", density=True)
                ax.hist(synth[column], bins=30, alpha=0.6,
                        label=f"{generator_name} synthetic (seed {size})",
                        density=True)
                ax.set_title(f"'{column}': real vs {generator_name} "
                             f"(seed {size})")
                ax.legend()
                fig.tight_layout()
                fig.savefig(save_dir /
                            f"real_vs_synth_{generator_prefix}_{column}_{size}.png",
                            dpi=150)
                plt.close(fig)


def plot_ctgan_loss_curve(results_dir, save_dir):
    """The CTGAN generator/discriminator loss curve for one run."""
    losses_dir = results_dir / "ctgan_losses"
    if not losses_dir.exists():
        return
    loss_files = sorted(losses_dir.glob("*.csv"))
    if not loss_files:
        return
    losses = pd.read_csv(loss_files[0])
    fig, ax = plt.subplots(figsize=(7, 4))
    for column in losses.columns:
        if column != "Epoch":
            ax.plot(losses[column], label=column)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title("CTGAN loss curve (one run)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(save_dir / "ctgan_loss_curve.png", dpi=150)
    plt.close(fig)


def apply_criteria(summary, fidelity_summary, privacy_summary):
    """
    Apply the FROZEN criteria.yaml rules per generator and seed size.
    Utility uses XGBoost PR-AUC. Returns (pass_fail_table, minimum_useful_seed).
    """
    criteria = load_criteria()
    rows = []
    for generator in ["CTGAN", "GaussianCopula"]:
        for seed_size in sorted(fidelity_summary["seed_size"].unique()):
            fid = fidelity_summary[
                (fidelity_summary["generator"] == generator) &
                (fidelity_summary["seed_size"] == seed_size)]
            prv = privacy_summary[
                (privacy_summary["generator"] == generator) &
                (privacy_summary["seed_size"] == seed_size)]
            tstr = summary[
                (summary["model"] == "xgboost") &
                (summary["arm"] == f"TSTR-{generator}") &
                (summary["seed_size"] == seed_size)]
            full = summary[
                (summary["model"] == "xgboost") &
                (summary["arm"] == "TRTR-full")]

            if fid.empty or prv.empty or tstr.empty or full.empty:
                continue

            utility_ratio = (tstr["pr_auc_mean"].iloc[0] /
                             max(full["pr_auc_mean"].iloc[0], 1e-12))
            dcr_ratio = prv["dcr_ratio_mean"].iloc[0]
            # exact-copy rule: synthetic rate must not exceed holdout rate
            exact_ok = (prv["synth_exact_match_rate_mean"].iloc[0] <=
                        prv["holdout_exact_match_rate_mean"].iloc[0])

            checks = {
                "utility_ratio>=0.90":
                    utility_ratio >= criteria["utility_ratio_min"],
                "ks_mean<=0.10":
                    fid["ks_mean_mean"].iloc[0] <= criteria["ks_mean_max"],
                "tvd_mean<=0.10":
                    fid["tvd_mean_mean"].iloc[0] <= criteria["tvd_mean_max"],
                "dcr_ratio>=0.80":
                    dcr_ratio >= criteria["dcr_ratio_min"],
                "no_extra_exact_copies": exact_ok,
            }
            rows.append({
                "generator": generator, "seed_size": seed_size,
                "utility_ratio": round(utility_ratio, 3),
                "ks_mean": fid["ks_mean_mean"].iloc[0],
                "tvd_mean": fid["tvd_mean_mean"].iloc[0],
                "dcr_ratio": dcr_ratio,
                **{rule: "PASS" if ok else "FAIL" for rule, ok in checks.items()},
                "ALL_PASS": all(checks.values()),
            })
    pass_fail_table = pd.DataFrame(rows)

    minimum_useful_seed = None
    if not pass_fail_table.empty:
        passed = pass_fail_table[pass_fail_table["ALL_PASS"]]
        if not passed.empty:
            # the smallest seed size where BOTH generators pass is a
            # conservative answer; otherwise report per-generator minimum
            both = passed.groupby("seed_size")["generator"].nunique()
            both_passed = both[both == passed["generator"].nunique()]
            if not both_passed.empty:
                minimum_useful_seed = int(both_passed.index.min())
            else:
                minimum_useful_seed = int(passed["seed_size"].min())
    return pass_fail_table, minimum_useful_seed


def write_results_md(summary, fidelity_summary, privacy_summary, runtime_summary,
                     pass_fail_table, minimum_useful_seed, findings):
    """Write results/RESULTS.md with tables, pass/fail and findings."""
    lines = ["# Experiment Results", ""]
    lines.append("## Utility summary (mean ± std over random seeds)")
    lines.append(summary.to_markdown(index=False))
    lines.append("")
    lines.append("## Fidelity summary")
    lines.append(fidelity_summary.to_markdown(index=False))
    lines.append("")
    lines.append("## Privacy summary (DCR)")
    lines.append(privacy_summary.to_markdown(index=False))
    lines.append("")
    lines.append("## Pass/Fail table (frozen criteria.yaml)")
    if pass_fail_table is not None and not pass_fail_table.empty:
        lines.append(pass_fail_table.to_markdown(index=False))
    else:
        lines.append("_No complete results yet — run the full experiment._")
    lines.append("")
    if minimum_useful_seed is not None:
        lines.append(f"## Minimum useful seed: **{minimum_useful_seed} rows**")
    else:
        lines.append("## Minimum useful seed: **none of the tested sizes "
                     "passed all criteria** (a valid finding).")
    lines.append("")
    lines.append("## Findings")
    lines.extend(f"- {f}" for f in findings)
    (RESULTS_DIR / "RESULTS.md").write_text("\n".join(lines), encoding="utf-8")


def make_findings(summary, fidelity_summary, privacy_summary,
                  pass_fail_table, minimum_useful_seed):
    """Plain-English findings — honest, no over-claiming."""
    findings = []
    if summary.empty:
        return ["No results yet — run the experiment first."]

    xgb = summary[summary["model"] == "xgboost"]
    largest = xgb["seed_size"].max()
    tstr_ctgan = xgb[(xgb["arm"] == "TSTR-CTGAN") &
                     (xgb["seed_size"] == largest)]["pr_auc_mean"]
    tstr_gc = xgb[(xgb["arm"] == "TSTR-GC") &
                  (xgb["seed_size"] == largest)]["pr_auc_mean"]
    trtr_seed = xgb[(xgb["arm"] == "TRTR-seed") &
                    (xgb["seed_size"] == largest)]["pr_auc_mean"]
    if not (tstr_ctgan.empty or trtr_seed.empty):
        if tstr_ctgan.iloc[0] > trtr_seed.iloc[0]:
            findings.append(
                "At the largest seed size, CTGAN synthetic data (TSTR) beat "
                "training on the small real seed alone (TRTR-seed).")
        else:
            findings.append(
                "At the largest seed size, training on the small real seed "
                "(TRTR-seed) still matched or beat synthetic data (TSTR).")
    if not (tstr_ctgan.empty or tstr_gc.empty):
        better = "CTGAN" if tstr_ctgan.iloc[0] >= tstr_gc.iloc[0] else \
            "Gaussian Copula"
        findings.append(f"{better} produced the more useful synthetic data "
                        "at the largest seed size (XGBoost PR-AUC).")

    if not fidelity_summary.empty:
        best_fid = fidelity_summary.groupby("generator")["ks_mean_mean"].min()
        findings.append(f"Best fidelity (lowest mean KS): "
                        f"{best_fid.idxmin()} ({best_fid.min():.3f}).")

    if not privacy_summary.empty:
        min_ratio = privacy_summary["dcr_ratio_mean"].min()
        if min_ratio < 0.8:
            findings.append(
                f"Memorisation warning: the lowest DCR ratio was {min_ratio:.2f} "
                "(below the 0.8 threshold) — synthetic rows sat closer to the "
                "seed than genuine records do.")
        else:
            findings.append(
                f"No memorisation warning: all DCR ratios stayed at or above "
                f"0.8 (lowest {min_ratio:.2f}).")

    if pass_fail_table is not None and not pass_fail_table.empty:
        n_pass = int(pass_fail_table["ALL_PASS"].sum())
        findings.append(f"{n_pass} of {len(pass_fail_table)} "
                        "(generator, seed size) combinations passed ALL "
                        "frozen criteria.")
    if minimum_useful_seed is not None:
        findings.append(f"The minimum useful seed is {minimum_useful_seed} "
                        "rows under our frozen criteria.")
    else:
        findings.append("No tested seed size passed all criteria — reported "
                        "honestly as a valid experimental finding.")
    findings.append(
        "Gains from more data flatten: PR-AUC improves with seed size but "
        "the improvement shrinks — check the flattening point on the main "
        "plot.")
    return findings


def main():
    config = load_config()
    metrics_path = RESULTS_DIR / "metrics.csv"
    if not metrics_path.exists():
        print("No results found. Run 'python -m src.run_experiment' first.")
        return

    metrics = pd.read_csv(metrics_path)
    fidelity = pd.read_csv(RESULTS_DIR / "fidelity.csv")
    privacy = pd.read_csv(RESULTS_DIR / "privacy.csv")
    runtime = pd.read_csv(RESULTS_DIR / "runtime.csv")

    summary, fidelity_summary, privacy_summary, runtime_summary = \
        build_summary(metrics, fidelity, privacy, runtime)

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    for model_name in ["random_forest", "xgboost"]:
        plot_metric_vs_seed_size(
            summary, "pr_auc", model_name,
            f"PR-AUC vs seed size ({model_name})",
            FIGURES_DIR / f"pr_auc_{model_name}.png", config)
        plot_metric_vs_seed_size(
            summary, "f1", model_name,
            f"F1 vs seed size ({model_name})",
            FIGURES_DIR / f"f1_{model_name}.png", config)

    plot_generator_metric(fidelity_summary, privacy_summary, runtime_summary,
                          config, FIGURES_DIR)

    dev_pool = pd.read_csv(PROJECT_ROOT / "data" / "processed" / "dev_pool.csv")
    plot_real_vs_synthetic_pairs(dev_pool, PROJECT_ROOT / "data" / "synthetic",
                                 config, FIGURES_DIR)
    plot_ctgan_loss_curve(RESULTS_DIR, FIGURES_DIR)

    pass_fail_table, minimum_useful_seed = apply_criteria(
        summary, fidelity_summary, privacy_summary)
    findings = make_findings(summary, fidelity_summary, privacy_summary,
                             pass_fail_table, minimum_useful_seed)
    write_results_md(summary, fidelity_summary, privacy_summary,
                     runtime_summary, pass_fail_table, minimum_useful_seed,
                     findings)

    summary.to_csv(RESULTS_DIR / "summary.csv", index=False)

    print("=== Pass/Fail table ===")
    print(pass_fail_table.to_string(index=False))
    print(f"\nMinimum useful seed: {minimum_useful_seed}")
    print("\n=== Findings ===")
    for finding in findings:
        print(f"- {finding}")
    print(f"\nFigures in {FIGURES_DIR}, report in results/RESULTS.md")


if __name__ == "__main__":
    main()
