"""
fidelity.py — Step 6: how closely does synthetic data copy the real data?

Why this file exists: a synthetic dataset is only useful if it resembles the
real one. We compare synthetic vs the DEV POOL (our stand-in for the real
population) with four measures:

  * KS statistic   — for NUMERICAL columns: the biggest gap between the two
                     cumulative distributions. 0 = identical shapes.
  * TVD            — for CATEGORICAL columns: half the total difference in
                     category proportions. 0 = identical proportions.
  * chi-square     — a second opinion on categorical counts (we use the
                     statistic for logging; decisions use KS/TVD because
                     p-values swing wildly with sample size).
  * correlation difference — do pairs of numerical columns move together in
                     the synthetic data the way they do in real data?
  * category coverage — % of real categories that appear at least once in
                     the synthetic data (small seeds may miss rare ones).

NOTE: the reference "real" data is the dev pool, NOT the locked test set.
"""

import numpy as np
import pandas as pd
from scipy import stats

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def ks_per_column(real, synth, numerical_cols):
    """
    Kolmogorov-Smirnov statistic per numerical column.
    Returns a DataFrame: column, ks_statistic, p_value.
    """
    rows = []
    for column in numerical_cols:
        statistic, p_value = stats.ks_2samp(real[column], synth[column])
        rows.append({"column": column,
                     "ks_statistic": round(statistic, 4),
                     "p_value": round(p_value, 6)})
    return pd.DataFrame(rows)


def tvd_per_column(real, synth, categorical_cols):
    """
    Total Variation Distance per categorical column:
    TVD = 0.5 * sum(|real proportion - synthetic proportion|)
    over the union of categories seen in either dataset.
    """
    rows = []
    for column in categorical_cols:
        real_props = real[column].value_counts(normalize=True)
        synth_props = synth[column].value_counts(normalize=True)
        all_categories = set(real_props.index) | set(synth_props.index)
        tvd = 0.5 * sum(
            abs(real_props.get(cat, 0) - synth_props.get(cat, 0))
            for cat in all_categories)
        rows.append({"column": column, "tvd": round(tvd, 4)})
    return pd.DataFrame(rows)


def chi_square_per_column(real, synth, categorical_cols):
    """
    Chi-square test per categorical column on the 2 x k count table.
    Categories with zero count in BOTH datasets are dropped first (a 2xk
    table with an all-zero column is invalid).
    """
    rows = []
    for column in categorical_cols:
        counts = pd.DataFrame({
            "real": real[column].value_counts(),
            "synth": synth[column].value_counts(),
        }).fillna(0)
        counts = counts[(counts["real"] > 0) | (counts["synth"] > 0)]
        try:
            chi2, p_value, _, _ = stats.chi2_contingency(counts.T)
        except ValueError:
            chi2, p_value = float("nan"), float("nan")
        rows.append({"column": column,
                     "chi2": round(chi2, 2),
                     "p_value": p_value})
    return pd.DataFrame(rows)


def correlation_difference(real, synth, numerical_cols):
    """
    Mean absolute difference between the upper triangles of the real and
    synthetic Pearson correlation matrices. 0 = relationships preserved.
    """
    real_corr = real[numerical_cols].corr().to_numpy()
    synth_corr = synth[numerical_cols].corr().to_numpy()
    upper_indices = np.triu_indices_from(real_corr, k=1)
    if len(upper_indices[0]) == 0:
        # fewer than two numerical columns -> no pairs to compare
        return 0.0
    differences = np.abs(real_corr[upper_indices] - synth_corr[upper_indices])
    return float(np.mean(differences))


def category_coverage(real, synth, categorical_cols):
    """
    % of real categories that appear at least once in the synthetic data,
    per column and overall.
    """
    per_column = {}
    for column in categorical_cols:
        real_categories = set(real[column].dropna().unique())
        synth_categories = set(synth[column].dropna().unique())
        covered = len(real_categories & synth_categories)
        per_column[column] = round(100 * covered / max(len(real_categories), 1), 2)
    overall = round(np.mean(list(per_column.values())), 2) if per_column else 100.0
    return per_column, overall


def evaluate_fidelity(real, synth, numerical_cols, categorical_cols):
    """
    Run all fidelity measures. Returns (per_column_table, summary_dict).
    """
    ks_table = ks_per_column(real, synth, numerical_cols)
    tvd_table = tvd_per_column(real, synth, categorical_cols)
    corr_diff = correlation_difference(real, synth, numerical_cols)
    coverage_per_column, coverage_overall = category_coverage(
        real, synth, categorical_cols)

    per_column = pd.concat([
        ks_table.set_index("column"),
        tvd_table.set_index("column"),
    ], axis=1)
    per_column["coverage_pct"] = pd.Series(coverage_per_column)

    summary = {
        "ks_mean": round(ks_table["ks_statistic"].mean(), 4),
        "tvd_mean": round(tvd_table["tvd"].mean(), 4),
        "corr_diff": round(corr_diff, 4),
        "coverage_mean": coverage_overall,
    }
    return per_column, summary


def plot_real_vs_synthetic(real, synth, column, save_path):
    """
    One comparison figure for one column:
      numerical -> overlaid histograms
      categorical -> side-by-side bar chart (top categories)
    """
    fig, ax = plt.subplots(figsize=(7, 4))
    if pd.api.types.is_numeric_dtype(real[column]):
        ax.hist(real[column], bins=30, alpha=0.6, label="real", density=True)
        ax.hist(synth[column], bins=30, alpha=0.6, label="synthetic", density=True)
        ax.set_ylabel("Density")
    else:
        top = real[column].value_counts().head(10).index
        positions = np.arange(len(top))
        real_props = real[column].value_counts(normalize=True).reindex(top).fillna(0)
        synth_props = synth[column].value_counts(normalize=True).reindex(top).fillna(0)
        ax.bar(positions - 0.2, real_props, width=0.4, label="real")
        ax.bar(positions + 0.2, synth_props, width=0.4, label="synthetic")
        ax.set_xticks(positions)
        ax.set_xticklabels(top, rotation=45)
        ax.set_ylabel("Proportion")
    ax.set_title(f"Real vs synthetic: '{column}'")
    ax.legend()
    fig.tight_layout()
    fig.savefig(save_path)
    plt.close(fig)


def main():
    """Demo: compare the Step 4 Gaussian Copula output (seed 500) vs dev pool."""
    from src.utils import PROJECT_ROOT, get_column_types, load_config
    config = load_config()
    target = config["dataset"]["target"]
    dev_pool = pd.read_csv(PROJECT_ROOT / "data" / "processed" / "dev_pool.csv")
    synth = pd.read_csv(PROJECT_ROOT / "data" / "synthetic" /
                        "rs0" / "gc_500.csv")
    numerical_cols, categorical_cols = get_column_types(dev_pool, target)

    per_column, summary = evaluate_fidelity(dev_pool, synth,
                                            numerical_cols, categorical_cols)
    print(per_column)
    print("\nSummary:", summary)

    examples_dir = PROJECT_ROOT / "figures" / "examples"
    examples_dir.mkdir(parents=True, exist_ok=True)
    plot_real_vs_synthetic(dev_pool, synth, "age",
                           examples_dir / "example_age.png")
    plot_real_vs_synthetic(dev_pool, synth, "workclass",
                           examples_dir / "example_workclass.png")
    print(f"Example plots saved to {examples_dir}")


if __name__ == "__main__":
    main()
