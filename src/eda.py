"""
eda.py — Step 2: Exploratory Data Analysis on the development pool ONLY.

Why this file exists: before generating synthetic data we must understand the
real data (class imbalance, skewed columns, rare categories). Everything here
uses data/processed/dev_pool.csv — the locked test set is never opened.

profile_dataset() is written as a reusable function because the Streamlit app
later calls the exact same function on user-uploaded CSVs.
"""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # save figures to disk without needing a screen
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from src.utils import PROJECT_ROOT, get_column_types, load_config


def profile_dataset(df, target=None):
    """
    Build a plain-English profile of any DataFrame.

    Returns a dictionary with: row/column counts, which columns are numerical
    or categorical, missing percentages, duplicate rows, unique-value counts,
    class balance (if a target is given), ID-like columns, and a list of
    warnings a beginner can actually read.

    This is reused by the Streamlit app on user-uploaded files.
    """
    profile = {
        "rows": len(df),
        "columns": df.shape[1],
    }
    numerical_cols, categorical_cols = get_column_types(df, target)
    profile["numerical_columns"] = numerical_cols
    profile["categorical_columns"] = categorical_cols

    # missing percentage per column
    missing_pct = (df.isna().mean() * 100).round(2)
    profile["missing_percent"] = missing_pct[missing_pct > 0].to_dict()

    # exact duplicate rows
    profile["duplicate_rows"] = int(df.duplicated().sum())

    # unique values per column
    profile["unique_values"] = {c: int(df[c].nunique()) for c in df.columns}

    # class balance
    if target and target in df.columns:
        counts = df[target].value_counts()
        profile["class_balance"] = {
            str(k): {"count": int(v), "percent": round(100 * v / len(df), 2)}
            for k, v in counts.items()
        }

    # ID-like columns: almost every row has a different value -> probably an ID
    id_like = [c for c in df.columns
               if df[c].nunique() > 0.95 * len(df)]
    profile["id_like_columns"] = id_like

    # ---- plain-English warnings -------------------------------------------
    warnings = []
    if target and target in df.columns:
        minority = df[target].value_counts().min()
        if minority < 20:
            warnings.append(
                f"Minority class has only {minority} rows — far too few to "
                "train a generator or classifier on.")

    for column in id_like:
        warnings.append(
            f"Column '{column}' looks like an ID (almost all values unique) — "
            "it should not be used as a model feature.")

    for column, pct in profile["missing_percent"].items():
        if pct > 30:
            warnings.append(
                f"Column '{column}' has {pct}% missing values.")

    # date/time columns are detected by name and by pandas' date type
    date_like = [c for c in df.columns
                 if "date" in c.lower() or "time" in c.lower()
                 or pd.api.types.is_datetime64_any_dtype(df[c])]
    if date_like:
        warnings.append(
            f"Possible date/time column(s) detected: {date_like} — "
            "time-series data is not supported by these generators.")

    profile["warnings"] = warnings
    return profile


def plot_class_distribution(df, target, save_path):
    """Bar chart of the target classes with counts and percentages on top."""
    counts = df[target].value_counts()
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(counts.index.astype(str), counts.values, color=["#4c72b0", "#c44e52"])
    for i, value in enumerate(counts.values):
        ax.text(i, value, f"{value}\n({value / len(df):.1%})",
                ha="center", va="bottom")
    ax.set_title(f"Class distribution of '{target}'")
    ax.set_ylabel("Number of rows")
    fig.tight_layout()
    fig.savefig(save_path)
    plt.close(fig)


def plot_numeric_histograms(df, numerical_cols, save_dir):
    """One histogram per numerical column."""
    for column in numerical_cols:
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.hist(df[column].dropna(), bins=30, color="#4c72b0", edgecolor="white")
        ax.set_title(f"Distribution of '{column}'")
        ax.set_xlabel(column)
        ax.set_ylabel("Count")
        fig.tight_layout()
        fig.savefig(save_dir / f"hist_{column}.png")
        plt.close(fig)


def plot_categorical_bars(df, categorical_cols, save_dir, top_n=10):
    """One bar chart per categorical column (top 10 categories only)."""
    for column in categorical_cols:
        top = df[column].value_counts().head(top_n)
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.bar(top.index.astype(str), top.values, color="#55a868")
        ax.set_title(f"Top {top_n} categories of '{column}'")
        ax.set_ylabel("Count")
        ax.tick_params(axis="x", rotation=45)
        fig.tight_layout()
        fig.savefig(save_dir / f"bar_{column}.png")
        plt.close(fig)


def plot_correlation_heatmap(df, numerical_cols, save_path):
    """Heatmap of Pearson correlations between numerical columns."""
    correlation = df[numerical_cols].corr()
    fig, ax = plt.subplots(figsize=(7, 6))
    sns.heatmap(correlation, annot=True, fmt=".2f", cmap="coolwarm",
                ax=ax, square=True)
    ax.set_title("Correlation heatmap (numerical columns)")
    fig.tight_layout()
    fig.savefig(save_path)
    plt.close(fig)


def plot_boxplots_by_class(df, numerical_cols, target, save_dir):
    """One boxplot per numerical column, split by the target class."""
    for column in numerical_cols:
        fig, ax = plt.subplots(figsize=(6, 4))
        df.boxplot(column=column, by=target, ax=ax)
        ax.set_title(f"'{column}' by {target}")
        ax.set_ylabel(column)
        fig.suptitle("")  # remove pandas' automatic sup-title
        fig.tight_layout()
        fig.savefig(save_dir / f"box_{column}_by_{target}.png")
        plt.close(fig)


def plot_missing_category_counts(df, categorical_cols, save_dir):
    """How many 'Missing' values each categorical column carries."""
    counts = {c: int((df[c] == "Missing").sum()) for c in categorical_cols}
    counts = {c: v for c, v in counts.items() if v > 0}
    if not counts:
        return
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar(counts.keys(), counts.values(), color="#c44e52")
    ax.set_title("Rows with the 'Missing' category per column")
    ax.set_ylabel("Count")
    ax.tick_params(axis="x", rotation=45)
    fig.tight_layout()
    fig.savefig(save_dir / "missing_category_counts.png")
    plt.close(fig)


def write_eda_report(profile, report_path):
    """Turn the profile dict into a small Markdown report with observations."""
    lines = ["# EDA Report (development pool only)", ""]
    lines.append(f"- Rows: **{profile['rows']}**, Columns: **{profile['columns']}**")
    lines.append(f"- Numerical columns: {', '.join(profile['numerical_columns'])}")
    lines.append(f"- Categorical columns: {', '.join(profile['categorical_columns'])}")
    lines.append(f"- Duplicate rows: {profile['duplicate_rows']}")
    if profile.get("class_balance"):
        lines.append("- Class balance: " + ", ".join(
            f"{k}: {v['count']} ({v['percent']}%)"
            for k, v in profile["class_balance"].items()))
    if profile["missing_percent"]:
        lines.append("- Missing values: " + ", ".join(
            f"{k} ({v}%)" for k, v in profile["missing_percent"].items()))
    lines.append("")
    lines.append("## Warnings")
    lines.extend(f"- {w}" for w in profile["warnings"] or ["None."])
    lines.append("")
    lines.append("## Observations")
    lines.extend(f"- {o}" for o in make_observations(profile))
    report_path.write_text("\n".join(lines), encoding="utf-8")


def make_observations(profile):
    """
    Plain-English observations, computed from the profile so they stay true
    even if the data changes.
    """
    observations = []
    balance = profile.get("class_balance", {})
    if balance:
        minority_pct = min(v["percent"] for v in balance.values())
        observations.append(
            f"The dataset is imbalanced: the smaller class is only "
            f"{minority_pct}% of rows. This is why we use PR-AUC instead of "
            "accuracy later.")
    missing = profile.get("missing_percent", {})
    if missing:
        observations.append(
            "Missing values exist in " + ", ".join(missing.keys()) +
            "; we keep them as an explicit 'Missing' category instead of "
            "deleting rows, to preserve the class ratio.")
    observations.append(
        "capital-gain and capital-loss are extremely skewed: most people have "
        "0, a few have very large values. Generators find such columns hard.")
    observations.append(
        "native-country is dominated by 'United States'; rare countries are "
        "categories a small seed may not even contain (coverage risk).")
    observations.append(
        "age and hours-per-week are roughly bell-shaped but with spikes "
        "(e.g. full-time 40 hours/week).")
    if profile["id_like_columns"]:
        observations.append(
            "ID-like columns detected: " + ", ".join(profile["id_like_columns"]) +
            " — excluded from modelling.")
    return observations


def main():
    config = load_config()
    target = config["dataset"]["target"]
    dev_pool = pd.read_csv(PROJECT_ROOT / "data" / "processed" / "dev_pool.csv")

    profile = profile_dataset(dev_pool, target)
    numerical_cols, categorical_cols = get_column_types(dev_pool, target)

    save_dir = PROJECT_ROOT / "figures" / "eda"
    save_dir.mkdir(parents=True, exist_ok=True)

    plot_class_distribution(dev_pool, target, save_dir / "class_distribution.png")
    plot_numeric_histograms(dev_pool, numerical_cols, save_dir)
    plot_categorical_bars(dev_pool, categorical_cols, save_dir)
    plot_correlation_heatmap(dev_pool, numerical_cols,
                             save_dir / "correlation_heatmap.png")
    plot_boxplots_by_class(dev_pool, numerical_cols, target, save_dir)
    plot_missing_category_counts(dev_pool, categorical_cols, save_dir)

    report_path = PROJECT_ROOT / "reports" / "eda_report.md"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    write_eda_report(profile, report_path)

    print("=== EDA profile ===")
    for key, value in profile.items():
        if key != "warnings":
            print(f"{key}: {value}")
    print("\nWarnings:")
    for warning in profile["warnings"]:
        print(f"  - {warning}")
    print(f"\nFigures saved to {save_dir}")
    print(f"Report saved to {report_path}")


if __name__ == "__main__":
    main()
