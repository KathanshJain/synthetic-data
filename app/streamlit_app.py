"""
app/streamlit_app.py — Step 11: the Streamlit web app.

Why this file exists: the app presents the project in three tabs:
  1. About the Project   — the story, pipeline and quality checks.
  2. Experiment Results  — the real numbers from results/ and figures/.
  3. Try It Yourself     — upload any CSV, profile it, generate synthetic
                           data with the generator of your choice.

SAFETY: this app NEVER reads the locked test set. All heavy functions are
imported from src/ so there is no duplicated logic.
"""

import pandas as pd
import streamlit as st

from src.eda import profile_dataset
from src.fidelity import evaluate_fidelity, plot_real_vs_synthetic
from src.generators import (
    generate,
    run_smote_nc,
    train_ctgan,
    train_gaussian_copula,
    validate_synthetic,
)
from src.privacy import compute_dcr
from src.utils import PROJECT_ROOT, get_column_types, load_config

st.set_page_config(page_title="ML on Artificially Forged Data",
                   page_icon="🕵️", layout="wide")


@st.cache_data
def load_summary():
    """Load results/summary.csv if the experiment has been run."""
    path = PROJECT_ROOT / "results" / "summary.csv"
    if path.exists():
        return pd.read_csv(path)
    return None


def tab_about():
    """Tab 1: the story of the project."""
    st.title("Machine Learning on Artificially Forged Data 🕵️")
    st.markdown("""
### The forger and the detective

Imagine an organisation with a table of personal records. It cannot share the
full table, so your team receives only a **small sample**. The question: can
a **forger** (a generative model) learn the patterns in that small sample and
produce artificial records good enough for a **detective** (an ML model) to
train on — and still perform well on genuine, unseen data?

### Our research question

> How much real data does a generator need to produce synthetic tabular data
> that is realistic, useful for training ML models, and does not copy real records?

### The pipeline

```
raw Adult CSV → clean → 80/20 split (dev pool + LOCKED test set, SHA-256)
→ nested stratified seeds (100…2000 rows, 3 random seeds)
→ generators: CTGAN | Gaussian Copula | SMOTE-NC
→ three quality checks:
     FIDELITY  (KS, TVD, chi-square, correlations, coverage)
     UTILITY   (RF + XGBoost, 5 arms, PR-AUC on the locked test set)
     PRIVACY   (Distance to Closest Record vs a real holdout)
→ minimum useful seed (frozen criteria.yaml)
```

### Three separate quality checks

| Check | Question | Tools |
|-------|----------|-------|
| Fidelity | Does synthetic data *look like* the real data? | KS, TVD, chi², correlation diff, coverage |
| Utility | Do models trained on it *work* on real data? | TRTR vs TSTR, PR-AUC |
| Privacy | Does it *copy* real records? | DCR vs holdout baseline |

### Team roles
- **Kathansh** — data analysis track: data, EDA, seeds, fidelity, privacy.
- **Aryan** — GenAI & ML track: generators, models, experiment runner, app.
""")


def tab_results():
    """Tab 2: the real experiment results."""
    st.title("Experiment Results 📊")
    summary = load_summary()
    if summary is None:
        st.warning("No results yet! Run the experiment first:  \n"
                   "`python -m src.run_experiment` (or `--quick` for a smoke "
                   "test), then `python -m src.analyze`.")
        return

    st.markdown("Pick a model and an arm to see the numbers.")
    col1, col2 = st.columns(2)
    model = col1.selectbox("Model", ["xgboost", "random_forest"])
    arm = col2.selectbox("Arm", ["TRTR-full", "TRTR-seed", "SMOTE-NC",
                                 "TSTR-CTGAN", "TSTR-GC"])

    filtered = summary[(summary["model"] == model) & (summary["arm"] == arm)]
    st.dataframe(filtered, use_container_width=True)

    figures_dir = PROJECT_ROOT / "figures" / "results"
    if figures_dir.exists():
        st.subheader("Main plots")
        for figure_name in [f"pr_auc_{model}.png", "fidelity_vs_seed_size.png",
                            "dcr_ratio_vs_seed_size.png",
                            "coverage_vs_seed_size.png"]:
            figure_path = figures_dir / figure_name
            if figure_path.exists():
                st.image(str(figure_path), caption=figure_name)

    # minimum useful seed in a highlighted box
    results_md = PROJECT_ROOT / "results" / "RESULTS.md"
    if results_md.exists():
        st.subheader("Full report")
        st.markdown(results_md.read_text(encoding="utf-8"))


def tab_try_it_yourself():
    """Tab 3: upload a CSV and generate synthetic data yourself."""
    st.title("Try It Yourself 🧪")
    st.caption("This tab never touches the locked test set.")

    uploaded_file = st.file_uploader("Upload any CSV", type="csv")
    config = load_config()
    target = config["dataset"]["target"]

    if st.button("Use sample: Adult dev pool"):
        sample = pd.read_csv(PROJECT_ROOT / "data" / "processed" / "dev_pool.csv")
        st.session_state["data"] = sample
    if uploaded_file is not None:
        st.session_state["data"] = pd.read_csv(uploaded_file)

    if "data" not in st.session_state:
        st.info("Upload a CSV above, or click the sample button.")
        return

    df = st.session_state["data"]

    # ---- profile ----
    st.subheader("1. Profile of your data")
    profile = profile_dataset(df, target=None)
    col1, col2, col3 = st.columns(3)
    col1.metric("Rows", profile["rows"])
    col2.metric("Columns", profile["columns"])
    col3.metric("Duplicate rows", profile["duplicate_rows"])
    for warning in profile["warnings"]:
        st.warning(warning)

    # ---- refuse or warn on problematic data ----
    if profile["rows"] < 50:
        st.error("Fewer than 50 rows — too little data to generate from.")
        return

    # ---- settings ----
    st.subheader("2. Generation settings")
    columns = list(df.columns)
    target_column = st.selectbox("Target column (or 'none')",
                                 ["(none)"] + columns)
    seed_size = st.slider("Seed size (rows used to train the generator)",
                          50, min(len(df), 2000), min(500, len(df)))
    generator_name = st.selectbox(
        "Generator", ["Gaussian Copula (fast default)", "CTGAN (slow)",
                      "SMOTE-NC (binary target only)"])
    n_synth_rows = st.number_input("Number of synthetic rows", 100, 10000, 1000)

    ctgan_steps = 200
    if "CTGAN" in generator_name:
        st.warning("CTGAN training may take a few minutes on a laptop.")
        ctgan_steps = st.slider("CTGAN training steps", 100, 3000, 200, step=100)

    if st.button("Generate synthetic data"):
        with st.spinner("Working..."):
            # build the seed (stratified if there is a binary target)
            if target_column != "(none)" and df[target_column].nunique() == 2:
                seed_df, rest = train_test_split_local(df, target_column,
                                                       seed_size)
            else:
                seed_df = df.sample(n=seed_size, random_state=0)
                rest = df.drop(seed_df.index)
            numerical_cols, categorical_cols = get_column_types(
                seed_df, target_column if target_column != "(none)" else None)

            if "SMOTE-NC" in generator_name:
                if target_column == "(none)" or df[target_column].nunique() != 2:
                    st.error("SMOTE-NC needs a binary target column.")
                    return
                minority = seed_df[target_column].value_counts().min()
                if minority < 10:
                    st.error(f"Minority class has only {minority} rows in the "
                             "seed — need at least 10.")
                    return
                synthetic_df, _ = run_smote_nc(
                    seed_df, numerical_cols, categorical_cols, target_column,
                    config["generators"]["smote"]["k_neighbors"], 0)
            elif "CTGAN" in generator_name:
                quick_config = dict(config)
                quick_config["generators"] = dict(config["generators"])
                quick_config["generators"]["ctgan"] = {
                    "total_training_steps": ctgan_steps,
                    "batch_size": config["generators"]["ctgan"]["batch_size"],
                    "use_gpu": False,
                }
                synthesizer, _ = train_ctgan(seed_df, quick_config, 0)
                synthetic_df, _ = generate(synthesizer, int(n_synth_rows))
            else:
                synthesizer, _ = train_gaussian_copula(seed_df, config, 0)
                synthetic_df, _ = generate(synthesizer, int(n_synth_rows))

        # ---- results ----
        st.subheader("3. Results")
        st.write("**Preview of synthetic data:**")
        st.dataframe(synthetic_df.head(), use_container_width=True)

        if target_column != "(none)" and target_column in synthetic_df.columns:
            comparison = pd.DataFrame({
                "real seed %": seed_df[target_column].value_counts(normalize=True),
                "synthetic %": synthetic_df[target_column].value_counts(
                    normalize=True),
            })
            st.bar_chart(comparison)

        per_column, fidelity_summary = evaluate_fidelity(
            seed_df.drop(columns=["row_id"], errors="ignore"),
            synthetic_df, numerical_cols, categorical_cols)
        st.write("**Fidelity summary:**", fidelity_summary)
        with st.expander("Per-column fidelity table"):
            st.dataframe(per_column)

        plot_column = st.selectbox("Compare a column (real vs synthetic)",
                                   numerical_cols + categorical_cols)
        plot_real_vs_synthetic(seed_df, synthetic_df, plot_column,
                               PROJECT_ROOT / "figures" / "app_plot.png")
        st.image(str(PROJECT_ROOT / "figures" / "app_plot.png"))

        # DCR: holdout = uploaded rows NOT in the seed
        if len(rest) >= 10:
            dcr = compute_dcr(seed_df, synthetic_df, rest.head(1000),
                              numerical_cols, categorical_cols)
            st.write("**Privacy (DCR) summary:**")
            st.json({key: value for key, value in dcr.items()})

        warnings = validate_synthetic(synthetic_df, seed_df,
                                      numerical_cols, categorical_cols)
        for warning in warnings:
            st.warning(warning)

        csv_bytes = synthetic_df.to_csv(index=False).encode()
        st.download_button("Download synthetic CSV", csv_bytes,
                           "synthetic_data.csv", "text/csv")


def train_test_split_local(df, target_column, seed_size):
    """Tiny stratified seed extraction for the app (local helper)."""
    from sklearn.model_selection import train_test_split
    fraction = min(seed_size / len(df), 0.9)
    seed_df, rest = train_test_split(df, train_size=fraction,
                                     random_state=0,
                                     stratify=df[target_column])
    return seed_df, rest


def main():
    tabs = st.tabs(["About the Project", "Experiment Results",
                    "Try It Yourself"])
    with tabs[0]:
        tab_about()
    with tabs[1]:
        tab_results()
    with tabs[2]:
        tab_try_it_yourself()


if __name__ == "__main__":
    main()
