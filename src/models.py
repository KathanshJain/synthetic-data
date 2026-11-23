"""
models.py — Step 8: the classifiers and the five training "arms".

Why this file exists: to answer "is synthetic data useful?", we train models
on different kinds of data and compare them on the SAME real test set:

  | Arm         | Training data                          | Purpose                  |
  |-------------|----------------------------------------|--------------------------|
  | TRTR-full   | whole dev pool (real)                  | reference benchmark      |
  | TRTR-seed   | small real seed only                   | small-data baseline      |
  | SMOTE-NC    | seed + SMOTE minority rows             | oversampling baseline    |
  | TSTR-CTGAN  | CTGAN-generated rows only              | main synthetic experiment|
  | TSTR-GC     | Gaussian-Copula rows only              | statistical generator    |

  TRTR = Train Real Test Real.   TSTR = Train Synthetic Test Real.

Primary metric: PR-AUC (average precision). The positive class (>50K) is only
~24% of rows, so accuracy is misleading — a model that always says "<=50K"
scores 76% accuracy while being useless. PR-AUC focuses exactly on how well
the minority class is found.

Encoders are fitted ONLY on each arm's training data (never on the test set),
which is what a real deployment would do and prevents evaluation leakage.
"""

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from xgboost import XGBClassifier

from src.utils import PROJECT_ROOT, get_column_types, load_config


def build_pipeline(model_name, numerical_cols, categorical_cols, config,
                   random_state):
    """
    One sklearn Pipeline: one-hot encode categoricals, pass numerical columns
    through, then fit the chosen classifier. Everything is inside a Pipeline
    so the encoder is fitted on training data only, automatically.
    """
    preprocessor = ColumnTransformer([
        ("categorical", OneHotEncoder(handle_unknown="ignore",
                                      sparse_output=False), categorical_cols),
        ("numerical", "passthrough", numerical_cols),
    ])

    if model_name == "random_forest":
        model_settings = config["models"]["random_forest"]
        model = RandomForestClassifier(
            n_estimators=model_settings["n_estimators"],
            random_state=random_state,
            n_jobs=-1,
        )
    elif model_name == "xgboost":
        model_settings = config["models"]["xgboost"]
        model = XGBClassifier(
            n_estimators=model_settings["n_estimators"],
            max_depth=model_settings["max_depth"],
            learning_rate=model_settings["learning_rate"],
            eval_metric="logloss",
            random_state=random_state,
            n_jobs=-1,
        )
    else:
        raise ValueError(f"Unknown model: {model_name}")

    return Pipeline([("preprocess", preprocessor), ("model", model)])


def train_and_evaluate(train_df, test_df, model_name, target, config,
                       random_state):
    """
    Train one model on train_df, evaluate on test_df, return all metrics.

    If the training set contains only one class, training is impossible —
    we return NaN metrics instead of crashing (this can happen for tiny
    seeds if stratification ever fails).
    """
    numerical_cols, categorical_cols = get_column_types(train_df, target)
    X_train = train_df.drop(columns=[target, "row_id"], errors="ignore")
    y_train = train_df[target]
    X_test = test_df.drop(columns=[target, "row_id"], errors="ignore")
    y_test = test_df[target]

    if y_train.nunique() < 2:
        print(f"WARNING: training data has only one class — skipping "
              f"{model_name}.")
        return {metric: float("nan") for metric in
                ["pr_auc", "roc_auc", "f1", "precision", "recall", "accuracy",
                 "train_time", "n_train_rows", "train_positive_rate"]}

    pipeline = build_pipeline(model_name, numerical_cols, categorical_cols,
                              config, random_state)
    import time
    start_time = time.time()
    pipeline.fit(X_train, y_train)
    train_time = time.time() - start_time

    probabilities = pipeline.predict_proba(X_test)[:, 1]
    predictions = (probabilities >= 0.5).astype(int)

    return {
        "pr_auc": round(float(average_precision_score(y_test, probabilities)), 4),
        "roc_auc": round(float(roc_auc_score(y_test, probabilities)), 4),
        "f1": round(float(f1_score(y_test, predictions)), 4),
        "precision": round(float(precision_score(y_test, predictions,
                                                 zero_division=0)), 4),
        "recall": round(float(recall_score(y_test, predictions)), 4),
        "accuracy": round(float(accuracy_score(y_test, predictions)), 4),
        "train_time": round(train_time, 2),
        "n_train_rows": int(len(train_df)),
        "train_positive_rate": round(float(y_train.mean()), 4),
    }


def main():
    """
    Demo WITHOUT touching the locked test set.

    We split the dev pool into a development-training part and a validation
    part, train TRTR-full and TRTR-seed on the training part only, and print
    validation metrics. The locked test set stays sealed until the full
    experiment runs with frozen settings.
    """
    from src.utils import load_locked_test  # noqa: F401 (NOT used!)
    config = load_config()
    target = config["dataset"]["target"]

    dev_pool = pd.read_csv(PROJECT_ROOT / "data" / "processed" / "dev_pool.csv")
    train_part, validation_part = train_test_split_stratified(
        dev_pool, target, test_fraction=0.2, random_state=0)
    seed_500 = pd.read_csv(PROJECT_ROOT / "data" / "seeds" / "rs0" / "seed_500.csv")

    print(f"Random-guess PR-AUC (positive prevalence in validation): "
          f"{validation_part[target].mean():.4f}\n")

    for model_name in ["random_forest", "xgboost"]:
        full_metrics = train_and_evaluate(
            train_part, validation_part, model_name, target, config, 0)
        seed_metrics = train_and_evaluate(
            seed_500, validation_part, model_name, target, config, 0)
        print(f"--- {model_name} (validation on real dev-pool holdout) ---")
        print(f"  TRTR-full : PR-AUC={full_metrics['pr_auc']} "
              f"ROC-AUC={full_metrics['roc_auc']}")
        print(f"  TRTR-seed : PR-AUC={seed_metrics['pr_auc']} "
              f"ROC-AUC={seed_metrics['roc_auc']}")


def train_test_split_stratified(df, target, test_fraction, random_state):
    """Small local stratified split helper (dev pool -> train/validation)."""
    from sklearn.model_selection import train_test_split
    train_part, validation_part = train_test_split(
        df, test_size=test_fraction, random_state=random_state,
        stratify=df[target])
    return train_part, validation_part


if __name__ == "__main__":
    main()
