export type Model = "xgboost" | "random_forest";
export type Metric = "pr_auc" | "roc_auc" | "f1" | "precision" | "recall" | "accuracy";
export type Metrics = Record<Metric, number>;
export interface Health {
  status: string;
  tests_passing: number;
  tests_total: number;
  version: string;
}
export interface Feature {
  name: string;
  type: string;
  unique: number;
  missing_pct: number;
}
export interface Overview {
  rows_raw: number;
  duplicates_removed: number;
  rows_clean: number;
  dev_rows: number;
  test_rows: number;
  positive_rate_dev: number;
  positive_rate_test: number;
  n_features: number;
  n_numerical: number;
  n_categorical: number;
  features: Feature[];
  locked: { sha256: string; verified: boolean; created: string; n_rows: number };
}
export interface Distribution {
  type: string;
  bins?: (string | number)[];
  categories?: string[];
  counts: number[];
}
export interface Seeds {
  sizes: number[];
  random_seeds: number[];
  nested_ok: boolean;
  class_counts: Record<string, { majority: number; minority: number }>;
}
export interface Summary {
  arm: string;
  seed_size: number;
  metric_means: Metrics;
  metric_stds: Metrics;
  n_runs: number;
}
export interface Reference {
  trtr_full: Record<Model, Metrics>;
  random_guess_pr_auc: number;
}
export interface Fidelity {
  overall: {
    generator: string;
    seed_size: number;
    ks_mean: number;
    tvd_mean: number;
    chi2_mean: number;
    corr_diff: number;
    coverage: number;
    stds: Record<string, number>;
  }[];
  per_column: {
    generator: string;
    seed_size: number;
    column: string;
    metric: string;
    value: number;
  }[];
}
export interface Privacy {
  generator: string;
  seed_size: number;
  dcr_median_synth: number;
  dcr_median_holdout: number;
  dcr_ratio: number;
  exact_rate_synth: number;
  exact_rate_holdout: number;
  p05_synth: number;
  p05_holdout: number;
}
export interface Histogram {
  bins: (string | number)[];
  synthetic: number[];
  holdout: number[];
}
export interface Runtime {
  generator: string;
  seed_size: number;
  seconds_mean: number;
  seconds_std: number;
}
export interface LossCurve {
  step: number[];
  generator_loss: number[];
  discriminator_loss: number[];
}
export interface Criteria {
  rules: { id: string; label: string; threshold: number; direction: string }[];
  evaluations: {
    generator: string;
    seed_size: number;
    values: Record<string, number>;
    passed: Record<string, boolean>;
    all_passed: boolean;
  }[];
  min_useful_seed: number | null;
  explanation: string;
}
export interface Comparison {
  type: string;
  bins?: (string | number)[];
  categories?: string[];
  real: number[];
  synthetic: number[];
}
export interface Correlation {
  columns: string[];
  real: number[][];
  synthetic: number[][];
  diff: number[][];
}
export interface Projection {
  real: [number, number][];
  synthetic: [number, number][];
  explained_variance: number[];
}
export interface Upload {
  dataset_id: string;
  profile: {
    rows: number;
    columns: string[];
    types: Record<string, string>;
    missing: Record<string, number>;
    warnings: string[];
    class_balance: Record<string, number>;
  };
}
export interface Job {
  status: string;
  stage: string;
  progress: number;
  log: string[];
  result: unknown | null;
  error: string | null;
}
