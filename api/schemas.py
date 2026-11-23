"""
Pydantic schemas for every response, mirroring the frontend contract in
frontend/web/src/api/types.ts EXACTLY (field names and shapes).

The contract is the source of truth: if a shape disagrees, fix the backend,
never the contract.
"""

from typing import Any, Dict, List, Literal, Optional, Union

from pydantic import BaseModel, Field

# ---------------------------------------------------------------- health ---


class Health(BaseModel):
    status: str
    version: str
    tests_passing: int
    tests_total: int


# -------------------------------------------------------------- overview ---


class FeatureInfo(BaseModel):
    name: str
    type: str
    unique: int
    missing_pct: float


class LockedInfo(BaseModel):
    sha256: str
    verified: bool
    created: str
    n_rows: int


class Overview(BaseModel):
    rows_raw: int
    duplicates_removed: int
    rows_clean: int
    dev_rows: int
    test_rows: int
    positive_rate_dev: float
    positive_rate_test: float
    n_features: int
    n_numerical: int
    n_categorical: int
    features: List[FeatureInfo]
    locked: LockedInfo


# ------------------------------------------------------------------ seeds ---


class Seeds(BaseModel):
    sizes: List[int]
    random_seeds: List[int]
    nested_ok: bool
    class_counts: Dict[str, Dict[str, int]]  # size -> {majority, minority}


# ---------------------------------------------------------------- results ---


class SummaryItem(BaseModel):
    arm: str
    seed_size: int
    metric_means: Dict[str, float]
    metric_stds: Dict[str, float]
    n_runs: int


class Reference(BaseModel):
    trtr_full: Dict[str, Dict[str, float]]  # model -> metrics
    random_guess_pr_auc: float


class FidelityOverallItem(BaseModel):
    generator: str
    seed_size: int
    ks_mean: float
    tvd_mean: float
    chi2_mean: float
    corr_diff: float
    coverage: float
    stds: Dict[str, float]


class FidelityPerColumnItem(BaseModel):
    generator: str
    seed_size: int
    column: str
    metric: str
    value: float


class Fidelity(BaseModel):
    overall: List[FidelityOverallItem]
    per_column: List[FidelityPerColumnItem]


class PrivacyItem(BaseModel):
    generator: str
    seed_size: int
    dcr_median_synth: float
    dcr_median_holdout: float
    dcr_ratio: float
    exact_rate_synth: float
    exact_rate_holdout: float
    p05_synth: float
    p05_holdout: float


class DcrHistogram(BaseModel):
    bins: List[float]
    synthetic: List[int]
    holdout: List[int]


class LossCurve(BaseModel):
    step: List[int]
    generator_loss: List[float]
    discriminator_loss: List[float]


class RuntimeItem(BaseModel):
    generator: str
    seed_size: int
    seconds_mean: float
    seconds_std: float


class CriteriaRule(BaseModel):
    id: str
    label: str
    threshold: float
    direction: str  # "<=" or ">="


class CriteriaEvaluation(BaseModel):
    generator: str
    seed_size: int
    values: Dict[str, float]
    passed: Dict[str, bool]
    all_passed: bool


class Criteria(BaseModel):
    rules: List[CriteriaRule]
    evaluations: List[CriteriaEvaluation]
    min_useful_seed: Optional[int]
    explanation: str


# ---------------------------------------------------------------- compare ---


class Distribution(BaseModel):
    type: str
    bins: Optional[List[Union[str, float]]] = None
    categories: Optional[List[str]] = None
    counts: List[int]


class Correlation(BaseModel):
    columns: List[str]
    real: List[List[float]]
    synthetic: List[List[float]]
    diff: List[List[float]]


class Projection(BaseModel):
    real: List[List[float]]
    synthetic: List[List[float]]
    explained_variance: List[float]


# -------------------------------------------------------------------- lab ---


class Upload(BaseModel):
    dataset_id: str
    profile: Dict[str, Any]  # profile_dataset() output, verbatim


class Comparison(BaseModel):
    type: str
    bins: Optional[List[Union[str, float]]] = None
    categories: Optional[List[str]] = None
    real: List[float]
    synthetic: List[float]
class LabRunRequest(BaseModel):
    dataset_id: str
    target: str
    seed_size: int = Field(ge=50, le=100000)
    generator: Literal["CTGAN", "GaussianCopula", "SMOTE-NC"]
    synthetic_rows: int = Field(ge=100, le=100000)
    ctgan_steps: int = Field(ge=100, le=3000)


class Job(BaseModel):
    status: str  # queued|running|completed|failed
    stage: str
    progress: int
    log: List[str]
    result: Optional[Dict[str, Any]]
    error: Optional[str]
