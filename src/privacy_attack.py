"""
privacy_attack.py — EXTENSION (not part of the frozen criteria).

A simple distance-based membership inference check, labelled as an
EXTENSION: it is NOT a privacy guarantee and NOT part of the frozen
criteria.yaml rules.

The idea: if the generator memorised its training seed, the seed's own
records should sit unusually CLOSE to the synthetic data — closer than
genuine records the generator never saw. So we compute, for every seed row,
the distance to its nearest synthetic row, and the same for every holdout
row. If seed records are systematically closer, membership is easier to
infer.

We report the result as an AUC: the probability that a randomly chosen seed
record is closer to the synthetic data than a randomly chosen holdout record
(the Mann-Whitney statistic). AUC = 0.5 means no separation (no attack
signal); AUC near 1.0 means seed records are systematically closer — a
memorisation warning. Like DCR, this is a diagnostic only.
"""

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import OneHotEncoder, StandardScaler


def membership_inference_auc(seed_df, synth_df, holdout_df, numerical_cols,
                             categorical_cols):
    """
    Distance-based membership inference check, reported as AUC.

    Returns a dict with:
      attack_auc        — probability a random seed record sits closer to the
                          synthetic data than a random holdout record
                          (Mann-Whitney formulation; 0.5 = no separation).
      seed_median_dist, holdout_median_dist — the two distance distributions.
      n_seed, n_holdout — sizes of the two compared groups.
    """
    seed_features = seed_df.drop(columns=["row_id"], errors="ignore")
    synth_features = synth_df.drop(columns=["row_id"], errors="ignore")
    holdout_features = holdout_df.drop(columns=["row_id"], errors="ignore")

    # preprocessing is fitted on the SEED ONLY, matching the DCR check
    preprocessor = ColumnTransformer([
        ("numerical", StandardScaler(), numerical_cols),
        ("categorical", OneHotEncoder(handle_unknown="ignore",
                                      sparse_output=False), categorical_cols),
    ])
    seed_transformed = preprocessor.fit_transform(seed_features)
    synth_transformed = preprocessor.transform(synth_features)
    holdout_transformed = preprocessor.transform(holdout_features)

    # distance from every seed row to its NEAREST SYNTHETIC row
    neighbours = NearestNeighbors(n_neighbors=1)
    neighbours.fit(synth_transformed)
    seed_distances = neighbours.kneighbors(seed_transformed)[0].flatten()
    holdout_distances = neighbours.kneighbors(holdout_transformed)[0].flatten()

    # Mann-Whitney: fraction of (seed, holdout) pairs where the seed record
    # is strictly closer to the synthetic data (ties count half)
    seed_broad = seed_distances[:, None]
    holdout_broad = holdout_distances[None, :]
    wins = (seed_broad < holdout_broad).sum()
    ties = (seed_broad == holdout_broad).sum()
    pair_count = len(seed_distances) * len(holdout_distances)
    auc = float((wins + 0.5 * ties) / pair_count)

    return {
        "attack_auc": round(auc, 4),
        "seed_median_dist": round(float(np.median(seed_distances)), 4),
        "holdout_median_dist": round(float(np.median(holdout_distances)), 4),
        "n_seed": int(len(seed_distances)),
        "n_holdout": int(len(holdout_distances)),
    }
