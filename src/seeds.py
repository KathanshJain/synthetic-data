"""
seeds.py — Step 4: build nested stratified seed samples from the dev pool.

Why this file exists: our research question compares seed SIZES. To make that
comparison fair, the seeds must be
  * NESTED — the 100-row seed is inside the 250-row seed, which is inside the
    500-row seed, and so on. Think of Russian dolls: every size looks at the
    SAME rows, just more of them, so differences come from sample size alone.
  * STRATIFIED — each seed keeps the real income-class ratio, so a small seed
    does not accidentally contain almost no minority-class rows.

We also carve out a 'holdout' of real rows NOT in any seed: it is the honest
baseline for the privacy (DCR) check — genuine new records that the generator
never saw.
"""

import numpy as np
import pandas as pd

from src.utils import PROJECT_ROOT, get_column_types, load_config


def build_nested_seeds(dev_pool, target, seed_sizes, random_state):
    """
    Build one nested, stratified seed per requested size.

    How it works (simple version):
      1. For each class, shuffle that class's row indices ONCE with a fixed
         random generator (numpy default_rng(random_state)).
      2. Walk the sizes from smallest to largest. For each size n, compute
         how many rows of each class we need (n x class proportion), rounded,
         then adjusted so the counts add up to exactly n and NEVER decrease
         from the previous size (that is what makes the seeds nested).
      3. A seed is simply the FIRST k shuffled rows of each class.

    Returns {size: DataFrame} with row_id kept (row_id is our audit trail).
    """
    rng = np.random.default_rng(random_state)

    # class proportions, e.g. {0: 0.76, 1: 0.24}
    class_counts = dev_pool[target].value_counts()
    proportions = {cls: count / len(dev_pool)
                   for cls, count in class_counts.items()}

    # shuffle each class's row indices once — every seed reuses this order
    shuffled_indices = {}
    for cls in class_counts.index:
        cls_indices = dev_pool.index[dev_pool[target] == cls].to_numpy()
        rng.shuffle(cls_indices)
        shuffled_indices[cls] = cls_indices

    seeds = {}
    previous_counts = {cls: 0 for cls in proportions}
    for size in sorted(seed_sizes):  # smallest to largest
        # raw per-class counts, rounded
        counts = {cls: int(round(size * p)) for cls, p in proportions.items()}
        # fix rounding so counts sum exactly to size
        difference = size - sum(counts.values())
        largest_class = max(proportions, key=proportions.get)
        counts[largest_class] += difference
        # nesting: counts may never shrink vs the previous size
        for cls in counts:
            counts[cls] = max(counts[cls], previous_counts[cls])
        # (if the max() pushed us over 'size', trim the largest class back)
        overflow = sum(counts.values()) - size
        if overflow > 0:
            counts[largest_class] -= overflow

        selected = []
        for cls, count in counts.items():
            selected.extend(shuffled_indices[cls][:count])
        seed_df = dev_pool.loc[selected].copy()
        seeds[size] = seed_df
        previous_counts = counts
    return seeds


def check_nested(seeds):
    """True if every smaller seed's row_ids are inside the next larger seed."""
    sizes = sorted(seeds.keys())
    for smaller, larger in zip(sizes, sizes[1:]):
        smaller_ids = set(seeds[smaller]["row_id"])
        larger_ids = set(seeds[larger]["row_id"])
        if not smaller_ids.issubset(larger_ids):
            return False
    return True


def get_holdout(dev_pool, largest_seed, size, random_state):
    """
    Pick 'size' random rows from the dev pool that are NOT in the largest
    seed. These genuine, unseen-by-the-generator rows are the baseline for
    the DCR privacy check.
    """
    seed_ids = set(largest_seed["row_id"])
    remaining = dev_pool[~dev_pool["row_id"].isin(seed_ids)]
    return remaining.sample(n=size, random_state=random_state)


def main():
    config = load_config()
    target = config["dataset"]["target"]
    dev_pool = pd.read_csv(PROJECT_ROOT / "data" / "processed" / "dev_pool.csv")

    print(f"{'rs':>3} {'size':>6} {'class0':>7} {'class1':>7} {'minority%':>10}")
    for random_seed in config["seeds"]["random_seeds"]:
        seeds = build_nested_seeds(
            dev_pool, target,
            config["seeds"]["seed_sizes"], random_seed)

        assert check_nested(seeds), f"nesting broken for random seed {random_seed}"

        seeds_dir = PROJECT_ROOT / "data" / "seeds" / f"rs{random_seed}"
        seeds_dir.mkdir(parents=True, exist_ok=True)
        for size, seed_df in seeds.items():
            seed_df.to_csv(seeds_dir / f"seed_{size}.csv", index=False)
            minority_pct = 100 * seed_df[target].mean()
            class0 = int((seed_df[target] == 0).sum())
            class1 = int((seed_df[target] == 1).sum())
            print(f"{random_seed:>3} {size:>6} {class0:>7} {class1:>7} "
                  f"{minority_pct:>9.1f}%")


if __name__ == "__main__":
    main()
