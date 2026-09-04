#!/usr/bin/env python3

from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path("/projects/bentosprg6/linwx/research_project")

FULL_SOURCE = ROOT / "data/task5/task5_full_coverage.csv"
RANKED_SOURCE = ROOT / "data/task5/task5_affinity_ranked_final.csv"

OUTDIR = ROOT / "data/task6"
OUTDIR.mkdir(parents=True, exist_ok=True)

FULL_OUT = OUTDIR / "task6_finetune_manifest.csv"
PILOT_OUT = OUTDIR / "task6_pilot_manifest.csv"


# ============================================================
# FULL EXPERIMENTAL LABEL MANIFEST
# ============================================================

df = pd.read_csv(FULL_SOURCE)

df["kd_nm"] = pd.to_numeric(
    df["kd_nm"],
    errors="coerce",
)

# Build aptamer length robustly from whichever Task-5 field exists.
if "sequence_calculated_length" in df.columns:
    df["aptamer_length"] = pd.to_numeric(
        df["sequence_calculated_length"],
        errors="coerce",
    )
elif "reported_length" in df.columns:
    df["aptamer_length"] = pd.to_numeric(
        df["reported_length"],
        errors="coerce",
    )
else:
    df["aptamer_length"] = (
        df["sequence"]
        .fillna("")
        .astype(str)
        .str.len()
    )

usable = df[
    df["kd_nm"].notna()
    & (df["kd_nm"] > 0)
    & df["sequence"].notna()
    & df["target_normalized"].notna()
    & df["nucleic_acid_type"].isin(["DNA", "RNA"])
].copy()

# Boltz convention:
# log10(Kd expressed in micromolar)
usable["boltz_target"] = (
    np.log10(usable["kd_nm"]) - 3.0
)

# ============================================================
# GROUPED TRAIN / VALIDATION / TEST SPLIT
# Group by protein/target name to reduce leakage.
# ============================================================

targets = np.array(
    sorted(
        usable["target_normalized"]
        .dropna()
        .unique()
    )
)

rng = np.random.default_rng(42)
rng.shuffle(targets)

n_targets = len(targets)

n_train = int(n_targets * 0.80)
n_val = int(n_targets * 0.10)

train_targets = set(
    targets[:n_train]
)

val_targets = set(
    targets[n_train:n_train + n_val]
)

test_targets = set(
    targets[n_train + n_val:]
)


def assign_split(target):
    if target in train_targets:
        return "train"
    if target in val_targets:
        return "validation"
    return "test"


usable["split"] = (
    usable["target_normalized"]
    .map(assign_split)
)

keep = [
    "model_id",
    "aptamer_id",
    "serial_number",
    "aptamer_name",
    "nucleic_acid_type",
    "sequence",
    "aptamer_length",
    "target_normalized",
    "affinity_raw",
    "kd_nm",
    "boltz_target",
    "doi",
    "pubmed",
    "task5_target_validation",
    "task5_target_reason",
    "task5_result_status",
    "task5_coverage_group",
    "split",
]

keep = [
    c for c in keep
    if c in usable.columns
]

full_manifest = usable[
    keep
].copy()

full_manifest.to_csv(
    FULL_OUT,
    index=False,
)


# ============================================================
# PILOT SET
#
# Use successful Task-5 Boltz records because they already have:
#   - resolved protein target
#   - aptamer sequence
#   - target length
#   - YAML input
#   - experimental Kd
#
# This makes conversion to AF3 much easier.
# ============================================================

ranked = pd.read_csv(RANKED_SOURCE)

ranked["experimental_kd_nm"] = pd.to_numeric(
    ranked["experimental_kd_nm"],
    errors="coerce",
)

ranked["aptamer_length"] = pd.to_numeric(
    ranked["aptamer_length"],
    errors="coerce",
)

ranked["target_length"] = pd.to_numeric(
    ranked["target_length"],
    errors="coerce",
)

ranked["total_polymer_length"] = (
    ranked["aptamer_length"]
    + ranked["target_length"]
)

ranked["boltz_target"] = (
    np.log10(
        ranked["experimental_kd_nm"]
    )
    - 3.0
)

pilot_pool = ranked[
    ranked["experimental_kd_nm"].notna()
    & (ranked["experimental_kd_nm"] > 0)
    & ranked["sequence"].notna()
    & ranked["target_normalized"].notna()
    & ranked["nucleic_acid_type"].isin(["DNA", "RNA"])
    & (
        ranked["target_type"]
        .astype(str)
        .str.lower()
        == "protein"
    )
    & ranked["yaml_path"].notna()
    & ranked["total_polymer_length"].notna()
    & (ranked["total_polymer_length"] <= 384)
].copy()

# Bring the target-group split from the full manifest into
# the pilot records.
split_map = (
    full_manifest[
        ["target_normalized", "split"]
    ]
    .drop_duplicates(
        "target_normalized"
    )
    .set_index(
        "target_normalized"
    )["split"]
    .to_dict()
)

pilot_pool["split"] = (
    pilot_pool["target_normalized"]
    .map(split_map)
)

pilot_pool = pilot_pool[
    pilot_pool["split"].notna()
].copy()


def stratified_sample(pool, n_requested, seed):
    """Sample across the experimental affinity range."""

    if len(pool) <= n_requested:
        return pool.copy()

    pool = pool.copy()

    try:
        pool["affinity_bin"] = pd.qcut(
            pool["boltz_target"],
            q=min(
                8,
                pool["boltz_target"].nunique(),
            ),
            labels=False,
            duplicates="drop",
        )
    except ValueError:
        pool["affinity_bin"] = 0

    pieces = []

    groups = list(
        pool.groupby(
            "affinity_bin",
            dropna=False,
        )
    )

    if not groups:
        return pool.sample(
            n=n_requested,
            random_state=seed,
        )

    per_bin = max(
        1,
        int(
            np.ceil(
                n_requested / len(groups)
            )
        ),
    )

    for i, (_, group) in enumerate(groups):
        take = min(
            per_bin,
            len(group),
        )

        pieces.append(
            group.sample(
                n=take,
                random_state=seed + i,
            )
        )

    sampled = (
        pd.concat(
            pieces,
            ignore_index=False,
        )
        .drop_duplicates(
            "model_id"
        )
    )

    if len(sampled) > n_requested:
        sampled = sampled.sample(
            n=n_requested,
            random_state=seed,
        )

    elif len(sampled) < n_requested:
        remaining = pool[
            ~pool["model_id"].isin(
                sampled["model_id"]
            )
        ]

        extra_n = min(
            n_requested - len(sampled),
            len(remaining),
        )

        if extra_n:
            sampled = pd.concat(
                [
                    sampled,
                    remaining.sample(
                        n=extra_n,
                        random_state=seed + 100,
                    ),
                ]
            )

    return sampled


# 48 train / 8 validation / 8 test = 64 total pilot cases.
pilot_parts = []

requirements = [
    ("train", 48, 42),
    ("validation", 8, 142),
    ("test", 8, 242),
]

for split_name, requested, seed in requirements:

    subset = pilot_pool[
        pilot_pool["split"]
        == split_name
    ].copy()

    selected = stratified_sample(
        subset,
        requested,
        seed,
    )

    pilot_parts.append(selected)

pilot = (
    pd.concat(
        pilot_parts,
        ignore_index=True,
    )
    .drop_duplicates(
        "model_id"
    )
)

pilot = pilot.sort_values(
    ["split", "boltz_target"]
).reset_index(drop=True)

pilot.to_csv(
    PILOT_OUT,
    index=False,
)


# ============================================================
# QC
# ============================================================

print("===== FULL TASK 6 MANIFEST =====")
print("Rows:", len(full_manifest))
print(
    "Unique aptamers:",
    full_manifest["aptamer_id"].nunique(),
)
print(
    "Unique targets:",
    full_manifest["target_normalized"].nunique(),
)

print()
print("Split rows:")
print(
    full_manifest["split"]
    .value_counts()
    .to_string()
)

print()
print("Targets by split:")
print(
    full_manifest.groupby("split")[
        "target_normalized"
    ]
    .nunique()
    .to_string()
)

print()
print("Boltz target range:")
print(
    full_manifest["boltz_target"]
    .describe()
    .to_string()
)

print()
print("===== PILOT SOURCE POOL =====")
print("Eligible Task-5 successes:", len(pilot_pool))
print(
    "Eligible targets:",
    pilot_pool[
        "target_normalized"
    ].nunique(),
)

print()
print("===== PILOT MANIFEST =====")
print("Pilot rows:", len(pilot))

print()
print("Pilot split counts:")
print(
    pilot["split"]
    .value_counts()
    .to_string()
)

print()
print(
    "Pilot affinity range:",
    float(pilot["boltz_target"].min()),
    "to",
    float(pilot["boltz_target"].max()),
)

print(
    "Pilot max total polymer length:",
    float(
        pilot[
            "total_polymer_length"
        ].max()
    ),
)

print(
    "Pilot YAML files:",
    int(
        pilot["yaml_path"]
        .notna()
        .sum()
    ),
)

print()
print("Saved:")
print(FULL_OUT)
print(PILOT_OUT)
