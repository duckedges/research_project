#!/usr/bin/env python3

from pathlib import Path
import numpy as np
import pandas as pd
import yaml

ROOT = Path("/projects/bentosprg6/linwx/research_project")

ORIGINAL = ROOT / "data/task5/task5_run_manifest.csv"
RECOVERY = ROOT / "data/task5/task5_recovery_manifest.csv"
FULL_SPLITS = ROOT / "data/task6/task6_finetune_manifest.csv"

OUT = ROOT / "data/task6/task6_pilot_manifest.csv"

# ============================================================
# LOAD TASK-5 PREPARED INPUTS
# ============================================================

orig = pd.read_csv(ORIGINAL)
rec = pd.read_csv(RECOVERY)

orig["task5_source"] = "original"
rec["task5_source"] = "recovery"

common = sorted(set(orig.columns) | set(rec.columns))

orig = orig.reindex(columns=common)
rec = rec.reindex(columns=common)

df = pd.concat(
    [orig, rec],
    ignore_index=True,
)

print("===== TASK 6 PILOT-SPECIFIC GROUPED SPLIT =====")
print("Original prepared:", len(orig))
print("Recovery prepared:", len(rec))
print("Combined:", len(df))

# Keep the original full-manifest split only as provenance.
full = pd.read_csv(FULL_SPLITS)

full_split_map = (
    full[
        ["target_normalized", "split"]
    ]
    .dropna()
    .drop_duplicates("target_normalized")
    .set_index("target_normalized")["split"]
    .to_dict()
)

df["full_manifest_split"] = (
    df["target_normalized"]
    .map(full_split_map)
)

# ============================================================
# EXPERIMENTAL KD
# ============================================================

kd_col = None

for c in ["experimental_kd_nm", "kd_nm"]:
    if c in df.columns:
        kd_col = c
        break

if kd_col is None:
    raise RuntimeError("No experimental Kd column found.")

df["kd_nm"] = pd.to_numeric(
    df[kd_col],
    errors="coerce",
)

df["boltz_target"] = (
    np.log10(df["kd_nm"]) - 3.0
)

# ============================================================
# PARSE ACTUAL TASK-5 YAML SEQUENCES
# ============================================================

def parse_yaml(path_str):
    out = {
        "yaml_exists": False,
        "protein_sequence": None,
        "aptamer_sequence_yaml": None,
        "yaml_nucleic_acid_type": None,
        "protein_length": np.nan,
        "aptamer_length_actual": np.nan,
    }

    if pd.isna(path_str):
        return out

    path = Path(str(path_str))

    if not path.is_file():
        return out

    out["yaml_exists"] = True

    try:
        with open(path) as f:
            data = yaml.safe_load(f)
    except Exception:
        return out

    for entity in data.get("sequences", []):

        if "protein" in entity:
            seq = str(
                entity["protein"].get("sequence", "")
            )
            seq = "".join(seq.split()).upper()

            if seq:
                out["protein_sequence"] = seq
                out["protein_length"] = len(seq)

        elif "rna" in entity:
            seq = str(
                entity["rna"].get("sequence", "")
            )
            seq = "".join(seq.split()).upper()

            if seq:
                out["aptamer_sequence_yaml"] = seq
                out["aptamer_length_actual"] = len(seq)
                out["yaml_nucleic_acid_type"] = "RNA"

        elif "dna" in entity:
            seq = str(
                entity["dna"].get("sequence", "")
            )
            seq = "".join(seq.split()).upper()

            if seq:
                out["aptamer_sequence_yaml"] = seq
                out["aptamer_length_actual"] = len(seq)
                out["yaml_nucleic_acid_type"] = "DNA"

    return out


parsed = pd.DataFrame(
    df["yaml_path"].apply(parse_yaml).tolist()
)

df = pd.concat(
    [
        df.reset_index(drop=True),
        parsed.reset_index(drop=True),
    ],
    axis=1,
)

df["total_polymer_length"] = (
    df["protein_length"]
    + df["aptamer_length_actual"]
)

valid = df[
    df["yaml_exists"]
    & df["protein_sequence"].notna()
    & df["aptamer_sequence_yaml"].notna()
    & df["kd_nm"].notna()
    & (df["kd_nm"] > 0)
    & df["target_normalized"].notna()
].copy()

print()
print("===== PREPARED QC =====")
print("YAMLs found:", int(df["yaml_exists"].sum()))
print(
    "Protein complexes recovered:",
    int(df["protein_sequence"].notna().sum()),
)
print("Valid labeled protein complexes:", len(valid))
print(
    "Unique targets:",
    valid["target_normalized"].nunique(),
)

# ============================================================
# PILOT TARGET GROUP PARTITION
#
# We independently group the PILOT by target.
# No target may occur in more than one pilot split.
#
# Search for the smallest length cutoff that can support:
# 48 train / 8 validation / 8 test.
# ============================================================

REQUEST = {
    "train": 48,
    "validation": 8,
    "test": 8,
}

CUTOFFS = [
    384,
    512,
    640,
    768,
    1024,
]

rng = np.random.default_rng(42)


def find_group_partition(pool, tries=20000):
    """
    Randomized target-group partition search.

    Returns target sets for train/validation/test while requiring
    at least 48/8/8 available rows and zero target overlap.
    """

    counts = (
        pool.groupby("target_normalized")
        .size()
        .to_dict()
    )

    targets = np.array(
        list(counts.keys()),
        dtype=object,
    )

    if len(targets) < 3:
        return None

    best = None
    best_score = None

    for _ in range(tries):

        perm = rng.permutation(targets)

        val_targets = []
        test_targets = []

        val_rows = 0
        test_rows = 0

        i = 0

        # Build validation target group.
        while i < len(perm) and val_rows < REQUEST["validation"]:
            t = perm[i]
            val_targets.append(t)
            val_rows += counts[t]
            i += 1

        # Build test target group.
        while i < len(perm) and test_rows < REQUEST["test"]:
            t = perm[i]
            test_targets.append(t)
            test_rows += counts[t]
            i += 1

        train_targets = list(perm[i:])

        train_rows = sum(
            counts[t]
            for t in train_targets
        )

        if (
            val_rows >= REQUEST["validation"]
            and test_rows >= REQUEST["test"]
            and train_rows >= REQUEST["train"]
        ):
            # Prefer minimal excess validation/test rows.
            score = (
                abs(val_rows - REQUEST["validation"])
                + abs(test_rows - REQUEST["test"])
            )

            if best is None or score < best_score:
                best = {
                    "train": set(train_targets),
                    "validation": set(val_targets),
                    "test": set(test_targets),
                    "available_rows": {
                        "train": train_rows,
                        "validation": val_rows,
                        "test": test_rows,
                    },
                }
                best_score = score

                if score == 0:
                    break

    return best


chosen_cutoff = None
partition = None

print()
print("===== PILOT GROUP-PARTITION SEARCH =====")

for cutoff in CUTOFFS:

    pool = valid[
        valid["total_polymer_length"]
        <= cutoff
    ].copy()

    result = find_group_partition(pool)

    print(
        f"cutoff={cutoff}",
        f"rows={len(pool)}",
        f"targets={pool['target_normalized'].nunique()}",
        "partition=",
        None if result is None else result["available_rows"],
    )

    if result is not None:
        chosen_cutoff = cutoff
        partition = result
        break

if partition is None:
    raise RuntimeError(
        "Could not construct a target-separated 48/8/8 pilot "
        "through a 1024-residue cutoff."
    )

print()
print("Chosen cutoff:", chosen_cutoff)
print(
    "Available rows after target grouping:",
    partition["available_rows"],
)

pool = valid[
    valid["total_polymer_length"]
    <= chosen_cutoff
].copy()

# Pilot-specific split.
pool["split"] = None

for split_name in [
    "train",
    "validation",
    "test",
]:
    mask = pool["target_normalized"].isin(
        partition[split_name]
    )
    pool.loc[mask, "split"] = split_name

pool = pool[
    pool["split"].notna()
].copy()

# ============================================================
# STRATIFIED ROW SAMPLING WITHIN TARGET-SEPARATED POOLS
# ============================================================

def sample_affinity(frame, requested, seed):

    frame = frame.copy()

    if len(frame) < requested:
        raise RuntimeError(
            f"Need {requested}; only {len(frame)} rows."
        )

    n_unique = frame["boltz_target"].nunique()

    if n_unique >= 2:
        frame["affinity_bin"] = pd.qcut(
            frame["boltz_target"],
            q=min(8, n_unique),
            labels=False,
            duplicates="drop",
        )
    else:
        frame["affinity_bin"] = 0

    groups = list(
        frame.groupby(
            "affinity_bin",
            dropna=False,
        )
    )

    per_bin = max(
        1,
        int(np.ceil(requested / len(groups))),
    )

    pieces = []

    for i, (_, g) in enumerate(groups):
        pieces.append(
            g.sample(
                n=min(per_bin, len(g)),
                random_state=seed + i,
            )
        )

    selected = (
        pd.concat(pieces)
        .drop_duplicates("model_id")
    )

    if len(selected) > requested:
        selected = selected.sample(
            n=requested,
            random_state=seed + 100,
        )

    if len(selected) < requested:

        remaining = frame[
            ~frame["model_id"].isin(
                selected["model_id"]
            )
        ]

        selected = pd.concat(
            [
                selected,
                remaining.sample(
                    n=requested - len(selected),
                    random_state=seed + 200,
                ),
            ]
        )

    return selected


pieces = []

for split_name, requested, seed in [
    ("train", 48, 42),
    ("validation", 8, 142),
    ("test", 8, 242),
]:

    frame = pool[
        pool["split"] == split_name
    ]

    pieces.append(
        sample_affinity(
            frame,
            requested,
            seed,
        )
    )

pilot = pd.concat(
    pieces,
    ignore_index=True,
)

pilot["pilot_length_cutoff"] = chosen_cutoff

pilot = pilot.sort_values(
    ["split", "boltz_target"]
).reset_index(drop=True)

# ============================================================
# FINAL LEAKAGE CHECKS
# ============================================================

train_targets = set(
    pilot.loc[
        pilot["split"] == "train",
        "target_normalized",
    ]
)

val_targets = set(
    pilot.loc[
        pilot["split"] == "validation",
        "target_normalized",
    ]
)

test_targets = set(
    pilot.loc[
        pilot["split"] == "test",
        "target_normalized",
    ]
)

assert not (train_targets & val_targets)
assert not (train_targets & test_targets)
assert not (val_targets & test_targets)

assert len(pilot) == 64
assert (pilot["split"] == "train").sum() == 48
assert (pilot["split"] == "validation").sum() == 8
assert (pilot["split"] == "test").sum() == 8

pilot.to_csv(
    OUT,
    index=False,
)

print()
print("===== FINAL PILOT =====")
print("Rows:", len(pilot))

print()
print("Split counts:")
print(
    pilot["split"]
    .value_counts()
    .to_string()
)

print()
print("Targets per split:")
print(
    pilot.groupby("split")[
        "target_normalized"
    ]
    .nunique()
    .to_string()
)

print()
print(
    "Target overlap train/validation:",
    len(train_targets & val_targets),
)
print(
    "Target overlap train/test:",
    len(train_targets & test_targets),
)
print(
    "Target overlap validation/test:",
    len(val_targets & test_targets),
)

print()
print(
    "Length range:",
    int(pilot["total_polymer_length"].min()),
    "to",
    int(pilot["total_polymer_length"].max()),
)

print(
    "Affinity range:",
    float(pilot["boltz_target"].min()),
    "to",
    float(pilot["boltz_target"].max()),
)

print()
print("Nucleic acid types:")
print(
    pilot["yaml_nucleic_acid_type"]
    .value_counts()
    .to_string()
)

print()
print(
    "Original Task-5 cases:",
    int((pilot["task5_source"] == "original").sum()),
)

print(
    "Recovery Task-5 cases:",
    int((pilot["task5_source"] == "recovery").sum()),
)

print()
print("Saved:", OUT)

print()
print("===== SUCCESS =====")
