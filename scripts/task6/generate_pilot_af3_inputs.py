#!/usr/bin/env python3

import json
import re
from pathlib import Path

import pandas as pd


ROOT = Path("/projects/bentosprg6/linwx/research_project")

PILOT = ROOT / "data/task6/task6_pilot_manifest.csv"
OUTDIR = ROOT / "data/task6/af3_inputs"
MAP_OUT = ROOT / "data/task6/task6_af3_input_manifest.csv"

OUTDIR.mkdir(parents=True, exist_ok=True)


def clean_name(text):
    text = str(text).strip().lower()
    text = re.sub(r"[^a-z0-9_-]+", "_", text)
    text = re.sub(r"_+", "_", text)
    return text.strip("_")


df = pd.read_csv(PILOT)

print("===== TASK 6: GENERATE AF3 PILOT INPUTS =====")
print("Pilot rows:", len(df))

required = [
    "model_id",
    "protein_sequence",
    "aptamer_sequence_yaml",
    "yaml_nucleic_acid_type",
    "split",
    "kd_nm",
    "boltz_target",
    "total_polymer_length",
]

missing = [
    c for c in required
    if c not in df.columns
]

if missing:
    raise RuntimeError(
        f"Pilot manifest missing columns: {missing}"
    )

if len(df) != 64:
    raise RuntimeError(
        f"Expected 64 pilot rows, found {len(df)}"
    )

if df["model_id"].nunique() != 64:
    raise RuntimeError(
        "Pilot model_id values are not unique."
    )

# Remove only JSONs created by this script on a prior run.
for old in OUTDIR.glob("task6_*.json"):
    old.unlink()

records = []

for idx, row in df.reset_index(drop=True).iterrows():

    model_id = str(row["model_id"])

    protein_seq = (
        str(row["protein_sequence"])
        .replace(" ", "")
        .replace("\n", "")
        .upper()
    )

    aptamer_seq = (
        str(row["aptamer_sequence_yaml"])
        .replace(" ", "")
        .replace("\n", "")
        .upper()
    )

    na_type = (
        str(row["yaml_nucleic_acid_type"])
        .strip()
        .upper()
    )

    if na_type not in {"DNA", "RNA"}:
        raise RuntimeError(
            f"{model_id}: unsupported nucleic acid type "
            f"{na_type!r}"
        )

    if not protein_seq:
        raise RuntimeError(
            f"{model_id}: empty protein sequence"
        )

    if not aptamer_seq:
        raise RuntimeError(
            f"{model_id}: empty aptamer sequence"
        )

    # Basic canonical alphabet checks.
    protein_allowed = set(
        "ACDEFGHIKLMNPQRSTVWYBXZJUO"
    )

    bad_protein = (
        set(protein_seq) - protein_allowed
    )

    if bad_protein:
        raise RuntimeError(
            f"{model_id}: unsupported protein characters "
            f"{sorted(bad_protein)}"
        )

    if na_type == "DNA":
        bad_na = set(aptamer_seq) - set("ACGTN")
    else:
        bad_na = set(aptamer_seq) - set("ACGUN")

    if bad_na:
        raise RuntimeError(
            f"{model_id}: unsupported {na_type} characters "
            f"{sorted(bad_na)}"
        )

    input_name = (
        f"task6_{idx:04d}_"
        f"{clean_name(model_id)}"
    )

    protein_entity = {
        "protein": {
            "id": "A",
            "sequence": protein_seq,
            "unpairedMsa": "",
            "pairedMsa": "",
            "templates": [],
        }
    }

    if na_type == "DNA":
        aptamer_entity = {
            "dna": {
                "id": "B",
                "sequence": aptamer_seq,
            }
        }
    else:
        aptamer_entity = {
            "rna": {
                "id": "B",
                "sequence": aptamer_seq,
            }
        }

    af3_input = {
        "name": input_name,
        "modelSeeds": [42],
        "sequences": [
            protein_entity,
            aptamer_entity,
        ],
        "dialect": "alphafold3",
        "version": 4,
    }

    json_path = (
        OUTDIR / f"{input_name}.json"
    )

    with open(json_path, "w") as f:
        json.dump(
            af3_input,
            f,
            indent=2,
        )

    observed_total = (
        len(protein_seq)
        + len(aptamer_seq)
    )

    expected_total = int(
        row["total_polymer_length"]
    )

    if observed_total != expected_total:
        raise RuntimeError(
            f"{model_id}: sequence length mismatch: "
            f"JSON={observed_total}, "
            f"manifest={expected_total}"
        )

    records.append(
        {
            "task6_index": idx,
            "input_name": input_name,
            "json_path": str(json_path),
            "model_id": model_id,
            "aptamer_id": row.get(
                "aptamer_id",
                "",
            ),
            "target_normalized": row.get(
                "target_normalized",
                "",
            ),
            "nucleic_acid_type": na_type,
            "protein_length": len(
                protein_seq
            ),
            "aptamer_length": len(
                aptamer_seq
            ),
            "total_polymer_length": (
                observed_total
            ),
            "kd_nm": float(
                row["kd_nm"]
            ),
            "boltz_target": float(
                row["boltz_target"]
            ),
            "split": row["split"],
        }
    )


manifest = pd.DataFrame(records)

manifest.to_csv(
    MAP_OUT,
    index=False,
)

print()
print("===== GENERATION QC =====")
print("JSON files:", len(records))
print(
    "Unique input names:",
    manifest["input_name"].nunique(),
)
print(
    "Unique model IDs:",
    manifest["model_id"].nunique(),
)

print()
print("Split counts:")
print(
    manifest["split"]
    .value_counts()
    .to_string()
)

print()
print("Nucleic acid types:")
print(
    manifest["nucleic_acid_type"]
    .value_counts()
    .to_string()
)

print()
print(
    "Length range:",
    int(
        manifest[
            "total_polymer_length"
        ].min()
    ),
    "to",
    int(
        manifest[
            "total_polymer_length"
        ].max()
    ),
)

print()
print("Input directory:")
print(OUTDIR)

print()
print("Mapping manifest:")
print(MAP_OUT)

print()
print("===== SUCCESS =====")
print(
    "64 AF3 no-MSA pilot inputs generated."
)
