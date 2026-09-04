#!/usr/bin/env python3

import csv
import json
import re
from pathlib import Path
from collections import Counter

PROJECT = Path("/projects/bentosprg6/linwx/research_project")

MODEL_MANIFEST = Path(
    "/projects/bentosprg6/linwx/ra_project/task2_utexas/"
    "data/processed/model_manifest.csv"
)

TARGETS = PROJECT / "data/task5/targets/target_validated.csv"

OUTDIR = PROJECT / "data/task5/recovery_inputs"
MANIFEST_OUT = PROJECT / "data/task5/task5_recovery_manifest.csv"

OUTDIR.mkdir(parents=True, exist_ok=True)


def read_csv(path):
    with open(path, newline="", encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))


def clean_sequence(seq):
    return re.sub(r"\s+", "", seq or "").upper()


def q(s):
    return json.dumps(s, ensure_ascii=False)


manifest = read_csv(MODEL_MANIFEST)
targets = read_csv(TARGETS)

accepted = {
    (r.get("target_normalized") or "").strip(): r
    for r in targets
    if (r.get("validation_status") or "").strip() == "ACCEPT"
}

candidates = []

for r in manifest:
    if (r.get("aptamer_prep_status") or "").strip() != "READY_AFTER_MOD_REVIEW":
        continue

    target_name = (r.get("target_normalized") or "").strip()

    if target_name not in accepted:
        continue

    typ = (r.get("nucleic_acid_type") or "").strip().upper()
    seq = clean_sequence(r.get("sequence"))

    if typ == "DNA":
        valid = set(seq) <= set("ACGT")
    elif typ == "RNA":
        valid = set(seq) <= set("ACGU")
    else:
        valid = False

    if not seq or not valid:
        continue

    candidates.append((r, accepted[target_name]))

print("Candidate rows:", len(candidates))
assert len(candidates) == 83, \
    f"Expected 83 recovery candidates, found {len(candidates)}"

rows_out = []
breakdown = Counter()

for idx, (apt, target) in enumerate(candidates, 1):
    run_id = f"task5r_{idx:04d}"

    apt_type = (apt.get("nucleic_acid_type") or "").strip().upper()
    apt_seq = clean_sequence(apt.get("sequence"))

    target_type = (target.get("resolved_type") or "").strip().lower()
    target_name = (apt.get("target_normalized") or "").strip()

    yaml_path = OUTDIR / f"{run_id}.yaml"

    lines = [
        "version: 1",
        "sequences:",
    ]

    # Target is chain A.
    if target_type == "protein":
        protein_seq = clean_sequence(target.get("resolved_sequence"))

        if not protein_seq:
            raise RuntimeError(
                f"{run_id}: ACCEPT protein target has no protein sequence: "
                f"{target_name}"
            )

        lines += [
            "  - protein:",
            "      id: A",
            f"      sequence: {protein_seq}",
            "      msa: empty",
        ]

        target_length = len(protein_seq)
        target_source = target.get("source_id", "")

    elif target_type == "ligand":
        smiles = (target.get("resolved_smiles") or "").strip()

        if not smiles:
            raise RuntimeError(
                f"{run_id}: ACCEPT ligand target has no SMILES: "
                f"{target_name}"
            )

        lines += [
            "  - ligand:",
            "      id: A",
            f"      smiles: {q(smiles)}",
        ]

        target_length = 0
        target_source = target.get("source_id", "")

    else:
        raise RuntimeError(
            f"{run_id}: unexpected accepted target type {target_type!r}"
        )

    # Aptamer is affinity binder chain B.
    if apt_type == "DNA":
        lines += [
            "  - dna:",
            "      id: B",
            f"      sequence: {apt_seq}",
        ]
    elif apt_type == "RNA":
        lines += [
            "  - rna:",
            "      id: B",
            f"      sequence: {apt_seq}",
        ]
    else:
        raise RuntimeError(
            f"{run_id}: unsupported aptamer type {apt_type!r}"
        )

    lines += [
        "properties:",
        "  - affinity:",
        "      binder: B",
        "",
    ]

    yaml_path.write_text("\n".join(lines), encoding="utf-8")

    breakdown[(apt_type, target_type)] += 1

    rows_out.append({
        "run_id": run_id,
        "model_id": apt.get("model_id", ""),
        "aptamer_id": apt.get("aptamer_id", ""),
        "serial_number": apt.get("serial_number", ""),
        "aptamer_name": apt.get("aptamer_name", ""),
        "nucleic_acid_type": apt_type,
        "sequence": apt_seq,
        "aptamer_length": len(apt_seq),
        "target_normalized": target_name,
        "target_type": target_type,
        "target_length": target_length,
        "target_source_database": target.get("source_database", ""),
        "target_source_id": target_source,
        "experimental_affinity_raw": apt.get("affinity_raw", ""),
        "experimental_kd_nm": apt.get("kd_nm", ""),
        "doi": apt.get("doi", ""),
        "pubmed": apt.get("pubmed", ""),
        "original_aptamer_prep_status": apt.get("aptamer_prep_status", ""),
        "modeling_approximation": (
            "UNMODIFIED_CANONICAL_BACKBONE; "
            "experimental modification metadata not represented in Boltz input"
        ),
        "yaml_path": str(yaml_path),
    })

fields = list(rows_out[0].keys())

with open(MANIFEST_OUT, "w", newline="", encoding="utf-8") as f:
    w = csv.DictWriter(f, fieldnames=fields)
    w.writeheader()
    w.writerows(rows_out)

print()
print("========== TASK 5 RECOVERY INPUTS ==========")
print("Generated YAML files :", len(rows_out))
print("Manifest             :", MANIFEST_OUT)
print("Input directory      :", OUTDIR)

print("\nBreakdown:")
for (apt_type, target_type), n in sorted(breakdown.items()):
    print(f"  {apt_type} + {target_type}: {n}")

protein_rows = [
    r for r in rows_out
    if r["target_type"] == "protein"
]

large = [
    r for r in protein_rows
    if int(r["target_length"]) > 1500
]

print()
print("Protein complexes    :", len(protein_rows))
print("Large >1500 aa       :", len(large))

if large:
    print("\nLarge recovery cases:")
    for r in large:
        print(
            r["run_id"],
            "|",
            r["target_length"],
            "aa |",
            r["target_normalized"],
            "|",
            r["model_id"],
        )

print()
print("First YAML:", rows_out[0]["yaml_path"])
print(Path(rows_out[0]["yaml_path"]).read_text())
