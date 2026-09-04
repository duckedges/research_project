#!/usr/bin/env python3

import csv
import json
import re
from pathlib import Path
from collections import Counter

PROJECT = Path("/projects/bentosprg6/linwx/research_project")

APTAMER_CSV = Path(
    "/projects/bentosprg6/linwx/ra_project/task2_utexas/"
    "data/processed/aptamer_chains_direct_ready.csv"
)

TARGET_CSV = PROJECT / "data/task5/targets/target_validated.csv"

OUTDIR = PROJECT / "data/task5/boltz_inputs"
MANIFEST = PROJECT / "data/task5/task5_run_manifest.csv"

OUTDIR.mkdir(parents=True, exist_ok=True)


def read_csv(path):
    with open(path, newline="", encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))


def clean_sequence(seq):
    return re.sub(r"\s+", "", seq or "").upper()


def yaml_quote(s):
    # JSON double-quoted strings are also valid YAML.
    return json.dumps(s, ensure_ascii=False)


aptamers = read_csv(APTAMER_CSV)
targets = read_csv(TARGET_CSV)

accepted = {
    (r.get("target_normalized") or "").strip(): r
    for r in targets
    if r.get("validation_status") == "ACCEPT"
}

runnable = [
    r for r in aptamers
    if (r.get("target_normalized") or "").strip() in accepted
]

manifest_rows = []
counts = Counter()

for index, apt in enumerate(runnable, 1):

    target_name = (apt.get("target_normalized") or "").strip()
    target = accepted[target_name]

    apt_type = (apt.get("nucleic_acid_type") or "").strip().upper()
    apt_seq = clean_sequence(apt.get("sequence"))

    target_type = (target.get("resolved_type") or "").strip()

    run_id = f"task5_{index:04d}"
    yaml_path = OUTDIR / f"{run_id}.yaml"

    if apt_type == "DNA":
        apt_entity = "dna"
    elif apt_type == "RNA":
        apt_entity = "rna"
    else:
        raise RuntimeError(
            f"{run_id}: unsupported aptamer type {apt_type!r}"
        )

    lines = [
        "version: 1",
        "sequences:",
    ]

    # ----------------------------------------
    # Target chain A
    # ----------------------------------------
    if target_type == "protein":

        protein_seq = clean_sequence(
            target.get("resolved_sequence")
        )

        if not protein_seq:
            raise RuntimeError(
                f"{run_id}: accepted protein has no sequence"
            )

        lines += [
            "  - protein:",
            "      id: A",
            f"      sequence: {protein_seq}",
            "      msa: empty",
        ]

        target_length = len(protein_seq)

    elif target_type == "ligand":

        smiles = (target.get("resolved_smiles") or "").strip()

        if not smiles:
            raise RuntimeError(
                f"{run_id}: accepted ligand has no SMILES"
            )

        lines += [
            "  - ligand:",
            "      id: A",
            f"      smiles: {yaml_quote(smiles)}",
        ]

        target_length = ""

    else:
        raise RuntimeError(
            f"{run_id}: unsupported resolved target type "
            f"{target_type!r}"
        )

    # ----------------------------------------
    # Aptamer chain B
    # ----------------------------------------
    lines += [
        f"  - {apt_entity}:",
        "      id: B",
        f"      sequence: {apt_seq}",
        "properties:",
        "  - affinity:",
        "      binder: B",
        "",
    ]

    yaml_path.write_text(
        "\n".join(lines),
        encoding="utf-8"
    )

    row = {
        "run_index": index,
        "run_id": run_id,
        "model_id": apt.get("model_id", ""),
        "aptamer_id": apt.get("aptamer_id", ""),
        "serial_number": apt.get("serial_number", ""),
        "aptamer_name": apt.get("aptamer_name", ""),
        "nucleic_acid_type": apt_type,
        "aptamer_length": len(apt_seq),
        "target_normalized": target_name,
        "target_type": target_type,
        "target_length": target_length,
        "target_source_database": target.get(
            "source_database", ""
        ),
        "target_source_id": target.get("source_id", ""),
        "target_match": target.get("matched_name", ""),
        "resolution_method": target.get(
            "resolution_method", ""
        ),
        "validation_reason": target.get(
            "validation_reason", ""
        ),
        "experimental_affinity_raw": apt.get(
            "affinity_raw", ""
        ),
        "experimental_kd_nm": apt.get("kd_nm", ""),
        "doi": apt.get("doi", ""),
        "pubmed": apt.get("pubmed", ""),
        "yaml_path": str(yaml_path),
    }

    manifest_rows.append(row)

    counts[(apt_type, target_type)] += 1


fields = list(manifest_rows[0].keys())

with open(MANIFEST, "w", newline="", encoding="utf-8") as f:
    w = csv.DictWriter(f, fieldnames=fields)
    w.writeheader()
    w.writerows(manifest_rows)


print("========== TASK 5 INPUT GENERATION ==========")
print("Generated YAML files :", len(manifest_rows))
print("Run manifest         :", MANIFEST)
print("Input directory      :", OUTDIR)

print("\nBreakdown:")
for (apt_type, target_type), n in sorted(counts.items()):
    print(
        f"  {apt_type:3s} + {target_type:7s}: {n}"
    )

protein_lengths = [
    int(r["target_length"])
    for r in manifest_rows
    if r["target_length"]
]

aptamer_lengths = [
    int(r["aptamer_length"])
    for r in manifest_rows
]

print("\nLength checks:")
print("  Max aptamer length :", max(aptamer_lengths))
print(
    "  Max protein length :",
    max(protein_lengths) if protein_lengths else "N/A"
)

print("\nFirst YAML:", manifest_rows[0]["yaml_path"])
print("Last YAML :", manifest_rows[-1]["yaml_path"])
