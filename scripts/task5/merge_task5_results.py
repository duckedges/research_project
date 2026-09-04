#!/usr/bin/env python3

import csv
import math
from pathlib import Path

PROJECT = Path("/projects/bentosprg6/linwx/research_project")

ORIGINAL = PROJECT / "data/task5/task5_affinity_ranked.csv"
RECOVERY = PROJECT / "data/task5/task5_recovery_affinity.csv"

OUTPUT = PROJECT / "data/task5/task5_affinity_ranked_final.csv"


def read_csv(path):
    with open(path, newline="", encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))


original = read_csv(ORIGINAL)
recovery = read_csv(RECOVERY)

print("========== TASK 5 FINAL MERGE ==========")
print("Original successful :", len(original))
print("Recovery successful :", len(recovery))

assert len(original) == 337, \
    f"Expected 337 original successes, found {len(original)}"

assert len(recovery) == 83, \
    f"Expected 83 recovery successes, found {len(recovery)}"


# -------------------------------------------------
# Add provenance
# -------------------------------------------------

combined = []

for r in original:
    x = dict(r)
    x["result_group"] = "ORIGINAL_DIRECT_READY"
    combined.append(x)

for r in recovery:
    x = dict(r)
    x["result_group"] = "RECOVERY_UNMODIFIED_BACKBONE"
    combined.append(x)


# -------------------------------------------------
# QC identifiers
# -------------------------------------------------

run_ids = [r.get("run_id", "").strip() for r in combined]
model_ids = [r.get("model_id", "").strip() for r in combined]

assert all(run_ids), "Blank run_id found"
assert all(model_ids), "Blank model_id found"

assert len(run_ids) == len(set(run_ids)), \
    "Duplicate run_id detected"

assert len(model_ids) == len(set(model_ids)), \
    "Duplicate model_id detected"


# -------------------------------------------------
# QC affinity values
# -------------------------------------------------

for r in combined:
    raw = r.get("affinity_pred_value", "")

    try:
        value = float(raw)
    except Exception:
        raise RuntimeError(
            f"Invalid affinity_pred_value for "
            f"{r.get('run_id')}: {raw!r}"
        )

    if not math.isfinite(value):
        raise RuntimeError(
            f"Non-finite affinity_pred_value for "
            f"{r.get('run_id')}: {raw!r}"
        )


# -------------------------------------------------
# Sort lowest -> highest and rerank
# -------------------------------------------------

combined.sort(
    key=lambda r: float(r["affinity_pred_value"])
)

for rank, r in enumerate(combined, 1):
    r["rank"] = str(rank)


# -------------------------------------------------
# Build union of columns
# -------------------------------------------------

preferred = [
    "rank",
    "result_group",
    "run_id",
    "model_id",
    "aptamer_id",
    "serial_number",
    "aptamer_name",
    "nucleic_acid_type",
    "aptamer_length",
    "target_normalized",
    "target_type",
    "target_length",
    "affinity_pred_value",
    "affinity_probability_binary",
    "affinity_pred_value1",
    "affinity_probability_binary1",
    "affinity_pred_value2",
    "affinity_probability_binary2",
    "experimental_affinity_raw",
    "experimental_kd_nm",
    "modeling_approximation",
    "doi",
    "pubmed",
    "affinity_json_path",
]

all_fields = set()

for r in combined:
    all_fields.update(r.keys())

fields = []

for f in preferred:
    if f in all_fields and f not in fields:
        fields.append(f)

for f in list(original[0].keys()) + list(recovery[0].keys()):
    if f not in fields and f in all_fields:
        fields.append(f)


# -------------------------------------------------
# Write
# -------------------------------------------------

with open(OUTPUT, "w", newline="", encoding="utf-8") as f:
    w = csv.DictWriter(
        f,
        fieldnames=fields,
        extrasaction="ignore"
    )
    w.writeheader()

    for r in combined:
        w.writerow({
            field: r.get(field, "")
            for field in fields
        })


# -------------------------------------------------
# Final QC
# -------------------------------------------------

assert len(combined) == 420

assert [int(r["rank"]) for r in combined] == \
       list(range(1, 421))

values = [
    float(r["affinity_pred_value"])
    for r in combined
]

assert values == sorted(values)

print()
print("Merged successful rows :", len(combined))
print("Unique run IDs         :", len(set(run_ids)))
print("Unique model IDs       :", len(set(model_ids)))

print()
print("Result groups:")
for group in [
    "ORIGINAL_DIRECT_READY",
    "RECOVERY_UNMODIFIED_BACKBONE",
]:
    print(
        f"{group:32s}:",
        sum(r["result_group"] == group for r in combined)
    )

print()
print("========== LOWEST 10 ==========")

for r in combined[:10]:
    print(
        f"{int(r['rank']):3d} | "
        f"{float(r['affinity_pred_value']):10.6f} | "
        f"{r['result_group']:30s} | "
        f"{r.get('model_id','')} | "
        f"{r.get('target_normalized','')}"
    )

print()
print("========== HIGHEST 10 ==========")

for r in combined[-10:]:
    print(
        f"{int(r['rank']):3d} | "
        f"{float(r['affinity_pred_value']):10.6f} | "
        f"{r['result_group']:30s} | "
        f"{r.get('model_id','')} | "
        f"{r.get('target_normalized','')}"
    )

print()
print("Output:", OUTPUT)
print()
print("========== FINAL 420 MERGE PASSED ==========")
