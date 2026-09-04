#!/usr/bin/env python3

import csv
import json
from pathlib import Path

PROJECT = Path("/projects/bentosprg6/linwx/research_project")

MANIFEST = PROJECT / "data/task5/task5_run_manifest.csv"
RESULT_ROOT = PROJECT / "results/task5/batch"

RANKED_OUT = PROJECT / "data/task5/task5_affinity_ranked.csv"
STATUS_OUT = PROJECT / "data/task5/task5_all_status.csv"
SUMMARY_OUT = PROJECT / "data/task5/TASK5_PROGRESS.txt"


def read_csv(path):
    with open(path, newline="", encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))


manifest = read_csv(MANIFEST)

records = []
success = []
missing = []
invalid = []


for row in manifest:
    run_id = row["run_id"]

    expected_dir = RESULT_ROOT / run_id

    json_files = sorted(
        expected_dir.rglob(f"affinity_{run_id}.json")
    ) if expected_dir.exists() else []

    out = dict(row)

    out.update({
        "affinity_pred_value": "",
        "affinity_probability_binary": "",
        "affinity_pred_value1": "",
        "affinity_probability_binary1": "",
        "affinity_pred_value2": "",
        "affinity_probability_binary2": "",
        "result_status": "",
        "affinity_json_path": "",
    })

    if not json_files:
        out["result_status"] = "MISSING"
        missing.append(out)
        records.append(out)
        continue

    p = json_files[0]

    try:
        with open(p, encoding="utf-8") as f:
            data = json.load(f)

        primary = data.get("affinity_pred_value")

        if primary is None:
            raise ValueError("affinity_pred_value missing")

        out["affinity_pred_value"] = primary
        out["affinity_probability_binary"] = data.get(
            "affinity_probability_binary", ""
        )
        out["affinity_pred_value1"] = data.get(
            "affinity_pred_value1", ""
        )
        out["affinity_probability_binary1"] = data.get(
            "affinity_probability_binary1", ""
        )
        out["affinity_pred_value2"] = data.get(
            "affinity_pred_value2", ""
        )
        out["affinity_probability_binary2"] = data.get(
            "affinity_probability_binary2", ""
        )

        out["result_status"] = "SUCCESS"
        out["affinity_json_path"] = str(p)

        success.append(out)

    except Exception as e:
        out["result_status"] = f"INVALID_JSON: {e}"
        out["affinity_json_path"] = str(p)
        invalid.append(out)

    records.append(out)


# --------------------------------------------------
# Ranked successful predictions: lowest -> highest
# --------------------------------------------------

success.sort(
    key=lambda r: float(r["affinity_pred_value"])
)

ranked_fields = [
    "rank",
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
    "target_source_database",
    "target_source_id",
    "experimental_affinity_raw",
    "experimental_kd_nm",
    "affinity_pred_value",
    "affinity_probability_binary",
    "affinity_pred_value1",
    "affinity_probability_binary1",
    "affinity_pred_value2",
    "affinity_probability_binary2",
    "doi",
    "pubmed",
    "affinity_json_path",
]

with open(RANKED_OUT, "w", newline="", encoding="utf-8") as f:
    w = csv.DictWriter(f, fieldnames=ranked_fields)
    w.writeheader()

    for rank, r in enumerate(success, 1):
        row = {k: r.get(k, "") for k in ranked_fields}
        row["rank"] = rank
        w.writerow(row)


# --------------------------------------------------
# Full 342-case status table
# --------------------------------------------------

status_fields = [
    "run_id",
    "model_id",
    "aptamer_id",
    "aptamer_name",
    "nucleic_acid_type",
    "target_normalized",
    "target_type",
    "experimental_kd_nm",
    "affinity_pred_value",
    "affinity_probability_binary",
    "affinity_pred_value1",
    "affinity_probability_binary1",
    "affinity_pred_value2",
    "affinity_probability_binary2",
    "result_status",
    "affinity_json_path",
]

with open(STATUS_OUT, "w", newline="", encoding="utf-8") as f:
    w = csv.DictWriter(f, fieldnames=status_fields)
    w.writeheader()

    # Successful entries first in affinity ranking order,
    # then incomplete/invalid entries.
    for r in success + invalid + missing:
        w.writerow({
            k: r.get(k, "")
            for k in status_fields
        })


summary = f"""TASK 5 AFFINITY COLLECTION

Total planned complexes: {len(manifest)}
Successful affinity predictions: {len(success)}
Missing affinity outputs: {len(missing)}
Invalid affinity outputs: {len(invalid)}

Primary ranking field:
affinity_pred_value

Ranking direction:
lowest to highest

Ranked results:
{RANKED_OUT}

Full run status:
{STATUS_OUT}
"""

SUMMARY_OUT.write_text(summary, encoding="utf-8")

print("========== TASK 5 RESULT COLLECTION ==========")
print("Planned complexes :", len(manifest))
print("Successful        :", len(success))
print("Missing           :", len(missing))
print("Invalid           :", len(invalid))

if success:
    print()
    print("Lowest affinity_pred_value:")
    for r in success[:5]:
        print(
            f"  {r['run_id']} | "
            f"{float(r['affinity_pred_value']):.6f} | "
            f"{r['target_normalized']}"
        )

    print()
    print("Highest affinity_pred_value:")
    for r in success[-5:]:
        print(
            f"  {r['run_id']} | "
            f"{float(r['affinity_pred_value']):.6f} | "
            f"{r['target_normalized']}"
        )

print()
print("Ranked CSV :", RANKED_OUT)
print("Status CSV :", STATUS_OUT)
print("Summary    :", SUMMARY_OUT)
