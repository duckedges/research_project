#!/usr/bin/env python3

import csv
import json
import re
from pathlib import Path
from collections import defaultdict

PROJECT = Path("/projects/bentosprg6/linwx/research_project")

MANIFEST = PROJECT / "data/task5/task5_recovery_manifest.csv"
OUTPUT = PROJECT / "data/task5/task5_recovery_affinity.csv"

ROOTS = [
    PROJECT / "results/task5/recovery_smoke",
    PROJECT / "results/task5/recovery_batch",
    PROJECT / "results/task5/recovery_large",
]


def read_csv(path):
    with open(path, newline="", encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))


manifest = read_csv(MANIFEST)

print("========== INDEXING RECOVERY RESULTS ==========")

# Scan each results tree only ONCE.
index = defaultdict(list)

for root in ROOTS:
    count = 0

    if root.exists():
        for p in root.rglob("affinity_task5r_*.json"):
            m = re.fullmatch(
                r"affinity_(task5r_\d{4})\.json",
                p.name
            )

            if m:
                run_id = m.group(1)
                index[run_id].append(p)
                count += 1

    print(f"{root.name:20s}: {count}")

print("Total indexed JSONs:", sum(len(v) for v in index.values()))

records = []
missing = []
duplicate = []

for row in manifest:
    run_id = row["run_id"]

    matches = index.get(run_id, [])

    if len(matches) == 0:
        missing.append(run_id)
        continue

    if len(matches) > 1:
        duplicate.append((run_id, matches))
        continue

    json_path = matches[0]

    with open(json_path, encoding="utf-8") as f:
        data = json.load(f)

    out = dict(row)

    out.update({
        "affinity_pred_value":
            data.get("affinity_pred_value", ""),

        "affinity_probability_binary":
            data.get("affinity_probability_binary", ""),

        "affinity_pred_value1":
            data.get("affinity_pred_value1", ""),

        "affinity_probability_binary1":
            data.get("affinity_probability_binary1", ""),

        "affinity_pred_value2":
            data.get("affinity_pred_value2", ""),

        "affinity_probability_binary2":
            data.get("affinity_probability_binary2", ""),

        "result_status": "SUCCESS",
        "affinity_json_path": str(json_path),
    })

    records.append(out)


print()
print("========== RECOVERY RESULT COLLECTION ==========")
print("Recovery planned :", len(manifest))
print("Successful       :", len(records))
print("Missing          :", len(missing))
print("Duplicate JSONs  :", len(duplicate))

if missing:
    print("\nMISSING:")
    for x in missing:
        print(x)

if duplicate:
    print("\nDUPLICATES:")
    for run_id, paths in duplicate:
        print(run_id)
        for p in paths:
            print(" ", p)

assert len(manifest) == 83, len(manifest)
assert len(records) == 83, len(records)
assert len(missing) == 0, missing
assert len(duplicate) == 0, duplicate

records.sort(
    key=lambda r: float(r["affinity_pred_value"])
)

fields = list(records[0].keys())

with open(OUTPUT, "w", newline="", encoding="utf-8") as f:
    w = csv.DictWriter(f, fieldnames=fields)
    w.writeheader()
    w.writerows(records)

print()
print("Lowest recovery predictions:")
for r in records[:5]:
    print(
        f"{r['run_id']} | "
        f"{float(r['affinity_pred_value']):.6f} | "
        f"{r['target_normalized']}"
    )

print()
print("Highest recovery predictions:")
for r in records[-5:]:
    print(
        f"{r['run_id']} | "
        f"{float(r['affinity_pred_value']):.6f} | "
        f"{r['target_normalized']}"
    )

print()
print("Output:", OUTPUT)
print("========== RECOVERY COLLECTION PASSED ==========")
