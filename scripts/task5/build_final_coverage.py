#!/usr/bin/env python3

import csv
from pathlib import Path
from collections import Counter

PROJECT = Path("/projects/bentosprg6/linwx/research_project")

MODEL_MANIFEST = Path(
    "/projects/bentosprg6/linwx/ra_project/task2_utexas/"
    "data/processed/model_manifest.csv"
)

DIRECT_READY = Path(
    "/projects/bentosprg6/linwx/ra_project/task2_utexas/"
    "data/processed/aptamer_chains_direct_ready.csv"
)

TARGETS = PROJECT / "data/task5/targets/target_validated.csv"

ORIGINAL_STATUS = PROJECT / "data/task5/task5_all_status.csv"
RECOVERY_MANIFEST = PROJECT / "data/task5/task5_recovery_manifest.csv"
FINAL_RANKED = PROJECT / "data/task5/task5_affinity_ranked_final.csv"

OUTPUT = PROJECT / "data/task5/task5_full_coverage.csv"


def read_csv(path):
    with open(path, newline="", encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))


manifest = read_csv(MODEL_MANIFEST)
direct = read_csv(DIRECT_READY)
targets = read_csv(TARGETS)
original_status = read_csv(ORIGINAL_STATUS)
recovery = read_csv(RECOVERY_MANIFEST)
ranked = read_csv(FINAL_RANKED)


# -------------------------------------------------
# Indexes
# -------------------------------------------------

direct_ids = {
    (r.get("model_id") or "").strip()
    for r in direct
}

target_map = {
    (r.get("target_normalized") or "").strip(): r
    for r in targets
}

original_status_map = {
    (r.get("model_id") or "").strip(): r
    for r in original_status
}

recovery_ids = {
    (r.get("model_id") or "").strip()
    for r in recovery
}

ranked_map = {
    (r.get("model_id") or "").strip(): r
    for r in ranked
}


# -------------------------------------------------
# QC source counts
# -------------------------------------------------

assert len(manifest) == 1495, len(manifest)
assert len(direct) == 1217, len(direct)
assert len(original_status) == 342, len(original_status)
assert len(recovery) == 83, len(recovery)
assert len(ranked) == 420, len(ranked)

assert len(direct_ids) == 1217
assert len(recovery_ids) == 83
assert len(ranked_map) == 420

assert recovery_ids.isdisjoint(
    set(original_status_map)
), "Recovery overlaps original attempted set"


# -------------------------------------------------
# Build all-record accounting
# -------------------------------------------------

rows_out = []

for row in manifest:
    out = dict(row)

    model_id = (row.get("model_id") or "").strip()
    target_name = (row.get("target_normalized") or "").strip()
    prep_status = (row.get("aptamer_prep_status") or "").strip()

    target = target_map.get(target_name, {})

    target_validation = (
        target.get("validation_status") or ""
    ).strip()

    target_reason = (
        target.get("validation_reason") or ""
    ).strip()

    out["task5_target_validation"] = target_validation
    out["task5_target_reason"] = target_reason

    out["task5_attempted"] = "NO"
    out["task5_result_status"] = ""
    out["task5_coverage_group"] = ""
    out["task5_rank"] = ""
    out["affinity_pred_value"] = ""
    out["task5_notes"] = ""

    # ---------------------------------------------
    # Successful prediction — original or recovery
    # ---------------------------------------------

    if model_id in ranked_map:
        rr = ranked_map[model_id]

        out["task5_attempted"] = "YES"
        out["task5_result_status"] = "SUCCESS"
        out["task5_rank"] = rr.get("rank", "")
        out["affinity_pred_value"] = rr.get(
            "affinity_pred_value", ""
        )

        group = rr.get("result_group", "")

        if group == "ORIGINAL_DIRECT_READY":
            out["task5_coverage_group"] = \
                "SUCCESS_ORIGINAL_DIRECT_READY"

            out["task5_notes"] = \
                "Direct-ready aptamer representation."

        elif group == "RECOVERY_UNMODIFIED_BACKBONE":
            out["task5_coverage_group"] = \
                "SUCCESS_RECOVERY_UNMODIFIED_BACKBONE"

            out["task5_notes"] = (
                "Experimental modification metadata was not "
                "represented; modeled using the canonical "
                "unmodified DNA/RNA backbone."
            )

        else:
            raise RuntimeError(
                f"Unexpected ranked group for {model_id}: "
                f"{group!r}"
            )

    # ---------------------------------------------
    # Original attempted but GPU OOM
    # ---------------------------------------------

    elif model_id in original_status_map:
        old = original_status_map[model_id]

        run_id = (old.get("run_id") or "").strip()
        status = (old.get("result_status") or "").strip()

        KNOWN_GPU_OOM_RUNS = {
            "task5_0049",
            "task5_0050",
            "task5_0313",
            "task5_0314",
            "task5_0330",
        }

        assert run_id in KNOWN_GPU_OOM_RUNS, (
            "Unexpected non-successful original run",
            run_id,
            model_id,
            status,
        )

        out["task5_attempted"] = "YES"
        out["task5_result_status"] = "GPU_OOM"
        out["task5_coverage_group"] = "GPU_OOM_ORIGINAL"

        out["task5_notes"] = (
            "Boltz-2 structure prediction exceeded available "
            "GPU memory; no affinity prediction was produced. "
            "The intermediate status table may label this run "
            "as MISSING because no affinity JSON was generated."
        )

    # ---------------------------------------------
    # Not attempted
    # ---------------------------------------------

    else:
        out["task5_attempted"] = "NO"
        out["task5_result_status"] = "NOT_RUN"

        if model_id in direct_ids:
            # These were structurally direct-ready but target
            # validation prevented inclusion in the run set.

            if target_validation == "NOT_USABLE":
                out["task5_coverage_group"] = \
                    "NOT_RUN_TARGET_NOT_USABLE"

            elif target_validation == "REJECT":
                out["task5_coverage_group"] = \
                    "NOT_RUN_TARGET_REJECT"

            elif target_validation == "REVIEW":
                out["task5_coverage_group"] = \
                    "NOT_RUN_TARGET_REVIEW"

            elif target_validation == "ACCEPT":
                raise RuntimeError(
                    f"Direct-ready ACCEPT record unexpectedly "
                    f"not attempted: {model_id}"
                )

            else:
                out["task5_coverage_group"] = \
                    "NOT_RUN_TARGET_UNRESOLVED"

            out["task5_notes"] = (
                "Not submitted because target resolution/"
                "validation did not meet Task 5 run criteria."
            )

        else:
            out["task5_coverage_group"] = \
                "NOT_RUN_APTAMER_PREPARATION"

            out["task5_notes"] = (
                f"Not submitted because aptamer preparation "
                f"status was {prep_status or 'not direct-ready'}."
            )

    rows_out.append(out)


# -------------------------------------------------
# Final counts
# -------------------------------------------------

assert len(rows_out) == 1495

attempted = [
    r for r in rows_out
    if r["task5_attempted"] == "YES"
]

success = [
    r for r in rows_out
    if r["task5_result_status"] == "SUCCESS"
]

oom = [
    r for r in rows_out
    if r["task5_result_status"] == "GPU_OOM"
]

not_run = [
    r for r in rows_out
    if r["task5_result_status"] == "NOT_RUN"
]

assert len(attempted) == 425, len(attempted)
assert len(success) == 420, len(success)
assert len(oom) == 5, len(oom)
assert len(not_run) == 1070, len(not_run)

assert len(success) + len(oom) + len(not_run) == 1495


# -------------------------------------------------
# Write
# -------------------------------------------------

fields = list(rows_out[0].keys())

with open(OUTPUT, "w", newline="", encoding="utf-8") as f:
    w = csv.DictWriter(f, fieldnames=fields)
    w.writeheader()
    w.writerows(rows_out)


# -------------------------------------------------
# Report
# -------------------------------------------------

print("========== FINAL TASK 5 COVERAGE ==========")
print("Total UTexas model records :", len(rows_out))
print("Attempted                  :", len(attempted))
print("Successful                 :", len(success))
print("GPU_OOM                    :", len(oom))
print("Not run                    :", len(not_run))

print()
print("========== COVERAGE GROUPS ==========")

counts = Counter(
    r["task5_coverage_group"]
    for r in rows_out
)

for k, v in counts.most_common():
    print(f"{k:42s}: {v}")

print()
print("========== ATTEMPTED QC ==========")

print(
    "Original successes :",
    counts["SUCCESS_ORIGINAL_DIRECT_READY"]
)

print(
    "Recovery successes :",
    counts["SUCCESS_RECOVERY_UNMODIFIED_BACKBONE"]
)

print(
    "GPU OOM            :",
    counts["GPU_OOM_ORIGINAL"]
)

print()
print("Output:", OUTPUT)

print()
print("========== FULL 1495 COVERAGE PASSED ==========")
