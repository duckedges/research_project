#!/usr/bin/env python3

import csv
import math
import os
from itertools import permutations

TASK5 = "data/task5/task5_affinity_ranked_final.csv"
TASK6 = "data/task6/task6_head1_heldout_test_predictions.csv"
OUTDIR = "data/task7"

os.makedirs(OUTDIR, exist_ok=True)


def to_float(x):
    if x is None:
        return None
    x = str(x).strip()
    if x == "":
        return None
    try:
        v = float(x)
        return v if math.isfinite(v) else None
    except Exception:
        return None


def average_ranks(values):
    """
    Ascending ranks:
    smallest value = rank 1.
    Ties receive average rank.
    """
    n = len(values)
    order = sorted(range(n), key=lambda i: values[i])

    ranks = [0.0] * n
    pos = 0

    while pos < n:
        end = pos + 1
        v = values[order[pos]]

        while end < n and values[order[end]] == v:
            end += 1

        avg_rank = ((pos + 1) + end) / 2.0

        for j in range(pos, end):
            ranks[order[j]] = avg_rank

        pos = end

    return ranks


def pearson(x, y):
    n = len(x)
    mx = sum(x) / n
    my = sum(y) / n

    num = sum((a-mx)*(b-my) for a, b in zip(x, y))
    denx = math.sqrt(sum((a-mx)**2 for a in x))
    deny = math.sqrt(sum((b-my)**2 for b in y))

    if denx == 0 or deny == 0:
        return float("nan")

    return num / (denx * deny)


def spearman(x, y):
    return pearson(average_ranks(x), average_ranks(y))


def exact_permutation_pvalue(x, y):
    """
    Exact two-sided permutation p-value.
    Intended for the 8-case matched test set.
    """
    observed = abs(spearman(x, y))
    yranks = average_ranks(y)
    xranks = average_ranks(x)

    total = 0
    extreme = 0

    for perm in permutations(yranks):
        r = abs(pearson(xranks, perm))
        total += 1
        if r >= observed - 1e-12:
            extreme += 1

    return extreme / total


# ============================================================
# 1. LOAD TASK 5 STANDALONE BOLTZ RESULTS
# ============================================================

with open(TASK5, newline="") as f:
    task5_rows = list(csv.DictReader(f))

boltz = {}

for row in task5_rows:
    model_id = row.get("model_id", "").strip()

    kd = to_float(
        row.get("experimental_kd_nm")
        or row.get("kd_nm")
    )

    pred = to_float(
        row.get("affinity_pred_value")
    )

    if model_id and kd is not None and pred is not None:
        boltz[model_id] = {
            "model_id": model_id,
            "kd_nm": kd,
            "boltz_pred": pred,
            "task5_global_rank": row.get("rank", "")
        }


# ============================================================
# 2. LOAD TASK 6 HELD-OUT AF3 + AFFINITY RESULTS
# ============================================================

with open(TASK6, newline="") as f:
    task6_rows = list(csv.DictReader(f))


matched = []

for row in task6_rows:
    model_id = row["model_id"].strip()

    kd = to_float(row["kd_nm"])

    pretrained = to_float(row["pretrained_head1_pred"])
    calibrated = to_float(row["calibrated_head1_pred"])
    finetuned = to_float(row["finetuned_head1_pred"])
    ensemble = to_float(row["ensemble_calibrated_pred"])

    if model_id not in boltz:
        print("WARNING: held-out model not found in Task 5:", model_id)
        continue

    matched.append({
        "model_id": model_id,
        "nucleic_acid_type": row.get("nucleic_acid_type", ""),
        "total_polymer_length": row.get("total_polymer_length", ""),
        "kd_nm": kd,
        "boltz_pred": boltz[model_id]["boltz_pred"],
        "task5_global_rank": boltz[model_id]["task5_global_rank"],
        "af3_pretrained_head1": pretrained,
        "af3_calibrated_head1": calibrated,
        "af3_finetuned_head1": finetuned,
        "af3_ensemble_calibrated": ensemble,
    })


print(f"Matched held-out cases: {len(matched)} / {len(task6_rows)}")

if len(matched) != len(task6_rows):
    raise SystemExit("ERROR: Not all Task 6 held-out cases matched Task 5.")


# ============================================================
# 3. SORT EXPERIMENTAL KD LOWEST -> HIGHEST
# ============================================================

matched.sort(key=lambda r: r["kd_nm"])

kd_values = [r["kd_nm"] for r in matched]
boltz_values = [r["boltz_pred"] for r in matched]
af3_cal_values = [r["af3_calibrated_head1"] for r in matched]
af3_ft_values = [r["af3_finetuned_head1"] for r in matched]
af3_pre_values = [r["af3_pretrained_head1"] for r in matched]
af3_ens_values = [r["af3_ensemble_calibrated"] for r in matched]


# Lower Kd = stronger.
# Task 5 affinity_pred_value: lower value treated as stronger.
# Task 6 Head-1 predicts the log(Kd)-derived target:
# lower value therefore also treated as stronger.

exp_ranks = average_ranks(kd_values)
boltz_ranks = average_ranks(boltz_values)
af3_cal_ranks = average_ranks(af3_cal_values)
af3_ft_ranks = average_ranks(af3_ft_values)
af3_pre_ranks = average_ranks(af3_pre_values)
af3_ens_ranks = average_ranks(af3_ens_values)

for i, r in enumerate(matched):
    r["experimental_rank"] = exp_ranks[i]
    r["boltz_rank_matched8"] = boltz_ranks[i]
    r["af3_calibrated_rank"] = af3_cal_ranks[i]
    r["af3_finetuned_rank"] = af3_ft_ranks[i]
    r["af3_pretrained_rank"] = af3_pre_ranks[i]
    r["af3_ensemble_rank"] = af3_ens_ranks[i]


# ============================================================
# 4. PRIMARY MATCHED-8 SPEARMAN CORRELATIONS
# ============================================================

rho_boltz = spearman(kd_values, boltz_values)
rho_af3_cal = spearman(kd_values, af3_cal_values)

# Secondary Task 6 comparisons
rho_af3_ft = spearman(kd_values, af3_ft_values)
rho_af3_pre = spearman(kd_values, af3_pre_values)
rho_af3_ens = spearman(kd_values, af3_ens_values)

# Exact two-sided permutation p-values for n=8
p_boltz = exact_permutation_pvalue(kd_values, boltz_values)
p_af3_cal = exact_permutation_pvalue(kd_values, af3_cal_values)
p_af3_ft = exact_permutation_pvalue(kd_values, af3_ft_values)
p_af3_pre = exact_permutation_pvalue(kd_values, af3_pre_values)
p_af3_ens = exact_permutation_pvalue(kd_values, af3_ens_values)


# ============================================================
# 5. SUPPLEMENTAL FULL TASK 5 BOLTZ CORRELATION
# ============================================================

full_boltz_rows = list(boltz.values())

full_kd = [r["kd_nm"] for r in full_boltz_rows]
full_pred = [r["boltz_pred"] for r in full_boltz_rows]

rho_boltz_full = spearman(full_kd, full_pred)


# ============================================================
# 6. WRITE MATCHED 8 RANKING TABLE
# ============================================================

matched_csv = os.path.join(
    OUTDIR,
    "task7_matched8_affinity_rankings.csv"
)

fields = [
    "experimental_rank",
    "model_id",
    "nucleic_acid_type",
    "total_polymer_length",
    "kd_nm",
    "boltz_pred",
    "boltz_rank_matched8",
    "task5_global_rank",
    "af3_calibrated_head1",
    "af3_calibrated_rank",
    "af3_finetuned_head1",
    "af3_finetuned_rank",
    "af3_pretrained_head1",
    "af3_pretrained_rank",
    "af3_ensemble_calibrated",
    "af3_ensemble_rank",
]

with open(matched_csv, "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=fields)
    writer.writeheader()
    writer.writerows(matched)


# ============================================================
# 7. WRITE FULL EXPERIMENTAL KD SORT
# ============================================================

full_sorted = sorted(
    full_boltz_rows,
    key=lambda r: r["kd_nm"]
)

full_exp_ranks = average_ranks(
    [r["kd_nm"] for r in full_sorted]
)

full_boltz_ranks = average_ranks(
    [r["boltz_pred"] for r in full_sorted]
)

full_csv = os.path.join(
    OUTDIR,
    "task7_experimental_kd_sorted_all_boltz.csv"
)

with open(full_csv, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow([
        "experimental_rank",
        "model_id",
        "kd_nm",
        "boltz_affinity_pred_value",
        "boltz_rank_within_full_set",
        "task5_existing_global_rank"
    ])

    for i, r in enumerate(full_sorted):
        writer.writerow([
            full_exp_ranks[i],
            r["model_id"],
            r["kd_nm"],
            r["boltz_pred"],
            full_boltz_ranks[i],
            r["task5_global_rank"]
        ])


# ============================================================
# 8. WRITE SUMMARY
# ============================================================

summary_file = os.path.join(
    OUTDIR,
    "task7_spearman_results.txt"
)

summary = f"""TASK 7 — EXPERIMENTAL AFFINITY RANKING ANALYSIS

Interpretation:
- Lower experimental Kd = stronger binding.
- Rankings use ascending values, so rank 1 = strongest.
- Task 5 Boltz affinity_pred_value is ranked ascending.
- Task 6 AF3+affinity Head-1 predicts the log(Kd)-derived target,
  so Head-1 predictions are also ranked ascending.

PRIMARY FAIR COMPARISON: SAME 8 HELD-OUT COMPLEXES
------------------------------------------------

Standalone Boltz-2 vs experimental Kd
N = {len(matched)}
Spearman rho = {rho_boltz:.6f}
Exact two-sided permutation p = {p_boltz:.6f}

AF3 + calibrated pretrained Head-1 vs experimental Kd
N = {len(matched)}
Spearman rho = {rho_af3_cal:.6f}
Exact two-sided permutation p = {p_af3_cal:.6f}

SECONDARY AF3+AFFINITY RESULTS
-----------------------------

AF3 + fine-tuned Head-1
Spearman rho = {rho_af3_ft:.6f}
Exact p = {p_af3_ft:.6f}

AF3 + raw pretrained Head-1
Spearman rho = {rho_af3_pre:.6f}
Exact p = {p_af3_pre:.6f}

AF3 + calibrated ensemble
Spearman rho = {rho_af3_ens:.6f}
Exact p = {p_af3_ens:.6f}

SUPPLEMENTAL STANDALONE BOLTZ RESULT
------------------------------------

All usable Task 5 Boltz cases
N = {len(full_boltz_rows)}
Spearman rho = {rho_boltz_full:.6f}

Note:
The full-set Boltz correlation is supplemental and should not be directly
compared numerically with the AF3+affinity held-out correlation because
the sample sets differ. The matched-8 analysis is the fair head-to-head
comparison.
"""

with open(summary_file, "w") as f:
    f.write(summary)


# ============================================================
# 9. PRINT RESULTS
# ============================================================

print()
print("=" * 78)
print("TASK 7 RESULTS")
print("=" * 78)

print()
print("EXPERIMENTAL KD ORDER: LOWEST -> HIGHEST")
print("(rank 1 = strongest experimentally measured binding)")
print()

for r in matched:
    print(
        f'{int(r["experimental_rank"]):2d}. '
        f'{r["model_id"]:25s} '
        f'Kd = {r["kd_nm"]:12g} nM'
    )

print()
print("PRIMARY MATCHED-8 COMPARISON")
print("-" * 50)
print(
    f"Boltz-2 vs Kd:                  "
    f"rho = {rho_boltz:.6f}, p = {p_boltz:.6f}"
)
print(
    f"AF3 + calibrated Head-1 vs Kd: "
    f"rho = {rho_af3_cal:.6f}, p = {p_af3_cal:.6f}"
)

print()
print("SECONDARY AF3 RESULTS")
print("-" * 50)
print(
    f"Fine-tuned Head-1:              "
    f"rho = {rho_af3_ft:.6f}, p = {p_af3_ft:.6f}"
)
print(
    f"Raw pretrained Head-1:          "
    f"rho = {rho_af3_pre:.6f}, p = {p_af3_pre:.6f}"
)
print(
    f"Calibrated ensemble:            "
    f"rho = {rho_af3_ens:.6f}, p = {p_af3_ens:.6f}"
)

print()
print("SUPPLEMENTAL FULL BOLTZ SET")
print("-" * 50)
print(
    f"N = {len(full_boltz_rows)}, "
    f"rho = {rho_boltz_full:.6f}"
)

print()
print("Saved:")
print(" ", matched_csv)
print(" ", full_csv)
print(" ", summary_file)

