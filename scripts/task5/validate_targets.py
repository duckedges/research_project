#!/usr/bin/env python3

import csv
import re
from collections import Counter
from pathlib import Path

PROJECT = Path("/projects/bentosprg6/linwx/research_project")

INPUT = PROJECT / "data/task5/targets/target_resolved.csv"
OUTPUT = PROJECT / "data/task5/targets/target_validated.csv"


STOPWORDS = {
    "human", "mouse", "rat", "bovine", "porcine", "rabbit",
    "recombinant", "sigma", "synthetic",
    "protein", "peptide", "antigen",
    "isoform", "subunit", "chain",
    "domain", "fragment"
}


def norm(s):
    s = (s or "").lower()

    s = s.replace("α", " alpha ")
    s = s.replace("β", " beta ")
    s = s.replace("γ", " gamma ")

    s = re.sub(r"[^a-z0-9]+", " ", s)
    return " ".join(s.split())


def important_tokens(s):
    toks = norm(s).split()

    return {
        t for t in toks
        if len(t) >= 2
        and t not in STOPWORDS
    }


def has_extra_role(target, match):
    """
    Reject cases where UniProt found a receptor/binder/transporter
    OF the requested target instead of the requested target itself.
    """

    tn = norm(target)
    mn = norm(match)

    reasons = []

    # "non-receptor" is legitimate terminology in proteins such as PTP1B,
    # so don't treat that as a receptor false-positive.
    candidate_without_nonreceptor = mn.replace("non receptor", "")

    if (
        "receptor" in candidate_without_nonreceptor.split()
        and "receptor" not in tn.split()
    ):
        reasons.append("candidate_is_receptor")

    phrases = [
        ("binding protein", "candidate_is_binding_protein"),
        ("transporter", "candidate_is_transporter"),
        ("accessory factor", "candidate_is_accessory_factor"),
        ("interacting protein", "candidate_is_interacting_protein"),
    ]

    for phrase, reason in phrases:
        if phrase in mn and phrase not in tn:
            reasons.append(reason)

    return reasons


with open(INPUT, newline="", encoding="utf-8-sig") as f:
    rows = list(csv.DictReader(f))


new_fields = list(rows[0].keys()) + [
    "validation_status",
    "validation_reason",
    "target_token_coverage"
]

counts = Counter()
out_rows = []


for r in rows:

    original_status = (r.get("task5_status") or "").strip()
    rtype = (r.get("resolved_type") or "").strip()
    target = r.get("target_normalized") or ""
    match = r.get("matched_name") or ""
    method = r.get("resolution_method") or ""

    validation = "NOT_USABLE"
    reason = original_status or "UNRESOLVED"
    coverage = 0.0


    # --------------------------------------------------
    # PubChem ligand
    # --------------------------------------------------
    if (
        original_status == "RESOLVED"
        and rtype == "ligand"
        and (r.get("resolved_smiles") or "").strip()
        and r.get("source_database") == "PubChem"
    ):
        validation = "ACCEPT"
        reason = "PUBCHEM_NAME_WITH_SMILES"


    # --------------------------------------------------
    # UniProt protein
    # --------------------------------------------------
    elif (
        original_status == "RESOLVED"
        and rtype == "protein"
        and (r.get("resolved_sequence") or "").strip()
    ):

        target_tokens = important_tokens(target)
        match_tokens = important_tokens(match)

        if target_tokens:
            coverage = len(target_tokens & match_tokens) / len(target_tokens)

        extra_roles = has_extra_role(target, match)


        # Exact gene match is strongest evidence.
        if method.startswith("UNIPROT_EXACT_GENE"):
            validation = "ACCEPT"
            reason = "UNIPROT_EXACT_GENE"


        # Reject obvious target-vs-receptor/binder errors.
        elif extra_roles:
            validation = "REJECT"
            reason = ",".join(extra_roles)


        # Strong name agreement.
        elif coverage >= 0.60:
            validation = "ACCEPT"
            reason = f"STRONG_NAME_MATCH_{coverage:.3f}"


        # Borderline matches are kept separate rather than guessed.
        elif coverage >= 0.40:
            validation = "REVIEW"
            reason = f"BORDERLINE_NAME_MATCH_{coverage:.3f}"


        else:
            validation = "REJECT"
            reason = f"WEAK_NAME_MATCH_{coverage:.3f}"


    new = dict(r)

    new["validation_status"] = validation
    new["validation_reason"] = reason
    new["target_token_coverage"] = f"{coverage:.3f}"

    counts[validation] += 1
    out_rows.append(new)


with open(OUTPUT, "w", newline="", encoding="utf-8") as f:
    w = csv.DictWriter(f, fieldnames=new_fields)
    w.writeheader()
    w.writerows(out_rows)


print("========== TASK 5 VALIDATION SUMMARY ==========")

for k in ["ACCEPT", "REVIEW", "REJECT", "NOT_USABLE"]:
    print(f"{k:12s}: {counts[k]}")


print("\n========== ACCEPTED BY TYPE ==========")

accepted = [
    r for r in out_rows
    if r["validation_status"] == "ACCEPT"
]

for k, v in Counter(
    r.get("resolved_type", "") for r in accepted
).most_common():
    print(f"{k or '[blank]':12s}: {v}")


print("\n========== REVIEW / REJECT EXAMPLES ==========")

shown = 0

for r in out_rows:

    if r["validation_status"] not in {"REVIEW", "REJECT"}:
        continue

    print()
    print("TARGET :", r.get("target_normalized", ""))
    print("MATCH  :", (r.get("matched_name") or "")[:150])
    print("SOURCE :", r.get("source_id", ""))
    print("STATUS :", r["validation_status"])
    print("WHY    :", r["validation_reason"])

    shown += 1

    if shown >= 25:
        break


print()
print("Saved:", OUTPUT)
