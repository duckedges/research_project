#!/usr/bin/env python3

import csv
import json
import re
import time
import urllib.parse
import urllib.request
import urllib.error
from pathlib import Path

PROJECT = Path("/projects/bentosprg6/linwx/research_project")

INPUT = PROJECT / "data/task5/targets/target_triage.csv"
OUTPUT = PROJECT / "data/task5/targets/target_resolved.csv"

# Common species appearing in aptamer literature.
SPECIES = {
    "human": 9606,
    "homo sapiens": 9606,
    "mouse": 10090,
    "mus musculus": 10090,
    "rat": 10116,
    "rattus norvegicus": 10116,
    "rabbit": 9986,
    "bovine": 9913,
    "cow": 9913,
    "escherichia coli": 562,
    "e. coli": 562,
    "bacillus subtilis": 1423,
    "saccharomyces cerevisiae": 559292,
    "yeast": 559292,
}

PROTEIN_WORDS = [
    "protein", "factor", "receptor", "enzyme", "kinase",
    "thrombin", "immunoglobulin", "antigen", "polymerase",
    "transcriptase", "streptavidin", "hemagglutinin",
    "growth factor", "interleukin", "integrin",
    "protease", "toxin", "allergen", "chemokine",
    "lactoferrin", "elastase", "mammaglobin",
    "peptide", "antibody", "ribonuclease"
]

UNSUPPORTED_WORDS = [
    "cell line", " cells", "oocyst", "whole cell",
    "bacterium", "bacteria", "virion"
]


def get_json(url, timeout=20):
    req = urllib.request.Request(
        url,
        headers={"User-Agent": "RA-Boltz2-Task5/1.0"}
    )
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read().decode("utf-8"))


def get_text(url, timeout=20):
    req = urllib.request.Request(
        url,
        headers={"User-Agent": "RA-Boltz2-Task5/1.0"}
    )
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return r.read().decode("utf-8")


def norm(s):
    s = s.lower()
    s = s.replace("α", " alpha ")
    s = s.replace("β", " beta ")
    s = s.replace("γ", " gamma ")
    s = re.sub(r"[^a-z0-9]+", " ", s)
    return " ".join(s.split())


def detect_taxid(name):
    n = norm(name)
    for species, taxid in SPECIES.items():
        if norm(species) in n:
            return taxid, species
    return None, ""


def clean_target_name(name):
    s = name

    # Remove common vendor/source annotations.
    s = re.sub(
        r"\b(Sigma|Human|Mouse|Rat|Yeast|Bovine)\b",
        " ",
        s,
        flags=re.I
    )

    # Remove catalog/cell identifiers and empty punctuation.
    s = re.sub(r"\bCRL[- ]?\d+\b", " ", s, flags=re.I)
    s = re.sub(r"\s+", " ", s)

    return s.strip(" ,;-")


def looks_protein(name, triage):
    n = norm(name)

    if triage == "LIKELY_PROTEIN":
        return True

    return any(norm(x) in n for x in PROTEIN_WORDS)


def looks_unsupported(name, triage):
    n = norm(name)

    if triage in {"CELL_OR_CELL_LINE", "ORGANISM_OR_PARTICLE"}:
        return True

    return any(norm(x) in n for x in UNSUPPORTED_WORDS)


def pubchem_lookup(name):
    # Try the original target and then a version with parenthetical aliases removed.
    candidates = [name]

    stripped = re.sub(r"\([^)]*\)", " ", name)
    stripped = re.sub(r"\s+", " ", stripped).strip(" ,;-")

    if stripped and stripped != name:
        candidates.append(stripped)

    for q in candidates:
        encoded = urllib.parse.quote(q, safe="")
        url = (
            "https://pubchem.ncbi.nlm.nih.gov/rest/pug/"
            f"compound/name/{encoded}/property/CanonicalSMILES,IsomericSMILES/JSON"
        )

        try:
            data = get_json(url)
            props = data["PropertyTable"]["Properties"][0]

            smiles = (
                props.get("ConnectivitySMILES")
                or props.get("CanonicalSMILES")
                or props.get("SMILES")
                or props.get("IsomericSMILES")
                or ""
            )

            if smiles:
                return {
                    "type": "ligand",
                    "smiles": smiles,
                    "source_id": str(props.get("CID", "")),
                    "matched_name": q,
                    "method": "PUBCHEM_NAME"
                }
        except Exception:
            pass

        time.sleep(0.10)

    return None


def uniprot_lookup(name):
    taxid, species = detect_taxid(name)
    cleaned = clean_target_name(name)

    # Parenthetical abbreviations often contain useful gene symbols.
    abbreviations = re.findall(r"\(([A-Za-z0-9αβγ\-]{2,15})\)", name)

    # Try exact gene matching first where possible.
    if taxid:
        for abbr in abbreviations:
            gene = (
                abbr.replace("α", "A")
                    .replace("β", "B")
                    .replace("γ", "G")
                    .replace("-", "")
            )

            if re.fullmatch(r"[A-Za-z][A-Za-z0-9]{1,11}", gene):
                query = (
                    f"(gene_exact:{gene}) AND "
                    f"(organism_id:{taxid}) AND "
                    f"(reviewed:true)"
                )

                fields = (
                    "accession,id,protein_name,gene_names,"
                    "organism_id,length,sequence"
                )

                url = (
                    "https://rest.uniprot.org/uniprotkb/search?"
                    + urllib.parse.urlencode({
                        "query": query,
                        "format": "tsv",
                        "fields": fields,
                        "size": 5
                    })
                )

                try:
                    text = get_text(url)
                    rows = list(csv.DictReader(text.splitlines(), delimiter="\t"))

                    if len(rows) == 1 and rows[0].get("Sequence"):
                        r = rows[0]
                        return {
                            "type": "protein",
                            "sequence": r["Sequence"],
                            "source_id": r.get("Entry", ""),
                            "matched_name": r.get("Protein names", ""),
                            "method": "UNIPROT_EXACT_GENE"
                        }
                except Exception:
                    pass

                time.sleep(0.10)

    # Protein-name search.
    qname = re.sub(r"\([^)]*\)", " ", cleaned)
    qname = re.sub(r"\s+", " ", qname).strip(" ,;-")

    if len(qname) < 3:
        return None

    query = f'protein_name:"{qname}"'

    if taxid:
        query += f" AND organism_id:{taxid}"

    query += " AND reviewed:true"

    fields = (
        "accession,id,protein_name,gene_names,"
        "organism_id,length,sequence"
    )

    url = (
        "https://rest.uniprot.org/uniprotkb/search?"
        + urllib.parse.urlencode({
            "query": query,
            "format": "tsv",
            "fields": fields,
            "size": 10
        })
    )

    try:
        text = get_text(url)
        rows = list(csv.DictReader(text.splitlines(), delimiter="\t"))
    except Exception:
        return None

    if not rows:
        return None

    query_tokens = set(norm(qname).split())

    best = None
    best_score = -1

    for r in rows:
        seq = r.get("Sequence", "")
        pname = r.get("Protein names", "")

        if not seq:
            continue

        pnorm = norm(pname)
        candidate_tokens = set(pnorm.split())

        overlap = len(query_tokens & candidate_tokens)
        score = overlap / max(1, len(query_tokens))

        # Strong bonus if cleaned target appears directly in candidate name.
        if norm(qname) in pnorm:
            score += 1.0

        # Penalize common false positives unless requested.
        for bad in ["binding protein", "transporter", "homolog"]:
            if bad in pnorm and bad not in norm(qname):
                score -= 0.35

        if score > best_score:
            best_score = score
            best = r

    # Conservative threshold.
    if best and best_score >= 0.60:
        return {
            "type": "protein",
            "sequence": best.get("Sequence", ""),
            "source_id": best.get("Entry", ""),
            "matched_name": best.get("Protein names", ""),
            "method": f"UNIPROT_NAME_SCORE_{best_score:.3f}"
        }

    return None


with open(INPUT, newline="", encoding="utf-8-sig") as f:
    rows = list(csv.DictReader(f))

out_fields = list(rows[0].keys()) + [
    "resolved_type",
    "resolved_sequence",
    "resolved_smiles",
    "source_database",
    "source_id",
    "matched_name",
    "resolution_method",
    "task5_status"
]

resolved = []
counts = {}

for i, row in enumerate(rows, 1):
    name = (row.get("target_normalized") or "").strip()
    triage = (row.get("triage_type") or "").strip()

    result = None
    status = "UNRESOLVED"

    if looks_unsupported(name, triage):
        status = "UNSUPPORTED_COMPLEX_TARGET"

    elif triage == "NUCLEIC_ACID_TARGET":
        status = "UNRESOLVED_NUCLEIC_ACID_TARGET"

    else:
        # For clearly protein-like targets, try UniProt first.
        if looks_protein(name, triage):
            result = uniprot_lookup(name)

            # Some names that look protein-like may actually be chemicals/toxins.
            if result is None:
                result = pubchem_lookup(name)

        else:
            # Unknown names: PubChem first, then UniProt.
            result = pubchem_lookup(name)

            if result is None:
                result = uniprot_lookup(name)

        if result:
            status = "RESOLVED"

    new = dict(row)

    new["resolved_type"] = ""
    new["resolved_sequence"] = ""
    new["resolved_smiles"] = ""
    new["source_database"] = ""
    new["source_id"] = ""
    new["matched_name"] = ""
    new["resolution_method"] = ""
    new["task5_status"] = status

    if result:
        new["resolved_type"] = result["type"]

        if result["type"] == "protein":
            new["resolved_sequence"] = result.get("sequence", "")
            new["source_database"] = "UniProt"
        elif result["type"] == "ligand":
            new["resolved_smiles"] = result.get("smiles", "")
            new["source_database"] = "PubChem"

        new["source_id"] = result.get("source_id", "")
        new["matched_name"] = result.get("matched_name", "")
        new["resolution_method"] = result.get("method", "")

    counts[status] = counts.get(status, 0) + 1
    resolved.append(new)

    print(
        f"[{i:03d}/{len(rows)}] "
        f"{status:32s} "
        f"{name[:75]}",
        flush=True
    )

    time.sleep(0.05)

OUTPUT.parent.mkdir(parents=True, exist_ok=True)

with open(OUTPUT, "w", newline="", encoding="utf-8") as f:
    w = csv.DictWriter(f, fieldnames=out_fields)
    w.writeheader()
    w.writerows(resolved)

print()
print("========== TASK 5 RESOLUTION SUMMARY ==========")

for k, v in sorted(counts.items()):
    print(f"{k:32s} {v}")

print()
print("Saved:", OUTPUT)
