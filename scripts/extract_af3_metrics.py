from pathlib import Path
import json
import csv

ROOT = Path(
    "af3_output/utexas_eif4e_apt1_no_msa/"
    "utexas_eif4e_apt1_no_msa"
)

MODEL_ID = "UTEXAS_10000969_R1012"
TARGET = "Human eIF4E (P06730)"
APTAMER = "Apt 1"
MODE = "no_MSA_no_templates"

rows = []

for sample_dir in sorted(ROOT.glob("seed-42_sample-*")):

    f = next(sample_dir.glob("*_summary_confidences.json"))

    with open(f) as h:
        x = json.load(h)

    sample = sample_dir.name.split("sample-")[-1]

    chain_pair = x.get("chain_pair_iptm", [])
    chain_pae = x.get("chain_pair_pae_min", [])

    # A = protein, B = aptamer
    ab_iptm = ""
    ab_pae = ""

    if len(chain_pair) >= 2:
        ab_iptm = chain_pair[0][1]

    if len(chain_pae) >= 2:
        ab_pae = chain_pae[0][1]

    rows.append({
        "model_id": MODEL_ID,
        "target": TARGET,
        "aptamer": APTAMER,
        "mode": MODE,
        "seed": 42,
        "sample": sample,
        "ranking_score": x.get("ranking_score"),
        "ptm": x.get("ptm"),
        "iptm": x.get("iptm"),
        "protein_aptamer_iptm": ab_iptm,
        "protein_aptamer_pae_min": ab_pae,
        "has_clash": x.get("has_clash"),
    })

rows.sort(
    key=lambda r: float(r["ranking_score"]),
    reverse=True
)

out = Path("task3_results/af3_sample_metrics.csv")

with open(out, "w", newline="") as h:
    writer = csv.DictWriter(h, fieldnames=rows[0].keys())
    writer.writeheader()
    writer.writerows(rows)

print("\nAF3 SAMPLE RESULTS")
print("==================")

for r in rows:
    print(
        f"sample={r['sample']} "
        f"ranking={r['ranking_score']} "
        f"pTM={r['ptm']} "
        f"ipTM={r['iptm']} "
        f"A-B ipTM={r['protein_aptamer_iptm']} "
        f"A-B PAEmin={r['protein_aptamer_pae_min']}"
    )

print()
print("Best sample:", rows[0]["sample"])
print("Written:", out)
