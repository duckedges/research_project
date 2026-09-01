from pathlib import Path
import json
import csv

RUNS = [
    {
        "model_id": "UTEXAS_10000969_R1012",
        "target": "Human eIF4E (P06730)",
        "aptamer": "Apt 1",
        "aptamer_type": "DNA",
        "experimental_kd_nm": "",
        "job_id": "2952322",
        "runtime": "00:01:26",
        "root": Path(
            "af3_output/utexas_eif4e_apt1_no_msa/"
            "utexas_eif4e_apt1_no_msa"
        ),
    },
    {
        "model_id": "UTEXAS_10000342_R364",
        "target": "Mouse CCL2/MCP-1 (P10148 Q24-R96)",
        "aptamer": "ADR7",
        "aptamer_type": "RNA",
        "experimental_kd_nm": "0.18",
        "job_id": "2953660",
        "runtime": "00:01:17",
        "root": Path(
            "af3_output/utexas_adr7_ccl2_no_msa/"
            "utexas_adr7_ccl2_no_msa"
        ),
    },
]

OUTDIR = Path("task3_results")
OUTDIR.mkdir(exist_ok=True)

sample_rows = []
complex_rows = []

for run in RUNS:

    rows = []

    for sample_dir in sorted(run["root"].glob("seed-42_sample-*")):

        files = list(sample_dir.glob("*_summary_confidences.json"))

        if len(files) != 1:
            raise RuntimeError(
                f"Expected one summary file in {sample_dir}, found {len(files)}"
            )

        with open(files[0]) as f:
            x = json.load(f)

        sample = int(sample_dir.name.split("sample-")[-1])

        pair_iptm = x.get("chain_pair_iptm", [])
        pair_pae = x.get("chain_pair_pae_min", [])

        ab_iptm = ""
        ab_pae = ""

        if len(pair_iptm) >= 2:
            ab_iptm = pair_iptm[0][1]

        if len(pair_pae) >= 2:
            ab_pae = pair_pae[0][1]

        row = {
            "model_id": run["model_id"],
            "target": run["target"],
            "aptamer": run["aptamer"],
            "aptamer_type": run["aptamer_type"],
            "experimental_kd_nm": run["experimental_kd_nm"],
            "job_id": run["job_id"],
            "sample": sample,
            "ranking_score": x.get("ranking_score"),
            "ptm": x.get("ptm"),
            "iptm": x.get("iptm"),
            "protein_aptamer_iptm": ab_iptm,
            "protein_aptamer_pae_min": ab_pae,
            "has_clash": x.get("has_clash"),
        }

        rows.append(row)
        sample_rows.append(row)

    if len(rows) != 5:
        raise RuntimeError(
            f"{run['model_id']} has {len(rows)} samples instead of 5"
        )

    best = max(
        rows,
        key=lambda r: float(r["ranking_score"])
    )

    complex_rows.append({
        "model_id": run["model_id"],
        "target": run["target"],
        "aptamer": run["aptamer"],
        "aptamer_type": run["aptamer_type"],
        "experimental_kd_nm": run["experimental_kd_nm"],
        "job_id": run["job_id"],
        "runtime": run["runtime"],
        "best_sample": best["sample"],
        "best_ranking_score": best["ranking_score"],
        "best_ptm": best["ptm"],
        "best_iptm": best["iptm"],
        "best_protein_aptamer_iptm":
            best["protein_aptamer_iptm"],
        "best_protein_aptamer_pae_min":
            best["protein_aptamer_pae_min"],
        "has_clash": best["has_clash"],
        "af3_mode": "no_MSA_no_templates",
    })


def write_csv(path, rows):
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=rows[0].keys()
        )
        writer.writeheader()
        writer.writerows(rows)


write_csv(
    OUTDIR / "af3_all_sample_metrics.csv",
    sample_rows
)

write_csv(
    OUTDIR / "af3_complex_summary.csv",
    complex_rows
)


print("\n===== TASK 3 AF3 COMPLEX SUMMARY =====")

for r in complex_rows:
    print()
    print("Model:", r["model_id"])
    print("Complex:", r["target"], "+", r["aptamer"])
    print("Experimental Kd:", r["experimental_kd_nm"] or "not reported")
    print("Job:", r["job_id"])
    print("Runtime:", r["runtime"])
    print("Best sample:", r["best_sample"])
    print("Ranking score:", r["best_ranking_score"])
    print("pTM:", r["best_ptm"])
    print("ipTM:", r["best_iptm"])
    print(
        "Protein-aptamer ipTM:",
        r["best_protein_aptamer_iptm"]
    )
    print(
        "Protein-aptamer PAE min:",
        r["best_protein_aptamer_pae_min"]
    )
    print("Clash:", r["has_clash"])

print()
print("Written:")
print(" task3_results/af3_all_sample_metrics.csv")
print(" task3_results/af3_complex_summary.csv")
