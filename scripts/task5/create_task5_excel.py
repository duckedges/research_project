#!/usr/bin/env python3

import csv
from pathlib import Path
from collections import Counter

from openpyxl import Workbook
from openpyxl.styles import Font, PatternFill, Alignment, Border, Side
from openpyxl.formatting.rule import ColorScaleRule
from openpyxl.utils import get_column_letter
from openpyxl.worksheet.table import Table, TableStyleInfo


PROJECT = Path("/projects/bentosprg6/linwx/research_project")

RANKED = PROJECT / "data/task5/task5_affinity_ranked.csv"
STATUS = PROJECT / "data/task5/task5_all_status.csv"
TARGETS = PROJECT / "data/task5/targets/target_validated.csv"

OUTPUT = PROJECT / "data/task5/TASK5_Boltz2_Aptamer_Affinity_Results.xlsx"


def read_csv(path):
    with open(path, newline="", encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))


ranked = read_csv(RANKED)
status = read_csv(STATUS)
targets = read_csv(TARGETS)

counts = Counter(r["result_status"] for r in status)

accepted_targets = [
    r for r in targets
    if r.get("validation_status") == "ACCEPT"
]

# --------------------------------------------------
# Workbook
# --------------------------------------------------

wb = Workbook()
ws_summary = wb.active
ws_summary.title = "Summary"

ws_ranked = wb.create_sheet("Ranked_Affinity")
ws_attempts = wb.create_sheet("All_Attempts")
ws_targets = wb.create_sheet("Target_Resolution")


# --------------------------------------------------
# Common styles
# --------------------------------------------------

dark_fill = PatternFill("solid", fgColor="1F4E78")
light_fill = PatternFill("solid", fgColor="D9EAF7")
green_fill = PatternFill("solid", fgColor="E2F0D9")
red_fill = PatternFill("solid", fgColor="FCE4D6")
gray_fill = PatternFill("solid", fgColor="E7E6E6")

white_bold = Font(color="FFFFFF", bold=True)
bold = Font(bold=True)

thin_gray = Side(style="thin", color="D9E1F2")
border = Border(bottom=thin_gray)


def style_header(ws, row=1):
    for cell in ws[row]:
        cell.fill = dark_fill
        cell.font = white_bold
        cell.alignment = Alignment(
            horizontal="center",
            vertical="center",
            wrap_text=True
        )


def autosize(ws, max_width=42):
    widths = {}

    for row in ws.iter_rows():
        for cell in row:
            value = "" if cell.value is None else str(cell.value)
            widths[cell.column] = min(
                max(
                    widths.get(cell.column, 0),
                    len(value) + 2
                ),
                max_width
            )

    for col_idx, width in widths.items():
        ws.column_dimensions[
            get_column_letter(col_idx)
        ].width = max(10, width)


# --------------------------------------------------
# SUMMARY
# --------------------------------------------------

ws_summary["A1"] = "Task 5 — Boltz-2 Aptamer Affinity Prediction"
ws_summary["A1"].font = Font(bold=True, size=16, color="FFFFFF")
ws_summary["A1"].fill = dark_fill
ws_summary.merge_cells("A1:D1")

summary_rows = [
    ("Metric", "Value", "Interpretation", "Notes"),
    (
        "UTexas model records",
        1495,
        "Original prepared modeling records",
        "From Task 2 model manifest"
    ),
    (
        "Direct-ready aptamer chains",
        1217,
        "Aptamer sequences usable as Boltz inputs",
        "DNA/RNA sequence preparation complete"
    ),
    (
        "Target-resolved complexes attempted",
        342,
        "Complexes submitted to adapted Boltz-2",
        "135 accepted unique targets"
    ),
    (
        "Successful affinity predictions",
        counts.get("SUCCESS", 0),
        "Affinity JSON produced",
        "Included in Ranked_Affinity sheet"
    ),
    (
        "GPU OOM failures",
        counts.get("GPU_OOM", 0),
        "Structure prediction exceeded available GPU memory",
        "APC, MLL1, thyroglobulin oversized complexes"
    ),
    (
        "Successful prediction rate",
        counts.get("SUCCESS", 0) / len(status),
        "Among the 342 attempted complexes",
        ""
    ),
    (
        "Lowest affinity_pred_value",
        float(ranked[0]["affinity_pred_value"]),
        ranked[0]["run_id"],
        ranked[0]["target_normalized"]
    ),
    (
        "Highest affinity_pred_value",
        float(ranked[-1]["affinity_pred_value"]),
        ranked[-1]["run_id"],
        ranked[-1]["target_normalized"]
    ),
    (
        "Primary ranking field",
        "affinity_pred_value",
        "Sorted lowest → highest",
        "Boltz-2 model output"
    ),
]

for row in summary_rows:
    ws_summary.append(row)

for cell in ws_summary[2]:
    cell.fill = light_fill
    cell.font = bold

ws_summary["B7"].number_format = "0.0%"
ws_summary["B8"].number_format = "0.000000"
ws_summary["B9"].number_format = "0.000000"

ws_summary.append([])
ws_summary.append([
    "Important limitation",
    (
        "Boltz-2 affinity modeling was developed for "
        "protein–small-molecule complexes. The Task 4 code "
        "was modified so DNA/RNA aptamers can be designated "
        "as affinity binders. These aptamer affinity values "
        "demonstrate technical execution of the adapted "
        "workflow and should NOT be interpreted as validated "
        "experimental binding affinities."
    )
])

ws_summary["A13"].font = bold
ws_summary["A13"].fill = red_fill
ws_summary["B13"].alignment = Alignment(wrap_text=True, vertical="top")
ws_summary.merge_cells("B13:D15")

ws_summary.freeze_panes = "A3"


# --------------------------------------------------
# RANKED AFFINITY
# --------------------------------------------------

rank_fields = [
    "rank",
    "run_id",
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

ws_ranked.append(rank_fields)

numeric_fields = {
    "rank",
    "aptamer_length",
    "target_length",
    "experimental_kd_nm",
    "affinity_pred_value",
    "affinity_probability_binary",
    "affinity_pred_value1",
    "affinity_probability_binary1",
    "affinity_pred_value2",
    "affinity_probability_binary2",
}

for r in ranked:
    values = []

    for field in rank_fields:
        value = r.get(field, "")

        if field in numeric_fields and value not in ("", None):
            try:
                if field in {"rank", "aptamer_length", "target_length"}:
                    value = int(float(value))
                else:
                    value = float(value)
            except Exception:
                pass

        values.append(value)

    ws_ranked.append(values)

style_header(ws_ranked)
ws_ranked.freeze_panes = "A2"
ws_ranked.auto_filter.ref = ws_ranked.dimensions

for row in ws_ranked.iter_rows(min_row=2):
    for cell in row:
        cell.border = border
        cell.alignment = Alignment(vertical="top")

# Affinity columns
for col in ["O", "P", "Q", "R", "S", "T"]:
    for cell in ws_ranked[col][1:]:
        cell.number_format = "0.000000"

# Experimental Kd
for cell in ws_ranked["N"][1:]:
    cell.number_format = "0.000000"

# Color scale on primary prediction
ws_ranked.conditional_formatting.add(
    f"O2:O{ws_ranked.max_row}",
    ColorScaleRule(
        start_type="min",
        start_color="F8696B",
        mid_type="percentile",
        mid_value=50,
        mid_color="FFEB84",
        end_type="max",
        end_color="63BE7B"
    )
)

rank_table = Table(
    displayName="RankedAffinityTable",
    ref=ws_ranked.dimensions
)
rank_style = TableStyleInfo(
    name="TableStyleMedium2",
    showFirstColumn=False,
    showLastColumn=False,
    showRowStripes=True,
    showColumnStripes=False
)
rank_table.tableStyleInfo = rank_style
ws_ranked.add_table(rank_table)


# --------------------------------------------------
# ALL ATTEMPTS
# --------------------------------------------------

attempt_fields = [
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

ws_attempts.append(attempt_fields)

for r in status:
    vals = []

    for field in attempt_fields:
        value = r.get(field, "")

        if field in {
            "experimental_kd_nm",
            "affinity_pred_value",
            "affinity_probability_binary",
            "affinity_pred_value1",
            "affinity_probability_binary1",
            "affinity_pred_value2",
            "affinity_probability_binary2",
        } and value not in ("", None):
            try:
                value = float(value)
            except Exception:
                pass

        vals.append(value)

    ws_attempts.append(vals)

style_header(ws_attempts)
ws_attempts.freeze_panes = "A2"
ws_attempts.auto_filter.ref = ws_attempts.dimensions

for row in ws_attempts.iter_rows(min_row=2):
    status_cell = row[14]

    if status_cell.value == "GPU_OOM":
        for cell in row:
            cell.fill = red_fill
    else:
        status_cell.fill = green_fill

    for cell in row:
        cell.border = border
        cell.alignment = Alignment(vertical="top")

for col in ["H", "I", "J", "K", "L", "M", "N"]:
    for cell in ws_attempts[col][1:]:
        cell.number_format = "0.000000"

attempt_table = Table(
    displayName="AllAttemptsTable",
    ref=ws_attempts.dimensions
)
attempt_style = TableStyleInfo(
    name="TableStyleMedium9",
    showFirstColumn=False,
    showLastColumn=False,
    showRowStripes=True,
    showColumnStripes=False
)
attempt_table.tableStyleInfo = attempt_style
ws_attempts.add_table(attempt_table)


# --------------------------------------------------
# TARGET RESOLUTION
# --------------------------------------------------

target_fields = [
    "target_normalized",
    "aptamer_count",
    "direct_ready_aptamers",
    "triage_type",
    "resolved_type",
    "source_database",
    "source_id",
    "matched_name",
    "resolution_method",
    "task5_status",
    "validation_status",
    "validation_reason",
]

ws_targets.append(target_fields)

for r in targets:
    ws_targets.append([
        r.get(field, "")
        for field in target_fields
    ])

style_header(ws_targets)
ws_targets.freeze_panes = "A2"
ws_targets.auto_filter.ref = ws_targets.dimensions

for row in ws_targets.iter_rows(min_row=2):
    validation = row[10].value

    if validation == "ACCEPT":
        row[10].fill = green_fill
    elif validation == "REJECT":
        row[10].fill = red_fill
    elif validation == "REVIEW":
        row[10].fill = light_fill
    else:
        row[10].fill = gray_fill

    for cell in row:
        cell.alignment = Alignment(vertical="top", wrap_text=True)
        cell.border = border

target_table = Table(
    displayName="TargetResolutionTable",
    ref=ws_targets.dimensions
)
target_style = TableStyleInfo(
    name="TableStyleMedium4",
    showFirstColumn=False,
    showLastColumn=False,
    showRowStripes=True,
    showColumnStripes=False
)
target_table.tableStyleInfo = target_style
ws_targets.add_table(target_table)


# --------------------------------------------------
# Formatting / widths
# --------------------------------------------------

for ws in [ws_summary, ws_ranked, ws_attempts, ws_targets]:
    autosize(ws)

# Specific readable widths
ws_summary.column_dimensions["A"].width = 30
ws_summary.column_dimensions["B"].width = 32
ws_summary.column_dimensions["C"].width = 38
ws_summary.column_dimensions["D"].width = 48

for ws in [ws_ranked, ws_attempts]:
    for col in ["D", "E", "F", "G", "H", "I", "J"]:
        if col in ws.column_dimensions:
            pass

# Text-heavy columns
for col in ["H", "M", "W"]:
    if col in ws_ranked.column_dimensions:
        ws_ranked.column_dimensions[col].width = 36

ws_ranked.column_dimensions["H"].width = 42
ws_ranked.column_dimensions["W"].width = 55

ws_attempts.column_dimensions["F"].width = 42
ws_attempts.column_dimensions["P"].width = 55

ws_targets.column_dimensions["A"].width = 52
ws_targets.column_dimensions["H"].width = 55
ws_targets.column_dimensions["L"].width = 42

for ws in [ws_ranked, ws_attempts, ws_targets]:
    for row in ws.iter_rows():
        for cell in row:
            cell.alignment = Alignment(
                vertical="top",
                wrap_text=True
            )

wb.save(OUTPUT)

print("========== TASK 5 EXCEL CREATED ==========")
print("Workbook:", OUTPUT)
print("Ranked rows:", len(ranked))
print("All attempts:", len(status))
print("Accepted targets:", len(accepted_targets))
print("Sheets:", ", ".join(wb.sheetnames))
