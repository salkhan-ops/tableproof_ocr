from __future__ import annotations

import csv
import json
from pathlib import Path

import fitz
from openpyxl import Workbook


ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw_pdfs"
GT = ROOT / "data" / "ground_truth"
MANIFEST = ROOT / "data" / "span_benchmark_manifest.csv"


def main() -> None:
    RAW.mkdir(parents=True, exist_ok=True)
    GT.mkdir(parents=True, exist_ok=True)
    rows = []
    for spec in _specs():
        table_id = spec["table_id"]
        pdf_path = RAW / f"{table_id}.pdf"
        gt_xlsx = GT / f"{table_id}_ground_truth.xlsx"
        gt_json = GT / f"{table_id}_ground_truth.json"
        _write_pdf(pdf_path, spec)
        _write_xlsx(gt_xlsx, spec["matrix"], spec["merged_ranges"])
        payload = _ground_truth_payload(spec)
        gt_json.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        rows.append(
            {
                "table_id": table_id,
                "pdf_path": f"data/raw_pdfs/{table_id}.pdf",
                "page_number": 1,
                "bbox": ",".join(str(item) for item in spec["bbox"]),
                "category": spec["category"],
                "ground_truth_path": f"data/ground_truth/{table_id}_ground_truth.xlsx",
                "ground_truth_json_path": f"data/ground_truth/{table_id}_ground_truth.json",
                "expected_physical_rows": len(spec["matrix"]),
                "expected_physical_columns": max(len(row) for row in spec["matrix"]),
                "expected_logical_cells": len(payload["logical_cells"]),
                "expected_header_levels": spec.get("header_rows", 1),
                "expected_row_groups": len(spec.get("row_groups", [])),
                "expected_merged_ranges": json.dumps(spec["merged_ranges"], sort_keys=True),
                "expected_parent_child_spans": json.dumps(spec.get("parent_child_links", []), sort_keys=True),
                "expected_genuine_empty_cells": json.dumps(spec.get("genuine_empty_cells", []), sort_keys=True),
                "expected_unresolved_spans": json.dumps(spec.get("unresolved_spans", []), sort_keys=True),
                "difficulty_notes": spec["notes"],
                "manually_verified": "true",
            }
        )
    with MANIFEST.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _specs() -> list[dict]:
    return [
        _spec(
            "span_01",
            "three-level nested header",
            "Parent six-column header, two child groups, six leaves, and stub header row span.",
            [
                ["Segment", "Performance Summary", "", "", "", "", ""],
                ["", "Operating Measures", "", "", "Financial Measures", "", ""],
                ["", "Revenue", "Cost", "Margin", "Assets", "Debt", "Equity"],
                ["North", "100", "60", "40", "450", "210", "240"],
                ["South", "120", "70", "50", "510", "260", "250"],
                ["West", "90", "55", "35", "390", "150", "240"],
            ],
            [_span(0, 0, "Segment", 3, 1, True, True), _span(0, 1, "Performance Summary", 1, 6, True, False), _span(1, 1, "Operating Measures", 1, 3, True, False), _span(1, 4, "Financial Measures", 1, 3, True, False)],
            [60, 145, 218, 291, 370, 448, 526],
            (52, 72, 590, 205),
            3,
            parent_child_links=[["nested_header_span_0_1_1x6", "nested_header_span_1_1_1x3"], ["nested_header_span_0_1_1x6", "nested_header_span_1_4_1x3"]],
        ),
        _spec(
            "span_02",
            "asymmetric nested header",
            "One five-column parent with unequal two-column and three-column child groups plus wrapped parent text.",
            [
                ["Category", "Operational and financial overview", "", "", "", ""],
                ["", "Short horizon", "", "Long horizon", "", ""],
                ["", "Units", "Revenue", "Assets", "Debt", "Cash"],
                ["A", "12", "100", "450", "210", "90"],
                ["B", "9", "120", "500", "240", "85"],
            ],
            [_span(0, 0, "Category", 3, 1, True, True), _span(0, 1, "Operational and financial overview summary", 1, 5, True, False), _span(1, 1, "Short horizon", 1, 2, True, False), _span(1, 3, "Long horizon", 1, 3, True, False)],
            [58, 150, 240, 330, 420, 505],
            (50, 72, 590, 220),
            3,
            visual_rows=[
                ["Category", "Operational and financial overview", "", "", "", ""],
                ["", "summary", "", "", "", ""],
                ["", "Short horizon", "", "Long horizon", "", ""],
                ["", "Units", "Revenue", "Assets", "Debt", "Cash"],
                ["A", "12", "100", "450", "210", "90"],
                ["B", "9", "120", "500", "240", "85"],
            ],
        ),
        _spec(
            "span_03",
            "multiple adjacent header groups",
            "Adjacent top-level groups with widths one, two, and three, including an ordinary unmerged header.",
            [
                ["Item", "Standalone", "Regional Group", "", "Product Group", "", ""],
                ["", "Rate", "East", "West", "Alpha", "Beta", "Gamma"],
                ["A", "5", "100", "95", "30", "28", "26"],
                ["B", "6", "110", "98", "33", "29", "27"],
            ],
            [_span(0, 0, "Item", 2, 1, True, True), _span(0, 2, "Regional Group", 1, 2, True, False), _span(0, 4, "Product Group", 1, 3, True, False)],
            [60, 145, 245, 325, 405, 480, 552],
            (54, 72, 598, 178),
            2,
        ),
        _spec(
            "span_04",
            "complex row-spanning stub",
            "Two vertical category labels in the first stub column and a genuine empty numeric cell.",
            [
                ["Category", "Line item", "2024", "2025", "Change"],
                ["Operating", "Revenue", "100", "120", "20"],
                ["", "Cost", "60", "70", "10"],
                ["", "Margin", "40", "", "5"],
                ["Investing", "Capital spend", "30", "45", "15"],
                ["", "Acquisitions", "10", "12", "2"],
            ],
            [_span(1, 0, "Operating", 3, 1, False, True), _span(4, 0, "Investing", 2, 1, False, True)],
            [58, 165, 300, 380, 462],
            (50, 72, 540, 205),
            1,
            genuine_empty_cells=[[3, 3]],
            row_groups=[_span(1, 0, "Operating", 3, 1, False, True), _span(4, 0, "Investing", 2, 1, False, True)],
        ),
        _spec(
            "span_05",
            "nested row groups",
            "Top-level row groups in column one and nested subcategories in column two.",
            [
                ["Group", "Subgroup", "Metric", "2024", "2025"],
                ["Assets", "Current", "Cash", "20", "24"],
                ["", "", "Inventory", "12", "14"],
                ["", "Non-current", "Property", "90", "95"],
                ["", "", "Intangibles", "18", "20"],
                ["Liabilities", "Current", "Payables", "22", "25"],
                ["", "", "Accruals", "11", "13"],
            ],
            [_span(1, 0, "Assets", 4, 1, False, True), _span(1, 1, "Current", 2, 1, False, True), _span(3, 1, "Non-current", 2, 1, False, True), _span(5, 0, "Liabilities", 2, 1, False, True), _span(5, 1, "Current", 2, 1, False, True)],
            [58, 150, 270, 405, 485],
            (50, 72, 560, 225),
            1,
            row_groups=[_span(1, 0, "Assets", 4, 1, False, True), _span(1, 1, "Current", 2, 1, False, True), _span(3, 1, "Non-current", 2, 1, False, True), _span(5, 0, "Liabilities", 2, 1, False, True), _span(5, 1, "Current", 2, 1, False, True)],
        ),
        _spec(
            "span_06",
            "mixed row and column spans",
            "Top-left stub spans three header rows and two stub columns; body stub spans multiple rows.",
            [
                ["Descriptor", "", "Results", "", "", ""],
                ["", "", "Q1", "", "Q2", ""],
                ["", "", "Revenue", "Cost", "Revenue", "Cost"],
                ["Consumer", "Product A", "100", "70", "120", "80"],
                ["", "Product B", "90", "", "110", "75"],
            ],
            [_span(0, 0, "Descriptor", 3, 2, True, True), _span(0, 2, "Results", 1, 4, True, False), _span(1, 2, "Q1", 1, 2, True, False), _span(1, 4, "Q2", 1, 2, True, False), _span(3, 0, "Consumer", 2, 1, False, True)],
            [58, 165, 285, 365, 445, 525],
            (50, 72, 590, 190),
            3,
            genuine_empty_cells=[[4, 3]],
        ),
        _spec(
            "span_07",
            "staggered header depth",
            "Different columns have one, two, and three header levels with covered blank-looking coordinates.",
            [
                ["Item", "Actual", "Forecast", "", "Scenario", "", ""],
                ["", "", "Near term", "", "Base", "Upside", "Downside"],
                ["", "", "2024", "2025", "", "", ""],
                ["A", "10", "12", "14", "16", "18", "15"],
                ["B", "11", "13", "15", "17", "19", "16"],
            ],
            [_span(0, 0, "Item", 3, 1, True, True), _span(0, 1, "Actual", 3, 1, True, False), _span(0, 2, "Forecast", 1, 2, True, False), _span(1, 2, "Near term", 1, 2, True, False), _span(0, 4, "Scenario", 1, 3, True, False)],
            [60, 150, 240, 320, 405, 485, 560],
            (52, 72, 604, 195),
            3,
        ),
        _spec(
            "span_08",
            "regression panels and subpanels",
            "Panel rows, grouped model columns, standard-error rows, and a significance note outside grid.",
            [
                ["Variable", "Baseline Models", "", "Extended Models", "", ""],
                ["", "M1", "M2", "M3", "M4", "M5"],
                ["Panel A: Full Sample", "", "", "", "", ""],
                ["Treatment", "0.10***", "0.11***", "0.09**", "0.08**", "0.07*"],
                ["", "(0.02)", "(0.03)", "(0.04)", "(0.03)", "(0.04)"],
                ["Panel B: Restricted Sample", "", "", "", "", ""],
                ["Treatment", "0.12***", "0.13***", "0.10**", "0.09**", "0.08*"],
                ["", "(0.03)", "(0.03)", "(0.05)", "(0.04)", "(0.05)"],
            ],
            [_span(0, 0, "Variable", 2, 1, True, True), _span(0, 1, "Baseline Models", 1, 2, True, False), _span(0, 3, "Extended Models", 1, 3, True, False), _span(2, 0, "Panel A: Full Sample", 1, 6, True, False), _span(5, 0, "Panel B: Restricted Sample", 1, 6, True, False)],
            [56, 170, 250, 330, 410, 490],
            (48, 72, 590, 250),
            2,
            footnotes=["*** p<0.01, ** p<0.05, * p<0.10"],
        ),
        _spec(
            "span_09",
            "financial statement row groups",
            "Nested financial statement-style row groups and period column groups, without arithmetic validation.",
            [
                ["Line", "Periods", "", "", ""],
                ["", "2023", "", "2024", ""],
                ["", "Amount", "Share", "Amount", "Share"],
                ["Assets", "Current assets", "", "", ""],
                ["", "Cash", "20", "15%", "24"],
                ["", "Inventory", "12", "9%", "14"],
                ["", "Non-current assets", "", "", ""],
                ["", "Property", "90", "68%", "95"],
                ["Liabilities", "Current liabilities", "", "", ""],
                ["", "Payables", "22", "17%", "25"],
            ],
            [_span(0, 0, "Line", 3, 1, True, True), _span(0, 1, "Periods", 1, 4, True, False), _span(1, 1, "2023", 1, 2, True, False), _span(1, 3, "2024", 1, 2, True, False), _span(3, 0, "Assets", 5, 1, False, True), _span(3, 1, "Current assets", 3, 1, False, True), _span(6, 1, "Non-current assets", 2, 1, False, True), _span(8, 0, "Liabilities", 2, 1, False, True), _span(8, 1, "Current liabilities", 2, 1, False, True)],
            [56, 170, 285, 385, 485],
            (48, 72, 580, 280),
            3,
        ),
        _spec(
            "span_10",
            "intentionally ambiguous span",
            "A sparse header row where span and non-span interpretations are both plausible; must remain unresolved.",
            [
                ["Measure", "Combined", "", "Separate"],
                ["Alpha", "10", "11", "12"],
                ["Beta", "11", "12", "13"],
            ],
            [],
            [60, 190, 310, 430],
            (52, 72, 520, 160),
            1,
            unresolved_spans=[{"anchor_row": 0, "anchor_column": 1, "candidate_text": "Combined", "competing_interpretations": ["column span over empty metric", "ordinary sparse header with genuine empty column"]}],
            genuine_empty_cells=[[0, 2]],
        ),
    ]


def _spec(table_id: str, category: str, notes: str, matrix: list[list[str]], merged_ranges: list[dict], columns: list[float], bbox: tuple[int, int, int, int], header_rows: int, parent_child_links: list[list[str]] | None = None, row_groups: list[dict] | None = None, genuine_empty_cells: list[list[int]] | None = None, unresolved_spans: list[dict] | None = None, footnotes: list[str] | None = None, visual_rows: list[list[str]] | None = None) -> dict:
    return {
        "table_id": table_id,
        "category": category,
        "notes": notes,
        "matrix": matrix,
        "visual_rows": visual_rows or matrix,
        "merged_ranges": merged_ranges,
        "columns": columns,
        "bbox": bbox,
        "header_rows": header_rows,
        "parent_child_links": parent_child_links or [],
        "row_groups": row_groups or [],
        "genuine_empty_cells": genuine_empty_cells or [],
        "unresolved_spans": unresolved_spans or [],
        "footnotes": footnotes or [],
    }


def _span(row: int, col: int, text: str, row_span: int, col_span: int, is_header: bool, is_stub: bool) -> dict:
    return {
        "row_index": row,
        "column_index": col,
        "text": text,
        "row_span": row_span,
        "column_span": col_span,
        "covered_coordinates": [[r, c] for r in range(row, row + row_span) for c in range(col, col + col_span)],
        "is_header": is_header,
        "is_stub": is_stub,
    }


def _ground_truth_payload(spec: dict) -> dict:
    covered = {tuple(coord) for span in spec["merged_ranges"] for coord in span["covered_coordinates"][1:]}
    logical = []
    span_lookup = {(span["row_index"], span["column_index"]): span for span in spec["merged_ranges"]}
    for row_index, row in enumerate(spec["matrix"]):
        for column_index, text in enumerate(row):
            if (row_index, column_index) in covered:
                continue
            logical.append(
                span_lookup.get(
                    (row_index, column_index),
                    _span(row_index, column_index, text, 1, 1, row_index < spec["header_rows"], column_index == 0),
                )
            )
    return {
        "table_id": spec["table_id"],
        "physical_rows": len(spec["matrix"]),
        "physical_columns": max(len(row) for row in spec["matrix"]),
        "logical_cell_count": len(logical),
        "logical_cells": logical,
        "merged_ranges": spec["merged_ranges"],
        "expected_merged_ranges": spec["merged_ranges"],
        "genuine_empty_coordinates": spec["genuine_empty_cells"],
        "parent_child_hierarchy": spec["parent_child_links"],
        "sibling_relationships": [],
        "section_boundaries": [span for span in spec["merged_ranges"] if span["column_span"] == max(len(row) for row in spec["matrix"])],
        "row_group_boundaries": spec["row_groups"],
        "expected_unresolved_spans": spec["unresolved_spans"],
        "expected_text_in_anchor_cells": [{"row_index": cell["row_index"], "column_index": cell["column_index"], "text": cell["text"]} for cell in logical if cell["text"]],
        "header_rows": spec["header_rows"],
        "body_rows": len(spec["matrix"]) - spec["header_rows"],
        "notes_and_footnotes": spec["footnotes"],
        "manually_verified": True,
    }


def _write_pdf(path: Path, spec: dict) -> None:
    doc = fitz.open()
    page = doc.new_page(width=612, height=792)
    y = 82
    for row in spec["visual_rows"]:
        for text, x in zip(row, spec["columns"]):
            if text:
                page.insert_text((x, y), text, fontsize=8.5)
        y += 15
    for note in spec.get("footnotes", []):
        y += 6
        page.insert_text((spec["columns"][0], y), note, fontsize=7.5)
    doc.save(path)
    doc.close()


def _write_xlsx(path: Path, matrix: list[list[str]], spans: list[dict]) -> None:
    wb = Workbook()
    ws = wb.active
    for row_index, row in enumerate(matrix, start=1):
        for col_index, text in enumerate(row, start=1):
            ws.cell(row_index, col_index, text)
    for span in spans:
        if span["row_span"] > 1 or span["column_span"] > 1:
            ws.merge_cells(
                start_row=span["row_index"] + 1,
                start_column=span["column_index"] + 1,
                end_row=span["row_index"] + span["row_span"],
                end_column=span["column_index"] + span["column_span"],
            )
    path.parent.mkdir(parents=True, exist_ok=True)
    wb.save(path)


if __name__ == "__main__":
    main()
