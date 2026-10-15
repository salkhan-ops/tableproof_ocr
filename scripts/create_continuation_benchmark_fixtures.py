from __future__ import annotations

import csv
import json
from pathlib import Path

import fitz


ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw_pdfs"
GT = ROOT / "data" / "ground_truth"
MANIFEST = ROOT / "data" / "continuation_benchmark_manifest.csv"


def main() -> None:
    RAW.mkdir(parents=True, exist_ok=True)
    GT.mkdir(parents=True, exist_ok=True)
    specs = _specs()
    rows = []
    for idx, spec in enumerate(specs, start=1):
        table_id = f"continuation_{idx:02d}"
        pdf_path = RAW / f"{table_id}.pdf"
        gt_path = GT / f"{table_id}_ground_truth.csv"
        raw_path = GT / f"{table_id}_raw_visual_lines.json"
        meta_path = GT / f"{table_id}_ground_truth.json"
        _write_pdf(pdf_path, spec)
        _write_csv(gt_path, spec["ground_truth"])
        raw_path.write_text(json.dumps(spec["raw_visual_lines"], indent=2) + "\n", encoding="utf-8")
        meta = {
            "physical_rows": spec["physical_rows"],
            "physical_columns": spec["physical_columns"],
            "logical_cell_count": len(spec["logical_cells"]),
            "logical_cells": spec["logical_cells"],
            "raw_visual_lines": spec["raw_visual_lines"],
            "ordinary_cells": spec["ordinary_cells"],
            "multi_line_cells": spec["multi_line_cells"],
            "row_spans": spec["row_spans"],
            "column_spans": spec["column_spans"],
            "section_headers": spec["section_headers"],
            "notes_and_footnotes": spec.get("expected_footnotes", []),
            "expected_empty_cells": spec.get("expected_empty_cells", []),
            "expected_continuations": spec.get("expected_continuations", []),
            "expected_unresolved_lines": spec.get("expected_unresolved_lines", []),
            "expected_footnotes": spec.get("expected_footnotes", []),
            "expected_merged_ranges": spec.get("expected_merged_ranges", []),
            "merged_ranges": spec.get("expected_merged_ranges", []),
            "source_word_mappings": spec.get("source_word_mappings", []),
            "case_types": spec.get("case_types", []),
        }
        meta_path.write_text(json.dumps(meta, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        rows.append(
            {
                "table_id": table_id,
                "pdf_path": f"data/raw_pdfs/{table_id}.pdf",
                "page_number": 1,
                "bbox": spec.get("bbox") or "54,72,470,260",
                "category": spec["category"],
                "ground_truth_path": f"data/ground_truth/{table_id}_ground_truth.csv",
                "raw_visual_lines_path": f"data/ground_truth/{table_id}_raw_visual_lines.json",
                "expected_physical_rows": len(spec["ground_truth"]),
                "expected_physical_columns": max(len(row) for row in spec["ground_truth"]),
                "expected_logical_cells": len(spec["logical_cells"]),
                "expected_merged_ranges": json.dumps(spec.get("expected_merged_ranges", [])),
                "expected_continuations": json.dumps(spec.get("expected_continuations", [])),
                "expected_unresolved_lines": json.dumps(spec.get("expected_unresolved_lines", [])),
                "expected_empty_cells": json.dumps(spec.get("expected_empty_cells", [])),
                "difficulty_notes": spec["notes"],
                "manually_verified": "true",
                "ground_truth_json_path": f"data/ground_truth/{table_id}_ground_truth.json",
            }
        )
    with MANIFEST.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _specs() -> list[dict]:
    return [
        _spec("wrapped first-column label", [["Label", "2023", "2024"], ["Operating and administrative expenses", "100", "120"]], [["Label", "2023", "2024"], ["Operating and administrative", "100", "120"], ["expenses", "", ""]], [_cont([2], 1, 0, "CONTINUATION_OF_SPECIFIC_CELL", "SPACE_JOIN", "Operating and administrative expenses")], case_types=["Wrapped first-column label"]),
        _spec("three-line first-column label", [["Label", "2023", "2024"], ["General, administrative, marketing, and distribution expenses", "100", "120"]], [["Label", "2023", "2024"], ["General, administrative,", "100", "120"], ["marketing, and distribution", "", ""], ["expenses", "", ""]], [_cont([2], 1, 0, "CONTINUATION_OF_SPECIFIC_CELL", "SPACE_JOIN", "General, administrative, marketing, and distribution"), _cont([3], 2, 0, "CONTINUATION_OF_SPECIFIC_CELL", "SPACE_JOIN", "marketing, and distribution expenses")], case_types=["Three-line first-column label"]),
        _spec("wrapped middle-column cell", [["Metric", "Current period value", "Prior value"], ["Revenue", "100", "90"]], [["Metric", "Current period", "Prior value"], ["", "value", ""], ["Revenue", "100", "90"]], [_cont([1], 0, 1, "MULTI_LINE_HEADER", "SPACE_JOIN", "Current period value")], case_types=["Wrapped middle-column cell"]),
        _spec("two wrapped cells same row", [["Header", "Measure"], ["Product description continued", "Measurement description continued"], ["Alpha", "Rate"]], [["Header", "Measure"], ["Product description", "Measurement description"], ["continued", "continued"], ["Alpha", "Rate"]], [_cont([2], 1, 0, "CONTINUATION_OF_SPECIFIC_CELL", "SPACE_JOIN", "Product description continued"), _cont([2], 1, 1, "CONTINUATION_OF_SPECIFIC_CELL", "SPACE_JOIN", "Measurement description continued")], case_types=["Two wrapped cells in the same row"]),
        _spec("multi-line simple header", [["Three-month return", "Twelve-month return"], ["5%", "12%"]], [["Three-month", "Twelve-month"], ["return", "return"], ["5%", "12%"]], [_cont([1], 0, 0, "MULTI_LINE_HEADER", "SPACE_JOIN", "Three-month return"), _cont([1], 0, 1, "MULTI_LINE_HEADER", "SPACE_JOIN", "Twelve-month return")], case_types=["Multi-line simple header"]),
        _spec("hyphenated line break", [["Label", "Value"], ["international revenue", "100"], ["long-term assets", "200"]], [["Label", "Value"], ["inter-", "100"], ["national revenue", ""], ["long-", "200"], ["term assets", ""]], [_cont([2], 1, 0, "CONTINUATION_OF_SPECIFIC_CELL", "HYPHEN_REMOVAL_JOIN", "international revenue"), _cont([4], 3, 0, "CONTINUATION_OF_SPECIFIC_CELL", "PRESERVE_HYPHEN_JOIN", "long-term assets")], case_types=["Hyphen-removal case", "Hyphen-preservation case"]),
        _spec("coefficient standard error separate", [["Variable", "Model"], ["Variable A", "0.125***"], ["", "(0.045)"]], [["Variable", "Model"], ["Variable A", "0.125***"], ["", "(0.045)"]], [], case_types=["Regression coefficient and standard-error pattern"]),
        _spec("significance legend footnote", [["Variable", "Model"], ["Variable A", "0.125***"]], [["Variable", "Model"], ["Variable A", "0.125***"], ["*** p<0.01, ** p<0.05, * p<0.10", ""]], [], expected_footnotes=["*** p<0.01, ** p<0.05, * p<0.10"], case_types=["Significance-star legend outside the grid", "Footnote close to the final row"]),
        _spec("genuine all-text final row", [["Item", "Status"], ["Alpha", "Open"], ["Total observations available", "Internal"]], [["Item", "Status"], ["Alpha", "Open"], ["Total observations available", "Internal"]], [], case_types=["Genuine all-text table row"]),
        _spec("ambiguous unresolved line", [["Item", "Value"], ["Alpha", "100"]], [["Item", "Value"], ["Alpha", "100"], ["maybe continued.", ""]], [], expected_unresolved=[2], case_types=["An intentionally ambiguous case that must remain unresolved"]),
        _spec(
            "wrapped final-column cell",
            [
                ["Item", "Units", "Status", "Explanation"],
                ["Product A", "12", "Active", "Available throughout the reporting period"],
                ["Product B", "9", "Closed", "Discontinued"],
                ["Product C", "15", "Active", "Available"],
                ["Product D", "18", "Active", "Deferred"],
            ],
            [
                ["Item", "Units", "Status", "Explanation"],
                ["Product A", "12", "Active", "Available throughout the"],
                ["", "", "", "reporting period"],
                ["Product B", "9", "Closed", "Discontinued"],
                ["Product C", "15", "Active", "Available"],
                ["Product D", "18", "Active", "Deferred"],
            ],
            [_cont([2], 1, 3, "CONTINUATION_OF_SPECIFIC_CELL", "SPACE_JOIN", "Available throughout the reporting period")],
            case_types=["Wrapped final-column cell"],
            columns=[64, 184, 284, 390],
            bbox="56,74,610,220",
            font_size=8.8,
            row_spacing=14.5,
        ),
        _spec(
            "multi-level spanning header with wrapped text",
            [
                ["Category", "Short-term operating performance measures", "", "Long-term financial performance measures", ""],
                ["", "Revenue", "Cost", "Assets", "Liabilities"],
                ["A", "100", "70", "450", "210"],
                ["B", "120", "80", "500", "240"],
                ["C", "90", "65", "410", "190"],
            ],
            [
                ["Category", "Short-term operating", "", "Long-term financial", ""],
                ["", "performance measures", "", "performance measures", ""],
                ["", "Revenue", "Cost", "Assets", "Liabilities"],
                ["A", "100", "70", "450", "210"],
                ["B", "120", "80", "500", "240"],
                ["C", "90", "65", "410", "190"],
            ],
            [
                _cont([1], 0, 1, "MULTI_LINE_HEADER", "SPACE_JOIN", "Short-term operating performance measures"),
                _cont([1], 0, 3, "MULTI_LINE_HEADER", "SPACE_JOIN", "Long-term financial performance measures"),
            ],
            case_types=["Multi-level spanning header with wrapped text"],
            expected_merged_ranges=[
                _span(0, 0, "Category", 2, 1, True, True),
                _span(0, 1, "Short-term operating performance measures", 1, 2, True, False),
                _span(0, 3, "Long-term financial performance measures", 1, 2, True, False),
            ],
            row_spans=[_span(0, 0, "Category", 2, 1, True, True)],
            column_spans=[
                _span(0, 1, "Short-term operating performance measures", 1, 2, True, False),
                _span(0, 3, "Long-term financial performance measures", 1, 2, True, False),
            ],
            columns=[60, 158, 260, 360, 468],
            bbox="52,74,590,230",
            font_size=8.5,
            row_spacing=15.5,
        ),
        _spec(
            "panel section headers",
            [
                ["Variable", "Mean", "Median", "N"],
                ["Panel A: Full Sample", "", "", ""],
                ["Revenue", "10.2", "9.8", "200"],
                ["Assets", "45.1", "42.0", "200"],
                ["Panel B: Restricted Sample", "", "", ""],
                ["Revenue", "11.5", "10.7", "120"],
                ["Assets", "48.3", "44.2", "120"],
            ],
            [
                ["Variable", "Mean", "Median", "N"],
                ["Panel A: Full Sample", "", "", ""],
                ["Revenue", "10.2", "9.8", "200"],
                ["Assets", "45.1", "42.0", "200"],
                ["Panel B: Restricted Sample", "", "", ""],
                ["Revenue", "11.5", "10.7", "120"],
                ["Assets", "48.3", "44.2", "120"],
            ],
            [],
            case_types=["Section header such as Panel A"],
            expected_merged_ranges=[
                _span(1, 0, "Panel A: Full Sample", 1, 4, True, False),
                _span(4, 0, "Panel B: Restricted Sample", 1, 4, True, False),
            ],
            column_spans=[
                _span(1, 0, "Panel A: Full Sample", 1, 4, True, False),
                _span(4, 0, "Panel B: Restricted Sample", 1, 4, True, False),
            ],
            section_headers=[
                _span(1, 0, "Panel A: Full Sample", 1, 4, True, False),
                _span(4, 0, "Panel B: Restricted Sample", 1, 4, True, False),
            ],
            columns=[62, 218, 318, 418],
            bbox="54,74,540,250",
            font_size=9.2,
            row_spacing=15.0,
        ),
        _spec(
            "missing numeric cell alongside wrapped text",
            [
                ["Category", "Current", "Prior", "Change"],
                ["Long-term operating and administrative expenses", "125", "", "8.5%"],
                ["Revenue", "300", "275", "9.1%"],
                ["Assets", "500", "490", "2.0%"],
            ],
            [
                ["Category", "Current", "Prior", "Change"],
                ["Long-term operating and", "125", "", "8.5%"],
                ["administrative expenses", "", "", ""],
                ["Revenue", "300", "275", "9.1%"],
                ["Assets", "500", "490", "2.0%"],
            ],
            [_cont([2], 1, 0, "CONTINUATION_OF_SPECIFIC_CELL", "SPACE_JOIN", "Long-term operating and administrative expenses")],
            case_types=["Missing numeric cell alongside wrapped text"],
            expected_empty_cells=[[1, 2]],
            columns=[58, 270, 370, 470],
            bbox="50,74,560,220",
            font_size=8.7,
            row_spacing=14.0,
        ),
    ]


def _cont(source_lines: list[int], target_row: int, target_column: int, continuation_type: str, joining_mode: str, final_text: str) -> dict:
    return {
        "source_visual_line_indices": source_lines,
        "target_row": target_row,
        "target_column": target_column,
        "continuation_type": continuation_type,
        "expected_joining_mode": joining_mode,
        "expected_final_text": final_text,
    }


def _span(row_index: int, column_index: int, text: str, row_span: int, column_span: int, is_header: bool, is_stub: bool) -> dict:
    return {
        "row_index": row_index,
        "column_index": column_index,
        "text": text,
        "row_span": row_span,
        "column_span": column_span,
        "covered_coordinates": [
            [row, column]
            for row in range(row_index, row_index + row_span)
            for column in range(column_index, column_index + column_span)
        ],
        "is_header": is_header,
        "is_stub": is_stub,
    }


def _spec(category: str, ground_truth: list[list[str]], visual_rows: list[list[str]], continuations: list[dict], expected_footnotes: list[str] | None = None, expected_unresolved: list[int] | None = None, case_types: list[str] | None = None, expected_merged_ranges: list[dict] | None = None, row_spans: list[dict] | None = None, column_spans: list[dict] | None = None, section_headers: list[dict] | None = None, expected_empty_cells: list[list[int]] | None = None, columns: list[float] | None = None, bbox: str | None = None, font_size: float = 9.0, row_spacing: float = 16.0) -> dict:
    physical_columns = max(len(row) for row in ground_truth)
    expected_merged_ranges = expected_merged_ranges or []
    covered_nonanchors = {
        tuple(coord)
        for span in expected_merged_ranges
        for coord in span.get("covered_coordinates", [])
        if tuple(coord) != (span["row_index"], span["column_index"])
    }
    logical_cells = [
        {
            "row_index": row_index,
            "column_index": column_index,
            "text": text,
            "row_span": 1,
            "column_span": 1,
            "covered_coordinates": [[row_index, column_index]],
            "is_header": row_index == 0,
            "is_stub": column_index == 0 and row_index > 0,
        }
        for row_index, row in enumerate(ground_truth)
        for column_index, text in enumerate(row)
        if (row_index, column_index) not in covered_nonanchors
    ]
    return {
        "category": category,
        "notes": category,
        "ground_truth": ground_truth,
        "visual_rows": visual_rows,
        "raw_visual_lines": [{"line_index": i, "cells": row} for i, row in enumerate(visual_rows)],
        "expected_continuations": continuations,
        "expected_footnotes": expected_footnotes or [],
        "expected_unresolved_lines": expected_unresolved or [],
        "expected_empty_cells": expected_empty_cells or [],
        "case_types": case_types or [],
        "physical_rows": len(ground_truth),
        "physical_columns": physical_columns,
        "logical_cells": logical_cells,
        "ordinary_cells": logical_cells,
        "multi_line_cells": continuations,
        "row_spans": row_spans or [],
        "column_spans": column_spans or [],
        "section_headers": section_headers or [],
        "expected_merged_ranges": expected_merged_ranges,
        "columns": columns,
        "bbox": bbox,
        "font_size": font_size,
        "row_spacing": row_spacing,
    }


def _write_pdf(path: Path, spec: dict) -> None:
    columns = spec.get("columns") or [60, 260, 430, 520]
    doc = fitz.open()
    page = doc.new_page(width=612, height=792)
    y = 82
    for row in spec["visual_rows"]:
        for text, x in zip(row, columns):
            if text:
                page.insert_text((x, y), text, fontsize=spec.get("font_size", 9.0))
        y += spec.get("row_spacing", 16.0)
    doc.save(path)
    doc.close()


def _write_csv(path: Path, rows: list[list[str]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        csv.writer(handle).writerows(rows)


if __name__ == "__main__":
    main()
