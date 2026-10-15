from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

from benchmark_runner import run_benchmark
from continuation_benchmark_runner import run_continuation_benchmark
from span_benchmark_runner import run_span_benchmark


ROOT = Path(__file__).resolve().parents[1]
EVAL = ROOT / "outputs" / "evaluation"
ORIGINAL = ROOT / "data" / "benchmark_manifest.csv"
CONTINUATION = ROOT / "data" / "continuation_benchmark_manifest.csv"
SPAN = ROOT / "data" / "span_benchmark_manifest.csv"


def run_complex_span_audit() -> dict[str, Any]:
    original = run_benchmark(ORIGINAL)
    continuation = run_continuation_benchmark(CONTINUATION)
    span = run_span_benchmark(SPAN)
    original_rows = _read_csv(EVAL / "benchmark_table_results.csv")
    continuation_rows = _read_csv(EVAL / "continuation_table_results.csv")
    span_rows = _read_csv(EVAL / "span_table_results.csv")
    inventory = _inventory()
    combined_rows = (
        [_tag(row, "original") for row in original_rows]
        + [_tag(row, "continuation") for row in continuation_rows]
        + [_tag(row, "span") for row in span_rows]
    )
    summary = _summary(combined_rows, original["summary"], continuation["summary"], span["summary"], inventory)
    _write_csv(EVAL / "complex_span_inventory.csv", inventory)
    _write_csv(EVAL / "complex_span_table_results.csv", combined_rows)
    (EVAL / "complex_span_summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    _write_report(EVAL / "complex_span_benchmark_report.md", combined_rows, summary)
    return {"summary": summary, "table_results": combined_rows}


def _inventory() -> list[dict[str, Any]]:
    rows = []
    for group, manifest in [("original", ORIGINAL), ("continuation", CONTINUATION), ("span", SPAN)]:
        for item in _read_csv(manifest):
            table_id = item["table_id"]
            rows.append(
                {
                    "table_id": table_id,
                    "benchmark_group": group,
                    "pdf_path": item["pdf_path"],
                    "ground_truth_path": item["ground_truth_path"],
                    "exists_pdf": (ROOT / item["pdf_path"]).exists(),
                    "exists_ground_truth": (ROOT / item["ground_truth_path"]).exists(),
                    "manually_verified": item.get("manually_verified", ""),
                    "duplicate_of": "",
                    "inventory_status": "OK",
                }
            )
    seen = {}
    for row in rows:
        if row["table_id"] in seen:
            row["duplicate_of"] = seen[row["table_id"]]
            row["inventory_status"] = "DUPLICATE"
        seen.setdefault(row["table_id"], row["table_id"])
        if not row["exists_pdf"]:
            row["inventory_status"] = "MISSING_PDF"
        if not row["exists_ground_truth"]:
            row["inventory_status"] = "MISSING_GROUND_TRUTH"
    return rows


def _summary(rows: list[dict[str, Any]], original: dict[str, Any], continuation: dict[str, Any], span: dict[str, Any], inventory: list[dict[str, Any]]) -> dict[str, Any]:
    completed = [row for row in rows if row.get("predicted_rows") not in {"", None}]
    status = "PASS"
    if span["tables_passed"] < 10 or len(rows) != 29:
        status = "PARTIAL"
    if not completed:
        status = "FAILED"
    return {
        "audit_status": status,
        "tables_total": len(rows),
        "unique_tables": len({row["table_id"] for row in rows}),
        "tables_evaluated": len(completed),
        "tables_passed": sum(1 for row in rows if row.get("status") == "PASS"),
        "tables_failed": sum(1 for row in rows if row.get("status") != "PASS"),
        "whole_table_exact_matches": sum(1 for row in completed if str(row.get("whole_table_exact_match")) == "True" or row.get("whole_table_exact_match") is True),
        "exact_cell_text_accuracy": _avg(completed, "exact_cell_text_accuracy"),
        "numeric_cell_accuracy": _avg(completed, "numeric_cell_accuracy"),
        "placement_rate": _avg(completed, "correct_row_and_column_placement_rate"),
        "empty_cell_accuracy": _avg(completed, "empty_cell_accuracy"),
        "corrections_required": sum(int(float(row.get("cells_requiring_manual_correction", 0) or 0)) for row in completed),
        "existing_tables_passed": original["tables_passed"] + continuation["tables_passed"],
        "existing_tables_failed": original["tables_failed"] + continuation["tables_failed"],
        "new_span_tables_passed": span["tables_passed"],
        "new_span_tables_failed": span["tables_failed"],
        "span_summary": span,
        "inventory_status": sorted({row["inventory_status"] for row in inventory}),
    }


def _write_report(path: Path, rows: list[dict[str, Any]], summary: dict[str, Any]) -> None:
    lines = ["# Complex Merged Span Audit", "", "## Summary", "", json.dumps(summary, indent=2, sort_keys=True), "", "## Tables"]
    for row in rows:
        lines.append(f"- {row.get('table_id')} ({row.get('benchmark_group')}): {row.get('status')}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _tag(row: dict[str, Any], group: str) -> dict[str, Any]:
    return {"benchmark_group": group, **row}


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fieldnames = sorted({key for row in rows for key in row}) if rows else ["table_id"]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _avg(rows: list[dict[str, Any]], key: str) -> float:
    values = [float(row[key]) for row in rows if row.get(key) not in {"", None}]
    return round(sum(values) / len(values), 6) if values else 0.0


if __name__ == "__main__":
    result = run_complex_span_audit()
    print(f"audit_status={result['summary']['audit_status']}")
    print(f"tables_passed={result['summary']['tables_passed']}")
    print(f"tables_failed={result['summary']['tables_failed']}")
    print("summary_json=outputs/evaluation/complex_span_summary.json")
    print("report=outputs/evaluation/complex_span_benchmark_report.md")
