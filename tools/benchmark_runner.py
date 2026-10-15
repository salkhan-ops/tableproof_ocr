from __future__ import annotations

import argparse
import csv
import json
import shutil
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from evaluator import evaluate_table_grid, load_ground_truth_table, table_grid_to_ground_truth, table_passes
from tableproof.models import BoundingBox
from tableproof.pdf_extractor import extract_table_area
from tableproof.table_reconstructor import reconstruct_from_words_file


@dataclass(frozen=True)
class BenchmarkEntry:
    table_id: str
    pdf_path: Path
    page_number: int
    bbox: BoundingBox
    category: str
    ground_truth_path: Path
    expected_rows: int
    expected_columns: int
    difficulty_notes: str
    born_digital: bool
    manually_verified: bool
    ground_truth_json_path: Path | None = None


def run_benchmark(
    manifest_path: Path | str = Path("data/benchmark_manifest.csv"),
    extraction_dir: Path | str = Path("outputs/extracted_tables"),
    evaluation_dir: Path | str = Path("outputs/evaluation"),
    diagnostics_dir: Path | str = Path("outputs/diagnostics"),
) -> dict[str, Any]:
    manifest_path = Path(manifest_path)
    extraction_dir = Path(extraction_dir)
    evaluation_dir = Path(evaluation_dir)
    diagnostics_dir = Path(diagnostics_dir)
    extraction_dir.mkdir(parents=True, exist_ok=True)
    evaluation_dir.mkdir(parents=True, exist_ok=True)
    diagnostics_dir.mkdir(parents=True, exist_ok=True)

    entries = load_manifest(manifest_path)
    table_results: list[dict[str, Any]] = []
    all_cell_errors: list[dict[str, Any]] = []

    for entry in entries:
        started = time.perf_counter()
        row: dict[str, Any] = {
            "table_id": entry.table_id,
            "category": entry.category,
            "expected_rows": entry.expected_rows,
            "expected_columns": entry.expected_columns,
            "difficulty_notes": entry.difficulty_notes,
            "status": "FAILED",
            "failure_reason": "",
        }
        try:
            extraction = extract_table_area(
                pdf_path=entry.pdf_path,
                page_number=entry.page_number,
                bbox=entry.bbox,
                output_dir=extraction_dir,
            )
            reconstruction = reconstruct_from_words_file(
                words_json_path=extraction.words_json_path,
                output_dir=evaluation_dir,
                debug=True,
            )
            outputs = _rename_outputs(entry.table_id, reconstruction, evaluation_dir, diagnostics_dir)
            filtered_grid = reconstruction.table_grid
            note_diagnostics = {
                "mode": "deterministic_table_boundary_in_reconstructor",
                "excluded_rows": reconstruction.table_grid.metrics.get("excluded_lines", []),
            }
            ground_truth = load_ground_truth_table(entry.ground_truth_path, entry.ground_truth_json_path)
            prediction = table_grid_to_ground_truth(filtered_grid)
            evaluation = evaluate_table_grid(entry.table_id, prediction, ground_truth)
            runtime = time.perf_counter() - started
            passed = table_passes(evaluation)
            warnings = [warning.code for warning in filtered_grid.warnings]

            row.update(
                evaluation.to_dict()
                | {
                    "status": "PASS" if passed else "FAIL",
                    "overall_reconstruction_confidence": reconstruction.table_grid.confidence,
                    "warnings": ";".join(warnings),
                    "note_rows_excluded": len(note_diagnostics["excluded_rows"]),
                    "note_exclusion_diagnostics": json.dumps(note_diagnostics, sort_keys=True),
                    "physical_rows": evaluation.physical_rows,
                    "physical_columns": evaluation.physical_columns,
                    "logical_cell_count": evaluation.logical_cell_count,
                    "expected_merged_ranges": json.dumps(evaluation.expected_merged_ranges or [], sort_keys=True),
                    "predicted_merged_ranges": json.dumps(evaluation.predicted_merged_ranges or [], sort_keys=True),
                    "exact_merged_range_accuracy": evaluation.exact_merged_range_accuracy,
                    "column_span_detection_precision": evaluation.column_span_detection_precision,
                    "column_span_detection_recall": evaluation.column_span_detection_recall,
                    "row_span_detection_precision": evaluation.row_span_detection_precision,
                    "row_span_detection_recall": evaluation.row_span_detection_recall,
                    "span_text_accuracy": evaluation.span_text_accuracy,
                    "covered_coordinate_accuracy": evaluation.covered_coordinate_accuracy,
                    "merged_cell_placement_accuracy": evaluation.merged_cell_placement_accuracy,
                    "percentage_spans_requiring_correction": evaluation.percentage_spans_requiring_correction,
                    "missed_ranges": json.dumps(evaluation.missed_ranges or [], sort_keys=True),
                    "false_ranges": json.dumps(evaluation.false_ranges or [], sort_keys=True),
                    "runtime_seconds": round(runtime, 6),
                    "xlsx_path": str(outputs["xlsx"]),
                    "csv_path": str(outputs["csv"]),
                    "structured_json_path": str(outputs["structured_json"]),
                    "diagnostic_png_path": str(outputs["diagnostic_png"]),
                    "diagnostics_json_path": str(outputs["diagnostics_json"]),
                    "cell_comparison_path": str(outputs["cell_comparison"]),
                    "ground_truth_preview": _preview(ground_truth.matrix),
                    "predicted_preview": _preview(prediction.matrix),
                }
            )
            _augment_diagnostics(outputs["diagnostics_json"], note_diagnostics, evaluation)
            _write_cell_comparison(outputs["cell_comparison"], entry.table_id, prediction.matrix, ground_truth.matrix, evaluation.cell_errors)
            all_cell_errors.extend(evaluation.cell_errors)
        except Exception as exc:  # noqa: BLE001 - benchmark must continue and record failures.
            runtime = time.perf_counter() - started
            row.update(
                {
                    "predicted_rows": "",
                    "predicted_columns": "",
                    "exact_row_count_match": False,
                    "exact_column_count_match": False,
                    "exact_cell_text_accuracy": 0.0,
                    "non_empty_cell_text_accuracy": 0.0,
                    "numeric_cell_accuracy": 0.0,
                    "correct_row_and_column_placement_rate": 0.0,
                    "empty_cell_accuracy": 0.0,
                    "whole_table_exact_match": False,
                    "cells_requiring_manual_correction": "",
                    "percentage_cells_requiring_correction": "",
                    "overall_reconstruction_confidence": 0.0,
                    "warnings": "",
                    "note_rows_excluded": "",
                    "note_exclusion_diagnostics": "",
                    "physical_rows": "",
                    "physical_columns": "",
                    "logical_cell_count": "",
                    "expected_merged_ranges": "",
                    "predicted_merged_ranges": "",
                    "exact_merged_range_accuracy": 0.0,
                    "column_span_detection_precision": 0.0,
                    "column_span_detection_recall": 0.0,
                    "row_span_detection_precision": 0.0,
                    "row_span_detection_recall": 0.0,
                    "span_text_accuracy": 0.0,
                    "covered_coordinate_accuracy": 0.0,
                    "merged_cell_placement_accuracy": 0.0,
                    "percentage_spans_requiring_correction": 0.0,
                    "missed_ranges": "",
                    "false_ranges": "",
                    "error_categories": "OTHER_STRUCTURE_ERROR",
                    "runtime_seconds": round(runtime, 6),
                    "failure_reason": f"{type(exc).__name__}: {exc}",
                }
            )
        table_results.append(row)

    summary = _aggregate(table_results, all_cell_errors)
    _write_table_results(evaluation_dir / "benchmark_table_results.csv", table_results)
    _write_cell_errors(evaluation_dir / "benchmark_cell_errors.csv", all_cell_errors)
    _write_json(evaluation_dir / "benchmark_summary.json", summary)
    _write_report(evaluation_dir / "benchmark_report.md", table_results, summary)
    return {"table_results": table_results, "summary": summary}


def load_manifest(path: Path | str) -> list[BenchmarkEntry]:
    entries: list[BenchmarkEntry] = []
    with Path(path).open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            entries.append(
                BenchmarkEntry(
                    table_id=row["table_id"],
                    pdf_path=Path(row["pdf_path"]),
                    page_number=int(row["page_number"]),
                    bbox=BoundingBox.from_csv(row["bbox"]),
                    category=row["category"],
                    ground_truth_path=Path(row["ground_truth_path"]),
                    expected_rows=int(row["expected_rows"]),
                    expected_columns=int(row["expected_columns"]),
                    difficulty_notes=row["difficulty_notes"],
                    born_digital=row["born_digital"].strip().lower() == "true",
                    manually_verified=row["manually_verified"].strip().lower() == "true",
                    ground_truth_json_path=Path(row["ground_truth_json_path"])
                    if row.get("ground_truth_json_path")
                    else None,
                )
            )
    return entries


def _rename_outputs(table_id: str, reconstruction: Any, evaluation_dir: Path, diagnostics_dir: Path) -> dict[str, Path]:
    outputs = {
        "xlsx": evaluation_dir / f"{table_id}_predicted.xlsx",
        "csv": evaluation_dir / f"{table_id}_predicted.csv",
        "structured_json": evaluation_dir / f"{table_id}_predicted.json",
        "diagnostic_png": diagnostics_dir / f"{table_id}_diagnostic.png",
        "diagnostics_json": diagnostics_dir / f"{table_id}_diagnostics.json",
        "cell_comparison": evaluation_dir / f"{table_id}_cell_comparison.csv",
    }
    shutil.copyfile(reconstruction.xlsx_path, outputs["xlsx"])
    shutil.copyfile(reconstruction.csv_path, outputs["csv"])
    shutil.copyfile(reconstruction.structured_json_path, outputs["structured_json"])
    shutil.copyfile(reconstruction.diagnostic_png_path, outputs["diagnostic_png"])
    shutil.copyfile(reconstruction.diagnostics_json_path, outputs["diagnostics_json"])
    return outputs


def _write_cell_comparison(
    path: Path,
    table_id: str,
    predicted: list[list[str]],
    ground_truth: list[list[str]],
    cell_errors: list[dict[str, Any]],
) -> None:
    error_lookup = {(item["row_index"], item["column_index"]): item["error_type"] for item in cell_errors}
    rows = max(len(predicted), len(ground_truth))
    columns = max(
        max((len(row) for row in predicted), default=0),
        max((len(row) for row in ground_truth), default=0),
    )
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["table_id", "row_index", "column_index", "expected", "predicted", "match", "error_type"],
        )
        writer.writeheader()
        for row_index in range(rows):
            for column_index in range(columns):
                expected = _matrix_get(ground_truth, row_index, column_index)
                pred = _matrix_get(predicted, row_index, column_index)
                writer.writerow(
                    {
                        "table_id": table_id,
                        "row_index": row_index,
                        "column_index": column_index,
                        "expected": expected,
                        "predicted": pred,
                        "match": expected == pred,
                        "error_type": error_lookup.get((row_index, column_index), ""),
                    }
                )


def _aggregate(table_results: list[dict[str, Any]], cell_errors: list[dict[str, Any]]) -> dict[str, Any]:
    completed = [row for row in table_results if row.get("predicted_rows") != ""]
    return {
        "tables_total": len(table_results),
        "tables_passed": sum(1 for row in table_results if row["status"] == "PASS"),
        "tables_failed": sum(1 for row in table_results if row["status"] != "PASS"),
        "whole_table_exact_matches": sum(1 for row in completed if row.get("whole_table_exact_match")),
        "exact_cell_text_accuracy": _avg(completed, "exact_cell_text_accuracy"),
        "numeric_cell_accuracy": _avg(completed, "numeric_cell_accuracy"),
        "correct_placement_rate": _avg(completed, "correct_row_and_column_placement_rate"),
        "empty_cell_accuracy": _avg(completed, "empty_cell_accuracy"),
        "cells_requiring_correction": sum(
            int(row.get("cells_requiring_manual_correction") or 0) for row in completed
        ),
        "percentage_requiring_correction": _avg(completed, "percentage_cells_requiring_correction"),
        "exact_merged_range_accuracy": _avg(completed, "exact_merged_range_accuracy"),
        "column_span_detection_precision": _avg(completed, "column_span_detection_precision"),
        "column_span_detection_recall": _avg(completed, "column_span_detection_recall"),
        "row_span_detection_precision": _avg(completed, "row_span_detection_precision"),
        "row_span_detection_recall": _avg(completed, "row_span_detection_recall"),
        "span_text_accuracy": _avg(completed, "span_text_accuracy"),
        "covered_coordinate_accuracy": _avg(completed, "covered_coordinate_accuracy"),
        "merged_cell_placement_accuracy": _avg(completed, "merged_cell_placement_accuracy"),
        "percentage_spans_requiring_correction": _avg(completed, "percentage_spans_requiring_correction"),
        "average_confidence": _avg(completed, "overall_reconstruction_confidence"),
        "total_runtime_seconds": round(sum(float(row.get("runtime_seconds") or 0) for row in table_results), 6),
        "error_categories": sorted({error["error_type"] for error in cell_errors}),
        "benchmark_status": _benchmark_status(table_results),
        "top_failure_modes": _top_failure_modes(table_results, cell_errors),
    }


def _benchmark_status(table_results: list[dict[str, Any]]) -> str:
    passed = sum(1 for row in table_results if row["status"] == "PASS")
    if passed == len(table_results):
        return "PASS"
    if passed > 0:
        return "PARTIAL"
    return "FAILED"


def _top_failure_modes(table_results: list[dict[str, Any]], cell_errors: list[dict[str, Any]]) -> list[str]:
    counts: dict[str, int] = {}
    for error in cell_errors:
        counts[error["error_type"]] = counts.get(error["error_type"], 0) + 1
    for row in table_results:
        if row.get("status") == "PASS":
            continue
        for category in str(row.get("error_categories", "")).split(";"):
            if category:
                counts[category] = counts.get(category, 0) + 1
    modes = [item[0] for item in sorted(counts.items(), key=lambda item: (-item[1], item[0]))[:3]]
    warning_counts: dict[str, int] = {}
    for row in table_results:
        for warning in str(row.get("warnings", "")).split(";"):
            if warning:
                warning_counts[warning] = warning_counts.get(warning, 0) + 1
    for warning, _count in sorted(warning_counts.items(), key=lambda item: (-item[1], item[0])):
        label = f"WARNING_{warning}"
        if label not in modes:
            modes.append(label)
        if len(modes) == 3:
            break
    while len(modes) < 3:
        modes.append("NO_ADDITIONAL_FAILURE_MODE_OBSERVED")
    return modes[:3]


def _write_table_results(path: Path, rows: list[dict[str, Any]]) -> None:
    fieldnames = sorted({key for row in rows for key in row.keys()})
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _write_cell_errors(path: Path, rows: list[dict[str, Any]]) -> None:
    fieldnames = ["table_id", "row_index", "column_index", "expected", "predicted", "error_type"]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _write_report(path: Path, table_results: list[dict[str, Any]], summary: dict[str, Any]) -> None:
    lines = [
        "# TableProof OCR Benchmark Report",
        "",
        "## Aggregate Results",
        "",
        f"- Benchmark status: {summary['benchmark_status']}",
        f"- Tables passed: {summary['tables_passed']}",
        f"- Tables failed: {summary['tables_failed']}",
        f"- Exact cell-text accuracy: {summary['exact_cell_text_accuracy']}",
        f"- Numeric-cell accuracy: {summary['numeric_cell_accuracy']}",
        f"- Correct placement rate: {summary['correct_placement_rate']}",
        f"- Empty-cell accuracy: {summary['empty_cell_accuracy']}",
        f"- Cells requiring correction: {summary['cells_requiring_correction']}",
        f"- Exact merged-range accuracy: {summary['exact_merged_range_accuracy']}",
        f"- Column-span precision/recall: {summary['column_span_detection_precision']} / {summary['column_span_detection_recall']}",
        f"- Row-span precision/recall: {summary['row_span_detection_precision']} / {summary['row_span_detection_recall']}",
        "",
        "## Tables",
    ]
    for row in table_results:
        lines.extend(
            [
                "",
                f"### {row['table_id']}: {row['category']}",
                "",
                f"- Notes: {row['difficulty_notes']}",
                f"- Expected dimensions: {row['expected_rows']} x {row['expected_columns']}",
                f"- Predicted dimensions: {row.get('predicted_rows')} x {row.get('predicted_columns')}",
                f"- Status: {row['status']}",
                f"- Metric results: exact={row.get('exact_cell_text_accuracy')}, numeric={row.get('numeric_cell_accuracy')}, placement={row.get('correct_row_and_column_placement_rate')}",
                f"- Warnings: {row.get('warnings', '')}",
                f"- Error categories: {row.get('error_categories', '')}",
                f"- Manual corrections required: {row.get('cells_requiring_manual_correction', '')}",
                f"- Physical dimensions: {row.get('physical_rows', '')} x {row.get('physical_columns', '')}",
                f"- Logical cells: {row.get('logical_cell_count', '')}",
                f"- Expected merged ranges: {row.get('expected_merged_ranges', '[]')}",
                f"- Predicted merged ranges: {row.get('predicted_merged_ranges', '[]')}",
                f"- Exact merged-range accuracy: {row.get('exact_merged_range_accuracy', '')}",
                f"- Span warnings: {row.get('warnings', '')}",
                "",
                "Ground truth preview:",
                "",
                "```text",
                row.get("ground_truth_preview", ""),
                "```",
                "",
                "Predicted preview:",
                "",
                "```text",
                row.get("predicted_preview", ""),
                "```",
            ]
        )
        if row.get("failure_reason"):
            lines.append(f"- Failure reason: {row['failure_reason']}")
    lines.extend(
        [
            "",
            "## Three Most Important Structural Weaknesses",
            "",
        ]
    )
    for index, mode in enumerate(summary["top_failure_modes"], start=1):
        lines.append(f"{index}. {mode}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_json(path: Path, payload: object) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _augment_diagnostics(path: Path, note_diagnostics: dict[str, Any], evaluation: Any) -> None:
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["note_row_exclusion"] = note_diagnostics
    payload["merged_cell_metrics"] = {
        "expected_merged_ranges": evaluation.expected_merged_ranges or [],
        "predicted_merged_ranges": evaluation.predicted_merged_ranges or [],
        "exact_merged_range_accuracy": evaluation.exact_merged_range_accuracy,
        "column_span_detection_precision": evaluation.column_span_detection_precision,
        "column_span_detection_recall": evaluation.column_span_detection_recall,
        "row_span_detection_precision": evaluation.row_span_detection_precision,
        "row_span_detection_recall": evaluation.row_span_detection_recall,
        "span_text_accuracy": evaluation.span_text_accuracy,
        "covered_coordinate_accuracy": evaluation.covered_coordinate_accuracy,
        "merged_cell_placement_accuracy": evaluation.merged_cell_placement_accuracy,
        "percentage_spans_requiring_correction": evaluation.percentage_spans_requiring_correction,
        "missed_ranges": evaluation.missed_ranges or [],
        "false_ranges": evaluation.false_ranges or [],
    }
    _write_json(path, payload)


def _preview(matrix: list[list[str]], limit: int = 8) -> str:
    return "\n".join(" | ".join(row) for row in matrix[:limit])


def _matrix_get(matrix: list[list[str]], row_index: int, column_index: int) -> str:
    if row_index >= len(matrix) or column_index >= len(matrix[row_index]):
        return ""
    return matrix[row_index][column_index]


def _avg(rows: list[dict[str, Any]], key: str) -> float:
    values = [float(row[key]) for row in rows if row.get(key) not in {"", None}]
    return round(sum(values) / len(values), 6) if values else 0.0


def main() -> int:
    parser = argparse.ArgumentParser(description="Run the five-table born-digital benchmark.")
    parser.add_argument("--manifest", type=Path, default=Path("data/benchmark_manifest.csv"))
    args = parser.parse_args()
    result = run_benchmark(args.manifest)
    summary = result["summary"]
    print(f"benchmark_status={summary['benchmark_status']}")
    print(f"tables_passed={summary['tables_passed']}")
    print(f"tables_failed={summary['tables_failed']}")
    print("summary_json=outputs/evaluation/benchmark_summary.json")
    print("report=outputs/evaluation/benchmark_report.md")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
