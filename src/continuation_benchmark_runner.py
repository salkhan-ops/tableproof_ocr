from __future__ import annotations

import csv
import json
import time
from pathlib import Path
from typing import Any

from evaluator import evaluate_table_grid, load_ground_truth_table, table_grid_to_ground_truth
from models import BoundingBox
from pdf_extractor import extract_table_area
from table_reconstructor import reconstruct_from_words_file


def run_continuation_benchmark(
    manifest_path: Path | str = Path("data/continuation_benchmark_manifest.csv"),
    extraction_dir: Path | str = Path("outputs/extracted_tables"),
    evaluation_dir: Path | str = Path("outputs/evaluation"),
) -> dict[str, Any]:
    manifest_path = Path(manifest_path)
    extraction_dir = Path(extraction_dir)
    evaluation_dir = Path(evaluation_dir)
    rows = list(csv.DictReader(manifest_path.open(encoding="utf-8")))
    table_results = []
    continuation_results = []
    errors = []
    for manifest_row in rows:
        started = time.perf_counter()
        table_id = manifest_row["table_id"]
        try:
            extraction = extract_table_area(
                manifest_row["pdf_path"],
                int(manifest_row["page_number"]),
                BoundingBox.from_csv(manifest_row["bbox"]),
                extraction_dir,
            )
            reconstruction = reconstruct_from_words_file(extraction.words_json_path, evaluation_dir, debug=True)
            gt = load_ground_truth_table(manifest_row["ground_truth_path"], manifest_row.get("ground_truth_json_path"))
            prediction = table_grid_to_ground_truth(reconstruction.table_grid)
            eval_result = evaluate_table_grid(table_id, prediction, gt)
            expected = json.loads(manifest_row["expected_continuations"])
            unresolved_expected = json.loads(manifest_row["expected_unresolved_lines"])
            predicted_decisions = reconstruction.table_grid.metrics.get("selected_decisions", [])
            unresolved_pred = reconstruction.table_grid.metrics.get("unresolved_lines", [])
            cont = _continuation_metrics(table_id, expected, predicted_decisions, unresolved_expected, unresolved_pred)
            no_word_loss = reconstruction.table_grid.metrics.get("no_word_loss", {})
            runtime = round(time.perf_counter() - started, 6)
            table_results.append(
                {
                    **eval_result.to_dict(),
                    **cont,
                    "table_id": table_id,
                    "status": "PASS" if eval_result.whole_table_exact_match and cont["false_positive_continuations"] == 0 else "FAIL",
                    "runtime_seconds": runtime,
                    "xlsx_path": str(reconstruction.xlsx_path),
                    "json_path": str(reconstruction.structured_json_path),
                    "diagnostics_json_path": str(reconstruction.diagnostics_json_path),
                    "raw_word_count": no_word_loss.get("raw_word_count", 0),
                    "lost_word_count": len(no_word_loss.get("lost_word_indices", [])),
                    "duplicated_word_count": len(no_word_loss.get("duplicated_word_indices", [])),
                    "no_word_loss_passed": no_word_loss.get("passed", False),
                }
            )
            continuation_results.extend(_decision_rows(table_id, expected, predicted_decisions))
        except Exception as exc:  # noqa: BLE001
            errors.append({"table_id": table_id, "error": f"{type(exc).__name__}: {exc}"})
            table_results.append({"table_id": table_id, "status": "FAIL", "error": errors[-1]["error"]})
    summary = _summary(table_results)
    _write_csv(evaluation_dir / "continuation_table_results.csv", table_results)
    _write_csv(evaluation_dir / "continuation_results.csv", continuation_results)
    _write_csv(evaluation_dir / "continuation_errors.csv", errors)
    (evaluation_dir / "continuation_summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    _write_report(evaluation_dir / "continuation_report.md", table_results, summary)
    return {"table_results": table_results, "summary": summary}


def _continuation_metrics(table_id: str, expected: list[dict], predicted: list[dict], expected_unresolved: list[int], unresolved_pred: list[dict]) -> dict[str, Any]:
    pred_keys = {_key(decision["selected_candidate"]): decision for decision in predicted}
    expected_keys = {_expected_key(item): item for item in expected}
    tp = set(pred_keys) & set(expected_keys)
    false_pos = set(pred_keys) - set(expected_keys)
    missed = set(expected_keys) - set(pred_keys)
    joining_matches = sum(1 for key in tp if pred_keys[key]["selected_candidate"].get("joining_mode") == expected_keys[key]["expected_joining_mode"])
    text_matches = sum(1 for key in tp if pred_keys[key].get("final_text") == expected_keys[key]["expected_final_text"])
    unresolved_lines = {item.get("line_index") for item in unresolved_pred}
    return {
        "expected_continuations": len(expected),
        "predicted_continuations": len(predicted),
        "true_positive_continuations": len(tp),
        "false_positive_continuations": len(false_pos),
        "missed_continuations": len(missed),
        "continuation_precision": _rate(len(tp), len(predicted)) if predicted else (1.0 if not expected else 0.0),
        "continuation_recall": _rate(len(tp), len(expected)) if expected else 1.0,
        "continuation_target_accuracy": _rate(len(tp), len(expected)) if expected else 1.0,
        "continuation_type_accuracy": _rate(len(tp), len(expected)) if expected else 1.0,
        "joining_mode_accuracy": _rate(joining_matches, len(expected)) if expected else 1.0,
        "final_merged_text_accuracy": _rate(text_matches, len(expected)) if expected else 1.0,
        "false_merge_rate": _rate(len(false_pos), max(len(predicted), 1)),
        "missed_merge_rate": _rate(len(missed), max(len(expected), 1)),
        "expected_unresolved_lines": len(expected_unresolved),
        "unresolved_case_accuracy": _rate(len(set(expected_unresolved) & unresolved_lines), len(expected_unresolved)) if expected_unresolved else 1.0,
    }


def _key(candidate: dict) -> tuple[Any, ...]:
    return (tuple(candidate.get("source_line_indices", [])), candidate.get("target_row_index"), candidate.get("target_column_index"))


def _expected_key(item: dict) -> tuple[Any, ...]:
    return (tuple(item["source_visual_line_indices"]), item["target_row"], item["target_column"])


def _decision_rows(table_id: str, expected: list[dict], predicted: list[dict]) -> list[dict[str, Any]]:
    return [
        {
            "table_id": table_id,
            "candidate_id": decision["selected_candidate"]["candidate_id"],
            "source_line_indices": decision["selected_candidate"]["source_line_indices"],
            "target_row": decision["selected_candidate"]["target_row_index"],
            "target_column": decision["selected_candidate"]["target_column_index"],
            "joining_mode": decision["selected_candidate"].get("joining_mode"),
            "final_text": decision.get("final_text"),
            "confidence": decision.get("confidence"),
        }
        for decision in predicted
    ]


def _summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    completed = [row for row in rows if "expected_continuations" in row]
    return {
        "tables_total": len(rows),
        "tables_passed": sum(1 for row in rows if row.get("status") == "PASS"),
        "tables_failed": sum(1 for row in rows if row.get("status") != "PASS"),
        "expected_continuations": sum(int(row.get("expected_continuations", 0)) for row in completed),
        "predicted_continuations": sum(int(row.get("predicted_continuations", 0)) for row in completed),
        "true_positive_continuations": sum(int(row.get("true_positive_continuations", 0)) for row in completed),
        "false_positive_continuations": sum(int(row.get("false_positive_continuations", 0)) for row in completed),
        "missed_continuations": sum(int(row.get("missed_continuations", 0)) for row in completed),
        "precision": _avg(completed, "continuation_precision"),
        "recall": _avg(completed, "continuation_recall"),
        "target_accuracy": _avg(completed, "continuation_target_accuracy"),
        "joining_mode_accuracy": _avg(completed, "joining_mode_accuracy"),
        "final_merged_text_accuracy": _avg(completed, "final_merged_text_accuracy"),
        "false_merges": sum(int(row.get("false_positive_continuations", 0)) for row in completed),
        "missed_merges": sum(int(row.get("missed_continuations", 0)) for row in completed),
        "whole_table_exact_matches": sum(1 for row in completed if row.get("whole_table_exact_match")),
        "exact_cell_text_accuracy": _avg(completed, "exact_cell_text_accuracy"),
        "numeric_cell_accuracy": _avg(completed, "numeric_cell_accuracy"),
        "placement_rate": _avg(completed, "correct_row_and_column_placement_rate"),
        "empty_cell_accuracy": _avg(completed, "empty_cell_accuracy"),
        "cells_requiring_correction": sum(int(row.get("cells_requiring_manual_correction", 0) or 0) for row in completed),
        "lost_word_count": sum(int(row.get("lost_word_count", 0)) for row in completed),
        "duplicated_word_count": sum(int(row.get("duplicated_word_count", 0)) for row in completed),
        "tables_with_no_word_loss": sum(1 for row in completed if str(row.get("no_word_loss_passed")) == "True" or row.get("no_word_loss_passed") is True),
    }


def _write_report(path: Path, rows: list[dict[str, Any]], summary: dict[str, Any]) -> None:
    lines = ["# Continuation Benchmark Report", "", json.dumps(summary, indent=2), ""]
    for row in rows:
        lines.append(f"## {row.get('table_id')} - {row.get('status')}")
        lines.append(json.dumps(row, indent=2, sort_keys=True))
        lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fieldnames = sorted({key for row in rows for key in row}) if rows else ["table_id"]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _avg(rows: list[dict[str, Any]], key: str) -> float:
    values = [float(row[key]) for row in rows if key in row]
    return _rate(sum(values), len(values)) if values else 0.0


def _rate(num: float, den: float) -> float:
    return round(num / den, 6) if den else 0.0


if __name__ == "__main__":
    result = run_continuation_benchmark()
    print(f"continuation_tables_passed={result['summary']['tables_passed']}")
    print(f"continuation_tables_failed={result['summary']['tables_failed']}")
    print("summary_json=outputs/evaluation/continuation_summary.json")
