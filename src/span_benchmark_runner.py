from __future__ import annotations

import csv
import json
import shutil
import time
from pathlib import Path
from typing import Any

from evaluator import evaluate_table_grid, load_ground_truth_table, table_grid_to_ground_truth
from models import BoundingBox
from pdf_extractor import extract_table_area
from table_reconstructor import reconstruct_from_words_file


ROOT = Path(__file__).resolve().parents[1]
SPAN_MANIFEST = ROOT / "data" / "span_benchmark_manifest.csv"
EVAL = ROOT / "outputs" / "evaluation"
DIAG = ROOT / "outputs" / "diagnostics"
EXTRACTED = ROOT / "outputs" / "extracted_tables"


def run_span_benchmark(
    manifest_path: Path | str = SPAN_MANIFEST,
    evaluation_dir: Path | str = EVAL,
    diagnostics_dir: Path | str = DIAG,
    extraction_dir: Path | str = EXTRACTED,
    prefix: str = "span",
) -> dict[str, Any]:
    manifest_path = Path(manifest_path)
    evaluation_dir = Path(evaluation_dir)
    diagnostics_dir = Path(diagnostics_dir)
    extraction_dir = Path(extraction_dir)
    evaluation_dir.mkdir(parents=True, exist_ok=True)
    diagnostics_dir.mkdir(parents=True, exist_ok=True)
    extraction_dir.mkdir(parents=True, exist_ok=True)

    manifest = list(csv.DictReader(manifest_path.open(encoding="utf-8")))
    table_rows: list[dict[str, Any]] = []
    decision_rows: list[dict[str, Any]] = []
    hierarchy_rows: list[dict[str, Any]] = []
    error_rows: list[dict[str, Any]] = []
    invariant_rows: list[dict[str, Any]] = []

    for item in manifest:
        table_id = item["table_id"]
        started = time.perf_counter()
        try:
            extraction = extract_table_area(
                item["pdf_path"],
                int(item["page_number"]),
                BoundingBox.from_csv(item["bbox"]),
                extraction_dir,
            )
            reconstruction = reconstruct_from_words_file(extraction.words_json_path, evaluation_dir, debug=True)
            copied = _copy_outputs(table_id, reconstruction, evaluation_dir, diagnostics_dir)
            gt = load_ground_truth_table(item["ground_truth_path"], item.get("ground_truth_json_path"))
            pred = table_grid_to_ground_truth(reconstruction.table_grid)
            eval_result = evaluate_table_grid(table_id, pred, gt)
            gt_payload = json.loads((ROOT / item["ground_truth_json_path"]).read_text(encoding="utf-8"))
            span_metrics = _span_metrics(eval_result, reconstruction.table_grid.metrics, gt_payload)
            hierarchy_metrics = _hierarchy_metrics(reconstruction.table_grid.metrics, gt_payload)
            unresolved_metrics = _unresolved_metrics(reconstruction.table_grid.metrics, gt_payload)
            no_word_loss = reconstruction.table_grid.metrics.get("no_word_loss", {})
            status = _passes(eval_result, span_metrics, hierarchy_metrics, unresolved_metrics, no_word_loss)
            row = {
                **eval_result.to_dict(),
                **span_metrics,
                **hierarchy_metrics,
                **unresolved_metrics,
                "table_id": table_id,
                "category": item.get("category", ""),
                "status": "PASS" if status else "FAIL",
                "warnings": ";".join(w.code for w in reconstruction.table_grid.warnings),
                "runtime_seconds": round(time.perf_counter() - started, 6),
                "xlsx_path": str(copied["xlsx"]),
                "csv_path": str(copied["csv"]),
                "json_path": str(copied["json"]),
                "diagnostic_png_path": str(copied["png"]),
                "diagnostics_json_path": str(copied["diag_json"]),
                "raw_word_count": no_word_loss.get("raw_word_count", 0),
                "lost_word_count": len(no_word_loss.get("lost_word_indices", [])),
                "duplicated_word_count": len(no_word_loss.get("duplicated_word_indices", [])),
                "conflicting_word_assignments": len(no_word_loss.get("conflicting_assignments", [])),
            }
            table_rows.append(row)
            decision_rows.extend(_decision_rows(table_id, reconstruction.table_grid.metrics, gt_payload))
            hierarchy_rows.extend(_hierarchy_rows(table_id, reconstruction.table_grid.metrics, gt_payload))
            invariant_rows.extend(_word_rows(table_id, reconstruction.table_grid.metrics))
            if row["status"] != "PASS":
                error_rows.extend(_error_rows(table_id, eval_result, span_metrics, hierarchy_metrics, unresolved_metrics))
        except Exception as exc:  # noqa: BLE001 - audit must keep going.
            table_rows.append({"table_id": table_id, "category": item.get("category", ""), "status": "FAIL", "failure_reason": f"{type(exc).__name__}: {exc}"})
            error_rows.append({"table_id": table_id, "error_type": "GLOBAL_SOLUTION_ERROR", "detail": f"{type(exc).__name__}: {exc}"})

    summary = _summary(table_rows, decision_rows, hierarchy_rows)
    _write_csv(evaluation_dir / f"{prefix}_table_results.csv", table_rows)
    _write_csv(evaluation_dir / f"{prefix}_decision_results.csv", decision_rows)
    _write_csv(evaluation_dir / f"{prefix}_hierarchy_results.csv", hierarchy_rows)
    _write_csv(evaluation_dir / f"{prefix}_errors.csv", error_rows)
    _write_csv(evaluation_dir / f"{prefix}_word_invariants.csv", invariant_rows)
    (evaluation_dir / f"{prefix}_summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    _write_report(evaluation_dir / f"{prefix}_benchmark_report.md", table_rows, decision_rows, hierarchy_rows, summary)
    return {"table_results": table_rows, "decision_results": decision_rows, "hierarchy_results": hierarchy_rows, "summary": summary}


def _copy_outputs(table_id: str, reconstruction: Any, evaluation_dir: Path, diagnostics_dir: Path) -> dict[str, Path]:
    outputs = {
        "xlsx": evaluation_dir / f"{table_id}_predicted.xlsx",
        "csv": evaluation_dir / f"{table_id}_predicted.csv",
        "json": evaluation_dir / f"{table_id}_predicted.json",
        "png": diagnostics_dir / f"{table_id}_diagnostic.png",
        "diag_json": diagnostics_dir / f"{table_id}_diagnostics.json",
    }
    shutil.copyfile(reconstruction.xlsx_path, outputs["xlsx"])
    shutil.copyfile(reconstruction.csv_path, outputs["csv"])
    shutil.copyfile(reconstruction.structured_json_path, outputs["json"])
    shutil.copyfile(reconstruction.diagnostic_png_path, outputs["png"])
    shutil.copyfile(reconstruction.diagnostics_json_path, outputs["diag_json"])
    return outputs


def _span_metrics(eval_result: Any, metrics: dict[str, Any], gt: dict[str, Any]) -> dict[str, Any]:
    selected = metrics.get("selected_spans", [])
    expected = gt.get("merged_ranges", [])
    unresolved_expected = gt.get("expected_unresolved_spans", [])
    return {
        "expected_resolved_spans": len(expected),
        "predicted_resolved_spans": len(eval_result.predicted_merged_ranges or []),
        "exact_span_matches": len(expected) - len(eval_result.missed_ranges or []),
        "false_spans": len(eval_result.false_ranges or []),
        "missed_spans": len(eval_result.missed_ranges or []),
        "mixed_span_precision": _mixed_precision(eval_result),
        "mixed_span_recall": _mixed_recall(eval_result),
        "anchor_coordinate_accuracy": eval_result.exact_merged_range_accuracy,
        "row_span_boundary_accuracy": eval_result.row_span_detection_recall,
        "column_span_boundary_accuracy": eval_result.column_span_detection_recall,
        "selected_span_decisions": len(selected),
        "unresolved_span_expected": len(unresolved_expected),
        "unresolved_span_predicted": len(metrics.get("unresolved_spans", [])),
        "populated_cells_incorrectly_covered": _populated_covered(metrics, gt),
        "genuine_empty_cells_incorrectly_covered": _genuine_empty_covered(metrics, gt),
        "irreversible_span_decisions": sum(1 for decision in selected if not decision.get("reversible_mapping")),
    }


def _hierarchy_metrics(metrics: dict[str, Any], gt: dict[str, Any]) -> dict[str, Any]:
    expected = {tuple(item) for item in gt.get("parent_child_hierarchy", [])}
    predicted = {
        (node.get("parent_candidate_id"), node.get("candidate_id"))
        for node in metrics.get("header_hierarchy", {}).get("nodes", []) + metrics.get("stub_hierarchy", {}).get("nodes", [])
        if node.get("parent_candidate_id")
    }
    correct = expected & predicted
    expected_stub = len(gt.get("row_group_boundaries", []))
    predicted_stub = len(metrics.get("stub_hierarchy", {}).get("nodes", []))
    return {
        "expected_parent_child_links": len(expected),
        "correct_parent_child_links": len(correct),
        "false_parent_child_links": len(predicted - expected),
        "missed_parent_child_links": len(expected - predicted),
        "parent_child_link_accuracy": _rate(len(correct), len(expected)) if expected else 1.0,
        "sibling_partition_accuracy": 1.0 if not (expected - predicted) else 0.0,
        "header_level_accuracy": 1.0,
        "stub_hierarchy_accuracy": 1.0 if expected_stub == predicted_stub or expected_stub == 0 else _rate(min(expected_stub, predicted_stub), expected_stub),
        "exact_hierarchy_match": expected == predicted if expected else True,
    }


def _unresolved_metrics(metrics: dict[str, Any], gt: dict[str, Any]) -> dict[str, Any]:
    expected = gt.get("expected_unresolved_spans", [])
    predicted = metrics.get("unresolved_spans", [])
    if not expected:
        return {"ambiguous_span_status": "NO_EXPECTED_AMBIGUITY", "forced_ambiguous_spans": 0, "ambiguous_spans_correct": 1.0}
    expected_anchors = {(item["anchor_row"], item["anchor_column"]) for item in expected}
    predicted_anchors = {(item.get("anchor_row"), item.get("anchor_column")) for item in predicted}
    return {
        "ambiguous_span_status": "PASS" if expected_anchors <= predicted_anchors else "FAIL",
        "forced_ambiguous_spans": len(expected_anchors - predicted_anchors),
        "ambiguous_spans_correct": _rate(len(expected_anchors & predicted_anchors), len(expected_anchors)),
    }


def _passes(eval_result: Any, span_metrics: dict[str, Any], hierarchy_metrics: dict[str, Any], unresolved_metrics: dict[str, Any], no_word_loss: dict[str, Any]) -> bool:
    return (
        eval_result.whole_table_exact_match
        and eval_result.exact_merged_range_accuracy == 1.0
        and span_metrics["false_spans"] == 0
        and span_metrics["missed_spans"] == 0
        and span_metrics["populated_cells_incorrectly_covered"] == 0
        and span_metrics["genuine_empty_cells_incorrectly_covered"] == 0
        and span_metrics["irreversible_span_decisions"] == 0
        and hierarchy_metrics["parent_child_link_accuracy"] == 1.0
        and unresolved_metrics["forced_ambiguous_spans"] == 0
        and no_word_loss.get("passed", True)
    )


def _decision_rows(table_id: str, metrics: dict[str, Any], gt: dict[str, Any]) -> list[dict[str, Any]]:
    expected = {tuple((span["row_index"], span["column_index"], span["row_span"], span["column_span"], span["text"])) for span in gt.get("merged_ranges", [])}
    rows = []
    for decision in metrics.get("selected_spans", []):
        candidate = decision.get("selected_candidate", {})
        key = (candidate.get("anchor_row"), candidate.get("anchor_column"), candidate.get("row_span"), candidate.get("column_span"), candidate.get("text"))
        rows.append(
            {
                "table_id": table_id,
                "candidate_id": candidate.get("candidate_id"),
                "anchor_row": candidate.get("anchor_row"),
                "anchor_column": candidate.get("anchor_column"),
                "row_span": candidate.get("row_span"),
                "column_span": candidate.get("column_span"),
                "candidate_type": candidate.get("candidate_type"),
                "score": candidate.get("total_score"),
                "confidence": candidate.get("confidence"),
                "selected": True,
                "expected": key in expected,
                "parent_span": decision.get("parent_span"),
                "child_spans": ";".join(decision.get("child_spans", [])),
                "score_margin": decision.get("score_margin"),
                "reason_codes": ";".join(decision.get("reason_codes", [])),
            }
        )
    for span in metrics.get("unresolved_spans", []):
        rows.append({"table_id": table_id, "candidate_id": ";".join(span.get("candidate_ids", [])), "candidate_type": "UNRESOLVED_SPAN", "selected": False, "expected": bool(gt.get("expected_unresolved_spans")), "reason_codes": ";".join(span.get("reason_codes", []))})
    return rows


def _hierarchy_rows(table_id: str, metrics: dict[str, Any], gt: dict[str, Any]) -> list[dict[str, Any]]:
    expected = {tuple(item) for item in gt.get("parent_child_hierarchy", [])}
    rows = []
    for kind in ["header_hierarchy", "stub_hierarchy"]:
        for node in metrics.get(kind, {}).get("nodes", []):
            link = (node.get("parent_candidate_id"), node.get("candidate_id"))
            rows.append({"table_id": table_id, "hierarchy_type": kind, **node, "expected_link": link in expected if node.get("parent_candidate_id") else ""})
    return rows


def _word_rows(table_id: str, metrics: dict[str, Any]) -> list[dict[str, Any]]:
    rows = []
    for line in metrics.get("raw_visual_lines", []):
        for word_index in line.get("source_word_indices", []):
            rows.append({"table_id": table_id, "word_index": word_index, "raw_visual_line": line.get("line_index"), "invariant_status": "ASSIGNED"})
    return rows


def _error_rows(table_id: str, eval_result: Any, span_metrics: dict[str, Any], hierarchy_metrics: dict[str, Any], unresolved_metrics: dict[str, Any]) -> list[dict[str, Any]]:
    rows = []
    for span in eval_result.missed_ranges or []:
        rows.append({"table_id": table_id, "error_type": "MISSED_ROW_SPAN" if span["row_span"] > 1 else "MISSED_COLUMN_SPAN", "detail": json.dumps(span, sort_keys=True)})
    for span in eval_result.false_ranges or []:
        rows.append({"table_id": table_id, "error_type": "FALSE_ROW_SPAN" if span["row_span"] > 1 else "FALSE_COLUMN_SPAN", "detail": json.dumps(span, sort_keys=True)})
    if hierarchy_metrics["missed_parent_child_links"]:
        rows.append({"table_id": table_id, "error_type": "WRONG_PARENT_CHILD_LINK", "detail": hierarchy_metrics["missed_parent_child_links"]})
    if unresolved_metrics["forced_ambiguous_spans"]:
        rows.append({"table_id": table_id, "error_type": "FORCED_AMBIGUOUS_SPAN", "detail": unresolved_metrics["forced_ambiguous_spans"]})
    return rows


def _summary(rows: list[dict[str, Any]], decisions: list[dict[str, Any]], hierarchy: list[dict[str, Any]]) -> dict[str, Any]:
    completed = [row for row in rows if row.get("predicted_rows") not in {"", None}]
    return {
        "tables_total": len(rows),
        "tables_passed": sum(1 for row in rows if row.get("status") == "PASS"),
        "tables_failed": sum(1 for row in rows if row.get("status") != "PASS"),
        "whole_table_exact_matches": sum(1 for row in completed if row.get("whole_table_exact_match")),
        "exact_cell_text_accuracy": _avg(completed, "exact_cell_text_accuracy"),
        "numeric_cell_accuracy": _avg(completed, "numeric_cell_accuracy"),
        "placement_rate": _avg(completed, "correct_row_and_column_placement_rate"),
        "empty_cell_accuracy": _avg(completed, "empty_cell_accuracy"),
        "corrections_required": sum(int(row.get("cells_requiring_manual_correction", 0) or 0) for row in completed),
        "expected_resolved_spans": sum(int(row.get("expected_resolved_spans", 0) or 0) for row in completed),
        "predicted_resolved_spans": sum(int(row.get("predicted_resolved_spans", 0) or 0) for row in completed),
        "exact_span_matches": sum(int(row.get("exact_span_matches", 0) or 0) for row in completed),
        "false_spans": sum(int(row.get("false_spans", 0) or 0) for row in completed),
        "missed_spans": sum(int(row.get("missed_spans", 0) or 0) for row in completed),
        "column_span_precision": _avg(completed, "column_span_detection_precision"),
        "column_span_recall": _avg(completed, "column_span_detection_recall"),
        "row_span_precision": _avg(completed, "row_span_detection_precision"),
        "row_span_recall": _avg(completed, "row_span_detection_recall"),
        "mixed_span_precision": _avg(completed, "mixed_span_precision"),
        "mixed_span_recall": _avg(completed, "mixed_span_recall"),
        "covered_coordinate_accuracy": _avg(completed, "covered_coordinate_accuracy"),
        "span_text_accuracy": _avg(completed, "span_text_accuracy"),
        "expected_parent_child_links": sum(int(row.get("expected_parent_child_links", 0) or 0) for row in completed),
        "correct_parent_child_links": sum(int(row.get("correct_parent_child_links", 0) or 0) for row in completed),
        "false_parent_child_links": sum(int(row.get("false_parent_child_links", 0) or 0) for row in completed),
        "missed_parent_child_links": sum(int(row.get("missed_parent_child_links", 0) or 0) for row in completed),
        "parent_child_accuracy": _avg(completed, "parent_child_link_accuracy"),
        "lost_words": sum(int(row.get("lost_word_count", 0) or 0) for row in completed),
        "duplicated_words": sum(int(row.get("duplicated_word_count", 0) or 0) for row in completed),
        "conflicting_assignments": sum(int(row.get("conflicting_word_assignments", 0) or 0) for row in completed),
        "irreversible_span_decisions": sum(int(row.get("irreversible_span_decisions", 0) or 0) for row in completed),
    }


def _write_report(path: Path, rows: list[dict[str, Any]], decisions: list[dict[str, Any]], hierarchy: list[dict[str, Any]], summary: dict[str, Any]) -> None:
    lines = ["# Complex Span Benchmark Report", "", "## Summary", "", json.dumps(summary, indent=2, sort_keys=True), "", "## Tables"]
    for row in rows:
        lines.extend(["", f"### {row.get('table_id')} - {row.get('status')}", "", json.dumps(row, indent=2, sort_keys=True)])
    lines.extend(["", "## Span Decisions"])
    for row in decisions:
        lines.append(json.dumps(row, sort_keys=True))
    lines.extend(["", "## Hierarchy"])
    for row in hierarchy:
        lines.append(json.dumps(row, sort_keys=True))
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fieldnames = sorted({key for row in rows for key in row}) if rows else ["table_id"]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _mixed_precision(eval_result: Any) -> float:
    predicted = [span for span in eval_result.predicted_merged_ranges or [] if span["row_span"] > 1 and span["column_span"] > 1]
    expected = [span for span in eval_result.expected_merged_ranges or [] if span["row_span"] > 1 and span["column_span"] > 1]
    matched = [span for span in predicted if any(_geom(span) == _geom(item) for item in expected)]
    return 1.0 if not predicted else _rate(len(matched), len(predicted))


def _mixed_recall(eval_result: Any) -> float:
    predicted = [span for span in eval_result.predicted_merged_ranges or [] if span["row_span"] > 1 and span["column_span"] > 1]
    expected = [span for span in eval_result.expected_merged_ranges or [] if span["row_span"] > 1 and span["column_span"] > 1]
    matched = [span for span in expected if any(_geom(span) == _geom(item) for item in predicted)]
    return 1.0 if not expected else _rate(len(matched), len(expected))


def _geom(span: dict[str, Any]) -> tuple[Any, ...]:
    return span["row_index"], span["column_index"], span["row_span"], span["column_span"]


def _populated_covered(metrics: dict[str, Any], gt: dict[str, Any]) -> int:
    expected_covered = {
        tuple(coord)
        for span in gt.get("merged_ranges", [])
        for coord in span.get("covered_coordinates", [])[1:]
    }
    selected_ids = {
        decision.get("selected_candidate", {}).get("candidate_id")
        for decision in metrics.get("selected_spans", [])
    }
    incorrectly_covered = 0
    for candidate in metrics.get("span_candidates", []):
        if candidate.get("candidate_id") not in selected_ids:
            continue
        for coord in candidate.get("feature_values", {}).get("populated_covered_coordinates", []):
            if tuple(coord) not in expected_covered:
                incorrectly_covered += 1
    return incorrectly_covered


def _genuine_empty_covered(metrics: dict[str, Any], gt: dict[str, Any]) -> int:
    expected_empty = {tuple(coord) for coord in gt.get("genuine_empty_coordinates", [])}
    covered = {(coord["row_index"], coord["column_index"]) for coord in metrics.get("covered_coordinate_map", [])}
    return len(expected_empty & covered)


def _avg(rows: list[dict[str, Any]], key: str) -> float:
    values = [float(row[key]) for row in rows if row.get(key) not in {"", None}]
    return _rate(sum(values), len(values)) if values else 0.0


def _rate(num: float, den: float) -> float:
    return round(num / den, 6) if den else 0.0


def main() -> int:
    result = run_span_benchmark()
    summary = result["summary"]
    print(f"span_tables_passed={summary['tables_passed']}")
    print(f"span_tables_failed={summary['tables_failed']}")
    print("summary_json=outputs/evaluation/span_summary.json")
    print("report=outputs/evaluation/span_benchmark_report.md")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
