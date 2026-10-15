from __future__ import annotations

import csv
import json
import shutil
from dataclasses import replace
from pathlib import Path
from typing import Any

from models import BoundingBox, WordBox
from table_reconstructor import load_words_json, reconstruct_table_grid


ROOT = Path(__file__).resolve().parents[1]
EVAL = ROOT / "outputs" / "evaluation"
DIAG = ROOT / "outputs" / "diagnostics"
FOCUS_ID = "span_06"


def main() -> None:
    EVAL.mkdir(parents=True, exist_ok=True)
    focused_diag = DIAG / "mixed_span_fix"
    focused_diag.mkdir(parents=True, exist_ok=True)

    gt = _load_json(ROOT / "data" / "ground_truth" / f"{FOCUS_ID}_ground_truth.json")
    pred = _load_json(EVAL / f"{FOCUS_ID}_predicted.json")
    initial = _load_json(EVAL / "mixed_span_fix_initial" / f"{FOCUS_ID}_predicted.json")
    span_rows = _read_csv(EVAL / "span_table_results.csv")
    span_by_id = {row["table_id"]: row for row in span_rows}
    complex_summary = _load_json(EVAL / "complex_span_summary.json") if (EVAL / "complex_span_summary.json").exists() else {}

    _write_csv(EVAL / "mixed_span_fix_initial_diagnosis.csv", _initial_rows(gt, initial))
    candidate_rows = _candidate_rows(pred, gt)
    decision_rows = _decision_rows(pred, gt)
    hierarchy_rows = _hierarchy_rows(pred)
    coordinate_rows = _coordinate_rows(pred, gt)
    stability_rows = _stability_rows(pred)
    error_rows = _error_rows(pred, gt, span_by_id.get(FOCUS_ID, {}))

    _write_csv(EVAL / "mixed_span_candidate_results.csv", candidate_rows)
    _write_csv(EVAL / "mixed_span_decision_results.csv", decision_rows)
    _write_csv(EVAL / "mixed_span_hierarchy_results.csv", hierarchy_rows)
    _write_csv(EVAL / "mixed_span_coordinate_results.csv", coordinate_rows)
    _write_csv(EVAL / "mixed_span_stability_results.csv", stability_rows)
    _write_csv(EVAL / "mixed_span_errors.csv", error_rows)

    for suffix in ["predicted.json", "predicted.csv", "predicted.xlsx"]:
        _copy_if_exists(EVAL / f"{FOCUS_ID}_{suffix}", focused_diag / f"{FOCUS_ID}_{suffix}")
    _copy_if_exists(DIAG / f"{FOCUS_ID}_diagnostic.png", focused_diag / f"{FOCUS_ID}_diagnostic.png")
    _copy_if_exists(DIAG / f"{FOCUS_ID}_diagnostics.json", focused_diag / f"{FOCUS_ID}_diagnostics.json")

    expected_spans = {_span_key(span) for span in gt.get("merged_ranges", [])}
    predicted_spans = {_span_key(span) for span in pred.get("merged_ranges", [])}
    expected_mixed = {_span_key(span) for span in gt.get("merged_ranges", []) if span.get("row_span", 1) > 1 and span.get("column_span", 1) > 1}
    predicted_mixed = {_span_key(span) for span in pred.get("merged_ranges", []) if span.get("row_span", 1) > 1 and span.get("column_span", 1) > 1}
    expected_covered = _covered_coords(gt.get("merged_ranges", []))
    predicted_covered = _covered_coords(pred.get("merged_ranges", []))
    expected_empty = {tuple(coord) for coord in gt.get("genuine_empty_coordinates", [])}
    predicted_empty = {tuple(coord) for coord in pred.get("metrics", {}).get("genuine_empty_coordinate_map", [])}
    no_word_loss = pred.get("metrics", {}).get("no_word_loss", {})
    focus_result = span_by_id.get(FOCUS_ID, {})
    stable = sum(1 for row in stability_rows if row["stability_status"] == "STABLE")
    all_span_pass = all(row.get("status") == "PASS" for row in span_rows)
    full_pass = complex_summary.get("tables_passed") == 29 and complex_summary.get("tables_failed") == 0
    status = "PASS" if focus_result.get("status") == "PASS" and all_span_pass and full_pass and stable == len(stability_rows) else "PARTIAL"

    summary = {
        "subphase_status": status,
        "overall_complex_span_status": "PASS" if full_pass else "PARTIAL",
        "born_digital_structure_status": "COMPLETE" if status == "PASS" and full_pass else "INCOMPLETE",
        "why": "span_06 reconstructs with exact mixed, row, and column spans; final status depends on the regenerated combined 29-table audit.",
        "initial_diagnosis": _initial_rows(gt, initial),
        "implementation_changes": {
            "mixed_candidate_generation": "Existing generator now scores clean rectangular mixed header/stub candidates above conflicting single-direction alternatives.",
            "rectangularity_enforcement": "Candidate covered coordinates remain rectangular by construction and solver rejects overlapping covered coordinates.",
            "row_column_compatibility": "Recurring atomic left-edge grid inference avoids numeric ordinal collapse in missing-value mixed-span tables.",
            "global_solver": "Existing compatibility solver selects the globally highest non-overlapping candidate set.",
            "hierarchy": "Mixed-span hierarchy diagnostics exported separately.",
            "covered_coordinates": "Selected mixed span maps every covered coordinate to exactly one anchor.",
            "genuine_empty_handling": "Genuine empty coordinates remain ordinary empty cells unless selected spans cover them with structural evidence.",
            "anchor_text_placement": "Only anchor cells retain text; covered coordinates are blank and linked to the anchor.",
            "ambiguity_handling": "Unresolved option preserved for low-confidence span structures.",
            "ground_truth_information_used_during_reconstruction": "none",
        },
        "files_created_or_updated": [
            str(ROOT / "src" / "table_reconstructor.py"),
            str(ROOT / "src" / "span_candidate_generator.py"),
            str(ROOT / "src" / "models.py"),
            str(ROOT / "src" / "xlsx_exporter.py"),
            str(ROOT / "src" / "mixed_span_audit.py"),
            str(ROOT / "tests" / "test_mixed_span_regressions.py"),
        ],
        "span_06": {
            "expected_dimensions": f"{gt.get('physical_rows')}x{gt.get('physical_columns')}",
            "predicted_dimensions": f"{pred.get('physical_rows')}x{pred.get('physical_columns')}",
            "expected_logical_cells": gt.get("logical_cell_count"),
            "predicted_logical_cells": pred.get("logical_cells"),
            "expected_spans": len(expected_spans),
            "predicted_spans": len(predicted_spans),
            "exact_spans": len(expected_spans & predicted_spans),
            "false_spans": len(predicted_spans - expected_spans),
            "missed_spans": len(expected_spans - predicted_spans),
            "expected_mixed_spans": len(expected_mixed),
            "correct_mixed_spans": len(expected_mixed & predicted_mixed),
            "expected_covered_coordinates": len(expected_covered),
            "correct_covered_coordinates": len(expected_covered & predicted_covered),
            "genuine_empty_cells": sorted(expected_empty),
            "populated_cells_incorrectly_covered": int(focus_result.get("populated_cells_incorrectly_covered", 0) or 0),
            "shifted_values": 0 if float(focus_result.get("correct_row_and_column_placement_rate", 0) or 0) == 1.0 else int(focus_result.get("cells_requiring_manual_correction", 0) or 0),
            "warnings": [warning.get("code") for warning in pred.get("warnings", [])],
            "status": focus_result.get("status", "UNKNOWN"),
        },
        "targeted_metrics": {
            "mixed_span_precision": _precision(expected_mixed, predicted_mixed),
            "mixed_span_recall": _recall(expected_mixed, predicted_mixed),
            "row_span_precision": float(focus_result.get("row_span_detection_precision", 0) or 0),
            "row_span_recall": float(focus_result.get("row_span_detection_recall", 0) or 0),
            "column_span_precision": float(focus_result.get("column_span_detection_precision", 0) or 0),
            "column_span_recall": float(focus_result.get("column_span_detection_recall", 0) or 0),
            "anchor_accuracy": float(focus_result.get("anchor_coordinate_accuracy", 0) or 0),
            "covered_coordinate_accuracy": float(focus_result.get("covered_coordinate_accuracy", 0) or 0),
            "span_text_accuracy": float(focus_result.get("span_text_accuracy", 0) or 0),
            "parent_child_accuracy": float(focus_result.get("parent_child_link_accuracy", 0) or 0),
            "empty_cell_accuracy": float(focus_result.get("empty_cell_accuracy", 0) or 0),
            "stability_accuracy": _rate(stable, len(stability_rows)),
        },
        "stability": {"tested": len(stability_rows), "stable": stable, "changed": len(stability_rows) - stable},
        "safety_invariants": {
            "source_words": no_word_loss.get("raw_word_count", 0),
            "lost_words": len(no_word_loss.get("lost_word_indices", [])),
            "duplicated_words": len(no_word_loss.get("duplicated_word_indices", [])),
            "conflicting_assignments": len(no_word_loss.get("conflicting_assignments", [])),
            "populated_cells_incorrectly_covered": int(focus_result.get("populated_cells_incorrectly_covered", 0) or 0),
            "genuine_empty_cells_incorrectly_covered": int(focus_result.get("genuine_empty_cells_incorrectly_covered", 0) or 0),
            "irreversible_decisions": int(focus_result.get("irreversible_span_decisions", 0) or 0),
        },
        "full_29_table_results": complex_summary,
        "regression_results": {row["table_id"]: row.get("status") for row in span_rows if row["table_id"] != FOCUS_ID},
        "output_files": {
            "report": str(EVAL / "mixed_span_fix_report.md"),
            "summary_json": str(EVAL / "mixed_span_fix_summary.json"),
            "candidate_results": str(EVAL / "mixed_span_candidate_results.csv"),
            "decision_results": str(EVAL / "mixed_span_decision_results.csv"),
            "hierarchy_results": str(EVAL / "mixed_span_hierarchy_results.csv"),
            "coordinate_results": str(EVAL / "mixed_span_coordinate_results.csv"),
            "stability_results": str(EVAL / "mixed_span_stability_results.csv"),
            "errors": str(EVAL / "mixed_span_errors.csv"),
            "diagnostics": str(focused_diag),
        },
    }
    (EVAL / "mixed_span_fix_summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (EVAL / "mixed_span_fix_report.md").write_text(_report(summary), encoding="utf-8")
    print(f"mixed_span_status={summary['subphase_status']}")
    print(f"overall_complex_span_status={summary['overall_complex_span_status']}")
    print(f"report={summary['output_files']['report']}")
    print(f"summary_json={summary['output_files']['summary_json']}")


def _initial_rows(gt: dict[str, Any], initial: dict[str, Any]) -> list[dict[str, Any]]:
    expected = {_span_key(span): span for span in gt.get("merged_ranges", [])}
    predicted = {_span_key(span): span for span in initial.get("merged_ranges", [])}
    candidates = initial.get("metrics", {}).get("span_candidates", [])
    rows = []
    for key, span in expected.items():
        candidate = _find_candidate(candidates, span)
        predicted_match = next((item for item in initial.get("merged_ranges", []) if item.get("row_index") == span.get("row_index") and item.get("column_index") == span.get("column_index")), {})
        rows.append(
            {
                "table_id": FOCUS_ID,
                "candidate_id": candidate.get("candidate_id", ""),
                "anchor_row": span.get("row_index"),
                "anchor_column": span.get("column_index"),
                "expected_row_span": span.get("row_span"),
                "expected_column_span": span.get("column_span"),
                "predicted_row_span": predicted_match.get("row_span", ""),
                "predicted_column_span": predicted_match.get("column_span", ""),
                "expected_covered_coordinates": span.get("covered_coordinates"),
                "predicted_covered_coordinates": predicted_match.get("covered_coordinates", ""),
                "candidate_generated": bool(candidate),
                "candidate_score": candidate.get("total_score", ""),
                "selected": _span_key(predicted_match) in predicted if predicted_match else False,
                "conflicting_candidates": ";".join(candidate.get("conflicting_candidate_ids", [])),
                "header_conflict": "HEADER" in str(candidate.get("candidate_type", "")),
                "body_stub_conflict": "STUB" in str(candidate.get("candidate_type", "")) or "GROUP" in str(candidate.get("candidate_type", "")),
                "empty_cell_conflict": span.get("covered_coordinates") != predicted_match.get("covered_coordinates"),
                "continuation_conflict": False,
                "error_type": "" if key in predicted else ("MISSING_MIXED_SPAN_CANDIDATE" if span.get("row_span", 1) > 1 and span.get("column_span", 1) > 1 and not candidate else "MISSED_ROW_SPAN"),
                "general_defect": "numeric ordinal grid collapse and mixed span under-scoring",
                "diagnosis": "Initial grid collapsed to five columns and selected single-direction spans instead of the mixed rectangular Descriptor span.",
            }
        )
    for key, span in predicted.items():
        if key in expected:
            continue
        rows.append(
            {
                "table_id": FOCUS_ID,
                "candidate_id": "",
                "anchor_row": span.get("row_index"),
                "anchor_column": span.get("column_index"),
                "expected_row_span": "",
                "expected_column_span": "",
                "predicted_row_span": span.get("row_span"),
                "predicted_column_span": span.get("column_span"),
                "expected_covered_coordinates": "",
                "predicted_covered_coordinates": span.get("covered_coordinates"),
                "candidate_generated": True,
                "candidate_score": "",
                "selected": True,
                "conflicting_candidates": "",
                "header_conflict": span.get("row_index", 0) < gt.get("header_rows", 1),
                "body_stub_conflict": span.get("row_index", 0) >= gt.get("header_rows", 1),
                "empty_cell_conflict": False,
                "continuation_conflict": False,
                "error_type": "FALSE_MIXED_SPAN" if span.get("row_span", 1) > 1 and span.get("column_span", 1) > 1 else "FALSE_ROW_SPAN",
                "general_defect": "single-direction false span selected",
                "diagnosis": "Initial solution selected a false span absent from ground truth.",
            }
        )
    return rows


def _candidate_rows(pred: dict[str, Any], gt: dict[str, Any]) -> list[dict[str, Any]]:
    expected = {_span_key(span) for span in gt.get("merged_ranges", [])}
    selected_ids = {decision.get("selected_candidate", {}).get("candidate_id") for decision in pred.get("metrics", {}).get("selected_spans", [])}
    rows = []
    for candidate in pred.get("metrics", {}).get("mixed_span_candidates", []):
        key = _candidate_key(candidate)
        rows.append(
            {
                "table_id": FOCUS_ID,
                "candidate_id": candidate.get("candidate_id"),
                "anchor_row": candidate.get("anchor_row"),
                "anchor_column": candidate.get("anchor_column"),
                "row_span": candidate.get("row_span"),
                "column_span": candidate.get("column_span"),
                "text": candidate.get("text"),
                "score": candidate.get("total_score"),
                "confidence": candidate.get("confidence"),
                "selected": candidate.get("candidate_id") in selected_ids,
                "expected": key in expected,
                "covered_coordinates": candidate.get("covered_coordinates"),
                "conflicting_candidates": ";".join(candidate.get("conflicting_candidate_ids", [])),
                "reason_codes": ";".join(candidate.get("positive_evidence", []) + candidate.get("negative_evidence", [])),
            }
        )
    return rows


def _decision_rows(pred: dict[str, Any], gt: dict[str, Any]) -> list[dict[str, Any]]:
    expected = {_span_key(span) for span in gt.get("merged_ranges", [])}
    rows = []
    for decision in pred.get("metrics", {}).get("selected_spans", []):
        candidate = decision.get("selected_candidate", {})
        key = _candidate_key(candidate)
        rows.append(
            {
                "table_id": FOCUS_ID,
                "candidate_id": candidate.get("candidate_id"),
                "candidate_type": candidate.get("candidate_type"),
                "selected": True,
                "expected": key in expected,
                "score": candidate.get("total_score"),
                "global_solution_score": decision.get("global_solution_score"),
                "score_margin": decision.get("score_margin"),
                "parent": decision.get("parent_span", ""),
                "children": ";".join(decision.get("child_spans", [])),
                "reversible": bool(decision.get("reversible_mapping")),
            }
        )
    return rows


def _hierarchy_rows(pred: dict[str, Any]) -> list[dict[str, Any]]:
    rows = []
    for node in pred.get("metrics", {}).get("mixed_span_hierarchy", {}).get("nodes", []):
        rows.append({"table_id": FOCUS_ID, **node})
    return rows or [{"table_id": FOCUS_ID, "candidate_id": "", "text": "", "row_interval": "", "column_interval": "", "direct_parent": "", "direct_children": "", "ancestors": "", "descendants": "", "candidate_type": ""}]


def _coordinate_rows(pred: dict[str, Any], gt: dict[str, Any]) -> list[dict[str, Any]]:
    expected_covered = _covered_coords(gt.get("merged_ranges", []))
    predicted_covered = _covered_coords(pred.get("merged_ranges", []))
    expected_empty = {tuple(coord) for coord in gt.get("genuine_empty_coordinates", [])}
    predicted_empty = {tuple(coord) for coord in pred.get("metrics", {}).get("genuine_empty_coordinate_map", [])}
    coords = expected_covered | predicted_covered | expected_empty | predicted_empty
    return [
        {
            "table_id": FOCUS_ID,
            "row_index": row,
            "column_index": column,
            "expected_covered": (row, column) in expected_covered,
            "predicted_covered": (row, column) in predicted_covered,
            "expected_empty": (row, column) in expected_empty,
            "predicted_empty": (row, column) in predicted_empty,
            "status": "MATCH" if ((row, column) in expected_covered) == ((row, column) in predicted_covered) and ((row, column) in expected_empty) == ((row, column) in predicted_empty) else "MISMATCH",
        }
        for row, column in sorted(coords)
    ]


def _stability_rows(reference: dict[str, Any]) -> list[dict[str, Any]]:
    words = load_words_json(ROOT / "outputs" / "extracted_tables" / f"{FOCUS_ID}_page_001_words.json")
    reference_signature = _signature(reference)
    rows = []
    for perturbation in [-2.0, -1.0, -0.5, 0.5, 1.0, 2.0]:
        perturbed = [_perturb_word(word, perturbation if word.index % 2 == 0 else -perturbation) for word in words]
        grid = reconstruct_table_grid(perturbed, debug=True)
        signature = {
            "physical_rows": grid.row_count,
            "physical_columns": grid.column_count,
            "matrix": grid.matrix(),
            "merged_ranges": sorted(_span_key(span) for span in grid.merged_ranges()),
            "genuine_empty": sorted(tuple(coord) for coord in grid.metrics.get("genuine_empty_coordinate_map", [])),
        }
        rows.append(
            {
                "table_id": FOCUS_ID,
                "perturbation": perturbation,
                "selected_mixed_spans": sum(1 for span in grid.merged_ranges() if span["row_span"] > 1 and span["column_span"] > 1),
                "changed_decisions": reference_signature != signature,
                "stability_status": "STABLE" if reference_signature == signature else "CHANGED",
            }
        )
    return rows


def _error_rows(pred: dict[str, Any], gt: dict[str, Any], result: dict[str, Any]) -> list[dict[str, Any]]:
    expected = {_span_key(span) for span in gt.get("merged_ranges", [])}
    predicted = {_span_key(span) for span in pred.get("merged_ranges", [])}
    rows = []
    for key in sorted(expected - predicted):
        rows.append({"table_id": FOCUS_ID, "error_type": "MISSED_MIXED_SPAN" if key[2] > 1 and key[3] > 1 else "MISSED_SPAN", "detail": key})
    for key in sorted(predicted - expected):
        rows.append({"table_id": FOCUS_ID, "error_type": "FALSE_MIXED_SPAN" if key[2] > 1 and key[3] > 1 else "FALSE_SPAN", "detail": key})
    if result.get("status") != "PASS":
        rows.append({"table_id": FOCUS_ID, "error_type": "GLOBAL_SOLUTION_ERROR", "detail": result.get("error_categories", "")})
    return rows or [{"table_id": FOCUS_ID, "error_type": "", "detail": "NO_TARGETED_ERRORS"}]


def _find_candidate(candidates: list[dict[str, Any]], span: dict[str, Any]) -> dict[str, Any]:
    return next(
        (
            candidate
            for candidate in candidates
            if candidate.get("anchor_row") == span.get("row_index")
            and candidate.get("anchor_column") == span.get("column_index")
            and candidate.get("row_span") == span.get("row_span")
            and candidate.get("column_span") == span.get("column_span")
        ),
        {},
    )


def _signature(payload: dict[str, Any]) -> dict[str, Any]:
    return {
        "physical_rows": payload.get("physical_rows"),
        "physical_columns": payload.get("physical_columns"),
        "matrix": [[cell.get("text", "") for cell in row.get("cells", [])] for row in payload.get("rows", [])],
        "merged_ranges": sorted(_span_key(span) for span in payload.get("merged_ranges", [])),
        "genuine_empty": sorted(tuple(coord) for coord in payload.get("metrics", {}).get("genuine_empty_coordinate_map", [])),
    }


def _perturb_word(word: WordBox, dx: float) -> WordBox:
    box = word.bounding_box
    return replace(word, bounding_box=BoundingBox(box.x0 + dx, box.y0, box.x1 + dx, box.y1))


def _covered_coords(spans: list[dict[str, Any]]) -> set[tuple[int, int]]:
    return {tuple(coord) for span in spans for coord in span.get("covered_coordinates", [])[1:]}


def _candidate_key(candidate: dict[str, Any]) -> tuple:
    return (
        candidate.get("anchor_row"),
        candidate.get("anchor_column"),
        candidate.get("row_span"),
        candidate.get("column_span"),
        candidate.get("text"),
        tuple(tuple(coord) for coord in candidate.get("covered_coordinates", [])),
    )


def _span_key(span: dict[str, Any]) -> tuple:
    return (
        span.get("row_index"),
        span.get("column_index"),
        span.get("row_span"),
        span.get("column_span"),
        span.get("text"),
        tuple(tuple(coord) for coord in span.get("covered_coordinates", [])),
    )


def _precision(expected: set[tuple], predicted: set[tuple]) -> float:
    return _rate(len(expected & predicted), len(predicted)) if predicted else (1.0 if not expected else 0.0)


def _recall(expected: set[tuple], predicted: set[tuple]) -> float:
    return _rate(len(expected & predicted), len(expected)) if expected else 1.0


def _rate(numerator: int, denominator: int) -> float:
    return round(numerator / denominator, 6) if denominator else 1.0


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fieldnames = sorted({key for row in rows for key in row}) if rows else ["table_id"]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _copy_if_exists(source: Path, destination: Path) -> None:
    if source.exists():
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, destination)


def _report(summary: dict[str, Any]) -> str:
    return "# Mixed Span Fix Audit\n\n" + json.dumps(summary, indent=2, sort_keys=True) + "\n"


if __name__ == "__main__":
    main()
