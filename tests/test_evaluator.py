from __future__ import annotations

from pathlib import Path

import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))

from evaluator import (
    ERROR_FALSE_COLUMN_SPAN,
    ERROR_MISSED_COLUMN_SPAN,
    ERROR_MISSED_ROW_SPAN,
    ERROR_COLUMN_MERGE,
    ERROR_EMPTY_CELL,
    ERROR_ROW_MERGE,
    ERROR_SUPERSCRIPT_OR_STAR,
    ERROR_TEXT_CONTENT,
    GroundTruthTable,
    evaluate_table,
    evaluate_table_grid,
)


def test_perfect_prediction() -> None:
    result = evaluate_table("t", [["A", "1"], ["B", "2"]], [["A", "1"], ["B", "2"]])

    assert result.whole_table_exact_match
    assert result.exact_cell_text_accuracy == 1.0
    assert result.numeric_cell_accuracy == 1.0


def test_wrong_row_placement() -> None:
    result = evaluate_table("t", [["A", "1"]], [["A", "1"], ["B", "2"]])

    assert ERROR_ROW_MERGE in result.error_categories
    assert not result.exact_row_count_match


def test_wrong_column_placement() -> None:
    result = evaluate_table("t", [["A"], ["B"]], [["A", "1"], ["B", "2"]])

    assert ERROR_COLUMN_MERGE in result.error_categories
    assert not result.exact_column_count_match


def test_missing_empty_cell() -> None:
    result = evaluate_table("t", [["A", "x", "2"]], [["A", "", "2"]])

    assert ERROR_EMPTY_CELL in result.error_categories
    assert result.empty_cell_accuracy == 0.0


def test_incorrect_minus_sign_is_not_normalized_away() -> None:
    result = evaluate_table("t", [["1.2"]], [["-1.2"]])

    assert ERROR_TEXT_CONTENT in result.error_categories
    assert result.numeric_cell_accuracy == 0.0


def test_missing_significance_star() -> None:
    result = evaluate_table("t", [["0.12"]], [["0.12*"]])

    assert ERROR_SUPERSCRIPT_OR_STAR in result.error_categories
    assert result.numeric_cell_accuracy == 0.0


def test_parentheses_mismatch() -> None:
    result = evaluate_table("t", [["0.10"]], [["(0.10)"]])

    assert ERROR_TEXT_CONTENT in result.error_categories


def test_dimension_mismatch() -> None:
    result = evaluate_table("t", [["A", "B", "C"]], [["A", "B"]])

    assert not result.exact_column_count_match
    assert result.cells_requiring_manual_correction == 1


def test_unicode_whitespace_normalization() -> None:
    result = evaluate_table("t", [["Alpha\u00a0  Beta"]], [["Alpha Beta"]])

    assert result.whole_table_exact_match


def test_deterministic_metric_results() -> None:
    first = evaluate_table("t", [["A", "1"], ["B", "x"]], [["A", "1"], ["B", "2"]])
    second = evaluate_table("t", [["A", "1"], ["B", "x"]], [["A", "1"], ["B", "2"]])

    assert first.to_dict() == second.to_dict()
    assert first.cell_errors == second.cell_errors


def span(row: int, col: int, text: str, rs: int, cs: int) -> dict:
    return {
        "row_index": row,
        "column_index": col,
        "text": text,
        "row_span": rs,
        "column_span": cs,
        "covered_coordinates": [[r, c] for r in range(row, row + rs) for c in range(col, col + cs)],
        "is_header": row < 2,
        "is_stub": col == 0 and row >= 2,
        "bounding_box": None,
    }


def gt(matrix, spans) -> GroundTruthTable:
    logical = [
        span(r, c, matrix[r][c], 1, 1)
        for r in range(len(matrix))
        for c in range(len(matrix[0]))
        if all([r, c] not in item["covered_coordinates"][1:] for item in spans)
    ]
    return GroundTruthTable(matrix, len(matrix), len(matrix[0]), logical + spans, spans)


def test_exact_merged_ranges_match() -> None:
    merged = [span(0, 1, "Q1 Results", 1, 2), span(2, 0, "Category A", 3, 1)]
    table = gt(
        [
            ["Category", "Q1 Results", ""],
            ["", "Revenue", "Cost"],
            ["Category A", "10", "7"],
            ["", "9", "6"],
            ["", "19", "13"],
        ],
        merged,
    )

    result = evaluate_table_grid("m", table, table)

    assert result.exact_merged_range_accuracy == 1.0
    assert result.column_span_detection_recall == 1.0
    assert result.row_span_detection_recall == 1.0


def test_missed_and_false_merged_ranges_are_classified() -> None:
    truth = gt([["Header", ""], ["A", "1"]], [span(0, 0, "Header", 1, 2)])
    pred = gt([["Header", ""], ["A", "1"]], [])

    result = evaluate_table_grid("m", pred, truth)

    assert ERROR_MISSED_COLUMN_SPAN in result.error_categories
    assert result.column_span_detection_recall == 0.0

    false_result = evaluate_table_grid("m", truth, pred)

    assert ERROR_FALSE_COLUMN_SPAN in false_result.error_categories
    assert false_result.column_span_detection_precision == 0.0


def test_missed_row_span_is_classified() -> None:
    truth = gt([["Stub", "A"], ["Group", "1"], ["", "2"]], [span(1, 0, "Group", 2, 1)])
    pred = gt([["Stub", "A"], ["Group", "1"], ["", "2"]], [])

    result = evaluate_table_grid("m", pred, truth)

    assert ERROR_MISSED_ROW_SPAN in result.error_categories
    assert result.row_span_detection_recall == 0.0


def test_covered_coordinate_is_not_penalized_as_empty() -> None:
    truth = gt([["Header", ""], ["A", "1"]], [span(0, 0, "Header", 1, 2)])
    pred = gt([["Header", ""], ["A", "1"]], [span(0, 0, "Header", 1, 2)])

    result = evaluate_table_grid("m", pred, truth)

    assert result.empty_cell_accuracy == 1.0
    assert result.covered_coordinate_accuracy == 1.0


def test_duplicate_text_across_span_needs_correction() -> None:
    truth = gt([["Header", ""], ["A", "1"]], [span(0, 0, "Header", 1, 2)])
    pred = gt([["Header", "Header"], ["A", "1"]], [])

    result = evaluate_table_grid("m", pred, truth)

    assert result.cells_requiring_manual_correction >= 1
