from __future__ import annotations

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pytest

from models import BoundingBox, WordBox
from table_reconstructor import reconstruct_table_grid


def wb(index: int, text: str, x0: float, y0: float, x1: float | None = None) -> WordBox:
    width = max(8.0, len(text) * 5.0)
    return WordBox(index=index, text=text, bounding_box=BoundingBox(x0, y0, x1 or x0 + width, y0 + 10))


def words_from_rows(rows: list[list[tuple[str, float]]], y_step: float = 16.0) -> list[WordBox]:
    words: list[WordBox] = []
    for row_index, row in enumerate(rows):
        y = 10 + row_index * y_step
        for text, x in row:
            if text:
                words.append(wb(len(words), text, x, y))
    return words


def selected(grid) -> list[dict]:
    return grid.metrics.get("selected_decisions", [])


def test_wrapped_first_column_cell_merges_without_losing_words() -> None:
    grid = reconstruct_table_grid(
        words_from_rows(
            [
                [("Description", 10), ("Value", 150)],
                [("Operating expenses", 10), ("100", 150)],
                [("continued", 10)],
            ]
        )
    )

    assert grid.matrix() == [["Description", "Value"], ["Operating expenses continued", "100"]]
    assert grid.metrics["no_word_loss"]["passed"] is True
    assert selected(grid)[0]["selected_candidate"]["joining_mode"] == "SPACE_JOIN"


def test_two_cells_wrapped_on_same_visual_line_merge_to_their_own_columns() -> None:
    grid = reconstruct_table_grid(
        words_from_rows(
            [
                [("Header", 10), ("Measure", 170)],
                [("Product description", 10), ("Measurement description", 170)],
                [("continued", 10), ("continued", 170)],
                [("Alpha", 10), ("Rate", 170)],
            ]
        )
    )

    assert grid.matrix()[1] == ["Product description continued", "Measurement description continued"]
    assert [(d["selected_candidate"]["target_column_index"], d["original_text"]) for d in selected(grid)] == [(0, "continued"), (1, "continued")]
    assert grid.metrics["no_word_loss"]["passed"] is True


def test_multiline_header_resolves_each_header_cell() -> None:
    grid = reconstruct_table_grid(
        words_from_rows(
            [
                [("Three-month", 10), ("Twelve-month", 170)],
                [("return", 10), ("return", 170)],
                [("5%", 10), ("12%", 170)],
            ]
        )
    )

    assert grid.matrix()[0] == ["Three-month return", "Twelve-month return"]
    assert len(selected(grid)) == 2


@pytest.mark.parametrize(
    ("left", "right", "expected", "mode"),
    [
        ("inter-", "national revenue", "international revenue", "HYPHEN_REMOVAL_JOIN"),
        ("long-", "term assets", "long-term assets", "PRESERVE_HYPHEN_JOIN"),
    ],
)
def test_hyphenated_line_break_modes(left: str, right: str, expected: str, mode: str) -> None:
    grid = reconstruct_table_grid(
        words_from_rows(
            [
                [("Label", 10), ("Value", 150)],
                [(left, 10), ("100", 150)],
                [(right, 10)],
            ]
        )
    )

    assert grid.matrix()[1][0] == expected
    assert selected(grid)[0]["selected_candidate"]["joining_mode"] == mode


def test_ambiguous_punctuated_trailing_line_is_reversible_and_unresolved() -> None:
    grid = reconstruct_table_grid(
        words_from_rows(
            [
                [("Item", 10), ("Value", 150)],
                [("Alpha", 10), ("100", 150)],
                [("maybe continued.", 10)],
            ]
        )
    )

    assert grid.matrix() == [["Item", "Value"], ["Alpha", "100"]]
    assert selected(grid) == []
    assert grid.metrics["unresolved_lines"][0]["line_index"] == 2
    assert grid.metrics["no_word_loss"]["passed"] is True


def test_significance_star_legend_is_excluded_as_note() -> None:
    grid = reconstruct_table_grid(
        words_from_rows(
            [
                [("Variable", 10), ("Model", 150)],
                [("Treatment", 10), ("0.125***", 150)],
                [("*** p<0.01, ** p<0.05, * p<0.10", 10)],
            ]
        )
    )

    assert grid.matrix() == [["Variable", "Model"], ["Treatment", "0.125***"]]
    assert grid.metrics["excluded_text_blocks"][0]["classification"] == "FOOTNOTE"
    assert "NOTE_PREFIX" in grid.metrics["excluded_text_blocks"][0]["reason_codes"]
