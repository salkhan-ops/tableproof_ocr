from __future__ import annotations

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from models import BoundingBox, WordBox
from table_reconstructor import reconstruct_table_grid


def wb(index: int, text: str, x0: float, y: float) -> WordBox:
    return WordBox(index=index, text=text, bounding_box=BoundingBox(x0, y, x0 + max(8, len(text) * 4.5), y + 10))


def words(rows: list[list[tuple[str, float]]], y_step: float = 15.0) -> list[WordBox]:
    output: list[WordBox] = []
    for row_index, row in enumerate(rows):
        y = 10 + row_index * y_step
        for text, x in row:
            if text:
                output.append(wb(len(output), text, x, y))
    return output


def decisions(grid) -> list[dict]:
    return grid.metrics.get("selected_decisions", [])


def test_wrapped_final_column_continuation_targets_rightmost_cell() -> None:
    grid = reconstruct_table_grid(words([
        [("Item", 60), ("Units", 180), ("Status", 280), ("Explanation", 390)],
        [("Product A", 60), ("12", 180), ("Active", 280), ("Available throughout the", 390)],
        [("reporting period", 390)],
        [("Product B", 60), ("9", 180), ("Closed", 280), ("Discontinued", 390)],
    ]))

    decision = decisions(grid)[0]["selected_candidate"]
    assert grid.column_count == 4
    assert grid.matrix()[1] == ["Product A", "12", "Active", "Available throughout the reporting period"]
    assert decision["target_row_index"] == 1
    assert decision["target_column_index"] == 3
    assert decision["candidate_type"] == "CONTINUATION_OF_SPECIFIC_CELL"


def test_wrapped_final_column_near_note_keeps_note_excluded() -> None:
    grid = reconstruct_table_grid(words([
        [("Item", 60), ("Units", 180), ("Status", 280), ("Explanation", 390)],
        [("Product A", 60), ("12", 180), ("Active", 280), ("Available throughout the", 390)],
        [("reporting period", 390)],
        [("Note: Values are draft.", 60)],
    ]))

    assert grid.matrix()[1][3] == "Available throughout the reporting period"
    assert grid.metrics["excluded_text_blocks"][0]["classification"] == "GENERAL_NOTE"


def test_multiline_spanning_headers_keep_exact_ranges() -> None:
    grid = reconstruct_table_grid(words([
        [("Category", 60), ("Short-term operating", 160), ("Long-term financial", 360)],
        [("performance measures", 160), ("performance measures", 360)],
        [("Revenue", 160), ("Cost", 260), ("Assets", 360), ("Liabilities", 468)],
        [("A", 60), ("100", 160), ("70", 260), ("450", 360), ("210", 468)],
    ]))

    spans = {(span["row_index"], span["column_index"], span["row_span"], span["column_span"], span["text"]) for span in grid.merged_ranges()}
    assert (0, 0, 2, 1, "Category") in spans
    assert (0, 1, 1, 2, "Short-term operating performance measures") in spans
    assert (0, 3, 1, 2, "Long-term financial performance measures") in spans


def test_wrapped_header_does_not_merge_with_body_text() -> None:
    grid = reconstruct_table_grid(words([
        [("Category", 60), ("Short-term operating", 160), ("Long-term financial", 360)],
        [("performance measures", 160), ("performance measures", 360)],
        [("Revenue", 160), ("Cost", 260), ("Assets", 360), ("Liabilities", 468)],
        [("Alpha", 60), ("100", 160), ("70", 260), ("450", 360), ("210", 468)],
    ]))

    assert grid.matrix()[1] == ["", "Revenue", "Cost", "Assets", "Liabilities"]
    assert grid.matrix()[2][0] == "Alpha"


def test_panel_section_headers_span_all_columns() -> None:
    grid = reconstruct_table_grid(words([
        [("Variable", 60), ("Mean", 220), ("Median", 320), ("N", 420)],
        [("Panel A: Full Sample", 60)],
        [("Revenue", 60), ("10.2", 220), ("9.8", 320), ("200", 420)],
        [("Panel B: Restricted Sample", 60)],
        [("Assets", 60), ("48.3", 220), ("44.2", 320), ("120", 420)],
    ]))

    spans = {(span["row_index"], span["column_span"], span["text"]) for span in grid.merged_ranges()}
    assert (1, 4, "Panel A: Full Sample") in spans
    assert (3, 4, "Panel B: Restricted Sample") in spans
    assert decisions(grid) == []


def test_sparse_genuine_data_row_is_not_section_header() -> None:
    grid = reconstruct_table_grid(words([
        [("Item", 60), ("Status", 220)],
        [("Alpha", 60), ("Open", 220)],
        [("Total observations available", 60), ("Internal", 220)],
    ]))

    assert grid.matrix()[2] == ["Total observations available", "Internal"]
    assert grid.merged_ranges() == []


def test_wrapped_cell_preserves_missing_middle_numeric_cell() -> None:
    grid = reconstruct_table_grid(words([
        [("Category", 60), ("Current", 270), ("Prior", 370), ("Change", 470)],
        [("Long-term operating and", 60), ("125", 270), ("8.5%", 470)],
        [("administrative expenses", 60)],
        [("Revenue", 60), ("300", 270), ("275", 370), ("9.1%", 470)],
    ]))

    assert grid.matrix()[1] == ["Long-term operating and administrative expenses", "125", "", "8.5%"]
    assert grid.rows[1].cells[2].covered_by is None
    assert grid.rows[1].cells[3].text == "8.5%"


def test_missing_middle_numeric_cell_without_wrapped_label_does_not_shift_values() -> None:
    grid = reconstruct_table_grid(words([
        [("Category", 60), ("Current", 270), ("Prior", 370), ("Change", 470)],
        [("Expenses", 60), ("125", 270), ("8.5%", 470)],
        [("Revenue", 60), ("300", 270), ("275", 370), ("9.1%", 470)],
    ]))

    assert grid.matrix()[1] == ["Expenses", "125", "", "8.5%"]


def test_wrapped_label_with_missing_final_numeric_cell_keeps_final_empty() -> None:
    grid = reconstruct_table_grid(words([
        [("Category", 60), ("Current", 270), ("Prior", 370), ("Change", 470)],
        [("Long-term operating and", 60), ("125", 270), ("120", 370)],
        [("administrative expenses", 60)],
        [("Revenue", 60), ("300", 270), ("275", 370), ("9.1%", 470)],
    ]))

    assert grid.matrix()[1] == ["Long-term operating and administrative expenses", "125", "120", ""]


def test_new_case_structures_are_deterministic_and_traceable() -> None:
    source = words([
        [("Item", 60), ("Units", 180), ("Status", 280), ("Explanation", 390)],
        [("Product A", 60), ("12", 180), ("Active", 280), ("Available throughout the", 390)],
        [("reporting period", 390)],
    ])

    first = reconstruct_table_grid(source)
    second = reconstruct_table_grid(source)

    assert first.to_dict() == second.to_dict()
    assert first.metrics["no_word_loss"]["passed"] is True
    assert first.metrics["raw_visual_lines"]
    assert first.metrics["post_merge_cells"]
