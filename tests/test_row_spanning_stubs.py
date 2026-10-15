from __future__ import annotations

from test_span_candidate_generator import words
from tableproof.table_reconstructor import reconstruct_table_grid


def test_nested_row_group_stubs_and_genuine_empty_cell() -> None:
    grid = reconstruct_table_grid(words([
        [("Group", 58), ("Subgroup", 150), ("Metric", 270), ("2024", 405)],
        [("Assets", 58), ("Current", 150), ("Cash", 270), ("20", 405)],
        [("Inventory", 270), ("", 405)],
        [("Non-current", 150), ("Property", 270), ("90", 405)],
        [("Intangibles", 270), ("", 405)],
    ]))
    spans = {(s["row_index"], s["column_index"], s["row_span"], s["text"]) for s in grid.merged_ranges()}
    assert any(row == 1 and col == 0 and span >= 2 and text == "Assets" for row, col, span, text in spans)
    assert grid.metrics["selected_spans"]
