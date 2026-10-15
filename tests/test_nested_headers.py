from __future__ import annotations

from test_span_candidate_generator import words
from table_reconstructor import reconstruct_table_grid


def test_asymmetric_child_group_widths_are_preserved() -> None:
    grid = reconstruct_table_grid(words([
        [("Category", 58), ("Overview", 150)],
        [("Short horizon", 150), ("Long horizon", 330)],
        [("Units", 150), ("Revenue", 240), ("Assets", 330), ("Debt", 420), ("Cash", 505)],
        [("A", 58), ("12", 150), ("100", 240), ("450", 330), ("210", 420), ("90", 505)],
    ]))
    spans = {(s["row_index"], s["column_index"], s["column_span"]) for s in grid.merged_ranges()}
    assert (0, 1, 5) in spans
    assert (1, 1, 2) in spans
    assert (1, 3, 3) in spans
