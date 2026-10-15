from __future__ import annotations

from test_span_candidate_generator import words
from table_reconstructor import reconstruct_table_grid


def test_sparse_all_text_row_does_not_become_span() -> None:
    grid = reconstruct_table_grid(words([
        [("Item", 60), ("Status", 220)],
        [("Alpha", 60), ("Open", 220)],
        [("Total observations available", 60), ("Internal", 220)],
    ]))
    assert grid.merged_ranges() == []


def test_raw_and_resolved_span_exports_exist() -> None:
    grid = reconstruct_table_grid(words([
        [("Category", 60), ("Q1 Results", 160), ("Q2 Results", 360)],
        [("Revenue", 160), ("Cost", 260), ("Revenue", 360), ("Cost", 468)],
        [("A", 60), ("100", 160), ("70", 260), ("120", 360), ("80", 468)],
    ]))
    assert grid.metrics["pre_span_grid"]
    assert grid.metrics["selected_global_solution"]
