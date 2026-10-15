from __future__ import annotations

import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))

from tableproof.table_reconstructor import load_words_json, reconstruct_table_grid


ROOT = Path(__file__).resolve().parents[1]


def _span_key(span: dict) -> tuple:
    return (
        span["row_index"],
        span["column_index"],
        span["row_span"],
        span["column_span"],
        span["text"],
        tuple(tuple(coord) for coord in span["covered_coordinates"]),
    )


def test_regression_panel_sparse_panel_words_do_not_create_model_columns() -> None:
    expected = json.loads((ROOT / "data/ground_truth/span_08_ground_truth.json").read_text(encoding="utf-8"))
    words = load_words_json(ROOT / "outputs/extracted_tables/span_08_page_001_words.json")

    grid = reconstruct_table_grid(words)

    assert grid.row_count == 8
    assert grid.column_count == 6
    assert {_span_key(span) for span in grid.merged_ranges()} == {
        _span_key(span) for span in expected["merged_ranges"]
    }
    assert grid.matrix()[2] == ["Panel A: Full Sample", "", "", "", "", ""]
    assert grid.matrix()[5] == ["Panel B: Restricted Sample", "", "", "", "", ""]


def test_compact_two_column_star_table_keeps_second_column_anchor() -> None:
    expected = json.loads((ROOT / "data/ground_truth/continuation_08_ground_truth.json").read_text(encoding="utf-8"))
    words = load_words_json(ROOT / "outputs/extracted_tables/continuation_08_page_001_words.json")

    grid = reconstruct_table_grid(words)

    assert grid.row_count == expected["physical_rows"]
    assert grid.column_count == expected["physical_columns"]
    assert grid.matrix() == [
        ["Variable", "Model"],
        ["Variable A", "0.125***"],
    ]
