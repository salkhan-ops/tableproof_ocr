from __future__ import annotations

from pathlib import Path
import json
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from table_reconstructor import load_words_json, reconstruct_table_grid


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


def test_span_09_hierarchical_body_stub_spans_are_exact() -> None:
    expected = json.loads((ROOT / "data/ground_truth/span_09_ground_truth.json").read_text(encoding="utf-8"))
    words = load_words_json(ROOT / "outputs/extracted_tables/span_09_page_001_words.json")
    grid = reconstruct_table_grid(words)

    assert grid.row_count == expected["physical_rows"]
    assert grid.column_count == expected["physical_columns"]
    assert {_span_key(span) for span in grid.merged_ranges()} == {
        _span_key(span) for span in expected["merged_ranges"]
    }


def test_categorical_value_row_does_not_absorb_following_same_level_label() -> None:
    words = load_words_json(ROOT / "outputs/extracted_tables/benchmark_02_page_001_words.json")
    grid = reconstruct_table_grid(words)
    spans = {_span_key(span) for span in grid.merged_ranges()}

    assert not any(span[0] == 5 and span[1] == 0 and span[2] > 1 for span in spans)
    assert grid.matrix()[5][0] == "Controls"
    assert grid.matrix()[6][0] == "Observations"
