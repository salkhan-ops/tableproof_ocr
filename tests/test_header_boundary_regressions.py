from __future__ import annotations

from dataclasses import replace
from pathlib import Path
import json
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from models import BoundingBox
from table_reconstructor import load_words_json, reconstruct_table_grid


ROOT = Path(__file__).resolve().parents[1]


def span_key(span: dict) -> tuple:
    return (
        span["row_index"],
        span["column_index"],
        span["row_span"],
        span["column_span"],
        span["text"],
        tuple(tuple(coord) for coord in span["covered_coordinates"]),
    )


def expected_spans(table_id: str) -> set[tuple]:
    path = ROOT / "data" / "ground_truth" / f"{table_id}_ground_truth.json"
    return {span_key(span) for span in json.loads(path.read_text(encoding="utf-8"))["merged_ranges"]}


def reconstruct_spans(table_id: str) -> set[tuple]:
    words_path = ROOT / "outputs" / "extracted_tables" / f"{table_id}_page_001_words.json"
    grid = reconstruct_table_grid(load_words_json(words_path))
    assert grid.row_count == int(json.loads((ROOT / "data" / "ground_truth" / f"{table_id}_ground_truth.json").read_text())["physical_rows"])
    assert grid.column_count == int(json.loads((ROOT / "data" / "ground_truth" / f"{table_id}_ground_truth.json").read_text())["physical_columns"])
    return {span_key(span) for span in grid.merged_ranges()}


def test_asymmetric_nested_header_boundaries_are_exact() -> None:
    assert reconstruct_spans("span_02") == expected_spans("span_02")


def test_adjacent_header_groups_remain_separated() -> None:
    assert reconstruct_spans("span_03") == expected_spans("span_03")


def test_staggered_header_depth_does_not_create_false_equalizing_spans() -> None:
    assert reconstruct_spans("span_07") == expected_spans("span_07")


def test_header_boundary_decisions_are_stable_under_small_coordinate_perturbations() -> None:
    for table_id in ["span_02", "span_03", "span_07"]:
        words = load_words_json(ROOT / "outputs" / "extracted_tables" / f"{table_id}_page_001_words.json")
        baseline = reconstruct_spans(table_id)
        for delta in [0.5, -1.0, 2.0]:
            shifted = [
                replace(
                    word,
                    bounding_box=BoundingBox(
                        word.bounding_box.x0 + delta,
                        word.bounding_box.y0 + delta / 4,
                        word.bounding_box.x1 + delta,
                        word.bounding_box.y1 + delta / 4,
                    ),
                )
                for word in words
            ]
            assert {span_key(span) for span in reconstruct_table_grid(shifted).merged_ranges()} == baseline
