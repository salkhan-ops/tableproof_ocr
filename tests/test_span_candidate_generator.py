from __future__ import annotations

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from models import BoundingBox, WordBox
from table_reconstructor import reconstruct_table_grid


def wb(index: int, text: str, x: float, y: float) -> WordBox:
    return WordBox(index, text, BoundingBox(x, y, x + max(8, len(text) * 4.5), y + 10))


def words(rows: list[list[tuple[str, float]]]) -> list[WordBox]:
    output = []
    for row_index, row in enumerate(rows):
        for text, x in row:
            output.append(wb(len(output), text, x, 10 + row_index * 15))
    return output


def test_three_level_nested_header_candidates_and_mappings() -> None:
    grid = reconstruct_table_grid(words([
        [("Segment", 60), ("Performance Summary", 145)],
        [("Operating Measures", 145), ("Financial Measures", 370)],
        [("Revenue", 145), ("Cost", 218), ("Margin", 291), ("Assets", 370), ("Debt", 448), ("Equity", 526)],
        [("North", 60), ("100", 145), ("60", 218), ("40", 291), ("450", 370), ("210", 448), ("240", 526)],
    ]))

    spans = {(s["row_index"], s["column_index"], s["row_span"], s["column_span"], s["text"]) for s in grid.merged_ranges()}
    assert (0, 1, 1, 6, "Performance Summary") in spans
    assert grid.metrics["span_candidates"]
    assert grid.metrics["span_word_mappings"]


def test_ambiguous_short_header_span_remains_unresolved() -> None:
    grid = reconstruct_table_grid(words([
        [("Measure", 60), ("Combined", 190), ("Separate", 430)],
        [("Alpha", 60), ("10", 190), ("11", 310), ("12", 430)],
        [("Beta", 60), ("11", 190), ("12", 310), ("13", 430)],
    ]))

    assert grid.merged_ranges() == []
    assert grid.metrics["unresolved_spans"]
