from __future__ import annotations

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from models import BoundingBox, SpanCandidate, SpanScoreComponents
from span_constraint_solver import solve_span_candidates


def cand(cid: str, col: int, span: int, score: float) -> SpanCandidate:
    components = SpanScoreComponents(2, score - 4, 1, 1, 1, 0, 0)
    return SpanCandidate(cid, 0, col, 1, span, [[0, c] for c in range(col, col + span)], "COLUMN_HEADER_SPAN", [], cid, BoundingBox(0, 0, 10, 10), {}, [], [], components, score, score / 10, [], [], [])


def test_solver_rejects_overlapping_siblings_deterministically() -> None:
    left = cand("left", 1, 2, 8.0)
    overlap = cand("overlap", 2, 2, 9.0)
    selected, rejected, conflicts, solution = solve_span_candidates([left, overlap])

    assert [item.candidate_id for item in selected] == ["overlap"]
    assert [item.candidate_id for item in rejected] == ["left"]
    assert conflicts
    assert solution.solver_runtime_seconds == 0.0
