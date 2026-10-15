from __future__ import annotations

import itertools

from models import GlobalSpanSolution, SpanCandidate, SpanConflict


def solve_span_candidates(candidates: list[SpanCandidate]) -> tuple[list[SpanCandidate], list[SpanCandidate], list[SpanConflict], GlobalSpanSolution]:
    selectable = [candidate for candidate in candidates if candidate.confidence >= 0.62 and not candidate.rejected_reason]
    conflicts = _conflicts(candidates)
    conflict_pairs = {
        tuple(sorted((conflict.left_candidate_id, conflict.right_candidate_id)))
        for conflict in conflicts
    }
    best: tuple[float, tuple[SpanCandidate, ...]] = (0.0, ())
    second = 0.0
    evaluated = 0
    pruned = 0
    if len(selectable) <= 18:
        iterable = itertools.chain.from_iterable(itertools.combinations(selectable, size) for size in range(len(selectable) + 1))
    else:
        iterable = (tuple(selectable[:limit]) for limit in range(len(selectable) + 1))
    for combo in iterable:
        evaluated += 1
        ids = {candidate.candidate_id for candidate in combo}
        if any(left in ids and right in ids for left, right in conflict_pairs):
            pruned += 1
            continue
        if _covered_overlap(combo):
            pruned += 1
            continue
        score = round(sum(candidate.total_score for candidate in combo) + _hierarchy_bonus(combo), 6)
        if score > best[0]:
            second = best[0]
            best = (score, combo)
        elif score > second:
            second = score
    selected = list(best[1])
    selected_ids = {candidate.candidate_id for candidate in selected}
    rejected = [candidate for candidate in candidates if candidate.candidate_id not in selected_ids]
    solution = GlobalSpanSolution(
        selected_candidate_ids=sorted(selected_ids),
        rejected_candidate_ids=sorted(candidate.candidate_id for candidate in rejected),
        total_score=best[0],
        second_best_score=second,
        score_margin=round(best[0] - second, 6),
        combinations_evaluated=evaluated,
        pruned_branches=pruned,
        solver_runtime_seconds=0.0,
        unresolved_conflicts=[],
    )
    return selected, rejected, conflicts, solution


def _conflicts(candidates: list[SpanCandidate]) -> list[SpanConflict]:
    rows = []
    coord_sets = {candidate.candidate_id: {tuple(coord) for coord in candidate.covered_coordinates} for candidate in candidates}
    for left, right in itertools.combinations(candidates, 2):
        if set(left.child_candidate_ids + left.parent_candidate_ids) & {right.candidate_id}:
            continue
        if coord_sets[left.candidate_id] & coord_sets[right.candidate_id]:
            rows.append(SpanConflict(left.candidate_id, right.candidate_id, "SPAN_OVERLAP_CONFLICT", "candidate covered coordinates overlap"))
    return rows


def _covered_overlap(combo: tuple[SpanCandidate, ...]) -> bool:
    covered = set()
    for candidate in combo:
        for coord in map(tuple, candidate.covered_coordinates[1:]):
            if coord in covered:
                return True
            covered.add(coord)
    return False


def _hierarchy_bonus(combo: tuple[SpanCandidate, ...]) -> float:
    ids = {candidate.candidate_id for candidate in combo}
    return round(sum(0.25 for candidate in combo for child in candidate.child_candidate_ids if child in ids), 6)
