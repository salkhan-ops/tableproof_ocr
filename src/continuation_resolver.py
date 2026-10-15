from __future__ import annotations

from dataclasses import replace
from typing import Any

from candidate_scorer import HIGH_CONFIDENCE_THRESHOLD, MIN_SCORE_MARGIN, rank_candidates, score_candidate
from line_classifier import line_feature_vectors, visual_lines_from_rows
from models import (
    BoundingBox,
    Cell,
    ContinuationCandidate,
    ContinuationDecision,
    ReconstructionWarning,
    TableRow,
    UnresolvedLine,
)


WARNING_AMBIGUOUS_CONTINUATION = "AMBIGUOUS_CONTINUATION"
WARNING_LOW_CONTINUATION_CONFIDENCE = "LOW_CONTINUATION_CONFIDENCE"
WARNING_CONTINUATION_MERGED = "CONTINUATION_MERGED"
WARNING_MULTIPLE_CONTINUATION_LINES_MERGED = "MULTIPLE_CONTINUATION_LINES_MERGED"
WARNING_POSSIBLE_WRAPPED_CELL = "POSSIBLE_WRAPPED_CELL"
WARNING_WRAPPED_HEADER_RESOLVED = "WRAPPED_HEADER_RESOLVED"
WARNING_POSSIBLE_SECTION_HEADER = "POSSIBLE_SECTION_HEADER"
WARNING_SECTION_HEADER_PRESERVED = "SECTION_HEADER_PRESERVED"
WARNING_POSSIBLE_VERTICAL_STUB_SPAN = "POSSIBLE_VERTICAL_STUB_SPAN"
WARNING_VERTICAL_STUB_SPAN_RESOLVED = "VERTICAL_STUB_SPAN_RESOLVED"
WARNING_POSSIBLE_COEFFICIENT_SE_PAIR = "POSSIBLE_COEFFICIENT_SE_PAIR"
WARNING_HYPHENATION_AMBIGUOUS = "HYPHENATION_AMBIGUOUS"
WARNING_HYPHEN_REMOVED = "HYPHEN_REMOVED"
WARNING_HYPHEN_PRESERVED = "HYPHEN_PRESERVED"
WARNING_UNMATCHED_PARENTHESES = "UNMATCHED_PARENTHESES"
WARNING_SYMBOL_ATTACHED_TO_CELL = "SYMBOL_ATTACHED_TO_CELL"
WARNING_POSSIBLE_NESTED_TABLE = "POSSIBLE_NESTED_TABLE"
WARNING_UNSUPPORTED_ROTATED_TEXT = "UNSUPPORTED_ROTATED_TEXT"
WARNING_POSSIBLE_TABLE_CONTINUATION_NEXT_PAGE = "POSSIBLE_TABLE_CONTINUATION_NEXT_PAGE"
WARNING_POSSIBLE_TABLE_CONTINUATION_PREVIOUS_PAGE = "POSSIBLE_TABLE_CONTINUATION_PREVIOUS_PAGE"
WARNING_CONTINUATION_CONFLICT_WITH_FOOTNOTE_CLASSIFIER = "CONTINUATION_CONFLICT_WITH_FOOTNOTE_CLASSIFIER"
WARNING_CONTINUATION_CONFLICT_WITH_SPAN_INFERENCE = "CONTINUATION_CONFLICT_WITH_SPAN_INFERENCE"
WARNING_UNRESOLVED_VISUAL_LINE = "UNRESOLVED_VISUAL_LINE"


def resolve_continuations(
    rows: list[TableRow],
    metrics: dict[str, Any],
    excluded_word_indices: set[int] | None = None,
) -> tuple[list[TableRow], list[ReconstructionWarning], dict[str, Any]]:
    excluded_word_indices = excluded_word_indices or set()
    raw_lines = visual_lines_from_rows(rows)
    feature_vectors = line_feature_vectors(rows, metrics.get("median_word_height", 10.0))
    feature_lookup = {vector.line_index: vector.feature_values for vector in feature_vectors}
    candidates_by_line: dict[int, list[ContinuationCandidate]] = {}
    selected: list[ContinuationDecision] = []
    unresolved: list[UnresolvedLine] = []
    warnings: list[ReconstructionWarning] = []
    rows_by_index = {row.row_index: row for row in rows}
    remove_rows: set[int] = set()
    selected_sources: dict[int, ContinuationDecision] = {}

    for row in rows:
        if row.row_index == 0:
            continue
        if any(word_index in excluded_word_indices for cell in row.cells for word_index in cell.source_word_indices):
            continue
        line_candidates = _generate_candidates(row, rows_by_index, feature_lookup.get(row.row_index, {}))
        ranked = rank_candidates(line_candidates)
        candidates_by_line[row.row_index] = ranked
        if not ranked:
            continue
        multi_select = _multi_cell_decisions_for_row(row, ranked)
        if multi_select:
            for decision_candidate, margin in multi_select:
                decision = _decision_from_candidate(decision_candidate, ranked, margin, row, rows_by_index)
                selected.append(decision)
                selected_sources[row.row_index] = decision
                warnings.append(
                    ReconstructionWarning(
                        WARNING_CONTINUATION_MERGED,
                        f"Merged visual line {row.row_index} into row {decision_candidate.target_row_index}, column {decision_candidate.target_column_index}",
                        row_index=decision_candidate.target_row_index,
                        column_index=decision_candidate.target_column_index,
                        word_indices=decision.source_word_indices,
                    )
                )
            remove_rows.add(row.row_index)
            continue
        best = ranked[0]
        second = ranked[1] if len(ranked) > 1 else None
        margin = round(best.score - (second.score if second else 0.0), 3)
        if best.candidate_type in {"CONTINUATION_OF_PREVIOUS_ROW", "CONTINUATION_OF_SPECIFIC_CELL", "MULTI_LINE_HEADER"} and best.score >= HIGH_CONFIDENCE_THRESHOLD and margin >= MIN_SCORE_MARGIN:
            decision = _decision_from_candidate(best, ranked, margin, row, rows_by_index)
            selected.append(decision)
            selected_sources[row.row_index] = decision
            remove_rows.add(row.row_index)
            warnings.append(
                ReconstructionWarning(
                    WARNING_CONTINUATION_MERGED,
                    f"Merged visual line {row.row_index} into row {best.target_row_index}, column {best.target_column_index}",
                    row_index=best.target_row_index,
                    column_index=best.target_column_index,
                    word_indices=decision.source_word_indices,
                )
            )
            if best.joining_mode == "HYPHEN_REMOVAL_JOIN":
                warnings.append(ReconstructionWarning(WARNING_HYPHEN_REMOVED, "Removed layout hyphen during continuation merge", word_indices=decision.source_word_indices))
            elif best.joining_mode == "PRESERVE_HYPHEN_JOIN":
                warnings.append(ReconstructionWarning(WARNING_HYPHEN_PRESERVED, "Preserved lexical hyphen during continuation merge", word_indices=decision.source_word_indices))
            elif best.joining_mode == "SYMBOL_ATTACHMENT":
                warnings.append(ReconstructionWarning(WARNING_SYMBOL_ATTACHED_TO_CELL, "Attached symbol/star continuation to target cell", word_indices=decision.source_word_indices))
        elif (
            best.candidate_type == "UNCERTAIN_UNRESOLVED_LINE"
            or margin < MIN_SCORE_MARGIN
            or (
                best.candidate_type
                in {"CONTINUATION_OF_PREVIOUS_ROW", "CONTINUATION_OF_SPECIFIC_CELL", "MULTI_LINE_HEADER"}
                and best.score < HIGH_CONFIDENCE_THRESHOLD
            )
        ):
            unresolved.append(
                UnresolvedLine(
                    line_index=row.row_index,
                    text=" ".join(cell.text for cell in row.cells if cell.text),
                    competing_interpretations=[candidate.candidate_type for candidate in ranked[:3]],
                    reason_codes=["LOW_SCORE_MARGIN" if margin < MIN_SCORE_MARGIN else "LOW_SCORE"],
                    confidence=best.confidence,
                )
            )
            warnings.append(ReconstructionWarning(WARNING_AMBIGUOUS_CONTINUATION, f"Visual line {row.row_index} left unresolved", row_index=row.row_index))
            remove_rows.add(row.row_index)
        elif best.candidate_type == "STANDALONE_TABLE_ROW" and feature_lookup.get(row.row_index, {}).get("terminal_punctuation") and feature_lookup.get(row.row_index, {}).get("is_sparse"):
            unresolved.append(
                UnresolvedLine(
                    line_index=row.row_index,
                    text=" ".join(cell.text for cell in row.cells if cell.text),
                    competing_interpretations=[candidate.candidate_type for candidate in ranked[:3]],
                    reason_codes=["SPARSE_PUNCTUATED_STANDALONE"],
                    confidence=best.confidence,
                )
            )
            warnings.append(ReconstructionWarning(WARNING_UNRESOLVED_VISUAL_LINE, f"Visual line {row.row_index} preserved as unresolved standalone text", row_index=row.row_index))
            remove_rows.add(row.row_index)

    target_updates = _redirect_chained_decisions(selected, selected_sources, remove_rows)
    resolved_rows = _apply_decisions(rows, remove_rows, target_updates)
    if len(selected) > 1:
        warnings.append(ReconstructionWarning(WARNING_MULTIPLE_CONTINUATION_LINES_MERGED, f"Merged {len(selected)} continuation lines"))
    diagnostics = {
        "raw_visual_lines": [line.to_dict() for line in raw_lines],
        "line_feature_vectors": [vector.to_dict() for vector in feature_vectors],
        "continuation_candidates": [candidate.to_dict() for candidates in candidates_by_line.values() for candidate in candidates],
        "candidate_scores": [
            {"candidate_id": candidate.candidate_id, "score": candidate.score}
            for candidates in candidates_by_line.values()
            for candidate in candidates
        ],
        "candidate_score_components": [
            {
                "candidate_id": candidate.candidate_id,
                "positive_evidence": candidate.positive_evidence,
                "negative_evidence": candidate.negative_evidence,
            }
            for candidates in candidates_by_line.values()
            for candidate in candidates
        ],
        "candidate_rankings": [
            {"line_index": line_index, "candidate_ids": [candidate.candidate_id for candidate in candidates]}
            for line_index, candidates in candidates_by_line.items()
        ],
        "selected_decisions": [decision.to_dict() for decision in selected],
        "rejected_candidates": [
            candidate.to_dict()
            for candidates in candidates_by_line.values()
            for candidate in candidates
            if all(candidate.candidate_id != decision.selected_candidate.candidate_id for decision in selected)
        ],
        "score_margins": [decision.score_margin for decision in selected],
        "joining_modes": [decision.selected_candidate.joining_mode for decision in selected],
        "pre_merge_cells": [row.to_dict() for row in rows],
        "post_merge_cells": [row.to_dict() for row in resolved_rows],
        "reversible_word_mapping": _word_mapping(resolved_rows, selected, excluded_word_indices),
        "unresolved_lines": [line.to_dict() for line in unresolved],
        "continuation_warning_details": [warning.to_dict() for warning in warnings],
        "thresholds_used": {"high_confidence": HIGH_CONFIDENCE_THRESHOLD, "min_score_margin": MIN_SCORE_MARGIN},
        "stability_checks": {"small_threshold_perturbation": "not_run_deterministic_baseline"},
        "no_word_loss": _no_word_loss(raw_lines, resolved_rows, excluded_word_indices, unresolved),
    }
    return resolved_rows, warnings, diagnostics


def _generate_candidates(row: TableRow, rows_by_index: dict[int, TableRow], features: dict[str, Any]) -> list[ContinuationCandidate]:
    previous = rows_by_index.get(row.row_index - 1)
    candidates: list[ContinuationCandidate] = []
    text = " ".join(cell.text for cell in row.cells if cell.text)
    populated = [cell for cell in row.cells if cell.text]
    allow_continuation = (
        previous is not None
        and populated
        and len(row.cells) > 1
        and not _looks_like_body_group_row(row, rows_by_index, features)
        and not features.get("section_label_shape")
        and (
            (len(populated) == 1 and not features.get("terminal_punctuation"))
            or (
                row.row_index == 1
                and [cell.column_index for cell in populated]
                == [cell.column_index for cell in previous.cells if cell.text]
                and not any(_is_numeric_like(cell.text) for cell in populated)
                and not any(_is_numeric_like(cell.text) for cell in previous.cells if cell.text)
                and all(cell.text[:1].islower() for cell in populated if cell.text)
            )
            or (
                len(populated) > 1
                and all(cell.text[:1].islower() for cell in populated if cell.text)
                and not any(_is_numeric_like(cell.text) for cell in populated)
                and all(
                    cell.column_index < len(previous.cells) and bool(previous.cells[cell.column_index].text)
                    for cell in populated
                )
                and not features.get("terminal_punctuation")
            )
            or (
                len(populated) == 1
                and len(populated) < len(row.cells)
                and previous is not None
                and len(populated) < sum(1 for cell in previous.cells if cell.text)
                and not any(_is_numeric_like(cell.text) for cell in populated)
                and not features.get("terminal_punctuation")
            )
            or (
                len(populated) == len(row.cells)
                and len(populated) > 1
                and all(cell.text[:1].islower() for cell in populated if cell.text)
                and not any(_is_numeric_like(cell.text) for cell in populated)
                and not features.get("terminal_punctuation")
            )
        )
    )
    if previous and populated and allow_continuation:
        for cell in populated:
            target_column = cell.column_index
            if target_column < len(previous.cells):
                target = previous.cells[target_column]
                parenthetical_continuation = target.text.count("(") > target.text.count(")") or cell.text.strip() == ")"
                symbol_attachment = features.get("significance_star_pattern", False)
                if _is_numeric_like(cell.text) and not (parenthetical_continuation or symbol_attachment):
                    continue
                enriched = dict(features)
                enriched.update(
                    {
                        "target_column_has_existing_text": bool(target.text),
                        "same_column_as_target": True,
                        "parenthetical_continuation": parenthetical_continuation,
                        "symbol_attachment": symbol_attachment,
                        "header_region": previous.row_index == 0,
                        "multi_cell_same_visual_line": len(populated) > 1,
                        "restores_row_pattern": len(populated) < len(row.cells),
                        "coefficient_se_separate_pattern": cell.text.startswith("(") and _neighbor_parenthetical_pattern(rows_by_index, row.row_index),
                    }
                )
                joining_mode = _joining_mode(target.text, cell.text)
                candidate_type = "MULTI_LINE_HEADER" if enriched["header_region"] else "CONTINUATION_OF_SPECIFIC_CELL"
                candidates.append(
                    score_candidate(
                        f"line{row.row_index}-prev-r{previous.row_index}-c{target_column}",
                        [row.row_index],
                        previous.row_index,
                        target_column,
                        candidate_type,
                        _join_text(target.text, cell.text, joining_mode),
                        target.row_span,
                        target.column_span,
                        enriched,
                        joining_mode,
                    )
                )
    candidates.append(
        score_candidate(
            f"line{row.row_index}-standalone",
            [row.row_index],
            row.row_index,
            None,
            "STANDALONE_TABLE_ROW",
            text,
            1,
            1,
            features,
            "SPACE_JOIN",
        )
    )
    candidates.append(
        score_candidate(
            f"line{row.row_index}-footnote",
            [row.row_index],
            None,
            None,
            "FOOTNOTE_OR_NOTE",
            text,
            1,
            1,
            features,
            "SPACE_JOIN",
        )
    )
    candidates.append(
        score_candidate(
            f"line{row.row_index}-unresolved",
            [row.row_index],
            None,
            None,
            "UNCERTAIN_UNRESOLVED_LINE",
            text,
            1,
            1,
            features,
            "SPACE_JOIN",
        )
    )
    return candidates


def _looks_like_body_group_row(row: TableRow, rows_by_index: dict[int, TableRow], features: dict[str, Any]) -> bool:
    populated = [cell for cell in row.cells if cell.text]
    if not populated or any(_is_numeric_like(cell.text) for cell in populated):
        return False
    if features.get("leading_lowercase") or all(cell.text[:1].islower() for cell in populated if cell.text):
        return False
    previous = rows_by_index.get(row.row_index - 1)
    following_rows = [rows_by_index[index] for index in range(row.row_index + 1, row.row_index + 4) if index in rows_by_index]
    previous_has_data = previous is not None and any(_is_numeric_like(cell.text) for cell in previous.cells)
    following_data_rows = [
        next_row
        for next_row in following_rows
        if sum(1 for cell in next_row.cells if _is_numeric_like(cell.text)) >= 2
    ]
    sparse_stub_only = (
        len(populated) <= 2
        and all(cell.column_index <= max(1, len(row.cells) // 3) for cell in populated)
        and bool(features.get("is_sparse"))
    )
    if not sparse_stub_only:
        return False
    same_or_deeper_label_follows = any(
        any(cell.text and not _is_numeric_like(cell.text) and cell.column_index >= populated[-1].column_index for cell in next_row.cells[: max(2, len(row.cells) // 2)])
        for next_row in following_rows
    )
    return previous_has_data and bool(following_data_rows) and same_or_deeper_label_follows


def _multi_cell_decisions_for_row(row: TableRow, ranked: list[ContinuationCandidate]) -> list[tuple[ContinuationCandidate, float]]:
    continuation_types = {"CONTINUATION_OF_PREVIOUS_ROW", "CONTINUATION_OF_SPECIFIC_CELL", "MULTI_LINE_HEADER"}
    eligible = [
        candidate
        for candidate in ranked
        if candidate.candidate_type in continuation_types
        and candidate.score >= HIGH_CONFIDENCE_THRESHOLD
        and candidate.target_column_index is not None
    ]
    populated = [cell for cell in row.cells if cell.text]
    if len(eligible) < 2 or len(populated) < 2:
        return []
    if not all(cell.text[:1].islower() for cell in populated if cell.text):
        return []
    top_score = eligible[0].score
    selected = []
    seen_columns: set[int] = set()
    for candidate in eligible:
        if top_score - candidate.score > 0.05:
            continue
        if candidate.target_column_index in seen_columns:
            continue
        seen_columns.add(candidate.target_column_index)
        selected.append(candidate)
    if len(selected) < 2:
        return []
    alternative_score = max((candidate.score for candidate in ranked if candidate.candidate_type not in continuation_types), default=0.0)
    margin = round(top_score - alternative_score, 3)
    if margin < MIN_SCORE_MARGIN:
        return []
    return [(candidate, margin) for candidate in selected]


def _is_numeric_like(text: str) -> bool:
    import re

    return bool(re.fullmatch(r"\(?[-+]?\d[\d,]*(\.\d+)?%?\)?(\*{1,3})?", text.strip()))


def _neighbor_parenthetical_pattern(rows_by_index: dict[int, TableRow], row_index: int) -> bool:
    parenthetical_count = 0
    for idx in [row_index - 2, row_index, row_index + 2]:
        row = rows_by_index.get(idx)
        if row and any(cell.text.startswith("(") for cell in row.cells):
            parenthetical_count += 1
    return parenthetical_count >= 2


def _joining_mode(left: str, right: str) -> str:
    if right in {"*", "**", "***", "†", "‡"}:
        return "SYMBOL_ATTACHMENT"
    if left.count("(") > left.count(")") or right == ")":
        return "PARENTHETICAL_JOIN"
    if left.endswith("-") and right[:1].islower():
        if left.lower().endswith(("long-", "short-", "well-")):
            return "PRESERVE_HYPHEN_JOIN"
        return "HYPHEN_REMOVAL_JOIN"
    if "\n" in left or "\n" in right:
        return "NEWLINE_PRESERVING_JOIN"
    return "SPACE_JOIN"


def _join_text(left: str, right: str, mode: str) -> str:
    if not left:
        return right
    if mode == "NO_SPACE_JOIN" or mode == "SYMBOL_ATTACHMENT":
        return f"{left}{right}"
    if mode == "HYPHEN_REMOVAL_JOIN":
        return f"{left[:-1]}{right}"
    if mode == "PRESERVE_HYPHEN_JOIN":
        return f"{left}{right}"
    if mode == "NEWLINE_PRESERVING_JOIN":
        return f"{left}\n{right}"
    if mode == "PARENTHETICAL_JOIN" and right == ")":
        return f"{left})"
    return f"{left} {right}".strip()


def _decision_from_candidate(
    best: ContinuationCandidate,
    ranked: list[ContinuationCandidate],
    margin: float,
    source_row: TableRow,
    rows_by_index: dict[int, TableRow],
) -> ContinuationDecision:
    target = rows_by_index[best.target_row_index].cells[best.target_column_index]
    source_cells = [
        cell
        for cell in source_row.cells
        if cell.text and (best.target_column_index is None or cell.column_index == best.target_column_index)
    ]
    if not source_cells:
        source_cells = [cell for cell in source_row.cells if cell.text]
    boxes = [
        cell.bounding_box.to_dict()
        for cell in [target] + source_cells
        if cell.bounding_box
    ]
    final_box = target.bounding_box
    for cell in source_cells:
        if cell.text and cell.bounding_box:
            final_box = final_box.union(cell.bounding_box) if final_box else cell.bounding_box
    source_word_indices = [word for cell in source_cells for word in cell.source_word_indices]
    original_text = " ".join(cell.text for cell in source_cells if cell.text)
    return ContinuationDecision(
        selected_candidate=best,
        all_considered_candidates=ranked,
        score_margin=margin,
        confidence=round(min(0.99, best.confidence + margin / 5.0), 3),
        reason_codes=best.positive_evidence,
        source_word_indices=source_word_indices,
        original_text=original_text,
        final_text=best.proposed_text,
        original_bounding_boxes=boxes,
        final_bounding_box=final_box,
        reversible_mapping={
            "source_line_indices": best.source_line_indices,
            "target_row_index": best.target_row_index,
            "target_column_index": best.target_column_index,
            "joining_mode": best.joining_mode,
        },
    )


def _apply_decisions(
    rows: list[TableRow],
    remove_rows: set[int],
    target_updates: dict[tuple[int, int], list[ContinuationDecision]],
) -> list[TableRow]:
    resolved: list[TableRow] = []
    for row in rows:
        if row.row_index in remove_rows:
            continue
        new_cells = []
        for cell in row.cells:
            updates = target_updates.get((row.row_index, cell.column_index), [])
            updated_cell = cell
            for decision in updates:
                append_text = decision.original_text
                joining_mode = decision.selected_candidate.joining_mode
                updated_cell = replace(
                    updated_cell,
                    text=_join_text(updated_cell.text, append_text, joining_mode),
                    bounding_box=decision.final_bounding_box,
                    source_word_indices=updated_cell.source_word_indices + decision.source_word_indices,
                    confidence=min(updated_cell.confidence, decision.confidence),
                )
            new_cells.append(updated_cell)
        resolved.append(TableRow(row_index=len(resolved), cells=[replace(cell, row_index=len(resolved)) for cell in new_cells], bounding_box=row.bounding_box, confidence=row.confidence, warnings=row.warnings))
    return resolved


def _redirect_chained_decisions(
    decisions: list[ContinuationDecision],
    selected_sources: dict[int, ContinuationDecision],
    remove_rows: set[int],
) -> dict[tuple[int, int], list[ContinuationDecision]]:
    updates: dict[tuple[int, int], list[ContinuationDecision]] = {}
    for decision in decisions:
        candidate = decision.selected_candidate
        target_row = candidate.target_row_index
        target_col = candidate.target_column_index
        while target_row in remove_rows and target_row in selected_sources:
            parent = selected_sources[target_row].selected_candidate
            target_row = parent.target_row_index
            target_col = parent.target_column_index if target_col is None else target_col
        updates.setdefault((target_row, target_col), []).append(decision)
    for key in updates:
        updates[key].sort(key=lambda item: item.selected_candidate.source_line_indices[0])
    return updates


def _word_mapping(
    rows: list[TableRow],
    decisions: list[ContinuationDecision],
    excluded_word_indices: set[int],
) -> dict[str, Any]:
    assignments: dict[int, list[dict[str, int]]] = {}
    for row in rows:
        for cell in row.cells:
            for word_index in cell.source_word_indices:
                assignments.setdefault(word_index, []).append({"row_index": cell.row_index, "column_index": cell.column_index})
    return {
        "resolved_cell_assignments": assignments,
        "excluded_word_indices": sorted(excluded_word_indices),
        "decision_word_indices": [word for decision in decisions for word in decision.source_word_indices],
    }


def _no_word_loss(
    raw_lines: list[Any],
    rows: list[TableRow],
    excluded_word_indices: set[int],
    unresolved: list[UnresolvedLine],
) -> dict[str, Any]:
    raw_words = {word for line in raw_lines for word in line.source_word_indices}
    assigned = {word for row in rows for cell in row.cells for word in cell.source_word_indices}
    unresolved_words: set[int] = set()
    unresolved_line_indices = {line.line_index for line in unresolved}
    for line in raw_lines:
        if line.line_index in unresolved_line_indices:
            unresolved_words.update(line.source_word_indices)
    accounted = assigned | excluded_word_indices | unresolved_words
    duplicates = sorted(
        word
        for word in assigned
        if sum(word in cell.source_word_indices for row in rows for cell in row.cells) > 1
    )
    return {
        "raw_word_count": len(raw_words),
        "assigned_word_count": len(assigned),
        "excluded_word_count": len(excluded_word_indices),
        "unresolved_word_count": len(unresolved_words),
        "lost_word_indices": sorted(raw_words - accounted),
        "duplicated_word_indices": duplicates,
        "passed": not (raw_words - accounted) and not duplicates,
    }
