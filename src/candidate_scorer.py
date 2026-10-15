from __future__ import annotations

from typing import Any

from models import ContinuationCandidate


HIGH_CONFIDENCE_THRESHOLD = 3.0
MEDIUM_CONFIDENCE_THRESHOLD = 2.35
MIN_SCORE_MARGIN = 0.45


def score_candidate(
    candidate_id: str,
    source_line_indices: list[int],
    target_row_index: int | None,
    target_column_index: int | None,
    candidate_type: str,
    proposed_text: str,
    proposed_row_span: int,
    proposed_column_span: int,
    feature_values: dict[str, Any],
    joining_mode: str,
) -> ContinuationCandidate:
    positive: list[str] = []
    negative: list[str] = []
    score = 0.0

    if candidate_type in {"CONTINUATION_OF_PREVIOUS_ROW", "CONTINUATION_OF_SPECIFIC_CELL", "MULTI_LINE_HEADER"}:
        if feature_values.get("has_single_populated_column") or feature_values.get("is_sparse"):
            score += 0.65
            positive.append("SPARSE_LINE")
        if feature_values.get("gap_from_previous_normalized", 99) <= 1.25:
            score += 0.65
            positive.append("CLOSE_TO_PREVIOUS_LINE")
        if feature_values.get("target_column_has_existing_text"):
            score += 0.55
            positive.append("TARGET_CELL_HAS_TEXT")
        if feature_values.get("same_column_as_target"):
            score += 0.75
            positive.append("COLUMN_ALIGNED_WITH_TARGET")
        if feature_values.get("parenthetical_continuation"):
            score += 0.8
            positive.append("PARENTHETICAL_CONTINUATION")
        if feature_values.get("symbol_attachment"):
            score += 1.0
            positive.append("SYMBOL_ATTACHMENT")
        if feature_values.get("header_region"):
            score += 1.15
            positive.append("HEADER_REGION")
        if feature_values.get("multi_cell_same_visual_line"):
            score += 1.15
            positive.append("MULTI_CELL_SAME_VISUAL_LINE")
        if feature_values.get("restores_row_pattern"):
            score += 0.6
            positive.append("RESTORES_ROW_PATTERN")
        if feature_values.get("note_source_prefix"):
            score -= 1.2
            negative.append("NOTE_PREFIX_CONTRADICTION")
        if feature_values.get("standalone_pattern_after_merge"):
            score -= 0.6
            negative.append("STANDALONE_PATTERN_AFTER_MERGE")
        if feature_values.get("coefficient_se_separate_pattern"):
            score -= 1.4
            negative.append("COEFFICIENT_SE_SEPARATE_PATTERN")
        if feature_values.get("terminal_punctuation") and not feature_values.get("trailing_hyphen"):
            score -= 0.8
            negative.append("TERMINAL_PUNCTUATION_STANDALONE_SIGNAL")

    elif candidate_type == "STANDALONE_TABLE_ROW":
        score += 1.0
        positive.append("PRESERVE_STRUCTURE")
        if not feature_values.get("is_sparse"):
            score += 0.8
            positive.append("FULL_ROW_PATTERN")
        if feature_values.get("section_label_shape"):
            score += 0.8
            positive.append("SECTION_LABEL_SHAPE")
        if feature_values.get("note_source_prefix"):
            score -= 0.8
            negative.append("NOTE_PREFIX")

    elif candidate_type in {"FOOTNOTE_OR_NOTE", "OUTSIDE_TABLE_TRAILING_TEXT"}:
        if feature_values.get("note_source_prefix"):
            score += 1.5
            positive.append("NOTE_PREFIX")
        if feature_values.get("sentence_like_prose"):
            score += 0.8
            positive.append("SENTENCE_LIKE_PROSE")
        if feature_values.get("is_final_line"):
            score += 0.4
            positive.append("TRAILING_LINE")
        if not feature_values.get("is_sparse"):
            score -= 0.9
            negative.append("FULL_ROW_PATTERN")

    elif candidate_type == "SECTION_OR_GROUP_HEADER":
        if feature_values.get("section_label_shape"):
            score += 1.4
            positive.append("SECTION_LABEL_SHAPE")
        if feature_values.get("alphabetic_token_ratio", 0) > 0.75:
            score += 0.4
            positive.append("ALPHA_HEAVY")

    elif candidate_type == "UNCERTAIN_UNRESOLVED_LINE":
        score += 0.5
        positive.append("SAFE_FALLBACK")

    confidence = max(0.0, min(0.99, score / 4.0))
    return ContinuationCandidate(
        candidate_id=candidate_id,
        source_line_indices=source_line_indices,
        target_row_index=target_row_index,
        target_column_index=target_column_index,
        candidate_type=candidate_type,
        proposed_text=proposed_text,
        proposed_row_span=proposed_row_span,
        proposed_column_span=proposed_column_span,
        feature_values=feature_values,
        positive_evidence=positive,
        negative_evidence=negative,
        score=round(score, 3),
        confidence=round(confidence, 3),
        rejected_reason="",
        decision_rank=0,
        joining_mode=joining_mode,
    )


def rank_candidates(candidates: list[ContinuationCandidate]) -> list[ContinuationCandidate]:
    ranked = sorted(candidates, key=lambda candidate: (-candidate.score, candidate.candidate_id))
    return [
        ContinuationCandidate(
            **{**candidate.__dict__, "decision_rank": index + 1}
        )
        for index, candidate in enumerate(ranked)
    ]
