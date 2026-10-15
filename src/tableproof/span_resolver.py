from __future__ import annotations

from dataclasses import replace
from typing import Any

from tableproof.header_hierarchy import build_header_hierarchy
from tableproof.models import Cell, CoveredCoordinate, ReconstructionWarning, SpanDecision, SpanResolution, TableRow, UnresolvedSpan
from tableproof.span_candidate_generator import generate_span_candidates
from tableproof.span_candidate_scorer import confidence_label
from tableproof.span_constraint_solver import solve_span_candidates
from tableproof.stub_hierarchy import build_stub_hierarchy


def resolve_spans(rows: list[TableRow], metrics: dict[str, Any] | None = None) -> SpanResolution:
    metrics = dict(metrics or {})
    candidates = generate_span_candidates(rows, metrics)
    selected, rejected, conflicts, solution = solve_span_candidates(candidates)
    selected, hiding = _drop_spans_hiding_words(rows, selected)
    rejected = [*rejected, *hiding]
    selected_ids = {candidate.candidate_id for candidate in selected}
    replacements: dict[tuple[int, int], Cell] = {}
    covered: dict[tuple[int, int], tuple[int, int]] = {}
    decisions: list[SpanDecision] = []
    warnings: list[ReconstructionWarning] = []

    by_anchor: dict[tuple[int, int], list] = {}
    for candidate in candidates:
        by_anchor.setdefault((candidate.anchor_row, candidate.anchor_column), []).append(candidate)

    for candidate in selected:
        anchor = rows[candidate.anchor_row].cells[candidate.anchor_column]
        covered_cells = [
            rows[r].cells[c]
            for r, c in map(tuple, candidate.covered_coordinates)
            if r < len(rows) and c < len(rows[r].cells)
        ]
        replacement = replace(
            anchor,
            text=candidate.text,
            bounding_box=candidate.bounding_box,
            source_word_indices=candidate.source_word_indices,
            row_span=candidate.row_span,
            column_span=candidate.column_span,
            is_header=anchor.is_header or candidate.anchor_row < int(metrics.get("estimated_header_rows", 1)) or "HEADER" in candidate.candidate_type or "SECTION" in candidate.candidate_type,
            is_stub=anchor.is_stub or "STUB" in candidate.candidate_type or "GROUP" in candidate.candidate_type,
            confidence=min(anchor.confidence, max(0.7, candidate.confidence)),
        )
        replacements[(anchor.row_index, anchor.column_index)] = replacement
        for row_index, column_index in map(tuple, candidate.covered_coordinates[1:]):
            covered[(row_index, column_index)] = (anchor.row_index, anchor.column_index)
        alternatives = sorted(by_anchor.get((candidate.anchor_row, candidate.anchor_column), []), key=lambda item: (-item.total_score, item.candidate_id))
        margin = candidate.total_score - (alternatives[1].total_score if len(alternatives) > 1 else solution.second_best_score)
        decisions.append(
            SpanDecision(
                selected_candidate=candidate,
                all_alternatives=alternatives,
                global_solution_score=solution.total_score,
                score_margin=round(margin, 6),
                reason_codes=[confidence_label(candidate.confidence), *candidate.positive_evidence],
                parent_span=next((parent for parent in candidate.parent_candidate_ids if parent in selected_ids), None),
                child_spans=[child for child in candidate.child_candidate_ids if child in selected_ids],
                covered_coordinates=candidate.covered_coordinates,
                original_cells=[cell.to_dict() for cell in covered_cells],
                resolved_cell=replace(anchor, row_span=candidate.row_span, column_span=candidate.column_span).to_dict(),
                reversible_mapping={
                    "candidate_id": candidate.candidate_id,
                    "anchor": [anchor.row_index, anchor.column_index],
                    "source_word_indices": candidate.source_word_indices,
                    "original_cells": [cell.to_dict() for cell in covered_cells],
                    "original_empty_coordinates": [
                        [cell.row_index, cell.column_index] for cell in covered_cells[1:] if not cell.text
                    ],
                    "final_text": candidate.text,
                },
            )
        )

    unresolved = [
        UnresolvedSpan(
            candidate_ids=[candidate.candidate_id],
            anchor_row=candidate.anchor_row,
            anchor_column=candidate.anchor_column,
            competing_interpretations=["apply_span", "preserve_pre_span_grid"],
            reason_codes=["UNRESOLVED_SPAN_STRUCTURE", *candidate.negative_evidence],
            confidence=candidate.confidence,
        )
        for candidate in rejected
        if 0.45 <= candidate.confidence < 0.62 and not candidate.rejected_reason
    ]
    warnings.extend(
        ReconstructionWarning(
            "SPAN_WOULD_HIDE_WORDS",
            f"Span candidate {candidate.candidate_id} was rejected: it covers cells with their own text",
            row_index=candidate.anchor_row,
            column_index=candidate.anchor_column,
        )
        for candidate in hiding
    )
    warnings.extend(
        ReconstructionWarning(
            "UNRESOLVED_SPAN_STRUCTURE",
            f"Span candidate {span.candidate_ids[0]} was preserved unresolved",
            row_index=span.anchor_row,
            column_index=span.anchor_column,
        )
        for span in unresolved
    )

    resolved_rows: list[TableRow] = []
    for row in rows:
        cells = []
        for cell in row.cells:
            key = (cell.row_index, cell.column_index)
            if key in replacements:
                cells.append(replacements[key])
            elif key in covered:
                cells.append(replace(cell, text="", bounding_box=None, source_word_indices=[], covered_by=covered[key], confidence=0.9))
            else:
                cells.append(replace(cell, row_span=1, column_span=1, covered_by=None) if cell.row_span > 1 or cell.column_span > 1 or cell.covered_by else cell)
        resolved_rows.append(replace(row, cells=cells))

    return SpanResolution(
        rows=resolved_rows,
        candidates=candidates,
        selected_decisions=decisions,
        rejected_spans=rejected,
        unresolved_spans=unresolved,
        conflicts=conflicts,
        header_hierarchy=build_header_hierarchy(selected),
        stub_hierarchy=build_stub_hierarchy(selected),
        global_solution=solution,
        covered_coordinate_map=[
            CoveredCoordinate(row, col, anchor[0], anchor[1]) for (row, col), anchor in sorted(covered.items())
        ],
        genuine_empty_coordinate_map=[
            [cell.row_index, cell.column_index]
            for row in resolved_rows
            for cell in row.cells
            if not cell.text and cell.covered_by is None
        ],
        warnings=warnings,
    )


def _drop_spans_hiding_words(rows: list[TableRow], selected: list) -> tuple[list, list]:
    """A span may cover a cell only if every word of that cell is part of the span.

    Otherwise the covered cell's own text (e.g. a line-item label under a section heading)
    would vanish from the output. Such candidates are rejected and the pre-span grid is kept.
    """
    kept, hiding = [], []
    for candidate in selected:
        own = set(candidate.source_word_indices)
        hides = False
        for r, c in map(tuple, candidate.covered_coordinates[1:]):
            if r < len(rows) and c < len(rows[r].cells):
                if set(rows[r].cells[c].source_word_indices) - own:
                    hides = True
                    break
        (hiding if hides else kept).append(candidate)
    return kept, hiding
