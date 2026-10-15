from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class BoundingBox:
    x0: float
    y0: float
    x1: float
    y1: float

    @classmethod
    def from_csv(cls, value: str) -> "BoundingBox":
        parts = [part.strip() for part in value.split(",")]
        if len(parts) != 4:
            raise ValueError("Bounding box must contain four comma-separated numbers: x0,y0,x1,y1")

        try:
            x0, y0, x1, y1 = (float(part) for part in parts)
        except ValueError as exc:
            raise ValueError("Bounding box coordinates must be numeric") from exc

        bbox = cls(x0=x0, y0=y0, x1=x1, y1=y1)
        bbox.validate()
        return bbox

    def validate(self) -> None:
        if self.x1 <= self.x0 or self.y1 <= self.y0:
            raise ValueError("Bounding box must satisfy x1 > x0 and y1 > y0")

    def to_dict(self) -> dict[str, float]:
        return asdict(self)

    @property
    def width(self) -> float:
        return self.x1 - self.x0

    @property
    def height(self) -> float:
        return self.y1 - self.y0

    @property
    def x_mid(self) -> float:
        return (self.x0 + self.x1) / 2

    @property
    def y_mid(self) -> float:
        return (self.y0 + self.y1) / 2

    def union(self, other: "BoundingBox") -> "BoundingBox":
        return BoundingBox(
            x0=min(self.x0, other.x0),
            y0=min(self.y0, other.y0),
            x1=max(self.x1, other.x1),
            y1=max(self.y1, other.y1),
        )


@dataclass(frozen=True)
class NativeWord:
    text: str
    bbox: BoundingBox
    block_no: int
    line_no: int
    word_no: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "text": self.text,
            "bbox": self.bbox.to_dict(),
            "block_no": self.block_no,
            "line_no": self.line_no,
            "word_no": self.word_no,
        }


@dataclass(frozen=True)
class WordBox:
    index: int
    text: str
    bounding_box: BoundingBox

    @classmethod
    def from_dict(cls, index: int, payload: dict[str, Any]) -> "WordBox":
        text = " ".join(str(payload.get("text", "")).split())
        bbox_payload = payload.get("bbox")
        if not isinstance(bbox_payload, dict):
            raise ValueError(f"Word {index} is missing a bbox object")
        bbox = BoundingBox(
            x0=float(bbox_payload["x0"]),
            y0=float(bbox_payload["y0"]),
            x1=float(bbox_payload["x1"]),
            y1=float(bbox_payload["y1"]),
        )
        bbox.validate()
        return cls(index=index, text=text, bounding_box=bbox)

    def to_dict(self) -> dict[str, Any]:
        return {
            "index": self.index,
            "text": self.text,
            "bounding_box": self.bounding_box.to_dict(),
        }


@dataclass(frozen=True)
class ReconstructionWarning:
    code: str
    message: str
    row_index: int | None = None
    column_index: int | None = None
    word_indices: list[int] | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "code": self.code,
            "message": self.message,
            "row_index": self.row_index,
            "column_index": self.column_index,
            "word_indices": self.word_indices or [],
        }


@dataclass(frozen=True)
class ExcludedTextBlock:
    text: str
    bounding_box: BoundingBox | None
    source_word_indices: list[int]
    classification: str
    confidence: float
    reason_codes: list[str]

    def to_dict(self) -> dict[str, Any]:
        return {
            "text": self.text,
            "bounding_box": self.bounding_box.to_dict() if self.bounding_box else None,
            "source_word_indices": self.source_word_indices,
            "classification": self.classification,
            "confidence": self.confidence,
            "reason_codes": self.reason_codes,
        }


@dataclass(frozen=True)
class FootnoteCandidate:
    row_index: int
    text: str
    classification: str
    confidence: float
    reason_codes: list[str]
    accepted_for_exclusion: bool

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class TableBoundaryDecision:
    candidate_table_start: int
    candidate_table_end: int
    accepted_table_rows: list[int]
    excluded_lines: list[ExcludedTextBlock]
    footnote_candidates: list[FootnoteCandidate]
    row_continuation_candidates: list[dict[str, Any]]
    boundary_feature_values: list[dict[str, Any]]
    boundary_decision_scores: list[dict[str, Any]]
    warning_suppression_reasons: list[dict[str, Any]]

    def to_dict(self) -> dict[str, Any]:
        return {
            "candidate_table_start": self.candidate_table_start,
            "candidate_table_end": self.candidate_table_end,
            "accepted_table_rows": self.accepted_table_rows,
            "excluded_lines": [line.to_dict() for line in self.excluded_lines],
            "footnote_candidates": [candidate.to_dict() for candidate in self.footnote_candidates],
            "row_continuation_candidates": self.row_continuation_candidates,
            "boundary_feature_values": self.boundary_feature_values,
            "boundary_decision_scores": self.boundary_decision_scores,
            "warning_suppression_reasons": self.warning_suppression_reasons,
        }


@dataclass(frozen=True)
class TableRegion:
    table_rows: list[int]
    footer_rows: list[int]
    excluded_blocks: list[ExcludedTextBlock]
    decision: TableBoundaryDecision

    def to_dict(self) -> dict[str, Any]:
        return {
            "table_rows": self.table_rows,
            "footer_rows": self.footer_rows,
            "excluded_blocks": [block.to_dict() for block in self.excluded_blocks],
            "decision": self.decision.to_dict(),
        }


@dataclass(frozen=True)
class VisualLine:
    line_index: int
    row_index: int
    text: str
    bounding_box: BoundingBox | None
    source_word_indices: list[int]
    populated_columns: list[int]

    def to_dict(self) -> dict[str, Any]:
        return {
            "line_index": self.line_index,
            "row_index": self.row_index,
            "text": self.text,
            "bounding_box": self.bounding_box.to_dict() if self.bounding_box else None,
            "source_word_indices": self.source_word_indices,
            "populated_columns": self.populated_columns,
        }


@dataclass(frozen=True)
class LineFeatureVector:
    line_index: int
    feature_values: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {"line_index": self.line_index, "feature_values": self.feature_values}


@dataclass(frozen=True)
class DecisionAlternative:
    candidate_id: str
    candidate_type: str
    score: float
    confidence: float
    rejected_reason: str
    decision_rank: int

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class ContinuationCandidate:
    candidate_id: str
    source_line_indices: list[int]
    target_row_index: int | None
    target_column_index: int | None
    candidate_type: str
    proposed_text: str
    proposed_row_span: int
    proposed_column_span: int
    feature_values: dict[str, Any]
    positive_evidence: list[str]
    negative_evidence: list[str]
    score: float
    confidence: float
    rejected_reason: str
    decision_rank: int
    joining_mode: str = "SPACE_JOIN"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class ContinuationDecision:
    selected_candidate: ContinuationCandidate
    all_considered_candidates: list[ContinuationCandidate]
    score_margin: float
    confidence: float
    reason_codes: list[str]
    source_word_indices: list[int]
    original_text: str
    final_text: str
    original_bounding_boxes: list[dict[str, float]]
    final_bounding_box: BoundingBox | None
    reversible_mapping: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {
            "selected_candidate": self.selected_candidate.to_dict(),
            "all_considered_candidates": [candidate.to_dict() for candidate in self.all_considered_candidates],
            "score_margin": self.score_margin,
            "confidence": self.confidence,
            "reason_codes": self.reason_codes,
            "source_word_indices": self.source_word_indices,
            "original_text": self.original_text,
            "final_text": self.final_text,
            "original_bounding_boxes": self.original_bounding_boxes,
            "final_bounding_box": self.final_bounding_box.to_dict() if self.final_bounding_box else None,
            "reversible_mapping": self.reversible_mapping,
        }


@dataclass(frozen=True)
class CellContinuation:
    decision: ContinuationDecision

    def to_dict(self) -> dict[str, Any]:
        return self.decision.to_dict()


@dataclass(frozen=True)
class HeaderContinuation:
    decision: ContinuationDecision

    def to_dict(self) -> dict[str, Any]:
        return self.decision.to_dict()


@dataclass(frozen=True)
class RowGroupCandidate:
    line_index: int
    text: str
    confidence: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class SectionHeaderCandidate:
    line_index: int
    text: str
    confidence: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class UnresolvedLine:
    line_index: int
    text: str
    competing_interpretations: list[str]
    reason_codes: list[str]
    confidence: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class CoveredCoordinate:
    row_index: int
    column_index: int
    anchor_row: int
    anchor_column: int
    coordinate_type: str = "COVERED"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class SpanScoreComponents:
    geometry: float
    structure: float
    hierarchy: float
    text_fit: float
    empty_coordinate_support: float
    contradiction_penalty: float
    complexity_penalty: float

    def total(self) -> float:
        return round(
            self.geometry
            + self.structure
            + self.hierarchy
            + self.text_fit
            + self.empty_coordinate_support
            - self.contradiction_penalty
            - self.complexity_penalty,
            6,
        )

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["total"] = self.total()
        return payload


@dataclass(frozen=True)
class SpanCandidate:
    candidate_id: str
    anchor_row: int
    anchor_column: int
    row_span: int
    column_span: int
    covered_coordinates: list[list[int]]
    candidate_type: str
    source_word_indices: list[int]
    text: str
    bounding_box: BoundingBox | None
    feature_values: dict[str, Any]
    positive_evidence: list[str]
    negative_evidence: list[str]
    score_components: SpanScoreComponents
    total_score: float
    confidence: float
    conflicting_candidate_ids: list[str]
    parent_candidate_ids: list[str]
    child_candidate_ids: list[str]
    rejected_reason: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "candidate_id": self.candidate_id,
            "anchor_row": self.anchor_row,
            "anchor_column": self.anchor_column,
            "row_span": self.row_span,
            "column_span": self.column_span,
            "covered_coordinates": self.covered_coordinates,
            "candidate_type": self.candidate_type,
            "source_word_indices": self.source_word_indices,
            "text": self.text,
            "bounding_box": self.bounding_box.to_dict() if self.bounding_box else None,
            "feature_values": self.feature_values,
            "positive_evidence": self.positive_evidence,
            "negative_evidence": self.negative_evidence,
            "score_components": self.score_components.to_dict(),
            "total_score": self.total_score,
            "confidence": self.confidence,
            "conflicting_candidate_ids": self.conflicting_candidate_ids,
            "parent_candidate_ids": self.parent_candidate_ids,
            "child_candidate_ids": self.child_candidate_ids,
            "rejected_reason": self.rejected_reason,
        }


@dataclass(frozen=True)
class SpanConflict:
    left_candidate_id: str
    right_candidate_id: str
    conflict_type: str
    reason: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class SpanHierarchyNode:
    candidate_id: str
    parent_candidate_id: str | None
    child_candidate_ids: list[str]
    hierarchy_level: int
    covered_rows: list[int]
    covered_columns: list[int]
    text: str
    confidence: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class HeaderHierarchy:
    nodes: list[SpanHierarchyNode]
    exact_partition: bool
    warnings: list[str]

    def to_dict(self) -> dict[str, Any]:
        return {
            "nodes": [node.to_dict() for node in self.nodes],
            "exact_partition": self.exact_partition,
            "warnings": self.warnings,
        }


@dataclass(frozen=True)
class StubHierarchy:
    nodes: list[SpanHierarchyNode]
    exact_partition: bool
    warnings: list[str]

    def to_dict(self) -> dict[str, Any]:
        return {
            "nodes": [node.to_dict() for node in self.nodes],
            "exact_partition": self.exact_partition,
            "warnings": self.warnings,
        }


@dataclass(frozen=True)
class SpanDecision:
    selected_candidate: SpanCandidate
    all_alternatives: list[SpanCandidate]
    global_solution_score: float
    score_margin: float
    reason_codes: list[str]
    parent_span: str | None
    child_spans: list[str]
    covered_coordinates: list[list[int]]
    original_cells: list[dict[str, Any]]
    resolved_cell: dict[str, Any]
    reversible_mapping: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {
            "selected_candidate": self.selected_candidate.to_dict(),
            "all_alternatives": [candidate.to_dict() for candidate in self.all_alternatives],
            "global_solution_score": self.global_solution_score,
            "score_margin": self.score_margin,
            "reason_codes": self.reason_codes,
            "parent_span": self.parent_span,
            "child_spans": self.child_spans,
            "covered_coordinates": self.covered_coordinates,
            "original_cells": self.original_cells,
            "resolved_cell": self.resolved_cell,
            "reversible_mapping": self.reversible_mapping,
        }


@dataclass(frozen=True)
class UnresolvedSpan:
    candidate_ids: list[str]
    anchor_row: int
    anchor_column: int
    competing_interpretations: list[str]
    reason_codes: list[str]
    confidence: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class GlobalSpanSolution:
    selected_candidate_ids: list[str]
    rejected_candidate_ids: list[str]
    total_score: float
    second_best_score: float
    score_margin: float
    combinations_evaluated: int
    pruned_branches: int
    solver_runtime_seconds: float
    unresolved_conflicts: list[str]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class SpanResolution:
    rows: list["TableRow"]
    candidates: list[SpanCandidate]
    selected_decisions: list[SpanDecision]
    rejected_spans: list[SpanCandidate]
    unresolved_spans: list[UnresolvedSpan]
    conflicts: list[SpanConflict]
    header_hierarchy: HeaderHierarchy
    stub_hierarchy: StubHierarchy
    global_solution: GlobalSpanSolution
    covered_coordinate_map: list[CoveredCoordinate]
    genuine_empty_coordinate_map: list[list[int]]
    warnings: list[ReconstructionWarning]

    def to_metrics(self) -> dict[str, Any]:
        mixed_candidates = [
            candidate for candidate in self.candidates if "MIXED" in candidate.candidate_type
        ]
        selected_mixed = [
            decision for decision in self.selected_decisions if "MIXED" in decision.selected_candidate.candidate_type
        ]
        rejected_mixed = [
            candidate for candidate in self.rejected_spans if "MIXED" in candidate.candidate_type
        ]
        mixed_nodes = [
            {
                "candidate_id": candidate.candidate_id,
                "text": candidate.text,
                "row_interval": [candidate.anchor_row, candidate.anchor_row + candidate.row_span - 1],
                "column_interval": [candidate.anchor_column, candidate.anchor_column + candidate.column_span - 1],
                "direct_parent": "",
                "direct_children": candidate.child_candidate_ids,
                "ancestors": candidate.parent_candidate_ids,
                "descendants": candidate.child_candidate_ids,
                "candidate_type": candidate.candidate_type,
            }
            for candidate in mixed_candidates
        ]
        return {
            "pre_span_grid": [[cell.to_dict() for cell in row.cells] for row in self.rows],
            "span_candidates": [candidate.to_dict() for candidate in self.candidates],
            "mixed_span_candidates": [candidate.to_dict() for candidate in mixed_candidates],
            "mixed_span_score_components": [
                {"candidate_id": candidate.candidate_id, **candidate.score_components.to_dict()}
                for candidate in mixed_candidates
            ],
            "span_score_components": [
                {"candidate_id": candidate.candidate_id, **candidate.score_components.to_dict()}
                for candidate in self.candidates
            ],
            "candidate_conflicts": [conflict.to_dict() for conflict in self.conflicts],
            "row_column_span_conflicts": [
                conflict.to_dict()
                for conflict in self.conflicts
                if any(candidate_id in {candidate.candidate_id for candidate in mixed_candidates} for candidate_id in [conflict.left_candidate_id, conflict.right_candidate_id])
            ],
            "candidate_parent_child_links": [
                {
                    "candidate_id": candidate.candidate_id,
                    "parents": candidate.parent_candidate_ids,
                    "children": candidate.child_candidate_ids,
                }
                for candidate in self.candidates
            ],
            "selected_global_solution": self.global_solution.to_dict(),
            "second_best_solution": {"score": self.global_solution.second_best_score},
            "best_global_solution": self.global_solution.to_dict(),
            "second_best_global_solution": {"score": self.global_solution.second_best_score},
            "global_solution_margin": self.global_solution.score_margin,
            "global_score_margin": self.global_solution.score_margin,
            "selected_spans": [decision.to_dict() for decision in self.selected_decisions],
            "selected_mixed_spans": [decision.to_dict() for decision in selected_mixed],
            "rejected_spans": [candidate.to_dict() for candidate in self.rejected_spans],
            "rejected_mixed_spans": [candidate.to_dict() for candidate in rejected_mixed],
            "unresolved_spans": [span.to_dict() for span in self.unresolved_spans],
            "header_hierarchy": self.header_hierarchy.to_dict(),
            "stub_hierarchy": self.stub_hierarchy.to_dict(),
            "mixed_span_hierarchy": {"nodes": mixed_nodes},
            "covered_coordinate_map": [coord.to_dict() for coord in self.covered_coordinate_map],
            "genuine_empty_coordinate_map": self.genuine_empty_coordinate_map,
            "anchor_text_assignments": [
                {
                    "candidate_id": decision.selected_candidate.candidate_id,
                    "anchor": [decision.selected_candidate.anchor_row, decision.selected_candidate.anchor_column],
                    "text": decision.selected_candidate.text,
                    "source_word_indices": decision.selected_candidate.source_word_indices,
                }
                for decision in self.selected_decisions
            ],
            "span_word_mappings": [decision.reversible_mapping for decision in self.selected_decisions],
            "solver_statistics": self.global_solution.to_dict(),
            "mixed_span_stability_results": [],
            "stability_results": {
                "deterministic": True,
                "small_perturbation_sensitivity": "not_recomputed",
            },
        }


@dataclass(frozen=True)
class Cell:
    row_index: int
    column_index: int
    text: str
    bounding_box: BoundingBox | None
    source_word_indices: list[int]
    row_span: int = 1
    column_span: int = 1
    covered_by: tuple[int, int] | None = None
    is_header: bool = False
    is_stub: bool = False
    confidence: float = 1.0
    warnings: list[ReconstructionWarning] | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "row_index": self.row_index,
            "column_index": self.column_index,
            "text": self.text,
            "bounding_box": self.bounding_box.to_dict() if self.bounding_box else None,
            "source_word_indices": self.source_word_indices,
            "row_span": self.row_span,
            "column_span": self.column_span,
            "covered_coordinates": self.covered_coordinates(),
            "covered_by": list(self.covered_by) if self.covered_by else None,
            "is_covered": self.covered_by is not None,
            "is_header": self.is_header,
            "is_stub": self.is_stub,
            "confidence": self.confidence,
            "warnings": [warning.to_dict() for warning in self.warnings or []],
        }

    def covered_coordinates(self) -> list[list[int]]:
        return [
            [row_index, column_index]
            for row_index in range(self.row_index, self.row_index + self.row_span)
            for column_index in range(self.column_index, self.column_index + self.column_span)
        ]


@dataclass(frozen=True)
class TableRow:
    row_index: int
    cells: list[Cell]
    bounding_box: BoundingBox | None
    confidence: float
    warnings: list[ReconstructionWarning] | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "row_index": self.row_index,
            "cells": [cell.to_dict() for cell in self.cells],
            "bounding_box": self.bounding_box.to_dict() if self.bounding_box else None,
            "confidence": self.confidence,
            "warnings": [warning.to_dict() for warning in self.warnings or []],
        }


@dataclass(frozen=True)
class TableGrid:
    rows: list[TableRow]
    column_boundaries: list[float]
    column_centers: list[float]
    bounding_box: BoundingBox | None
    confidence: float
    warnings: list[ReconstructionWarning]
    metrics: dict[str, Any]

    @property
    def row_count(self) -> int:
        return len(self.rows)

    @property
    def column_count(self) -> int:
        return len(self.column_centers)

    def matrix(self) -> list[list[str]]:
        return [[cell.text for cell in row.cells] for row in self.rows]

    def merged_ranges(self) -> list[dict[str, Any]]:
        ranges = []
        for row in self.rows:
            for cell in row.cells:
                if cell.covered_by is not None:
                    continue
                if cell.row_span > 1 or cell.column_span > 1:
                    ranges.append(
                        {
                            "row_index": cell.row_index,
                            "column_index": cell.column_index,
                            "text": cell.text,
                            "row_span": cell.row_span,
                            "column_span": cell.column_span,
                            "covered_coordinates": cell.covered_coordinates(),
                            "is_header": cell.is_header,
                            "is_stub": cell.is_stub,
                            "confidence": cell.confidence,
                        }
                    )
        return ranges

    def logical_cell_count(self) -> int:
        return sum(1 for row in self.rows for cell in row.cells if cell.covered_by is None)

    def to_dict(self) -> dict[str, Any]:
        return {
            "row_count": self.row_count,
            "column_count": self.column_count,
            "physical_rows": self.row_count,
            "physical_columns": self.column_count,
            "logical_cells": self.logical_cell_count(),
            "merged_ranges": self.merged_ranges(),
            "column_boundaries": self.column_boundaries,
            "column_centers": self.column_centers,
            "bounding_box": self.bounding_box.to_dict() if self.bounding_box else None,
            "confidence": self.confidence,
            "warnings": [warning.to_dict() for warning in self.warnings],
            "metrics": self.metrics,
            "table_region": self.metrics.get("table_region", {}),
            "excluded_text_blocks": self.metrics.get("excluded_text_blocks", []),
            "rows": [row.to_dict() for row in self.rows],
        }


@dataclass(frozen=True)
class PageMetadata:
    pdf_path: str
    page_number: int
    zero_based_page_index: int
    page_width: float
    page_height: float
    crop_bbox: BoundingBox
    render_dpi: int
    rendered_png_path: str
    native_words_json_path: str
    metadata_json_path: str
    word_count: int

    @classmethod
    def build(
        cls,
        pdf_path: Path,
        page_number: int,
        page_width: float,
        page_height: float,
        crop_bbox: BoundingBox,
        render_dpi: int,
        rendered_png_path: Path,
        native_words_json_path: Path,
        metadata_json_path: Path,
        word_count: int,
    ) -> "PageMetadata":
        return cls(
            pdf_path=str(pdf_path),
            page_number=page_number,
            zero_based_page_index=page_number - 1,
            page_width=page_width,
            page_height=page_height,
            crop_bbox=crop_bbox,
            render_dpi=render_dpi,
            rendered_png_path=str(rendered_png_path),
            native_words_json_path=str(native_words_json_path),
            metadata_json_path=str(metadata_json_path),
            word_count=word_count,
        )

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["crop_bbox"] = self.crop_bbox.to_dict()
        return data
