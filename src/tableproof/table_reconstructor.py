from __future__ import annotations

import csv
import json
import re
import statistics
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from tableproof.continuation_resolver import resolve_continuations
from tableproof.diagnostic_renderer import render_diagnostic_png
from tableproof.models import (
    ReconstructionOptions,
    BoundingBox,
    Cell,
    ExcludedTextBlock,
    FootnoteCandidate,
    ReconstructionWarning,
    TableBoundaryDecision,
    TableGrid,
    TableRegion,
    TableRow,
    WordBox,
)
from tableproof.span_resolver import resolve_spans
from tableproof.xlsx_exporter import export_table_to_xlsx


WARNING_AMBIGUOUS_ROW = "AMBIGUOUS_ROW"
WARNING_AMBIGUOUS_COLUMN = "AMBIGUOUS_COLUMN"
WARNING_POSSIBLE_MERGED_CELL = "POSSIBLE_MERGED_CELL"
WARNING_IRREGULAR_COLUMN_COUNT = "IRREGULAR_COLUMN_COUNT"
WARNING_LARGE_HORIZONTAL_GAP = "LARGE_HORIZONTAL_GAP"
WARNING_OVERLAPPING_CELLS = "OVERLAPPING_CELLS"
WARNING_UNASSIGNED_WORD = "UNASSIGNED_WORD"
WARNING_LOW_STRUCTURE_CONFIDENCE = "LOW_STRUCTURE_CONFIDENCE"
WARNING_MISSED_COLUMN_SPAN = "MISSED_COLUMN_SPAN"
WARNING_FALSE_COLUMN_SPAN = "FALSE_COLUMN_SPAN"
WARNING_WRONG_COLUMN_SPAN = "WRONG_COLUMN_SPAN"
WARNING_MISSED_ROW_SPAN = "MISSED_ROW_SPAN"
WARNING_FALSE_ROW_SPAN = "FALSE_ROW_SPAN"
WARNING_WRONG_ROW_SPAN = "WRONG_ROW_SPAN"
WARNING_OVERLAPPING_SPANS = "OVERLAPPING_SPANS"
WARNING_SPAN_BOUNDARY_ERROR = "SPAN_BOUNDARY_ERROR"
WARNING_TEXT_DUPLICATED_ACROSS_SPAN = "TEXT_DUPLICATED_ACROSS_SPAN"
WARNING_COVERED_CELL_TREATED_AS_EMPTY = "COVERED_CELL_TREATED_AS_EMPTY"
WARNING_AMBIGUOUS_SPAN = "AMBIGUOUS_SPAN"
WARNING_EXPECTED_EMPTY_CELL_PATTERN = "EXPECTED_EMPTY_CELL_PATTERN"
WARNING_TRIANGULAR_MATRIX_PATTERN = "TRIANGULAR_MATRIX_PATTERN"
WARNING_HEADER_SPAN_EXPLAINS_ROW_WIDTH = "HEADER_SPAN_EXPLAINS_ROW_WIDTH"
WARNING_POSSIBLE_FOOTNOTE = "POSSIBLE_FOOTNOTE"
WARNING_FOOTNOTE_EXCLUDED = "FOOTNOTE_EXCLUDED"
WARNING_UNCERTAIN_TABLE_BOUNDARY = "UNCERTAIN_TABLE_BOUNDARY"
WARNING_POSSIBLE_WRAPPED_CELL = "POSSIBLE_WRAPPED_CELL"
WARNING_ROW_CONTINUATION_MERGED = "ROW_CONTINUATION_MERGED"
WARNING_TRAILING_TEXT_EXCLUDED = "TRAILING_TEXT_EXCLUDED"
_UNSET = object()


class ReconstructionError(RuntimeError):
    """Raised when native words cannot be reconstructed into a table baseline."""


@dataclass(frozen=True)
class ReconstructionOutputs:
    table_grid: TableGrid
    xlsx_path: Path
    csv_path: Path
    structured_json_path: Path
    diagnostic_png_path: Path
    diagnostics_json_path: Path


@dataclass
class _CandidateCell:
    text: str
    bbox: BoundingBox
    word_indices: list[int]
    grouping_confidence: float
    warnings: list[ReconstructionWarning]


@dataclass(frozen=True)
class _WordFeature:
    word: WordBox
    row_index: int
    x0: float
    x1: float
    center_x: float
    gap_to_next: float | None
    normalized_gap_to_next: float | None
    token_type: str


@dataclass(frozen=True)
class _ColumnModel:
    centers: list[float]
    boundaries: list[float]
    anchors: list[dict[str, Any]]
    rejected: list[dict[str, Any]]
    method: str
    confidence: float
    warnings: list[ReconstructionWarning]


@dataclass
class _RawRow:
    words: list[WordBox]

    @property
    def bbox(self) -> BoundingBox:
        return _union_boxes([word.bounding_box for word in self.words])

    @property
    def y_mid(self) -> float:
        return statistics.median(word.bounding_box.y_mid for word in self.words)


def reconstruct_from_words_file(
    words_json_path: Path | str,
    output_dir: Path | str = Path("outputs/evaluation"),
    debug: bool = False,
    row_tolerance: float | None = None,
    column_tolerance: float | None = None,
) -> ReconstructionOutputs:
    words_json_path = Path(words_json_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    words = load_words_json(words_json_path)
    table_grid = reconstruct_table_grid(
        words,
        row_tolerance=row_tolerance,
        column_tolerance=column_tolerance,
        debug=debug,
    )

    stem = words_json_path.stem.replace("_words", "")
    xlsx_path = output_dir / f"{stem}_reconstructed.xlsx"
    csv_path = output_dir / f"{stem}_reconstructed.csv"
    structured_json_path = output_dir / f"{stem}_reconstructed.json"
    diagnostic_png_path = output_dir / f"{stem}_diagnostic.png"
    diagnostics_json_path = output_dir / f"{stem}_diagnostics.json"

    export_table_to_xlsx(table_grid, xlsx_path)
    _write_csv(table_grid, csv_path)
    _write_json(structured_json_path, table_grid.to_dict())
    render_diagnostic_png(table_grid, diagnostic_png_path)
    _write_json(diagnostics_json_path, _diagnostics_payload(table_grid))

    return ReconstructionOutputs(
        table_grid=table_grid,
        xlsx_path=xlsx_path,
        csv_path=csv_path,
        structured_json_path=structured_json_path,
        diagnostic_png_path=diagnostic_png_path,
        diagnostics_json_path=diagnostics_json_path,
    )


def load_words_json(words_json_path: Path | str) -> list[WordBox]:
    path = Path(words_json_path)
    if not path.exists():
        raise ReconstructionError(f"Words JSON does not exist: {path}")

    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ReconstructionError(f"Words JSON is invalid: {path}") from exc

    if not isinstance(payload, list):
        raise ReconstructionError("Words JSON must contain a list of word objects")

    words: list[WordBox] = []
    for index, item in enumerate(payload):
        if not isinstance(item, dict):
            raise ReconstructionError(f"Word {index} must be an object")
        word = WordBox.from_dict(index, item)
        if word.text:
            words.append(word)

    if not words:
        raise ReconstructionError("Words JSON contains no non-blank words")
    return words


def reconstruct_table_grid(
    words: list[WordBox],
    row_tolerance: float | None = None,
    column_tolerance: float | None = None,
    debug: bool = False,
    options: ReconstructionOptions | None = None,
) -> TableGrid:
    options = options or ReconstructionOptions()
    metrics = _adaptive_metrics(words, row_tolerance, column_tolerance)
    metrics["profile"] = options.profile
    metrics["keep_unresolved_lines"] = options.keeps_unresolved_lines
    raw_rows, row_warnings = _cluster_rows(words, metrics["row_tolerance"])
    raw_rows.sort(key=lambda row: (row.bbox.y0, row.bbox.x0))

    row_features = _word_features_for_rows(raw_rows, metrics)
    column_model_features = _row_features_without_trailing_notes(row_features)
    column_model = _infer_columns_from_atomic_words(column_model_features, metrics)
    rows, cell_warnings = _assign_and_group_words(
        raw_rows,
        row_features,
        column_model.centers,
        column_model.boundaries,
        metrics,
    )
    region = _infer_table_region(rows, metrics)
    rows = _rows_from_region(rows, region)
    excluded_word_indices = {
        word_index
        for block in region.excluded_blocks
        for word_index in block.source_word_indices
    }
    rows, continuation_warnings, continuation_metrics = resolve_continuations(
        rows,
        metrics,
        excluded_word_indices=excluded_word_indices,
    )
    metrics["estimated_header_rows"] = _estimate_header_rows(rows)
    pre_span_rows = rows
    span_resolution = resolve_spans(rows, metrics)
    rows = span_resolution.rows
    span_warnings = span_resolution.warnings

    excluded_row_indices = {
        score["row_index"]
        for score in region.decision.boundary_decision_scores
        if score.get("score", 0) >= 3.0
    }
    warnings = [
        warning
        for warning in row_warnings + column_model.warnings + cell_warnings + continuation_warnings + span_warnings
        if warning.row_index not in excluded_row_indices
    ]
    expected_columns = len(column_model.centers)
    structure_warnings, suppression_reasons = _calibrated_structure_warnings(rows, expected_columns)
    warnings.extend(structure_warnings)
    warnings.extend(
        ReconstructionWarning(
            WARNING_FOOTNOTE_EXCLUDED,
            f"Excluded {len(region.excluded_blocks)} trailing note/source/footer line(s)",
        )
        for _ in region.excluded_blocks[:1]
    )
    recurring_groups = metrics.get("recurring_x_position_groups", 0)
    if expected_columns == 1 and recurring_groups >= 3:
        warnings.append(
            ReconstructionWarning(
                WARNING_AMBIGUOUS_COLUMN,
                "Only one column was inferred despite multiple recurring x-position groups",
            )
        )

    bounding_box = _union_boxes([word.bounding_box for word in words])
    confidence = _overall_confidence(rows, warnings, expected_columns, column_model.confidence)
    if confidence < 0.65:
        warnings.append(
            ReconstructionWarning(
                WARNING_LOW_STRUCTURE_CONFIDENCE,
                f"Overall reconstruction confidence is low: {confidence:.2f}",
            )
        )

    # Account for words on the FINAL grid: a word placed before span resolution must still be
    # in some cell after it. (Counting earlier let hidden labels pass as "no word loss".)
    placed_before = {w for row in pre_span_rows for cell in row.cells for w in cell.source_word_indices}
    placed_after = {w for row in rows for cell in row.cells for w in cell.source_word_indices}
    hidden = sorted(placed_before - placed_after)
    no_word_loss = dict(continuation_metrics.get("no_word_loss", {}))
    no_word_loss["hidden_by_spans_word_indices"] = hidden
    no_word_loss["assigned_word_count"] = len(placed_after)
    no_word_loss["passed"] = bool(no_word_loss.get("passed", True)) and not hidden
    continuation_metrics["no_word_loss"] = no_word_loss
    if hidden:
        warnings.append(
            ReconstructionWarning(
                "WORDS_HIDDEN_BY_SPAN",
                f"{len(hidden)} word(s) are not in any output cell after span resolution",
                word_indices=hidden,
            )
        )

    metrics["debug_enabled"] = 1.0 if debug else 0.0
    metrics["table_region"] = region.to_dict()
    metrics["continuation_resolution"] = continuation_metrics
    metrics.update(continuation_metrics)
    span_metrics = span_resolution.to_metrics()
    span_metrics["pre_span_grid"] = [[cell.to_dict() for cell in row.cells] for row in pre_span_rows]
    metrics.update(span_metrics)
    metrics["excluded_text_blocks"] = [block.to_dict() for block in region.excluded_blocks]
    metrics["candidate_table_start"] = region.decision.candidate_table_start
    metrics["candidate_table_end"] = region.decision.candidate_table_end
    metrics["accepted_table_rows"] = region.decision.accepted_table_rows
    metrics["excluded_lines"] = [line.to_dict() for line in region.excluded_blocks]
    metrics["boundary_feature_values"] = region.decision.boundary_feature_values
    metrics["boundary_decision_scores"] = region.decision.boundary_decision_scores
    metrics["footnote_candidates"] = [candidate.to_dict() for candidate in region.decision.footnote_candidates]
    metrics["row_continuation_candidates"] = region.decision.row_continuation_candidates
    metrics["classification_reason_codes"] = sorted(
        {code for block in region.excluded_blocks for code in block.reason_codes}
    )
    metrics["warning_suppression_reasons"] = suppression_reasons + region.decision.warning_suppression_reasons
    metrics["candidate_column_anchors"] = column_model.anchors
    metrics["anchor_support_counts"] = [anchor["support_count"] for anchor in column_model.anchors]
    metrics["anchor_alignment_types"] = [anchor["alignment_type"] for anchor in column_model.anchors]
    metrics["candidate_boundaries"] = column_model.boundaries
    metrics["column_inference_method"] = column_model.method
    metrics["rejected_column_candidates"] = column_model.rejected
    metrics["per_row_word_gaps"] = _serialize_row_gaps(row_features)
    metrics["per_word_column_assignment"] = _serialize_word_assignments(rows)
    metrics["confidence_components"] = {
        "column_inference": column_model.confidence,
        "cell_mean": _mean([cell.confidence for row in rows for cell in row.cells if cell.text]),
        "warning_penalty": min(0.5, len(warnings) * 0.04),
        "overall": confidence,
    }
    metrics["merged_ranges"] = [
        {
            "row_index": cell.row_index,
            "column_index": cell.column_index,
            "text": cell.text,
            "row_span": cell.row_span,
            "column_span": cell.column_span,
            "covered_coordinates": cell.covered_coordinates(),
            "confidence": cell.confidence,
        }
        for row in rows
        for cell in row.cells
        if cell.covered_by is None and (cell.row_span > 1 or cell.column_span > 1)
    ]
    return TableGrid(
        rows=rows,
        column_boundaries=column_model.boundaries,
        column_centers=column_model.centers,
        bounding_box=bounding_box,
        confidence=confidence,
        warnings=warnings,
        metrics=metrics,
    )


def _infer_spans(rows: list[TableRow]) -> tuple[list[TableRow], list[ReconstructionWarning]]:
    warnings: list[ReconstructionWarning] = []
    if not rows:
        return rows, warnings
    row_count = len(rows)
    column_count = len(rows[0].cells)
    header_rows = _estimate_header_rows(rows)
    replacements: dict[tuple[int, int], Cell] = {}
    covered: dict[tuple[int, int], tuple[int, int]] = {}

    for row in rows:
        if not _looks_like_section_header_row(row):
            continue
        populated = [cell for cell in row.cells if cell.text]
        if not populated:
            continue
        anchor_cell = populated[0]
        merged_text = " ".join(cell.text for cell in populated)
        merged_indices = [word for cell in populated for word in cell.source_word_indices]
        merged_box = _union_boxes([cell.bounding_box for cell in populated if cell.bounding_box])
        anchor = _replace_cell(
            anchor_cell,
            text=merged_text,
            bounding_box=merged_box,
            source_word_indices=merged_indices,
            column_span=column_count,
            is_header=True,
            is_stub=False,
            confidence=min(anchor_cell.confidence, 0.86),
        )
        replacements[(anchor_cell.row_index, anchor_cell.column_index)] = anchor
        for covered_col in range(column_count):
            if covered_col != anchor_cell.column_index:
                covered[(anchor_cell.row_index, covered_col)] = (anchor_cell.row_index, anchor_cell.column_index)

    for row in rows[:header_rows]:
        col = 0
        while col < column_count:
            cell = row.cells[col]
            if not cell.text:
                col += 1
                continue
            span = 1
            probe = col + 1
            while probe < column_count and not row.cells[probe].text:
                span += 1
                probe += 1
            if span > 1:
                anchor = _replace_cell(
                    cell,
                    column_span=span,
                    is_header=True,
                    confidence=min(cell.confidence, 0.86),
                )
                replacements[(cell.row_index, cell.column_index)] = anchor
                for covered_col in range(col + 1, col + span):
                    covered[(cell.row_index, covered_col)] = (cell.row_index, cell.column_index)
            col += span

    for row in rows[:header_rows]:
        if not row.cells or not row.cells[0].text:
            continue
        has_wrapped_group_header = any(
            replacement.row_index == row.row_index
            and replacement.column_span > 1
            and len(replacement.text.split()) >= 3
            for replacement in replacements.values()
        )
        if not has_wrapped_group_header:
            continue
        span = 1
        probe = row.row_index + 1
        while (
            probe < header_rows
            and not rows[probe].cells[0].text
            and any(cell.text for cell in rows[probe].cells[1:])
        ):
            span += 1
            probe += 1
        if span > 1:
            cell = row.cells[0]
            replacements[(cell.row_index, cell.column_index)] = _replace_cell(
                replacements.get((cell.row_index, cell.column_index), cell),
                row_span=span,
                is_header=True,
                is_stub=True,
                confidence=min(cell.confidence, 0.86),
            )
            for covered_row in range(row.row_index + 1, row.row_index + span):
                covered[(covered_row, 0)] = (row.row_index, 0)

    for row_index in range(header_rows, row_count):
        cell = rows[row_index].cells[0]
        if not cell.text:
            continue
        span = 1
        probe = row_index + 1
        while (
            probe < row_count
            and not rows[probe].cells[0].text
            and any(body_cell.text for body_cell in rows[probe].cells[1:])
        ):
            span += 1
            probe += 1
        if span > 2:
            anchor = _replace_cell(
                cell,
                row_span=span,
                is_stub=True,
                confidence=min(cell.confidence, 0.82),
            )
            replacements[(cell.row_index, cell.column_index)] = anchor
            for covered_row in range(row_index + 1, row_index + span):
                covered[(covered_row, cell.column_index)] = (cell.row_index, cell.column_index)

    new_rows: list[TableRow] = []
    for row in rows:
        new_cells: list[Cell] = []
        for cell in row.cells:
            key = (cell.row_index, cell.column_index)
            if key in replacements:
                new_cells.append(replacements[key])
            elif key in covered:
                new_cells.append(
                    _replace_cell(
                        cell,
                        text="",
                        source_word_indices=[],
                        bounding_box=None,
                        covered_by=covered[key],
                        confidence=0.9,
                    )
                )
            else:
                adjusted_column_span = cell.column_span
                populated_in_row = sum(1 for row_cell in row.cells if row_cell.text)
                if (
                    cell.column_span > 1
                    and not _looks_like_section_header_row(row)
                    and (cell.row_index >= header_rows or populated_in_row > 1)
                ):
                    adjusted_column_span = 1
                new_cells.append(
                    _replace_cell(
                        cell,
                        column_span=adjusted_column_span,
                        is_header=cell.row_index < header_rows,
                        is_stub=cell.column_index == 0 and cell.row_index >= header_rows and bool(cell.text),
                    )
                )
        new_rows.append(
            TableRow(
                row_index=row.row_index,
                cells=new_cells,
                bounding_box=row.bounding_box,
                confidence=row.confidence,
                warnings=row.warnings,
            )
        )
    return new_rows, warnings


def _looks_like_section_header_row(row: TableRow) -> bool:
    if not row.cells or row.row_index == 0:
        return False
    populated = [cell for cell in row.cells if cell.text]
    if not populated:
        return False
    text = " ".join(cell.text for cell in populated).strip()
    if not text or any(is_numeric_text(cell.text) for cell in populated):
        return False
    if len(text.split()) < 2:
        return False
    if text.endswith("."):
        return False
    alpha_ratio = sum(1 for char in text if char.isalpha()) / max(len(text), 1)
    if ":" in text and len(populated) <= max(4, len(row.cells)):
        return True
    return len(populated) <= 2 and alpha_ratio >= 0.55 and text.isupper() and len(text.split()) <= 4


def _estimate_header_rows(rows: list[TableRow]) -> int:
    header_rows = 0
    for index, row in enumerate(rows):
        numeric_count = sum(1 for cell in row.cells if is_numeric_text(cell.text))
        if numeric_count == 0 and header_rows < 4:
            if (
                header_rows >= 2
                and row.cells
                and row.cells[0].text
                and sum(1 for cell in row.cells if cell.text) >= 2
            ):
                break
            header_rows += 1
            continue
        if (
            header_rows > 0
            and header_rows < 4
            and numeric_count > 0
            and row.cells
            and not row.cells[0].text
            and index + 1 < len(rows)
            and rows[index + 1].cells
            and (
                bool(rows[index + 1].cells[0].text)
                or not any(is_numeric_text(cell.text) for cell in rows[index + 1].cells)
            )
        ):
            header_rows += 1
            continue
        break
    return max(1, header_rows) if rows else 0


def is_numeric_text(text: str) -> bool:
    return bool(re.fullmatch(r"\(?[-+]?\d[\d,]*(\.\d+)?%?\)?(\*{1,3})?", text.strip()))


def _infer_table_region(rows: list[TableRow], metrics: dict[str, Any]) -> TableRegion:
    if not rows:
        decision = TableBoundaryDecision(0, -1, [], [], [], [], [], [], [])
        return TableRegion([], [], [], decision)

    features = [_boundary_features(row, rows, metrics) for row in rows]
    scores = []
    candidates: list[FootnoteCandidate] = []
    excluded_rows: set[int] = set()
    excluded_blocks: list[ExcludedTextBlock] = []
    table_end = len(rows) - 1

    for row in reversed(rows):
        feature = features[row.row_index]
        score, reason_codes, classification = _score_trailing_exclusion(feature)
        scores.append(
            {
                "row_index": row.row_index,
                "score": round(score, 3),
                "classification": classification,
                "reason_codes": reason_codes,
            }
        )
        if row.row_index < len(rows) - 2:
            break
        candidate = FootnoteCandidate(
            row_index=row.row_index,
            text=feature["text"],
            classification=classification,
            confidence=round(min(0.99, score / 5.0), 3),
            reason_codes=reason_codes,
            accepted_for_exclusion=score >= 3.0,
        )
        if score >= 2.0:
            candidates.append(candidate)
        if score >= 3.0:
            excluded_rows.add(row.row_index)
            excluded_blocks.append(
                ExcludedTextBlock(
                    text=feature["text"],
                    bounding_box=row.bounding_box,
                    source_word_indices=[
                        word_index for cell in row.cells for word_index in cell.source_word_indices
                    ],
                    classification=classification,
                    confidence=candidate.confidence,
                    reason_codes=reason_codes,
                )
            )
            table_end = row.row_index - 1
            continue
        break

    accepted = [row.row_index for row in rows if row.row_index not in excluded_rows]
    decision = TableBoundaryDecision(
        candidate_table_start=0,
        candidate_table_end=table_end,
        accepted_table_rows=accepted,
        excluded_lines=list(reversed(excluded_blocks)),
        footnote_candidates=list(reversed(candidates)),
        row_continuation_candidates=_row_continuation_candidates(rows, metrics),
        boundary_feature_values=features,
        boundary_decision_scores=list(reversed(scores)),
        warning_suppression_reasons=[],
    )
    return TableRegion(
        table_rows=accepted,
        footer_rows=[],
        excluded_blocks=list(reversed(excluded_blocks)),
        decision=decision,
    )


def _boundary_features(row: TableRow, rows: list[TableRow], metrics: dict[str, Any]) -> dict[str, Any]:
    text = " ".join(cell.text for cell in row.cells if cell.text).strip()
    row_widths = [other.bounding_box.width for other in rows if other.bounding_box is not None]
    median_width = _median_or_default(row_widths, row.bounding_box.width if row.bounding_box else 1.0)
    populated = sum(1 for cell in row.cells if cell.text)
    row_span = row.bounding_box.width if row.bounding_box else 0.0
    previous = rows[row.row_index - 1] if row.row_index > 0 else None
    vertical_gap = (
        row.bounding_box.y0 - previous.bounding_box.y1
        if row.bounding_box is not None and previous and previous.bounding_box is not None
        else 0.0
    )
    word_count = sum(len(cell.source_word_indices) for cell in row.cells)
    alpha_chars = sum(1 for char in text if char.isalpha())
    nonspace_chars = sum(1 for char in text if not char.isspace())
    alpha_ratio = alpha_chars / max(nonspace_chars, 1)
    prefix = _note_prefix(text)
    return {
        "row_index": row.row_index,
        "text": text,
        "populated_columns": populated,
        "column_count": len(row.cells),
        "row_width": row_span,
        "median_row_width": median_width,
        "width_ratio": row_span / max(median_width, 1.0),
        "vertical_gap": vertical_gap,
        "vertical_gap_ratio": vertical_gap / max(metrics.get("median_word_height", 1.0), 1.0),
        "word_count": word_count,
        "alpha_ratio": round(alpha_ratio, 3),
        "has_note_prefix": prefix is not None,
        "prefix": prefix,
        "sentence_punctuation": any(mark in text for mark in [":", ".", ";"]),
        "long_prose": word_count >= 5 and alpha_ratio >= 0.55,
        "has_numeric_cell": any(is_numeric_text(cell.text) for cell in row.cells),
    }


def _score_trailing_exclusion(feature: dict[str, Any]) -> tuple[float, list[str], str]:
    score = 0.0
    reasons = []
    if feature["has_note_prefix"]:
        score += 2.6 if feature.get("prefix") in {"*", "*, **", "***", "significance"} else 1.5
        reasons.append("NOTE_PREFIX")
    if feature["populated_columns"] <= max(1, feature["column_count"] // 2):
        score += 0.75
        reasons.append("SPARSE_COLUMNS")
    if feature["long_prose"]:
        score += 0.9
        reasons.append("LONG_PROSE")
    if feature["sentence_punctuation"]:
        score += 0.45
        reasons.append("SENTENCE_PUNCTUATION")
    if feature["alpha_ratio"] >= 0.65 and not feature["has_numeric_cell"]:
        score += 0.5
        reasons.append("ALPHA_HEAVY")
    if feature["vertical_gap_ratio"] >= 0.4:
        score += 0.4
        reasons.append("VERTICAL_GAP")
    if feature["width_ratio"] >= 1.15 and feature["populated_columns"] <= max(2, feature["column_count"] // 2):
        score += 0.4
        reasons.append("PROSE_SPANS_GRID")

    classification = "UNCERTAIN_TRAILING_TEXT"
    prefix = feature.get("prefix")
    if prefix in {"source", "sources"}:
        classification = "SOURCE_NOTE"
    elif prefix in {"standard errors", "significance", "p-values", "parentheses contain", "*", "*, **", "***"}:
        classification = "FOOTNOTE"
    elif prefix in {"note", "notes", "figures in"}:
        classification = "GENERAL_NOTE"
    elif score >= 3.0:
        classification = "CONTINUATION_TEXT"
    return score, reasons, classification


def _note_prefix(text: str) -> str | None:
    lowered = text.lower().strip()
    prefixes = [
        "note:",
        "notes:",
        "source:",
        "sources:",
        "standard errors",
        "significance",
        "p-values",
        "figures in",
        "parentheses contain",
        "*, **",
        "***",
        "*",
    ]
    for prefix in prefixes:
        if lowered.startswith(prefix):
            return prefix.rstrip(":")
    return None


def _row_continuation_candidates(rows: list[TableRow], metrics: dict[str, Any]) -> list[dict[str, Any]]:
    candidates = []
    for row in rows[1:]:
        previous = rows[row.row_index - 1]
        populated = sum(1 for cell in row.cells if cell.text)
        first_only = bool(row.cells and row.cells[0].text) and populated == 1
        numeric_previous = any(is_numeric_text(cell.text) for cell in previous.cells)
        gap = (
            row.bounding_box.y0 - previous.bounding_box.y1
            if row.bounding_box is not None and previous.bounding_box is not None
            else 0.0
        )
        score = 0.0
        reasons = []
        if first_only:
            score += 1.0
            reasons.append("FIRST_COLUMN_ONLY")
        if not numeric_previous:
            score += 0.5
            reasons.append("NO_NUMERIC_PREVIOUS_ROW")
        if gap <= metrics.get("median_word_height", 1.0) * 0.45:
            score += 0.5
            reasons.append("SMALL_VERTICAL_GAP")
        if score >= 1.0:
            candidates.append(
                {
                    "row_index": row.row_index,
                    "previous_row_index": previous.row_index,
                    "score": round(score, 3),
                    "reason_codes": reasons,
                    "merged": False,
                }
            )
    return candidates


def _rows_from_region(rows: list[TableRow], region: TableRegion) -> list[TableRow]:
    accepted = set(region.table_rows)
    if len(accepted) == len(rows):
        return rows
    kept = [row for row in rows if row.row_index in accepted]
    old_to_new = {row.row_index: index for index, row in enumerate(kept)}
    reindexed = []
    for new_index, row in enumerate(kept):
        new_cells = []
        for cell in row.cells:
            covered_by = (
                (old_to_new[cell.covered_by[0]], cell.covered_by[1])
                if cell.covered_by and cell.covered_by[0] in old_to_new
                else cell.covered_by
            )
            new_cells.append(
                Cell(
                    row_index=new_index,
                    column_index=cell.column_index,
                    text=cell.text,
                    bounding_box=cell.bounding_box,
                    source_word_indices=cell.source_word_indices,
                    row_span=cell.row_span,
                    column_span=cell.column_span,
                    covered_by=covered_by,
                    is_header=cell.is_header,
                    is_stub=cell.is_stub,
                    confidence=cell.confidence,
                    warnings=cell.warnings,
                )
            )
        reindexed.append(
            TableRow(
                row_index=new_index,
                cells=new_cells,
                bounding_box=row.bounding_box,
                confidence=row.confidence,
                warnings=row.warnings,
            )
        )
    return reindexed


def _calibrated_structure_warnings(
    rows: list[TableRow], expected_columns: int
) -> tuple[list[ReconstructionWarning], list[dict[str, Any]]]:
    warnings: list[ReconstructionWarning] = []
    suppressions: list[dict[str, Any]] = []
    triangular = _looks_like_triangular_matrix(rows)
    header_rows = _estimate_header_rows(rows)
    for row in rows:
        filled_count = sum(1 for cell in row.cells if cell.text)
        if not expected_columns or filled_count in {expected_columns, 0}:
            continue
        reason = None
        if any(cell.row_span > 1 or cell.column_span > 1 or cell.covered_by for cell in row.cells):
            reason = WARNING_HEADER_SPAN_EXPLAINS_ROW_WIDTH
        elif row.row_index < header_rows:
            reason = WARNING_HEADER_SPAN_EXPLAINS_ROW_WIDTH
        elif triangular:
            reason = WARNING_TRIANGULAR_MATRIX_PATTERN
        elif filled_count < expected_columns:
            reason = WARNING_EXPECTED_EMPTY_CELL_PATTERN

        if reason:
            suppressions.append(
                {
                    "row_index": row.row_index,
                    "suppressed_warning": WARNING_IRREGULAR_COLUMN_COUNT,
                    "reason": reason,
                    "filled_columns": filled_count,
                    "expected_columns": expected_columns,
                }
            )
            continue
        warnings.append(
            ReconstructionWarning(
                WARNING_IRREGULAR_COLUMN_COUNT,
                f"Row {row.row_index} has {filled_count} populated cells for {expected_columns} columns",
                row_index=row.row_index,
            )
        )
    return warnings, suppressions


def _looks_like_triangular_matrix(rows: list[TableRow]) -> bool:
    if len(rows) < 3:
        return False
    body = rows[1:]
    filled_after_stub = [
        sum(1 for cell in row.cells[1:] if cell.text)
        for row in body
        if row.cells and row.cells[0].text
    ]
    if len(filled_after_stub) < 3:
        return False
    return filled_after_stub == sorted(filled_after_stub)


def _replace_cell(
    cell: Cell,
    text: str | None = None,
    bounding_box: BoundingBox | None | object = _UNSET,
    source_word_indices: list[int] | None = None,
    row_span: int | None = None,
    column_span: int | None = None,
    covered_by: tuple[int, int] | None | object = _UNSET,
    is_header: bool | None = None,
    is_stub: bool | None = None,
    confidence: float | None = None,
) -> Cell:
    return Cell(
        row_index=cell.row_index,
        column_index=cell.column_index,
        text=cell.text if text is None else text,
        bounding_box=cell.bounding_box if bounding_box is _UNSET else bounding_box,
        source_word_indices=cell.source_word_indices if source_word_indices is None else source_word_indices,
        row_span=cell.row_span if row_span is None else row_span,
        column_span=cell.column_span if column_span is None else column_span,
        covered_by=cell.covered_by if covered_by is _UNSET else covered_by,
        is_header=cell.is_header if is_header is None else is_header,
        is_stub=cell.is_stub if is_stub is None else is_stub,
        confidence=cell.confidence if confidence is None else confidence,
        warnings=cell.warnings,
    )


def _adaptive_metrics(
    words: list[WordBox], row_tolerance: float | None, column_tolerance: float | None
) -> dict[str, float]:
    heights = [word.bounding_box.height for word in words]
    char_widths = [
        word.bounding_box.width / max(len(word.text), 1)
        for word in words
        if word.bounding_box.width > 0 and word.text
    ]
    median_height = _median_or_default(heights, 10.0)
    median_char_width = _median_or_default(char_widths, 5.0)
    sorted_words = sorted(words, key=lambda word: (word.bounding_box.y_mid, word.bounding_box.x0))
    horizontal_gaps = []
    for left, right in zip(sorted_words, sorted_words[1:]):
        if abs(left.bounding_box.y_mid - right.bounding_box.y_mid) <= median_height:
            gap = right.bounding_box.x0 - left.bounding_box.x1
            if gap >= 0:
                horizontal_gaps.append(gap)

    median_gap = _median_or_default(horizontal_gaps, median_char_width)
    cell_gap_threshold = max(median_char_width * 1.6, min(median_gap * 0.5, median_char_width * 4.0))

    return {
        "median_word_height": median_height,
        "median_char_width": median_char_width,
        "median_horizontal_gap": median_gap,
        "row_tolerance": row_tolerance if row_tolerance is not None else max(median_height * 0.55, 2.0),
        "column_tolerance": (
            column_tolerance if column_tolerance is not None else max(median_char_width * 2.2, 4.0)
        ),
        "cell_gap_threshold": cell_gap_threshold,
        "large_gap_threshold": max(cell_gap_threshold * 3.0, median_char_width * 6.0),
    }


def _cluster_rows(words: list[WordBox], row_tolerance: float) -> tuple[list[_RawRow], list[ReconstructionWarning]]:
    rows: list[_RawRow] = []
    warnings: list[ReconstructionWarning] = []
    for word in sorted(words, key=lambda item: (item.bounding_box.y_mid, item.bounding_box.x0)):
        candidates = []
        for row in rows:
            overlap = _vertical_overlap_ratio(word.bounding_box, row.bbox)
            if overlap >= 0.45 or abs(word.bounding_box.y_mid - row.y_mid) <= row_tolerance:
                candidates.append(row)
        if len(candidates) > 1:
            warnings.append(
                ReconstructionWarning(
                    WARNING_AMBIGUOUS_ROW,
                    f"Word {word.index} could fit multiple row clusters",
                    word_indices=[word.index],
                )
            )
        if candidates:
            candidates[0].words.append(word)
        else:
            rows.append(_RawRow(words=[word]))

    for row in rows:
        row.words.sort(key=lambda item: (item.bounding_box.x0, item.bounding_box.y0))
    return rows, warnings


def _candidate_cells_for_row(row_index: int, row: _RawRow, metrics: dict[str, float]) -> list[_CandidateCell]:
    candidates: list[_CandidateCell] = []
    current_words: list[WordBox] = []
    current_warnings: list[ReconstructionWarning] = []
    row_words = sorted(row.words, key=lambda word: word.bounding_box.x0)

    for word in row_words:
        if not current_words:
            current_words = [word]
            continue
        previous = current_words[-1]
        gap = word.bounding_box.x0 - previous.bounding_box.x1
        if gap <= metrics["cell_gap_threshold"]:
            current_words.append(word)
        else:
            if gap >= metrics["large_gap_threshold"]:
                current_warnings.append(
                    ReconstructionWarning(
                        WARNING_LARGE_HORIZONTAL_GAP,
                        f"Large gap of {gap:.1f} points before word {word.index}",
                        row_index=row_index,
                        word_indices=[word.index],
                    )
                )
            candidates.append(_candidate_from_words(current_words, 0.94, current_warnings))
            current_words = [word]
            current_warnings = []

    if current_words:
        candidates.append(_candidate_from_words(current_words, 0.94, current_warnings))
    return candidates


def _candidate_from_words(
    words: list[WordBox], confidence: float, warnings: list[ReconstructionWarning]
) -> _CandidateCell:
    return _CandidateCell(
        text=" ".join(word.text for word in words),
        bbox=_union_boxes([word.bounding_box for word in words]),
        word_indices=[word.index for word in words],
        grouping_confidence=confidence,
        warnings=warnings,
    )


def _word_features_for_rows(
    raw_rows: list[_RawRow], metrics: dict[str, Any]
) -> list[list[_WordFeature]]:
    row_features: list[list[_WordFeature]] = []
    for row_index, row in enumerate(raw_rows):
        features: list[_WordFeature] = []
        sorted_words = sorted(row.words, key=lambda word: (word.bounding_box.x0, word.bounding_box.y0))
        for index, word in enumerate(sorted_words):
            next_word = sorted_words[index + 1] if index + 1 < len(sorted_words) else None
            gap = next_word.bounding_box.x0 - word.bounding_box.x1 if next_word else None
            features.append(
                _WordFeature(
                    word=word,
                    row_index=row_index,
                    x0=word.bounding_box.x0,
                    x1=word.bounding_box.x1,
                    center_x=word.bounding_box.x_mid,
                    gap_to_next=gap,
                    normalized_gap_to_next=gap / metrics["median_char_width"] if gap is not None else None,
                    token_type=_token_type(word.text),
                )
            )
        row_features.append(features)
    return row_features


def _infer_columns_from_atomic_words(
    row_features: list[list[_WordFeature]], metrics: dict[str, Any]
) -> _ColumnModel:
    warnings: list[ReconstructionWarning] = []
    rejected: list[dict[str, Any]] = []
    anchors: list[dict[str, Any]] = []
    tolerance = metrics["column_tolerance"]
    recurring_groups = _recurring_position_groups(row_features, tolerance)
    metrics["recurring_x_position_groups"] = len(recurring_groups)

    body_rows = [row for row in row_features if any(feature.token_type in {"numeric", "percentage"} for feature in row)]
    numeric_rows = [row for row in body_rows if _numeric_features(row)]

    if len(numeric_rows) >= 2:
        first_words = [row[0] for row in body_rows if row] or [row[0] for row in row_features if row]
        text_anchor_x = statistics.median(feature.x0 for feature in first_words)
        leading_stub_anchors = _leading_sparse_body_label_anchors(
            row_features,
            text_anchor_x,
            tolerance,
            metrics,
        )
        anchors.extend(leading_stub_anchors)
        anchors.append(
            _anchor_payload(
                index=len(anchors),
                position=text_anchor_x,
                support_features=first_words,
                alignment_type="left",
                source="first-word-left-edge",
            )
        )

        max_numeric_count = max(len(_numeric_features(row)) for row in numeric_rows)
        numeric_anchors: list[dict[str, Any]] = []
        for ordinal in range(max_numeric_count):
            support = [nums[ordinal] for row in numeric_rows if len((nums := _numeric_features(row))) > ordinal]
            support = support + _aligned_non_numeric_support(row_features, support, tolerance)
            if len(support) < 2:
                rejected.append(
                    {
                        "reason": "insufficient_numeric_support",
                        "ordinal": ordinal,
                        "support_count": len(support),
                    }
                )
                continue
            left_spread = max(feature.x0 for feature in support) - min(feature.x0 for feature in support)
            right_spread = max(feature.x1 for feature in support) - min(feature.x1 for feature in support)
            center_spread = max(feature.center_x for feature in support) - min(feature.center_x for feature in support)
            if right_spread <= max(left_spread, center_spread):
                position = statistics.median(feature.x1 for feature in support)
                alignment_type = "right"
            elif center_spread <= left_spread:
                position = statistics.median(feature.center_x for feature in support)
                alignment_type = "center"
            else:
                position = statistics.median(feature.x0 for feature in support)
                alignment_type = "left"
            numeric_anchors.append(
                _anchor_payload(
                    index=len(anchors) + len(numeric_anchors),
                    position=position,
                    support_features=support,
                    alignment_type=alignment_type,
                    source=f"numeric-ordinal-{ordinal}",
                )
            )

        if numeric_anchors:
            anchors.extend(numeric_anchors)
            recurring_text_anchors = []
            if recurring_text_anchors:
                anchors.extend(recurring_text_anchors)
                boundaries = _boundaries_from_anchor_positions(
                    [anchor["position"] for anchor in anchors], row_features
                )
                return _ColumnModel(
                    centers=_centers_from_boundaries(boundaries),
                    boundaries=boundaries,
                    anchors=_renumber_anchors(anchors),
                    rejected=rejected,
                    method="numeric_and_recurring_text_anchors",
                    confidence=0.9,
                    warnings=warnings,
                )
            recurring_atomic_anchors = _recurring_atomic_column_anchors(
                row_features,
                tolerance,
                metrics,
            )
            if len(recurring_atomic_anchors) > len(anchors):
                boundaries = _boundaries_from_anchor_positions(
                    [anchor["position"] for anchor in recurring_atomic_anchors], row_features
                )
                return _ColumnModel(
                    centers=_centers_from_boundaries(boundaries),
                    boundaries=boundaries,
                    anchors=_renumber_anchors(recurring_atomic_anchors),
                    rejected=rejected,
                    method="recurring_atomic_left_edges_with_numeric_body",
                    confidence=0.91,
                    warnings=warnings,
                )
            header_supported = _header_supported_text_anchors(row_features, metrics, tolerance)
            if len(header_supported) > len(anchors):
                return _ColumnModel(
                    centers=_centers_from_boundaries(
                        _boundaries_from_anchor_positions(
                            [anchor["position"] for anchor in header_supported], row_features
                        )
                    ),
                    boundaries=_boundaries_from_anchor_positions(
                        [anchor["position"] for anchor in header_supported], row_features
                    ),
                    anchors=_renumber_anchors(header_supported),
                    rejected=rejected,
                    method="header_supported_recurring_text_anchors",
                    confidence=0.9,
                    warnings=warnings,
                )
            if leading_stub_anchors:
                boundaries = _boundaries_from_anchor_positions(
                    [anchor["position"] for anchor in anchors], row_features
                )
                centers = _centers_from_boundaries(boundaries)
                return _ColumnModel(
                    centers=centers,
                    boundaries=boundaries,
                    anchors=_renumber_anchors(anchors),
                    rejected=rejected,
                    method="body_numeric_and_sparse_stub_anchors",
                    confidence=0.92,
                    warnings=warnings,
                )
            boundaries = _boundaries_from_body_gaps(row_features, numeric_rows, len(numeric_anchors))
            centers = _centers_from_boundaries(boundaries)
            return _ColumnModel(
                centers=centers,
                boundaries=boundaries,
                anchors=_renumber_anchors(anchors),
                rejected=rejected,
                method="body_numeric_ordinal_anchors",
                confidence=0.94,
                warnings=warnings,
            )

    clusters = _cluster_atomic_positions(row_features, tolerance)
    supported_clusters = [cluster for cluster in clusters if len({feature.row_index for feature in cluster}) >= 2]
    numeric_supported_clusters = [
        cluster
        for cluster in supported_clusters
        if any(
            _has_model_column_token(feature) or any(_has_model_column_token(item) for item in row_features[feature.row_index])
            for feature in cluster
        )
    ]
    if len(numeric_supported_clusters) >= 3:
        supported_clusters = [
            cluster
            for cluster in supported_clusters
            if not _looks_like_sparse_prose_fragment_cluster(cluster, row_features, tolerance)
            and not _looks_like_short_label_suffix_cluster(cluster, row_features, metrics)
        ]
    if len(supported_clusters) >= 2:
        supported_clusters.sort(key=lambda cluster: statistics.median(feature.x0 for feature in cluster))
        cluster_positions = [statistics.median(feature.x0 for feature in cluster) for cluster in supported_clusters]
        if all(
            (right - left) <= metrics["large_gap_threshold"]
            for left, right in zip(cluster_positions, cluster_positions[1:])
        ):
            supported_clusters = []

    if len(supported_clusters) >= 2:
        supported_positions = [statistics.median(feature.x0 for feature in cluster) for cluster in supported_clusters]
        for cluster in clusters:
            if cluster in supported_clusters:
                continue
            position = statistics.median(feature.x0 for feature in cluster)
            if min(supported_positions) < position < max(supported_positions):
                rejected.append(
                    {
                        "reason": "singleton_between_supported_columns",
                        "position": round(position, 3),
                        "support_count": len(cluster),
                    }
                )
                warnings.append(
                    ReconstructionWarning(
                        WARNING_AMBIGUOUS_COLUMN,
                        f"Unsupported column candidate at x={position:.1f} sits between stable columns",
                        word_indices=[feature.word.index for feature in cluster],
                    )
                )
        for cluster in supported_clusters:
            anchors.append(
                _anchor_payload(
                    index=len(anchors),
                    position=statistics.median(feature.x0 for feature in cluster),
                    support_features=cluster,
                    alignment_type="left",
                    source="recurring-left-edge",
                )
            )
        centers = [anchor["position"] for anchor in anchors]
        boundaries = _boundaries_from_anchor_positions(centers, row_features)
        return _ColumnModel(
            centers=_centers_from_boundaries(boundaries),
            boundaries=boundaries,
            anchors=_renumber_anchors(anchors),
            rejected=rejected,
            method="recurring_atomic_left_edges",
            confidence=0.82,
            warnings=warnings,
        )

    all_features = [feature for row in row_features for feature in row]
    if not all_features:
        raise ReconstructionError("Unable to infer any table columns")

    anchors.append(
        _anchor_payload(
            index=0,
            position=statistics.median(feature.x0 for feature in all_features),
            support_features=all_features,
            alignment_type="left",
            source="single-column-fallback",
        )
    )
    for cluster in clusters:
        if len({feature.row_index for feature in cluster}) < 2:
            rejected.append(
                {
                    "reason": "insufficient_cross_row_support",
                    "position": round(statistics.median(feature.x0 for feature in cluster), 3),
                    "support_count": len(cluster),
                }
            )
    bbox = _union_boxes([feature.word.bounding_box for feature in all_features])
    return _ColumnModel(
        centers=[bbox.x_mid],
        boundaries=[bbox.x0, bbox.x1],
        anchors=anchors,
        rejected=rejected,
        method="single_column_fallback",
        confidence=0.72 if len(recurring_groups) <= 1 else 0.45,
        warnings=warnings,
    )


def _row_features_without_trailing_notes(row_features: list[list[_WordFeature]]) -> list[list[_WordFeature]]:
    filtered = list(row_features)
    for row in reversed(row_features[-2:]):
        text = " ".join(feature.word.text for feature in row).strip()
        if _note_prefix(text):
            filtered = [candidate for candidate in filtered if candidate is not row]
            continue
        break
    return filtered or row_features


def _recurring_text_column_anchors(
    row_features: list[list[_WordFeature]], existing_anchors: list[dict[str, Any]], tolerance: float
) -> list[dict[str, Any]]:
    existing_positions = [float(anchor["position"]) for anchor in existing_anchors]
    body_row_indices = {
        row[0].row_index
        for row in row_features
        if any(feature.token_type in {"numeric", "percentage"} for feature in row)
    }
    first_numeric_position = min(
        (float(anchor["position"]) for anchor in existing_anchors if str(anchor.get("source", "")).startswith("numeric")),
        default=float("inf"),
    )
    anchors = []
    for cluster in _recurring_position_groups(row_features, tolerance):
        text_features = [
            feature
            for feature in cluster
            if feature.token_type == "text" and feature.row_index in body_row_indices
        ]
        if len({feature.row_index for feature in text_features}) < 2:
            continue
        position = statistics.median(feature.x0 for feature in text_features)
        if position > first_numeric_position:
            continue
        if any(abs(position - existing) <= tolerance for existing in existing_positions):
            continue
        anchors.append(
            _anchor_payload(
                index=len(existing_anchors) + len(anchors),
                position=position,
                support_features=text_features,
                alignment_type="left",
                source="recurring-text-left-edge",
            )
        )
    anchors.sort(key=lambda anchor: anchor["position"])
    return anchors


def _recurring_atomic_column_anchors(
    row_features: list[list[_WordFeature]],
    tolerance: float,
    metrics: dict[str, Any],
) -> list[dict[str, Any]]:
    supported_clusters = [
        cluster
        for cluster in _cluster_atomic_positions(row_features, tolerance)
        if len({feature.row_index for feature in cluster}) >= 2
    ]
    numeric_supported_clusters = [
        cluster
        for cluster in supported_clusters
        if any(
            _has_model_column_token(feature) or any(_has_model_column_token(item) for item in row_features[feature.row_index])
            for feature in cluster
        )
    ]
    if len(numeric_supported_clusters) >= 3:
        supported_clusters = [
            cluster
            for cluster in supported_clusters
            if not _looks_like_sparse_prose_fragment_cluster(cluster, row_features, tolerance)
            and not _looks_like_short_label_suffix_cluster(cluster, row_features, metrics)
        ]
    if len(supported_clusters) < 2:
        return []
    supported_clusters.sort(key=lambda cluster: statistics.median(feature.x0 for feature in cluster))
    positions = [statistics.median(feature.x0 for feature in cluster) for cluster in supported_clusters]
    if all(
        (right - left) <= metrics["large_gap_threshold"]
        for left, right in zip(positions, positions[1:])
    ):
        return []
    return [
        _anchor_payload(
            index=index,
            position=statistics.median(feature.x0 for feature in cluster),
            support_features=cluster,
            alignment_type="left",
            source="recurring-atomic-left-edge-with-numeric-body",
        )
        for index, cluster in enumerate(supported_clusters)
    ]


def _looks_like_sparse_prose_fragment_cluster(
    cluster: list[_WordFeature],
    row_features: list[list[_WordFeature]],
    tolerance: float,
) -> bool:
    if any(feature.token_type in {"numeric", "percentage"} for feature in cluster):
        return False
    rows_by_index = {row[0].row_index: row for row in row_features if row}
    if not cluster:
        return False
    for feature in cluster:
        row = rows_by_index.get(feature.row_index, [])
        if any(item.token_type in {"numeric", "percentage"} for item in row):
            return False
        text_items = [item for item in row if item.token_type == "text"]
        if len(text_items) <= 1:
            return False
        if abs(feature.x0 - min(item.x0 for item in text_items)) <= tolerance:
            return False
    return True


def _looks_like_short_label_suffix_cluster(
    cluster: list[_WordFeature],
    row_features: list[list[_WordFeature]],
    metrics: dict[str, Any],
) -> bool:
    if not cluster or any(feature.token_type != "text" for feature in cluster):
        return False
    rows_by_index = {row[0].row_index: sorted(row, key=lambda item: item.x0) for row in row_features if row}
    max_gap = max(metrics.get("median_char_width", 4.0) * 2.5, 12.0)
    for feature in cluster:
        row = rows_by_index.get(feature.row_index, [])
        previous_text = None
        for item in row:
            if item.word.index == feature.word.index:
                break
            if item.token_type == "text":
                previous_text = item
        if previous_text is None or feature.x0 - previous_text.x1 > max_gap:
            return False
    return True


def _has_model_column_token(feature: _WordFeature) -> bool:
    text = feature.word.text.strip()
    return feature.token_type in {"numeric", "percentage"} or bool(re.search(r"\d", text))


def _assign_and_group_words(
    raw_rows: list[_RawRow],
    row_features: list[list[_WordFeature]],
    column_centers: list[float],
    column_boundaries: list[float],
    metrics: dict[str, Any],
) -> tuple[list[TableRow], list[ReconstructionWarning]]:
    rows: list[TableRow] = []
    warnings: list[ReconstructionWarning] = []
    column_count = len(column_centers)

    for row_index, raw_row in enumerate(raw_rows):
        grouped_words: list[list[WordBox]] = [[] for _ in range(column_count)]
        row_warnings: list[ReconstructionWarning] = []
        previous_feature: _WordFeature | None = None
        previous_column_index: int | None = None
        for feature in row_features[row_index]:
            column_index, confidence = _assign_word_to_column(feature, column_boundaries, metrics)
            if (
                previous_feature is not None
                and previous_column_index is not None
                and feature.token_type == "text"
                and previous_feature.token_type == "text"
                and feature.word.text[:1].islower()
                and feature.x0 - previous_feature.x1 <= metrics["cell_gap_threshold"]
                and feature.x0 <= column_boundaries[min(previous_column_index + 1, len(column_boundaries) - 1)] + metrics["median_char_width"] * 4
            ):
                column_index = previous_column_index
                confidence = max(confidence, 0.82)
            if confidence < 0.62:
                warning = ReconstructionWarning(
                    WARNING_AMBIGUOUS_COLUMN,
                    f"Word '{feature.word.text}' is close to a column boundary",
                    row_index=row_index,
                    column_index=column_index,
                    word_indices=[feature.word.index],
                )
                warnings.append(warning)
                row_warnings.append(warning)
            grouped_words[column_index].append(feature.word)
            previous_feature = feature
            previous_column_index = column_index

        cells: list[Cell] = []
        for column_index, words in enumerate(grouped_words):
            words.sort(key=lambda word: word.bounding_box.x0)
            cell_warnings: list[ReconstructionWarning] = []
            column_span = 1
            bbox = _union_boxes([word.bounding_box for word in words]) if words else None
            if bbox is not None:
                column_span, merge_confidence = _span_for_box(bbox, column_index, column_boundaries)
                if bbox.width <= metrics["median_char_width"] * 5:
                    column_span = 1
                    merge_confidence = 1.0
                if column_span > 1:
                    warning = ReconstructionWarning(
                        WARNING_POSSIBLE_MERGED_CELL,
                        f"Cell may span {column_span} columns",
                        row_index=row_index,
                        column_index=column_index,
                        word_indices=[word.index for word in words],
                    )
                    warnings.append(warning)
                    row_warnings.append(warning)
                    cell_warnings.append(warning)
                else:
                    merge_confidence = 1.0
                cell_confidence = round(min(_cell_grouping_confidence(words, metrics), merge_confidence), 3)
            else:
                cell_confidence = 0.9

            cells.append(
                Cell(
                    row_index=row_index,
                    column_index=column_index,
                    text=" ".join(word.text for word in words),
                    bounding_box=bbox,
                    source_word_indices=[word.index for word in words],
                    row_span=1,
                    column_span=column_span,
                    confidence=cell_confidence,
                    warnings=cell_warnings,
                )
            )

        rows.append(
            TableRow(
                row_index=row_index,
                cells=cells,
                bounding_box=raw_row.bbox,
                confidence=round(_mean([cell.confidence for cell in cells if cell.text] or [0.9]), 3),
                warnings=row_warnings,
            )
        )

    return rows, warnings


def _assign_word_to_column(
    feature: _WordFeature, boundaries: list[float], metrics: dict[str, Any]
) -> tuple[int, float]:
    width = feature.x1 - feature.x0
    x = feature.x0 if feature.token_type == "text" and width > metrics["median_char_width"] * 7 else feature.center_x
    for column_index in range(len(boundaries) - 1):
        if boundaries[column_index] <= x <= boundaries[column_index + 1]:
            distances = []
            if column_index > 0:
                distances.append(abs(x - boundaries[column_index]))
            if column_index < len(boundaries) - 2:
                distances.append(abs(x - boundaries[column_index + 1]))
            distance = min(distances) if distances else metrics["median_char_width"] * 3
            confidence = min(1.0, max(0.35, distance / max(metrics["median_char_width"], 1.0)))
            return column_index, confidence
    if x < boundaries[0]:
        return 0, 0.5
    return len(boundaries) - 2, 0.5


def _cell_grouping_confidence(words: list[WordBox], metrics: dict[str, Any]) -> float:
    if len(words) <= 1:
        return 0.97
    gaps = [
        right.bounding_box.x0 - left.bounding_box.x1
        for left, right in zip(words, words[1:])
    ]
    if any(gap > metrics["large_gap_threshold"] for gap in gaps):
        return 0.62
    if any(gap > metrics["cell_gap_threshold"] * 1.8 for gap in gaps):
        return 0.78
    return 0.94


def _span_for_box(bbox: BoundingBox, column_index: int, boundaries: list[float]) -> tuple[int, float]:
    if column_index >= len(boundaries) - 2:
        return 1, 1.0
    span = 1
    for next_boundary in boundaries[column_index + 1 : -1]:
        if bbox.x1 > next_boundary:
            span += 1
    if span <= 1:
        return 1, 1.0
    return span, 0.82


def _token_type(text: str) -> str:
    compact = text.strip()
    if re.fullmatch(r"[-+]?\d[\d,]*(\.\d+)?", compact):
        return "numeric"
    if re.fullmatch(r"[-+]?\d+(\.\d+)?%", compact):
        return "percentage"
    if re.fullmatch(r"\d{1,4}[-/]\d{1,2}([-/]\d{1,4})?", compact):
        return "date"
    if re.search(r"[A-Za-z]", compact):
        return "text"
    return "other"


def _numeric_features(row: list[_WordFeature]) -> list[_WordFeature]:
    return [feature for feature in row if feature.token_type in {"numeric", "percentage"}]


def _aligned_non_numeric_support(
    row_features: list[list[_WordFeature]], numeric_support: list[_WordFeature], tolerance: float
) -> list[_WordFeature]:
    if not numeric_support:
        return []
    target_x0 = statistics.median(feature.x0 for feature in numeric_support)
    target_center = statistics.median(feature.center_x for feature in numeric_support)
    support_rows = {feature.row_index for feature in numeric_support}
    aligned = []
    for row in row_features:
        if not row or row[0].row_index in support_rows:
            continue
        for feature in row:
            if feature.token_type in {"numeric", "percentage"}:
                continue
            if abs(feature.x0 - target_x0) <= tolerance or abs(feature.center_x - target_center) <= tolerance:
                aligned.append(feature)
                break
    return aligned


def _leading_sparse_body_label_anchors(
    row_features: list[list[_WordFeature]],
    first_body_anchor_x: float,
    tolerance: float,
    metrics: dict[str, Any],
) -> list[dict[str, Any]]:
    candidates: dict[int, list[_WordFeature]] = {}
    for row in row_features:
        if not row:
            continue
        numeric_count = sum(1 for feature in row if feature.token_type in {"numeric", "percentage"})
        if numeric_count:
            continue
        text_features = [feature for feature in row if feature.token_type == "text"]
        if not text_features or len(text_features) > 3:
            continue
        first = text_features[0]
        if first.x0 >= first_body_anchor_x - max(tolerance, metrics.get("median_char_width", 4.0) * 1.5):
            continue
        bucket = round(first.x0 / max(tolerance, 1.0))
        candidates.setdefault(bucket, []).append(first)
    anchors = []
    for cluster in candidates.values():
        row_support = {feature.row_index for feature in cluster}
        if len(row_support) < 2:
            continue
        position = statistics.median(feature.x0 for feature in cluster)
        if position >= first_body_anchor_x:
            continue
        anchors.append(
            _anchor_payload(
                index=len(anchors),
                position=position,
                support_features=cluster,
                alignment_type="left",
                source="sparse-body-stub-left-edge",
            )
        )
    return sorted(anchors, key=lambda anchor: anchor["position"])


def _header_supported_text_anchors(
    row_features: list[list[_WordFeature]], metrics: dict[str, Any], tolerance: float
) -> list[dict[str, Any]]:
    if not row_features:
        return []
    header_row = sorted(row_features[0], key=lambda feature: feature.x0)
    header_positions = []
    previous: _WordFeature | None = None
    for feature in header_row:
        if previous is None or feature.x0 - previous.x1 >= metrics["large_gap_threshold"]:
            header_positions.append(feature.x0)
        previous = feature
    if len(header_positions) < 3:
        return []
    clusters = _cluster_atomic_positions(row_features, tolerance)
    anchors = []
    for cluster in clusters:
        rows = {feature.row_index for feature in cluster}
        if len(rows) < 2:
            continue
        position = statistics.median(feature.x0 for feature in cluster)
        if not any(abs(position - header_x) <= tolerance for header_x in header_positions):
            continue
        anchors.append(
            _anchor_payload(
                index=len(anchors),
                position=position,
                support_features=cluster,
                alignment_type="left",
                source="header-supported-recurring-left-edge",
            )
        )
    anchors.sort(key=lambda anchor: anchor["position"])
    if len(anchors) < 3:
        return []
    return anchors


def _boundaries_from_body_gaps(
    row_features: list[list[_WordFeature]], numeric_rows: list[list[_WordFeature]], numeric_column_count: int
) -> list[float]:
    all_boxes = [feature.word.bounding_box for row in row_features for feature in row]
    min_x = min(box.x0 for box in all_boxes)
    max_x = max(box.x1 for box in all_boxes)
    boundaries = [min_x]
    for ordinal in range(numeric_column_count):
        gap_midpoints = []
        for row in numeric_rows:
            nums = _numeric_features(row)
            if len(nums) <= ordinal:
                continue
            current = nums[ordinal]
            previous = _previous_feature(row, current)
            if previous is not None:
                gap_midpoints.append((previous.x1 + current.x0) / 2)
        if gap_midpoints:
            boundaries.append(statistics.median(gap_midpoints))
    boundaries.append(max_x)
    clean = [boundaries[0]]
    for boundary in boundaries[1:]:
        if boundary > clean[-1]:
            clean.append(boundary)
    return clean


def _previous_feature(row: list[_WordFeature], feature: _WordFeature) -> _WordFeature | None:
    sorted_row = sorted(row, key=lambda item: item.x0)
    for index, item in enumerate(sorted_row):
        if item.word.index == feature.word.index and index > 0:
            return sorted_row[index - 1]
    return None


def _centers_from_boundaries(boundaries: list[float]) -> list[float]:
    return [(left + right) / 2 for left, right in zip(boundaries, boundaries[1:])]


def _cluster_atomic_positions(
    row_features: list[list[_WordFeature]], tolerance: float
) -> list[list[_WordFeature]]:
    features = [feature for row in row_features for feature in row]
    clusters: list[list[_WordFeature]] = []
    for feature in sorted(features, key=lambda item: item.x0):
        if not clusters or abs(feature.x0 - statistics.median(item.x0 for item in clusters[-1])) > tolerance:
            clusters.append([feature])
        else:
            clusters[-1].append(feature)
    return clusters


def _recurring_position_groups(row_features: list[list[_WordFeature]], tolerance: float) -> list[list[_WordFeature]]:
    return [
        cluster
        for cluster in _cluster_atomic_positions(row_features, tolerance)
        if len({feature.row_index for feature in cluster}) >= 2
    ]


def _boundaries_from_anchor_positions(positions: list[float], row_features: list[list[_WordFeature]]) -> list[float]:
    all_boxes = [feature.word.bounding_box for row in row_features for feature in row]
    positions = sorted(positions)
    boundaries = [min(box.x0 for box in all_boxes)]
    for left, right in zip(positions, positions[1:]):
        boundaries.append((left + right) / 2)
    boundaries.append(max(box.x1 for box in all_boxes))
    return boundaries


def _anchor_payload(
    index: int,
    position: float,
    support_features: list[_WordFeature],
    alignment_type: str,
    source: str,
) -> dict[str, Any]:
    return {
        "index": index,
        "position": round(position, 3),
        "support_count": len(support_features),
        "alignment_type": alignment_type,
        "source": source,
        "word_indices": [feature.word.index for feature in support_features],
        "texts": [feature.word.text for feature in support_features],
    }


def _renumber_anchors(anchors: list[dict[str, Any]]) -> list[dict[str, Any]]:
    anchors = sorted(anchors, key=lambda anchor: anchor["position"])
    for index, anchor in enumerate(anchors):
        anchor["index"] = index
    return anchors


def _assign_cells(
    candidate_rows: list[list[_CandidateCell]],
    column_centers: list[float],
    column_boundaries: list[float],
    metrics: dict[str, float],
) -> tuple[list[TableRow], list[ReconstructionWarning]]:
    rows: list[TableRow] = []
    warnings: list[ReconstructionWarning] = []
    column_count = len(column_centers)

    for row_index, candidates in enumerate(candidate_rows):
        slots: list[Cell | None] = [None] * column_count
        row_warnings: list[ReconstructionWarning] = []
        for candidate in candidates:
            column_index, distance = _nearest_column(candidate.bbox.x0, column_centers)
            confidence = max(0.35, 1.0 - (distance / max(metrics["column_tolerance"] * 3.0, 1.0)))
            candidate_warnings = list(candidate.warnings)

            if distance > metrics["column_tolerance"] * 1.75:
                warning = ReconstructionWarning(
                    WARNING_AMBIGUOUS_COLUMN,
                    f"Cell '{candidate.text}' is far from inferred column {column_index}",
                    row_index=row_index,
                    column_index=column_index,
                    word_indices=candidate.word_indices,
                )
                warnings.append(warning)
                row_warnings.append(warning)
                candidate_warnings.append(warning)
                confidence = min(confidence, 0.58)

            column_span, merge_confidence = _possible_column_span(
                candidate, column_index, column_boundaries, column_count
            )
            if column_span > 1:
                warning = ReconstructionWarning(
                    WARNING_POSSIBLE_MERGED_CELL,
                    f"Cell '{candidate.text}' may span {column_span} columns",
                    row_index=row_index,
                    column_index=column_index,
                    word_indices=candidate.word_indices,
                )
                warnings.append(warning)
                row_warnings.append(warning)
                candidate_warnings.append(warning)
                confidence = min(confidence, merge_confidence)

            cell = Cell(
                row_index=row_index,
                column_index=column_index,
                text=candidate.text,
                bounding_box=candidate.bbox,
                source_word_indices=candidate.word_indices,
                row_span=1,
                column_span=column_span,
                confidence=round(min(confidence, candidate.grouping_confidence), 3),
                warnings=candidate_warnings,
            )

            if slots[column_index] is not None:
                warning = ReconstructionWarning(
                    WARNING_OVERLAPPING_CELLS,
                    f"Multiple cells assigned to row {row_index}, column {column_index}",
                    row_index=row_index,
                    column_index=column_index,
                    word_indices=candidate.word_indices,
                )
                warnings.append(warning)
                row_warnings.append(warning)
                cell = Cell(
                    row_index=cell.row_index,
                    column_index=cell.column_index,
                    text=f"{slots[column_index].text} {cell.text}".strip(),
                    bounding_box=slots[column_index].bounding_box.union(cell.bounding_box)
                    if slots[column_index].bounding_box and cell.bounding_box
                    else cell.bounding_box,
                    source_word_indices=slots[column_index].source_word_indices + cell.source_word_indices,
                    row_span=1,
                    column_span=max(slots[column_index].column_span, cell.column_span),
                    confidence=min(slots[column_index].confidence, cell.confidence, 0.55),
                    warnings=(slots[column_index].warnings or []) + (cell.warnings or []) + [warning],
                )
            slots[column_index] = cell

        cells: list[Cell] = []
        row_boxes = [candidate.bbox for candidate in candidates]
        for column_index, maybe_cell in enumerate(slots):
            if maybe_cell is None:
                cells.append(
                    Cell(
                        row_index=row_index,
                        column_index=column_index,
                        text="",
                        bounding_box=None,
                        source_word_indices=[],
                        confidence=0.9,
                        warnings=[],
                    )
                )
            else:
                cells.append(maybe_cell)

        rows.append(
            TableRow(
                row_index=row_index,
                cells=cells,
                bounding_box=_union_boxes(row_boxes) if row_boxes else None,
                confidence=round(_mean([cell.confidence for cell in cells if cell.text] or [0.9]), 3),
                warnings=row_warnings,
            )
        )

    return rows, warnings


def _possible_column_span(
    candidate: _CandidateCell, column_index: int, boundaries: list[float], column_count: int
) -> tuple[int, float]:
    if column_count <= 1:
        return 1, 1.0
    right_boundary_index = min(column_index + 1, len(boundaries) - 1)
    natural_right = boundaries[right_boundary_index]
    if candidate.bbox.x1 <= natural_right:
        return 1, 1.0

    span = 1
    for boundary_index in range(column_index + 2, len(boundaries)):
        if candidate.bbox.x1 > boundaries[boundary_index - 1]:
            span += 1
    span = min(span, column_count - column_index)
    overflow = candidate.bbox.x1 - natural_right
    confidence = 0.82 if overflow > 0 else 1.0
    return max(span, 1), confidence


def _cluster_positions(values: list[float], tolerance: float) -> list[list[float]]:
    clusters: list[list[float]] = []
    for value in sorted(values):
        if not clusters or abs(value - statistics.median(clusters[-1])) > tolerance:
            clusters.append([value])
        else:
            clusters[-1].append(value)
    return clusters


def _boundaries_from_centers(centers: list[float], candidate_rows: list[list[_CandidateCell]]) -> list[float]:
    all_boxes = [candidate.bbox for row in candidate_rows for candidate in row]
    min_x = min(box.x0 for box in all_boxes)
    max_x = max(box.x1 for box in all_boxes)
    if len(centers) == 1:
        return [min_x, max_x]
    boundaries = [min_x]
    for left, right in zip(centers, centers[1:]):
        boundaries.append((left + right) / 2)
    boundaries.append(max_x)
    return boundaries


def _nearest_column(x0: float, centers: list[float]) -> tuple[int, float]:
    distances = [(index, abs(x0 - center)) for index, center in enumerate(centers)]
    return min(distances, key=lambda item: (item[1], item[0]))


def _vertical_overlap_ratio(left: BoundingBox, right: BoundingBox) -> float:
    overlap = max(0.0, min(left.y1, right.y1) - max(left.y0, right.y0))
    return overlap / max(min(left.height, right.height), 1.0)


def _union_boxes(boxes: list[BoundingBox]) -> BoundingBox:
    if not boxes:
        raise ValueError("Cannot union an empty list of boxes")
    box = boxes[0]
    for other in boxes[1:]:
        box = box.union(other)
    return box


def _median_or_default(values: list[float], default: float) -> float:
    clean = [value for value in values if value > 0]
    return float(statistics.median(clean)) if clean else default


def _mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def _overall_confidence(
    rows: list[TableRow],
    warnings: list[ReconstructionWarning],
    expected_columns: int,
    column_confidence: float,
) -> float:
    cell_confidences = [
        cell.confidence for row in rows for cell in row.cells if cell.text or expected_columns > 0
    ]
    base = min(_mean(cell_confidences) if cell_confidences else 0.0, column_confidence)
    penalty = min(0.5, len(warnings) * 0.04)
    return round(max(0.0, min(1.0, base - penalty)), 3)


def _serialize_row_gaps(row_features: list[list[_WordFeature]]) -> list[dict[str, Any]]:
    rows = []
    for row_index, features in enumerate(row_features):
        rows.append(
            {
                "row_index": row_index,
                "gaps": [
                    {
                        "word_index": feature.word.index,
                        "row_index": row_index,
                        "text": feature.word.text,
                        "x0": round(feature.x0, 3),
                        "x1": round(feature.x1, 3),
                        "center_x": round(feature.center_x, 3),
                        "gap_to_next": round(feature.gap_to_next, 3)
                        if feature.gap_to_next is not None
                        else None,
                        "normalized_gap_to_next": round(feature.normalized_gap_to_next, 3)
                        if feature.normalized_gap_to_next is not None
                        else None,
                        "token_type": feature.token_type,
                    }
                    for feature in features
                ],
            }
        )
    return rows


def _serialize_word_assignments(rows: list[TableRow]) -> list[dict[str, Any]]:
    assignments = []
    for row in rows:
        for cell in row.cells:
            for word_index in cell.source_word_indices:
                assignments.append(
                    {
                        "word_index": word_index,
                        "row_index": cell.row_index,
                        "column_index": cell.column_index,
                        "cell_text": cell.text,
                    }
                )
    return sorted(assignments, key=lambda item: item["word_index"])


def _diagnostics_payload(table_grid: TableGrid) -> dict[str, Any]:
    low_confidence_cells = []
    for row in table_grid.rows:
        for cell in row.cells:
            if cell.text and cell.confidence < 0.75:
                low_confidence_cells.append(cell.to_dict())
    return {
        "overall_confidence": table_grid.confidence,
        "row_count": table_grid.row_count,
        "column_count": table_grid.column_count,
        "candidate_column_anchors": table_grid.metrics.get("candidate_column_anchors", []),
        "anchor_support_counts": table_grid.metrics.get("anchor_support_counts", []),
        "anchor_alignment_types": table_grid.metrics.get("anchor_alignment_types", []),
        "candidate_boundaries": table_grid.metrics.get("candidate_boundaries", []),
        "per_row_word_gaps": table_grid.metrics.get("per_row_word_gaps", []),
        "per_word_column_assignment": table_grid.metrics.get("per_word_column_assignment", []),
        "column_inference_method": table_grid.metrics.get("column_inference_method"),
        "rejected_column_candidates": table_grid.metrics.get("rejected_column_candidates", []),
        "confidence_components": table_grid.metrics.get("confidence_components", {}),
        "candidate_table_start": table_grid.metrics.get("candidate_table_start"),
        "candidate_table_end": table_grid.metrics.get("candidate_table_end"),
        "accepted_table_rows": table_grid.metrics.get("accepted_table_rows", []),
        "excluded_lines": table_grid.metrics.get("excluded_lines", []),
        "boundary_feature_values": table_grid.metrics.get("boundary_feature_values", []),
        "boundary_decision_scores": table_grid.metrics.get("boundary_decision_scores", []),
        "footnote_candidates": table_grid.metrics.get("footnote_candidates", []),
        "row_continuation_candidates": table_grid.metrics.get("row_continuation_candidates", []),
        "classification_reason_codes": table_grid.metrics.get("classification_reason_codes", []),
        "warning_suppression_reasons": table_grid.metrics.get("warning_suppression_reasons", []),
        "merged_ranges": table_grid.merged_ranges(),
        "excluded_text_blocks": table_grid.metrics.get("excluded_text_blocks", []),
        "low_confidence_cells": low_confidence_cells,
        "warnings": [warning.to_dict() for warning in table_grid.warnings],
        "metrics": table_grid.metrics,
    }


def _write_csv(table_grid: TableGrid, path: Path) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerows(table_grid.matrix())


def _write_json(path: Path, payload: object) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
