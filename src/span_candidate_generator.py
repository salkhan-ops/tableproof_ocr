from __future__ import annotations

import itertools
from typing import Any

from models import BoundingBox, Cell, SpanCandidate, SpanScoreComponents, TableRow
from span_candidate_scorer import score_from_components


def generate_span_candidates(rows: list[TableRow], metrics: dict[str, Any] | None = None) -> list[SpanCandidate]:
    metrics = metrics or {}
    candidates: list[SpanCandidate] = []
    header_rows = int(metrics.get("estimated_header_rows") or _estimate_header_rows(rows))
    column_count = len(rows[0].cells) if rows else 0
    data_start = _infer_data_start_column(rows, header_rows)
    has_numeric_body = any(_is_numeric(cell.text) for row in rows for cell in row.cells)
    pass_through_context = _has_pass_through_context(rows, header_rows)

    for row in rows:
        if _looks_like_section_header(row):
            populated = [cell for cell in row.cells if cell.text]
            if len(populated) > 1:
                candidates.append(
                    _candidate_from_cells(
                        populated,
                        1,
                        column_count - populated[0].column_index,
                        "SECTION_HEADER_SPAN",
                        rows,
                        ["section_shape", "sparse_structural_row", "multi_fragment_section_anchor"],
                        [],
                        8.8,
                    )
                )
            else:
                candidates.append(_candidate(row.cells[0], 1, column_count, "SECTION_HEADER_SPAN", rows, ["section_shape", "sparse_structural_row"], [], 8.8))

    if not has_numeric_body and not candidates:
        return []

    for row in rows[:header_rows]:
        if row.row_index < header_rows - 1:
            for start, end in _header_text_runs(row, column_count, float(metrics.get("cell_gap_threshold", 18.0))):
                if start == 0:
                    continue
                blank_span = _blank_run_right(row, end, column_count)
                total_span = end - start + blank_span
                if total_span >= (end - start + 1):
                    support = _child_population_support(rows, row.row_index, start, total_span, header_rows)
                    header_support = _child_header_support(rows, row.row_index, start, total_span, header_rows)
                    if row.row_index > 0 and header_rows >= 3 and header_support == 0:
                        continue
                    if support:
                        candidate_type = "NESTED_HEADER_SPAN" if row.row_index + 1 < header_rows else "COLUMN_HEADER_SPAN"
                        candidates.append(
                            _candidate_from_cells(
                                row.cells[start : end + 1],
                                1,
                                total_span,
                                candidate_type,
                                rows,
                                ["multi_fragment_header_anchor", "child_columns_populated"],
                                [],
                                8.8 + min(1.0, support * 0.25),
                                {
                                    "header_fragment_count": end - start + 1,
                                    "child_population_support": support,
                                    "child_header_support": header_support,
                                },
                            )
                        )
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
            if span > 1 and row.row_index < max(1, header_rows - 1):
                support = _child_population_support(rows, row.row_index, col, span, header_rows)
                ambiguous_single_level = header_rows <= 1 and span == 2 and len(cell.text.split()) == 1
                if header_rows <= 1 and not ambiguous_single_level:
                    col += span
                    continue
                header_support = _child_header_support(rows, row.row_index, col, span, header_rows)
                if row.row_index > 0 and header_rows >= 3 and header_support == 0:
                    col += span
                    continue
                candidate_type = "NESTED_HEADER_SPAN" if row.row_index + 1 < header_rows else "COLUMN_HEADER_SPAN"
                base = 7.0 + min(1.2, support * 0.4)
                positive = ["header_sparsity_reduced", "child_columns_populated"] if support else ["header_sparsity_reduced"]
                negative = [] if support else ["weak_child_column_support"]
                if ambiguous_single_level:
                    base = 5.0
                    negative.append("ambiguous_short_single_level_header_span")
                candidates.append(
                    _candidate(
                        cell,
                        1,
                        span,
                        candidate_type,
                        rows,
                        positive,
                        negative,
                        base,
                        {"child_population_support": support, "child_header_support": header_support, "header_rows": header_rows},
                    )
                )
            col += max(span, 1)

    for row in rows[:header_rows]:
        for col, cell in enumerate(row.cells):
            if not cell.text:
                continue
            span = _blank_run_below(rows, row.row_index, col, limit=header_rows)
            rich_header_group = header_rows >= 3 or pass_through_context or any(
                candidate.column_span > 1 and len(candidate.text.split()) >= 3 and "HEADER" in candidate.candidate_type
                for candidate in candidates
            )
            if span > 1 and rich_header_group and row.row_index == 0:
                ctype = "STUB_HEADER_SPAN" if col == 0 else "COLUMN_HEADER_SPAN"
                candidates.append(
                    _candidate(
                        cell,
                        span,
                        1,
                        ctype,
                        rows,
                        ["header_pass_through", "header_level_consistency"],
                        [],
                        8.5,
                        {"header_rows": header_rows},
                    )
                )

    for row_index in range(header_rows, len(rows)):
        row = rows[row_index]
        if _looks_like_section_header(row):
            continue
        for col, cell in enumerate(row.cells[: max(1, data_start)]):
            if not cell.text:
                continue
            blank_span = _blank_run_below(rows, row_index, col, limit=len(rows))
            structural_span = _body_group_span_length(rows, row_index, col, data_start)
            span = max(blank_span, structural_span)
            if span <= 1:
                continue
            below_have_group_labels = any(
                any(next_cell.text for next_cell in rows[probe].cells[col + 1 : max(col + 2, data_start)])
                for probe in range(row_index + 1, row_index + span)
            )
            if span >= 3 or structural_span > 1 or (col + 1 < data_start and below_have_group_labels):
                ctype = "ROW_GROUP_SPAN" if col + 1 < data_start else "VERTICAL_STUB_SPAN"
                candidates.append(
                    _candidate(
                        cell,
                        span,
                        1,
                        ctype,
                        rows,
                        ["repeated_empty_stub_pattern", "body_columns_continue"],
                        [],
                        8.2 if span >= 3 else 7.2,
                        {"data_start_column": data_start, "row_group_depth": col + 1},
                    )
                )

    for row in rows[:header_rows]:
        for cell in row.cells:
            if not cell.text:
                continue
            col_span = _blank_run_right(row, cell.column_index, column_count)
            row_span = _blank_run_below(rows, row.row_index, cell.column_index, limit=header_rows)
            if row_span > 1 and col_span > 1:
                candidates.append(
                    _candidate(
                        cell,
                        row_span,
                        col_span,
                        "MIXED_ROW_COLUMN_SPAN",
                        rows,
                        ["row_and_column_empty_coordinates", "top_left_stub_shape"],
                        ["high_complexity"],
                        9.2,
                        {"mixed_span": True},
                    )
                )

    candidates = _dedupe(candidates)
    return _with_relationships(candidates)


def _candidate(
    cell: Cell,
    row_span: int,
    column_span: int,
    candidate_type: str,
    rows: list[TableRow],
    positive: list[str],
    negative: list[str],
    base_score: float,
    feature_values: dict[str, Any] | None = None,
) -> SpanCandidate:
    coords = [[r, c] for r in range(cell.row_index, cell.row_index + row_span) for c in range(cell.column_index, cell.column_index + column_span)]
    populated_covered = [
        (r, c)
        for r, c in (tuple(coord) for coord in coords[1:])
        if r < len(rows) and c < len(rows[r].cells) and rows[r].cells[c].text
    ]
    data_start = int((feature_values or {}).get("data_start_column", max(1, cell.column_index + 1)))
    anchor_has_nonnumeric_data = any(
        data_cell.text and not _is_numeric(data_cell.text)
        for data_cell in rows[cell.row_index].cells[data_start:]
    )
    label_child_coverage = (
        candidate_type in {"ROW_GROUP_SPAN", "VERTICAL_STUB_SPAN"}
        and populated_covered
        and not anchor_has_nonnumeric_data
        and all(
            not _is_numeric(rows[r].cells[c].text)
            for r, c in populated_covered
            if r < len(rows) and c < len(rows[r].cells)
        )
    )
    contradiction = 0.0 if label_child_coverage else (5.0 if populated_covered else 0.0)
    complexity = max(0, row_span * column_span - 2) * 0.15
    components = SpanScoreComponents(
        geometry=2.0,
        structure=base_score - 4.0,
        hierarchy=1.0 if "HEADER" in candidate_type or "GROUP" in candidate_type else 0.7,
        text_fit=1.0,
        empty_coordinate_support=min(1.5, (row_span * column_span - 1) * 0.35),
        contradiction_penalty=contradiction,
        complexity_penalty=complexity,
    )
    total, confidence = score_from_components(components)
    cid = f"{candidate_type.lower()}_{cell.row_index}_{cell.column_index}_{row_span}x{column_span}"
    if populated_covered and not label_child_coverage:
        negative = negative + ["covers_populated_cell"]
    if label_child_coverage:
        positive = positive + ["covers_child_stub_labels"]
    return SpanCandidate(
        candidate_id=cid,
        anchor_row=cell.row_index,
        anchor_column=cell.column_index,
        row_span=row_span,
        column_span=column_span,
        covered_coordinates=coords,
        candidate_type=candidate_type,
        source_word_indices=cell.source_word_indices,
        text=cell.text,
        bounding_box=cell.bounding_box,
        feature_values={
            "anchor_width": cell.bounding_box.width if cell.bounding_box else 0,
            "anchor_height": cell.bounding_box.height if cell.bounding_box else 0,
            "populated_covered_coordinates": [list(item) for item in populated_covered],
            **(feature_values or {}),
        },
        positive_evidence=positive,
        negative_evidence=negative,
        score_components=components,
        total_score=total,
        confidence=confidence,
        conflicting_candidate_ids=[],
        parent_candidate_ids=[],
        child_candidate_ids=[],
        rejected_reason="SPAN_COVERS_POPULATED_CELL" if populated_covered and not label_child_coverage else "",
    )


def _candidate_from_cells(
    cells: list[Cell],
    row_span: int,
    column_span: int,
    candidate_type: str,
    rows: list[TableRow],
    positive: list[str],
    negative: list[str],
    base_score: float,
    feature_values: dict[str, Any] | None = None,
) -> SpanCandidate:
    anchor = cells[0]
    coords = [[r, c] for r in range(anchor.row_index, anchor.row_index + row_span) for c in range(anchor.column_index, anchor.column_index + column_span)]
    source_tokens = []
    for cell in cells:
        tokens = cell.text.split()
        if len(tokens) == len(cell.source_word_indices):
            source_tokens.extend(zip(cell.source_word_indices, tokens))
        else:
            source_tokens.append((min(cell.source_word_indices or [10**9]), cell.text))
    source_tokens.sort(key=lambda item: item[0])
    source_indices = [index for index, _token in source_tokens]
    text = " ".join(token for _index, token in source_tokens)
    boxes = [cell.bounding_box for cell in cells if cell.bounding_box]
    bbox = boxes[0] if boxes else None
    for box in boxes[1:]:
        bbox = bbox.union(box) if bbox else box
    populated_covered = []
    anchor_fragment_coords = {(cell.row_index, cell.column_index) for cell in cells}
    for r, c in map(tuple, coords[1:]):
        if (r, c) in anchor_fragment_coords:
            continue
        if r < len(rows) and c < len(rows[r].cells) and rows[r].cells[c].text:
            populated_covered.append((r, c))
    components = SpanScoreComponents(
        geometry=2.2,
        structure=base_score - 4.0,
        hierarchy=1.2,
        text_fit=1.1,
        empty_coordinate_support=min(1.6, (row_span * column_span - len(cells)) * 0.35),
        contradiction_penalty=5.0 if populated_covered else 0.0,
        complexity_penalty=max(0, row_span * column_span - 2) * 0.12,
    )
    total, confidence = score_from_components(components)
    cid = f"{candidate_type.lower()}_{anchor.row_index}_{anchor.column_index}_{row_span}x{column_span}"
    return SpanCandidate(
        candidate_id=cid,
        anchor_row=anchor.row_index,
        anchor_column=anchor.column_index,
        row_span=row_span,
        column_span=column_span,
        covered_coordinates=coords,
        candidate_type=candidate_type,
        source_word_indices=source_indices,
        text=text,
        bounding_box=bbox,
        feature_values={
            "anchor_width": bbox.width if bbox else 0,
            "anchor_height": bbox.height if bbox else 0,
            "populated_covered_coordinates": [list(item) for item in populated_covered],
            **(feature_values or {}),
        },
        positive_evidence=positive,
        negative_evidence=negative + (["covers_populated_cell"] if populated_covered else []),
        score_components=components,
        total_score=total,
        confidence=confidence,
        conflicting_candidate_ids=[],
        parent_candidate_ids=[],
        child_candidate_ids=[],
        rejected_reason="SPAN_COVERS_POPULATED_CELL" if populated_covered else "",
    )


def _with_relationships(candidates: list[SpanCandidate]) -> list[SpanCandidate]:
    by_id = {candidate.candidate_id: candidate for candidate in candidates}
    parents: dict[str, list[str]] = {candidate.candidate_id: [] for candidate in candidates}
    children: dict[str, list[str]] = {candidate.candidate_id: [] for candidate in candidates}
    conflicts: dict[str, list[str]] = {candidate.candidate_id: [] for candidate in candidates}
    coord_sets = {candidate.candidate_id: {tuple(coord) for coord in candidate.covered_coordinates} for candidate in candidates}
    for left, right in itertools.combinations(candidates, 2):
        if _is_parent(left, right):
            children[left.candidate_id].append(right.candidate_id)
            parents[right.candidate_id].append(left.candidate_id)
            continue
        if _is_parent(right, left):
            children[right.candidate_id].append(left.candidate_id)
            parents[left.candidate_id].append(right.candidate_id)
            continue
        overlap = coord_sets[left.candidate_id] & coord_sets[right.candidate_id]
        if overlap:
            conflicts[left.candidate_id].append(right.candidate_id)
            conflicts[right.candidate_id].append(left.candidate_id)
    return [
        SpanCandidate(**{**candidate.__dict__, "conflicting_candidate_ids": sorted(conflicts[candidate.candidate_id]), "parent_candidate_ids": sorted(parents[candidate.candidate_id]), "child_candidate_ids": sorted(children[candidate.candidate_id])})
        for candidate in candidates
    ]


def _is_parent(parent: SpanCandidate, child: SpanCandidate) -> bool:
    if parent.anchor_row >= child.anchor_row:
        return False
    parent_cols = {coord[1] for coord in parent.covered_coordinates}
    child_cols = {coord[1] for coord in child.covered_coordinates}
    return child_cols <= parent_cols and parent.column_span > 1 and child.column_span > 1


def _dedupe(candidates: list[SpanCandidate]) -> list[SpanCandidate]:
    seen = set()
    output = []
    for candidate in candidates:
        key = (candidate.anchor_row, candidate.anchor_column, candidate.row_span, candidate.column_span, candidate.candidate_type)
        if key in seen:
            continue
        seen.add(key)
        output.append(candidate)
    return sorted(output, key=lambda c: (c.anchor_row, c.anchor_column, c.candidate_type, -c.total_score))


def _header_text_runs(row: TableRow, column_count: int, max_fragment_gap: float) -> list[tuple[int, int]]:
    runs = []
    populated = [index for index in range(column_count) if row.cells[index].text and not _is_numeric(row.cells[index].text)]
    for left_pos, start in enumerate(populated):
        for end in populated[left_pos + 1 :]:
            previous = row.cells[end - 1]
            current = row.cells[end]
            gap = (
                current.bounding_box.x0 - previous.bounding_box.x1
                if current.bounding_box is not None and previous.bounding_box is not None
                else max_fragment_gap + 1
            )
            if end == start + 1 and gap <= max_fragment_gap:
                runs.append((start, end))
            else:
                break
    start = None
    last = None
    for index in range(column_count):
        cell = row.cells[index]
        if cell.text and not _is_numeric(cell.text):
            if start is None:
                start = index
            last = index
            continue
        if start is not None and last is not None and last > start:
            runs.append((start, last))
        start = None
        last = None
    if start is not None and last is not None and last > start:
        runs.append((start, last))
    return runs


def _has_pass_through_context(rows: list[TableRow], header_rows: int) -> bool:
    if header_rows >= 3:
        return True
    if header_rows < 2 or not rows:
        return False
    top = rows[0]
    lower = rows[1]
    for index, cell in enumerate(top.cells):
        if index == 0 or not cell.text:
            continue
        right_blank = index + 1 < len(top.cells) and not top.cells[index + 1].text
        lower_same = index < len(lower.cells) and bool(lower.cells[index].text)
        if lower_same and not right_blank:
            return True
    return False


def _blank_run_right(row: TableRow, col: int, column_count: int) -> int:
    span = 1
    probe = col + 1
    while probe < column_count and not row.cells[probe].text:
        span += 1
        probe += 1
    return span


def _blank_run_below(rows: list[TableRow], row_index: int, col: int, limit: int) -> int:
    span = 1
    probe = row_index + 1
    while probe < limit and col < len(rows[probe].cells) and not rows[probe].cells[col].text and not _looks_like_section_header(rows[probe]):
        span += 1
        probe += 1
    return span


def _body_group_span_length(rows: list[TableRow], row_index: int, col: int, data_start: int) -> int:
    row = rows[row_index]
    if col >= len(row.cells) or not row.cells[col].text:
        return 1
    if any(_is_numeric(cell.text) for cell in row.cells):
        return 1
    span = 1
    for probe in range(row_index + 1, len(rows)):
        next_row = rows[probe]
        if _looks_like_section_header(next_row):
            break
        populated_stub_cols = [
            index
            for index, cell in enumerate(next_row.cells[: max(data_start, col + 1)])
            if cell.text and not _is_numeric(cell.text)
        ]
        numeric_count = sum(1 for cell in next_row.cells[data_start:] if _is_numeric(cell.text))
        if numeric_count == 0 and populated_stub_cols and min(populated_stub_cols) == col:
            break
        if numeric_count == 0 and populated_stub_cols and min(populated_stub_cols) < col:
            break
        if numeric_count == 0 and not populated_stub_cols:
            break
        if numeric_count >= 1 or populated_stub_cols:
            span += 1
            continue
        break
    return span


def _child_population_support(rows: list[TableRow], row_index: int, col: int, span: int, header_rows: int) -> int:
    support = 0
    for probe in range(row_index + 1, min(len(rows), header_rows + 2)):
        support += sum(1 for c in range(col, min(col + span, len(rows[probe].cells))) if rows[probe].cells[c].text)
    return support


def _child_header_support(rows: list[TableRow], row_index: int, col: int, span: int, header_rows: int) -> int:
    support = 0
    for probe in range(row_index + 1, min(len(rows), header_rows)):
        support += sum(1 for c in range(col, min(col + span, len(rows[probe].cells))) if rows[probe].cells[c].text)
    return support


def _infer_data_start_column(rows: list[TableRow], header_rows: int) -> int:
    max_col = len(rows[0].cells) if rows else 0
    first_numeric = max_col
    for row in rows[header_rows:]:
        for index, cell in enumerate(row.cells):
            if _is_numeric(cell.text):
                first_numeric = min(first_numeric, index)
                break
    return first_numeric if first_numeric != max_col else 1


def _estimate_header_rows(rows: list[TableRow]) -> int:
    header_rows = 0
    for index, row in enumerate(rows):
        numeric_count = sum(1 for cell in row.cells if _is_numeric(cell.text))
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
                or not any(_is_numeric(cell.text) for cell in rows[index + 1].cells)
            )
        ):
            header_rows += 1
            continue
        break
    return max(1, header_rows) if rows else 0


def _looks_like_section_header(row: TableRow) -> bool:
    populated = [cell for cell in row.cells if cell.text]
    if not populated or row.row_index == 0:
        return False
    text = " ".join(cell.text for cell in populated).strip()
    alpha = sum(1 for char in text if char.isalpha())
    if any(_is_numeric(cell.text) for cell in populated):
        return False
    if ":" in text and len(populated) <= max(4, len(row.cells)):
        return True
    return len(populated) <= 2 and len(text.split()) >= 2 and alpha / max(len(text), 1) > 0.5 and text.isupper()


def _is_numeric(text: str) -> bool:
    import re

    return bool(re.fullmatch(r"\(?[-+]?\d[\d,]*(\.\d+)?%?\)?(\*{1,3})?", text.strip()))
