from __future__ import annotations

from tableproof.models import HeaderHierarchy, SpanCandidate, SpanHierarchyNode


def build_header_hierarchy(selected: list[SpanCandidate]) -> HeaderHierarchy:
    nodes = []
    selected_ids = {candidate.candidate_id for candidate in selected}
    for candidate in selected:
        if candidate.column_span <= 1 or "HEADER" not in candidate.candidate_type:
            continue
        parents = [parent for parent in candidate.parent_candidate_ids if parent in selected_ids]
        children = [child for child in candidate.child_candidate_ids if child in selected_ids]
        nodes.append(
            SpanHierarchyNode(
                candidate_id=candidate.candidate_id,
                parent_candidate_id=parents[0] if parents else None,
                child_candidate_ids=children,
                hierarchy_level=candidate.anchor_row,
                covered_rows=sorted({coord[0] for coord in candidate.covered_coordinates}),
                covered_columns=sorted({coord[1] for coord in candidate.covered_coordinates}),
                text=candidate.text,
                confidence=candidate.confidence,
            )
        )
    warnings = []
    return HeaderHierarchy(nodes=nodes, exact_partition=not warnings, warnings=warnings)
