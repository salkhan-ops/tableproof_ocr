from __future__ import annotations

from tableproof.models import SpanCandidate, SpanHierarchyNode, StubHierarchy


def build_stub_hierarchy(selected: list[SpanCandidate]) -> StubHierarchy:
    nodes = []
    selected_ids = {candidate.candidate_id for candidate in selected}
    for candidate in selected:
        if candidate.row_span <= 1 or not any(token in candidate.candidate_type for token in ["STUB", "GROUP", "SECTION", "MIXED"]):
            continue
        parents = [parent for parent in candidate.parent_candidate_ids if parent in selected_ids]
        children = [child for child in candidate.child_candidate_ids if child in selected_ids]
        nodes.append(
            SpanHierarchyNode(
                candidate_id=candidate.candidate_id,
                parent_candidate_id=parents[0] if parents else None,
                child_candidate_ids=children,
                hierarchy_level=candidate.anchor_column,
                covered_rows=sorted({coord[0] for coord in candidate.covered_coordinates}),
                covered_columns=sorted({coord[1] for coord in candidate.covered_coordinates}),
                text=candidate.text,
                confidence=candidate.confidence,
            )
        )
    return StubHierarchy(nodes=nodes, exact_partition=True, warnings=[])
