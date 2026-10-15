from __future__ import annotations

from models import SpanScoreComponents


def confidence_label(confidence: float) -> str:
    if confidence >= 0.82:
        return "HIGH_CONFIDENCE_SPAN"
    if confidence >= 0.62:
        return "MEDIUM_CONFIDENCE_SPAN"
    if confidence >= 0.45:
        return "LOW_CONFIDENCE_SPAN"
    return "UNRESOLVED_SPAN"


def score_from_components(components: SpanScoreComponents) -> tuple[float, float]:
    total = components.total()
    confidence = max(0.0, min(0.99, total / 10.0))
    return round(total, 6), round(confidence, 6)
