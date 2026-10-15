from __future__ import annotations

import math
import re
from dataclasses import dataclass
from typing import Any

from tableproof.ocr_models import OcrWordBox


NUMERIC_RE = re.compile(r"^\(?[-+]?\d[\d,]*(\.\d+)?%?\)?(\*{1,3})?$")


@dataclass(frozen=True)
class WordMatch:
    gt_index: int | None
    pred_index: int | None
    iou: float
    center_error: float


def evaluate_ocr_words(table_id: str, expected: list[OcrWordBox], predicted: list[OcrWordBox]) -> tuple[dict[str, Any], dict[str, Any], list[dict[str, Any]]]:
    matches = match_words_by_geometry(expected, predicted)
    matched = [match for match in matches if match.gt_index is not None and match.pred_index is not None]
    expected_texts = [expected[match.gt_index].text for match in matched if match.gt_index is not None]
    predicted_texts = [predicted[match.pred_index].text for match in matched if match.pred_index is not None]
    exact = sum(1 for left, right in zip(expected_texts, predicted_texts) if _norm(left) == _norm(right))
    char_distance = sum(_levenshtein(_norm(left), _norm(right)) for left, right in zip(expected_texts, predicted_texts))
    char_total = sum(len(_norm(text)) for text in expected_texts)
    numeric_expected = [
        (left, right)
        for left, right in zip(expected_texts, predicted_texts)
        if NUMERIC_RE.match(left.strip())
    ]
    text_metrics = {
        "table_id": table_id,
        "expected_words": len(expected),
        "predicted_words": len(predicted),
        "matched_words": len(matched),
        "word_exact_accuracy": _rate(exact, len(expected)),
        "character_error_rate": _rate(char_distance, char_total),
        "numeric_token_accuracy": _token_accuracy(numeric_expected),
        "punctuation_accuracy": _char_class_accuracy(expected_texts, predicted_texts, set(".,;:")),
        "minus_sign_accuracy": _char_presence_accuracy(expected_texts, predicted_texts, "-"),
        "decimal_point_accuracy": _char_presence_accuracy(expected_texts, predicted_texts, "."),
        "percentage_sign_accuracy": _char_presence_accuracy(expected_texts, predicted_texts, "%"),
        "parenthesis_accuracy": _char_presence_accuracy(expected_texts, predicted_texts, "()"),
        "significance_star_accuracy": _char_presence_accuracy(expected_texts, predicted_texts, "*"),
    }
    geometry_metrics = {
        "table_id": table_id,
        "word_box_detection_precision": _rate(len(matched), len(predicted)),
        "word_box_detection_recall": _rate(len(matched), len(expected)),
        "average_matched_box_iou": round(sum(match.iou for match in matched) / len(matched), 6) if matched else 0.0,
        "average_center_position_error": round(sum(match.center_error for match in matched) / len(matched), 6) if matched else 0.0,
        "baseline_position_error": _baseline_error(expected, predicted, matched),
        "reading_order_accuracy": _reading_order_accuracy(expected, predicted, matched),
        "missed_words": sum(1 for match in matches if match.gt_index is not None and match.pred_index is None),
        "false_words": sum(1 for match in matches if match.gt_index is None and match.pred_index is not None),
        "split_word_errors": _split_errors(expected, predicted, matches),
        "merged_word_errors": _merged_errors(expected, predicted, matches),
    }
    error_rows = _error_rows(table_id, expected, predicted, matches)
    return text_metrics, geometry_metrics, error_rows


def match_words_by_geometry(expected: list[OcrWordBox], predicted: list[OcrWordBox], iou_threshold: float = 0.15) -> list[WordMatch]:
    pairs = []
    for gt_index, gt in enumerate(expected):
        for pred_index, pred in enumerate(predicted):
            score = _iou(gt, pred)
            if score >= iou_threshold:
                pairs.append((score, gt_index, pred_index))
    pairs.sort(reverse=True)
    used_gt = set()
    used_pred = set()
    matches: list[WordMatch] = []
    for score, gt_index, pred_index in pairs:
        if gt_index in used_gt or pred_index in used_pred:
            continue
        used_gt.add(gt_index)
        used_pred.add(pred_index)
        matches.append(
            WordMatch(
                gt_index=gt_index,
                pred_index=pred_index,
                iou=score,
                center_error=_center_error(expected[gt_index], predicted[pred_index]),
            )
        )
    for gt_index in range(len(expected)):
        if gt_index not in used_gt:
            matches.append(WordMatch(gt_index=gt_index, pred_index=None, iou=0.0, center_error=0.0))
    for pred_index in range(len(predicted)):
        if pred_index not in used_pred:
            matches.append(WordMatch(gt_index=None, pred_index=pred_index, iou=0.0, center_error=0.0))
    return sorted(matches, key=lambda item: (item.gt_index is None, item.gt_index if item.gt_index is not None else 10**9, item.pred_index if item.pred_index is not None else 10**9))


def classify_primary_error(text_metrics: dict[str, Any], geometry_metrics: dict[str, Any], table_metrics: dict[str, Any]) -> str:
    if table_metrics.get("whole_table_exact_match") in {True, "True"}:
        return ""
    if geometry_metrics.get("missed_words", 0):
        return "OCR_MISSED_WORD"
    if geometry_metrics.get("false_words", 0):
        return "OCR_FALSE_WORD"
    if text_metrics.get("numeric_token_accuracy", 1.0) < 1.0:
        return "OCR_NUMERIC_ERROR"
    if text_metrics.get("significance_star_accuracy", 1.0) < 1.0:
        return "OCR_STAR_ERROR"
    if text_metrics.get("word_exact_accuracy", 1.0) < 1.0:
        return "OCR_TEXT_ERROR"
    if geometry_metrics.get("average_center_position_error", 0.0) > 8.0:
        return "OCR_BOUNDING_BOX_SHIFT"
    return "STRUCTURE_ERROR_TRIGGERED_BY_NOISY_BOXES"


def _error_rows(table_id: str, expected: list[OcrWordBox], predicted: list[OcrWordBox], matches: list[WordMatch]) -> list[dict[str, Any]]:
    rows = []
    for match in matches:
        if match.gt_index is not None and match.pred_index is None:
            rows.append({"table_id": table_id, "error_type": "OCR_MISSED_WORD", "expected_text": expected[match.gt_index].text, "predicted_text": "", "iou": 0.0})
        elif match.gt_index is None and match.pred_index is not None:
            rows.append({"table_id": table_id, "error_type": "OCR_FALSE_WORD", "expected_text": "", "predicted_text": predicted[match.pred_index].text, "iou": 0.0})
        elif match.gt_index is not None and match.pred_index is not None:
            gt = expected[match.gt_index].text
            pred = predicted[match.pred_index].text
            if _norm(gt) != _norm(pred):
                rows.append({"table_id": table_id, "error_type": _text_error_type(gt, pred), "expected_text": gt, "predicted_text": pred, "iou": round(match.iou, 6)})
    return rows


def _text_error_type(expected: str, predicted: str) -> str:
    if expected.count("*") != predicted.count("*"):
        return "OCR_STAR_ERROR"
    if NUMERIC_RE.match(expected.strip()):
        return "OCR_NUMERIC_ERROR"
    if any(mark in expected for mark in ".,;:()%()-"):
        return "OCR_PUNCTUATION_ERROR"
    return "OCR_TEXT_ERROR"


def _iou(left: OcrWordBox, right: OcrWordBox) -> float:
    x0 = max(left.x0, right.x0)
    y0 = max(left.y0, right.y0)
    x1 = min(left.x1, right.x1)
    y1 = min(left.y1, right.y1)
    overlap = max(0.0, x1 - x0) * max(0.0, y1 - y0)
    left_area = max(0.0, left.x1 - left.x0) * max(0.0, left.y1 - left.y0)
    right_area = max(0.0, right.x1 - right.x0) * max(0.0, right.y1 - right.y0)
    union = left_area + right_area - overlap
    return overlap / union if union else 0.0


def _center_error(left: OcrWordBox, right: OcrWordBox) -> float:
    return math.hypot(((left.x0 + left.x1) / 2) - ((right.x0 + right.x1) / 2), ((left.y0 + left.y1) / 2) - ((right.y0 + right.y1) / 2))


def _baseline_error(expected: list[OcrWordBox], predicted: list[OcrWordBox], matches: list[WordMatch]) -> float:
    values = [
        abs(expected[match.gt_index].y1 - predicted[match.pred_index].y1)
        for match in matches
        if match.gt_index is not None and match.pred_index is not None
    ]
    return round(sum(values) / len(values), 6) if values else 0.0


def _reading_order_accuracy(expected: list[OcrWordBox], predicted: list[OcrWordBox], matches: list[WordMatch]) -> float:
    pairs = [(match.gt_index, match.pred_index) for match in matches if match.gt_index is not None and match.pred_index is not None]
    if not pairs:
        return 0.0 if expected else 1.0
    inversions = 0
    total = 0
    for left_index, (gt_left, pred_left) in enumerate(pairs):
        for gt_right, pred_right in pairs[left_index + 1 :]:
            total += 1
            if (gt_left - gt_right) * (pred_left - pred_right) < 0:
                inversions += 1
    return round(1.0 - _rate(inversions, total), 6) if total else 1.0


def _split_errors(expected: list[OcrWordBox], predicted: list[OcrWordBox], matches: list[WordMatch]) -> int:
    missed_texts = [_norm(expected[match.gt_index].text) for match in matches if match.gt_index is not None and match.pred_index is None]
    false_texts = [_norm(predicted[match.pred_index].text) for match in matches if match.gt_index is None and match.pred_index is not None]
    return sum(1 for text in missed_texts if len(text) > 1 and _norm("".join(false_texts)).find(text) >= 0)


def _merged_errors(expected: list[OcrWordBox], predicted: list[OcrWordBox], matches: list[WordMatch]) -> int:
    missed_texts = [_norm(expected[match.gt_index].text) for match in matches if match.gt_index is not None and match.pred_index is None]
    false_texts = [_norm(predicted[match.pred_index].text) for match in matches if match.gt_index is None and match.pred_index is not None]
    return sum(1 for text in false_texts if any(text == _norm(left + right) for left in missed_texts for right in missed_texts if left != right))


def _token_accuracy(pairs: list[tuple[str, str]]) -> float:
    if not pairs:
        return 1.0
    return _rate(sum(1 for left, right in pairs if _norm(left) == _norm(right)), len(pairs))


def _char_class_accuracy(expected: list[str], predicted: list[str], chars: set[str]) -> float:
    relevant = [(left, right) for left, right in zip(expected, predicted) if any(char in left for char in chars)]
    if not relevant:
        return 1.0
    return _rate(sum(1 for left, right in relevant if [c for c in left if c in chars] == [c for c in right if c in chars]), len(relevant))


def _char_presence_accuracy(expected: list[str], predicted: list[str], chars: str) -> float:
    relevant = [(left, right) for left, right in zip(expected, predicted) if any(char in left for char in chars)]
    if not relevant:
        return 1.0
    return _rate(sum(1 for left, right in relevant if all(left.count(char) == right.count(char) for char in chars)), len(relevant))


def _levenshtein(left: str, right: str) -> int:
    previous = list(range(len(right) + 1))
    for i, left_char in enumerate(left, 1):
        current = [i]
        for j, right_char in enumerate(right, 1):
            current.append(min(previous[j] + 1, current[j - 1] + 1, previous[j - 1] + (left_char != right_char)))
        previous = current
    return previous[-1]


def _norm(text: str) -> str:
    return " ".join(text.split())


def _rate(numerator: int | float, denominator: int | float) -> float:
    return round(float(numerator) / float(denominator), 6) if denominator else 1.0
