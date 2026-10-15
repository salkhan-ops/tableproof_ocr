from __future__ import annotations

from pathlib import Path
import sys

from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))

from tableproof.models import BoundingBox
from tableproof.ocr_adapter import (
    correct_significance_markers,
    image_bbox_to_pdf_bbox,
    native_words_to_ocr_ground_truth,
    ocr_words_to_structure_words,
    pdf_bbox_to_image_bbox,
)
from ocr_benchmark_runner import _make_variant
from ocr_evaluator import evaluate_ocr_words, match_words_by_geometry
from tableproof.ocr_models import OcrResult, OcrWordBox


def word(text: str, x0: float, y0: float, x1: float, y1: float) -> OcrWordBox:
    return OcrWordBox(text, x0, y0, x1, y1, 0.9, 1, "test", text, raw_text=text)


def test_ocr_adapter_schema_conversion_and_raw_preservation() -> None:
    result = OcrResult("test", "1", "image.png", 1, [word("100", 1, 2, 11, 12)], {"raw": True})
    structure_words = ocr_words_to_structure_words(result.words)

    assert result.raw_response == {"raw": True}
    assert structure_words[0].text == "100"
    assert structure_words[0].bounding_box == BoundingBox(1, 2, 11, 12)


def test_pdf_to_image_coordinate_round_trip() -> None:
    crop = BoundingBox(50, 70, 150, 170)
    pdf = BoundingBox(60, 90, 80, 110)
    image = pdf_bbox_to_image_bbox(pdf, crop, 2.0)

    assert image == BoundingBox(20, 40, 60, 80)
    assert image_bbox_to_pdf_bbox(image, crop, 2.0) == pdf


def test_native_words_to_ocr_ground_truth() -> None:
    native = [{"text": "Alpha", "bbox": {"x0": 60, "y0": 80, "x1": 90, "y1": 92}, "block_no": 0, "line_no": 1}]
    metadata = {"crop_bbox": {"x0": 50, "y0": 70, "x1": 150, "y1": 170}, "render_dpi": 144, "page_number": 1}

    words = native_words_to_ocr_ground_truth(native, metadata)

    assert words[0].text == "Alpha"
    assert words[0].x0 == 20
    assert words[0].line_hint == "0:1"


def test_word_matching_and_character_error_metrics() -> None:
    expected = [word("0.10***", 0, 0, 20, 10), word("(0.02)", 30, 0, 50, 10)]
    predicted = [word("0.10**", 0, 0, 20, 10), word("(0.02)", 30, 0, 50, 10), word("ghost", 80, 0, 100, 10)]

    text_metrics, geometry_metrics, errors = evaluate_ocr_words("t", expected, predicted)

    assert text_metrics["word_exact_accuracy"] == 0.5
    assert text_metrics["significance_star_accuracy"] == 0.0
    assert text_metrics["parenthesis_accuracy"] == 1.0
    assert geometry_metrics["false_words"] == 1
    assert any(error["error_type"] == "OCR_STAR_ERROR" for error in errors)


def test_missed_split_and_merged_word_classification() -> None:
    expected = [word("Longterm", 0, 0, 40, 10)]
    predicted = [word("Long", 80, 0, 98, 10), word("term", 100, 0, 120, 10)]

    matches = match_words_by_geometry(expected, predicted, iou_threshold=0.8)
    text_metrics, geometry_metrics, _errors = evaluate_ocr_words("t", expected, predicted)

    assert matches
    assert geometry_metrics["missed_words"] >= 1
    assert geometry_metrics["false_words"] >= 1


def test_correct_significance_markers_fixes_observed_tesseract_confusions() -> None:
    cases = {
        '0.118"°': "0.118**",
        "0.097°": "0.097*",
        '0.332"*°': "0.332***",
        "0.301**°": "0.301***",
        "0.287°**": "0.287***",
    }
    for ocr_text, expected in cases.items():
        assert correct_significance_markers(word(ocr_text, 0, 0, 20, 10)).text == expected


def test_correct_significance_markers_leaves_genuine_errors_and_correct_text_alone() -> None:
    already_correct = word("0.10***", 0, 0, 20, 10)
    digit_misread = word("4200", 0, 0, 20, 10)
    non_numeric = word("Q1", 0, 0, 20, 10)
    long_suffix = word("100°°°°", 0, 0, 20, 10)

    assert correct_significance_markers(already_correct) is already_correct
    assert correct_significance_markers(digit_misread) is digit_misread
    assert correct_significance_markers(non_numeric) is non_numeric
    assert correct_significance_markers(long_suffix) is long_suffix


def test_deterministic_image_degradation(tmp_path: Path) -> None:
    source = tmp_path / "source.png"
    Image.new("RGB", (80, 40), "white").save(source)
    words = [word("A", 10, 10, 20, 20)]
    first = tmp_path / "first.jpg"
    second = tmp_path / "second.jpg"

    first_words = _make_variant(source, first, words, 100, 0.0, 0.0, 45)
    second_words = _make_variant(source, second, words, 100, 0.0, 0.0, 45)

    assert first.read_bytes() == second.read_bytes()
    assert first_words == second_words
