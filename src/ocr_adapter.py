from __future__ import annotations

import json
import re
from dataclasses import replace
from pathlib import Path
from typing import Any

from models import BoundingBox, WordBox
from ocr_models import OcrResult, OcrWordBox


_NUMERIC_PREFIX_RE = re.compile(r"^\(?[-+]?\d[\d,]*(\.\d+)?%?\)?")
_STAR_CONFUSABLE_CHARS = set("*°\"'´`º·•˚")
_MAX_STAR_SUFFIX_LENGTH = 3


def correct_significance_markers(word: OcrWordBox) -> OcrWordBox:
    """Fix Tesseract's known confusion of trailing significance stars (*) with
    degree/quote-like glyphs (°, ", ', ...) on otherwise-numeric tokens.

    Scoped narrowly to a numeric-prefix + short confusable-only suffix so it
    cannot touch genuine digit misreads or unrelated punctuation.
    """
    match = _NUMERIC_PREFIX_RE.match(word.text)
    if not match:
        return word
    suffix = word.text[match.end() :]
    if not suffix or len(suffix) > _MAX_STAR_SUFFIX_LENGTH:
        return word
    if any(char not in _STAR_CONFUSABLE_CHARS for char in suffix):
        return word
    if set(suffix) == {"*"}:
        return word
    return replace(word, text=word.text[: match.end()] + "*" * len(suffix))


def normalise_ocr_words(result: OcrResult) -> list[OcrWordBox]:
    clean = []
    for word in result.words:
        text = " ".join(word.text.split())
        if not text:
            continue
        bbox = word.bounding_box
        bbox.validate()
        clean.append(
            OcrWordBox(
                text=text,
                x0=bbox.x0,
                y0=bbox.y0,
                x1=bbox.x1,
                y1=bbox.y1,
                confidence=word.confidence,
                page_number=word.page_number,
                source_engine=word.source_engine,
                source_token_id=word.source_token_id,
                line_hint=word.line_hint,
                rotation=word.rotation,
                raw_text=word.raw_text or word.text,
            )
        )
    return sorted(clean, key=lambda item: (item.y0, item.x0, item.source_token_id))


def ocr_words_to_structure_words(words: list[OcrWordBox]) -> list[WordBox]:
    return [word.to_word_box(index) for index, word in enumerate(normalise_ocr_words(OcrResult("", "", "", 1, words, {})))]


def save_normalised_words(words: list[OcrWordBox], path: Path | str) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps([word.to_dict() for word in words], indent=2, sort_keys=True) + "\n", encoding="utf-8")


def load_normalised_words(path: Path | str) -> list[OcrWordBox]:
    return [OcrWordBox.from_dict(item) for item in json.loads(Path(path).read_text(encoding="utf-8"))]


def pdf_bbox_to_image_bbox(bbox: BoundingBox, crop_bbox: BoundingBox, scale: float) -> BoundingBox:
    return BoundingBox(
        x0=(bbox.x0 - crop_bbox.x0) * scale,
        y0=(bbox.y0 - crop_bbox.y0) * scale,
        x1=(bbox.x1 - crop_bbox.x0) * scale,
        y1=(bbox.y1 - crop_bbox.y0) * scale,
    )


def image_bbox_to_pdf_bbox(bbox: BoundingBox, crop_bbox: BoundingBox, scale: float) -> BoundingBox:
    return BoundingBox(
        x0=bbox.x0 / scale + crop_bbox.x0,
        y0=bbox.y0 / scale + crop_bbox.y0,
        x1=bbox.x1 / scale + crop_bbox.x0,
        y1=bbox.y1 / scale + crop_bbox.y0,
    )


def native_words_to_ocr_ground_truth(
    native_words: list[dict[str, Any]],
    metadata: dict[str, Any],
    image_scale: float | None = None,
    page_number: int | None = None,
) -> list[OcrWordBox]:
    crop = BoundingBox(**metadata["crop_bbox"])
    scale = image_scale if image_scale is not None else float(metadata.get("render_dpi", 300)) / 72.0
    words = []
    for index, word in enumerate(native_words):
        bbox = pdf_bbox_to_image_bbox(BoundingBox(**word["bbox"]), crop, scale)
        words.append(
            OcrWordBox(
                text=str(word.get("text", "")),
                x0=bbox.x0,
                y0=bbox.y0,
                x1=bbox.x1,
                y1=bbox.y1,
                confidence=1.0,
                page_number=int(page_number or metadata.get("page_number", 1)),
                source_engine="native_ground_truth",
                source_token_id=f"native-{index}",
                line_hint=f"{word.get('block_no', '')}:{word.get('line_no', '')}",
                rotation=0.0,
                raw_text=str(word.get("text", "")),
            )
        )
    return words
