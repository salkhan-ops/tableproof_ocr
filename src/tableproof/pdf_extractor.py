"""Native PDF words and page rendering, under permissive licenses.

Words come from pdfplumber/pdfminer.six (MIT); rendering from pypdfium2 (Apache-2.0/BSD-3).
PyMuPDF (AGPL-3.0) is not used by the library. Coordinates are PDF points with the origin at
the top-left of the page, as before. A word belongs to a clip when its box intersects the clip
(the semantics PyMuPDF's ``clip`` had), so words straddling the edge are not cut.

:func:`extract_words` and :func:`render_page` are pure (no files written). The older
:func:`extract_table_area` keeps writing PNG/JSON artifacts for the CLI and benchmarks.
"""

from __future__ import annotations

import io
import json
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

import pdfplumber
import pypdfium2 as pdfium
from pdfminer.pdfparser import PDFSyntaxError

from tableproof.models import BoundingBox, NativeWord, PageMetadata

if TYPE_CHECKING:
    from PIL.Image import Image

DEFAULT_DPI = 300
_X_TOLERANCE = 1.5  # points between characters that still belong to one word
_Y_TOLERANCE = 3.0
_LINE_TOLERANCE = 2.0  # words whose tops differ by less than this share a line number

PdfSource = Path | str | bytes


class PdfExtractionError(RuntimeError):
    """Raised when the requested PDF extraction cannot be completed. Never carries content."""


@dataclass(frozen=True)
class ExtractionResult:
    table_png_path: Path
    words_json_path: Path
    metadata_json_path: Path
    word_count: int


@dataclass(frozen=True)
class PageWords:
    page_number: int
    page_width: float
    page_height: float
    words: list[NativeWord]


def _open_plumber(source: PdfSource) -> Any:
    try:
        if isinstance(source, bytes):
            return pdfplumber.open(io.BytesIO(source))
        return pdfplumber.open(Path(source))
    except (PDFSyntaxError, ValueError, OSError) as exc:
        raise PdfExtractionError("Unable to open the PDF") from exc


def extract_words(
    source: PdfSource, page_number: int, clip: BoundingBox | None = None
) -> PageWords:
    """Words on a 1-based page (optionally only those intersecting ``clip``)."""
    if page_number < 1:
        raise PdfExtractionError("Page number must be 1 or greater")
    with _open_plumber(source) as pdf:
        if not pdf.pages:
            raise PdfExtractionError("The PDF has no pages")
        if page_number > len(pdf.pages):
            raise PdfExtractionError(
                f"Page {page_number} is out of range for a {len(pdf.pages)}-page PDF"
            )
        page = pdf.pages[page_number - 1]
        x0, top, x1, bottom = (float(v) for v in page.bbox)
        if clip is not None:
            _validate_bbox_within_page(clip, x0, top, x1, bottom)
        raw = page.extract_words(
            x_tolerance=_X_TOLERANCE, y_tolerance=_Y_TOLERANCE, keep_blank_chars=False
        )
    selected = [w for w in raw if clip is None or _intersects(w, clip)]
    return PageWords(page_number, x1 - x0, bottom - top, _numbered(selected))


def _intersects(word: dict[str, Any], clip: BoundingBox) -> bool:
    x0, x1 = float(word["x0"]), float(word["x1"])
    top, bottom = float(word["top"]), float(word["bottom"])
    return x1 > clip.x0 and x0 < clip.x1 and bottom > clip.y0 and top < clip.y1


def _numbered(raw: list[dict[str, Any]]) -> list[NativeWord]:
    """Assign reading-order line and word numbers (pdfplumber has no block structure)."""
    words: list[NativeWord] = []
    line_no, word_no, last_top = -1, 0, None
    for item in sorted(raw, key=lambda w: (round(float(w["top"]), 1), float(w["x0"]))):
        top = float(item["top"])
        if last_top is None or abs(top - last_top) > _LINE_TOLERANCE:
            line_no, word_no, last_top = line_no + 1, 0, top
        words.append(
            NativeWord(
                text=str(item["text"]),
                bbox=BoundingBox(float(item["x0"]), top, float(item["x1"]), float(item["bottom"])),
                block_no=0,
                line_no=line_no,
                word_no=word_no,
            )
        )
        word_no += 1
    return words


def render_page(
    source: PdfSource,
    page_number: int,
    *,
    dpi: int = DEFAULT_DPI,
    clip: BoundingBox | None = None,
) -> Image:
    """Render a 1-based page (or a clip of it) to an RGB PIL image."""
    if dpi < 72:
        raise PdfExtractionError("DPI must be at least 72")
    try:
        document = pdfium.PdfDocument(source if isinstance(source, bytes) else str(source))
    except pdfium.PdfiumError as exc:
        raise PdfExtractionError("Unable to open the PDF") from exc
    try:
        if page_number < 1 or page_number > len(document):
            raise PdfExtractionError(f"Page {page_number} is out of range")
        page = document[page_number - 1]
        width, height = page.get_size()
        crop = (0.0, 0.0, 0.0, 0.0)
        if clip is not None:
            # pypdfium2 crops by margins (left, bottom, right, top) in points.
            crop = (clip.x0, height - clip.y1, width - clip.x1, clip.y0)
        return page.render(scale=dpi / 72, crop=crop).to_pil().convert("RGB")
    finally:
        document.close()


def extract_table_area(
    pdf_path: Path | str,
    page_number: int,
    bbox: BoundingBox | None = None,
    output_dir: Path | str = Path("outputs/extracted_tables"),
    dpi: int = DEFAULT_DPI,
) -> ExtractionResult:
    pdf_path = Path(pdf_path).expanduser().resolve()
    output_dir = Path(output_dir)
    if not pdf_path.is_file():
        raise PdfExtractionError("The PDF does not exist or is not a file")
    if dpi < 72:
        raise PdfExtractionError("DPI must be at least 72")
    output_dir.mkdir(parents=True, exist_ok=True)

    page_words = extract_words(pdf_path, page_number, bbox)
    crop_bbox = bbox or BoundingBox(0.0, 0.0, page_words.page_width, page_words.page_height)
    table_png_path, words_json_path, metadata_json_path = _artifact_paths(
        output_dir, pdf_path, page_number
    )
    render_page(pdf_path, page_number, dpi=dpi, clip=bbox).save(table_png_path)
    _write_json(words_json_path, [word.to_dict() for word in page_words.words])
    metadata = PageMetadata.build(
        pdf_path=pdf_path,
        page_number=page_number,
        page_width=page_words.page_width,
        page_height=page_words.page_height,
        crop_bbox=crop_bbox,
        render_dpi=dpi,
        rendered_png_path=table_png_path,
        native_words_json_path=words_json_path,
        metadata_json_path=metadata_json_path,
        word_count=len(page_words.words),
    )
    _write_json(metadata_json_path, metadata.to_dict())
    return ExtractionResult(
        table_png_path=table_png_path,
        words_json_path=words_json_path,
        metadata_json_path=metadata_json_path,
        word_count=len(page_words.words),
    )


def _validate_bbox_within_page(
    bbox: BoundingBox, x0: float, top: float, x1: float, bottom: float
) -> None:
    bbox.validate()
    if bbox.x0 < x0 or bbox.y0 < top:
        raise PdfExtractionError("Bounding box starts outside the page")
    if bbox.x1 > x1 or bbox.y1 > bottom:
        raise PdfExtractionError(
            f"Bounding box ends outside the page ({x1 - x0:.1f} x {bottom - top:.1f} points)"
        )


def _artifact_paths(output_dir: Path, pdf_path: Path, page_number: int) -> tuple[Path, Path, Path]:
    stem = f"{pdf_path.stem}_page_{page_number:03d}"
    return (
        output_dir / f"{stem}_table.png",
        output_dir / f"{stem}_words.json",
        output_dir / f"{stem}_metadata.json",
    )


def _write_json(path: Path, payload: object) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
