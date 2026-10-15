from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import fitz

from models import BoundingBox, NativeWord, PageMetadata


DEFAULT_DPI = 300


class PdfExtractionError(RuntimeError):
    """Raised when the requested PDF extraction cannot be completed."""


@dataclass(frozen=True)
class ExtractionResult:
    table_png_path: Path
    words_json_path: Path
    metadata_json_path: Path
    word_count: int


def extract_table_area(
    pdf_path: Path | str,
    page_number: int,
    bbox: BoundingBox | None = None,
    output_dir: Path | str = Path("outputs/extracted_tables"),
    dpi: int = DEFAULT_DPI,
) -> ExtractionResult:
    pdf_path = Path(pdf_path).expanduser().resolve()
    output_dir = Path(output_dir)

    if not pdf_path.exists():
        raise PdfExtractionError(f"PDF does not exist: {pdf_path}")
    if not pdf_path.is_file():
        raise PdfExtractionError(f"PDF path is not a file: {pdf_path}")
    if page_number < 1:
        raise PdfExtractionError("Page number must be 1 or greater")
    if dpi < 72:
        raise PdfExtractionError("DPI must be at least 72")

    output_dir.mkdir(parents=True, exist_ok=True)

    try:
        with fitz.open(pdf_path) as document:
            if document.page_count == 0:
                raise PdfExtractionError(f"PDF has no pages: {pdf_path}")
            if page_number > document.page_count:
                raise PdfExtractionError(
                    f"Page {page_number} is out of range for a {document.page_count}-page PDF"
                )

            page = document.load_page(page_number - 1)
            page_rect = page.rect
            crop_bbox = bbox or BoundingBox(
                x0=float(page_rect.x0),
                y0=float(page_rect.y0),
                x1=float(page_rect.x1),
                y1=float(page_rect.y1),
            )
            _validate_bbox_within_page(crop_bbox, page_rect)

            words = _extract_native_words(page, crop_bbox)
            table_png_path, words_json_path, metadata_json_path = _artifact_paths(
                output_dir, pdf_path, page_number
            )
            _render_crop(page, crop_bbox, table_png_path, dpi)
            _write_json(words_json_path, [word.to_dict() for word in words])

            metadata = PageMetadata.build(
                pdf_path=pdf_path,
                page_number=page_number,
                page_width=float(page_rect.width),
                page_height=float(page_rect.height),
                crop_bbox=crop_bbox,
                render_dpi=dpi,
                rendered_png_path=table_png_path,
                native_words_json_path=words_json_path,
                metadata_json_path=metadata_json_path,
                word_count=len(words),
            )
            _write_json(metadata_json_path, metadata.to_dict())
    except PdfExtractionError:
        raise
    except fitz.FileDataError as exc:
        raise PdfExtractionError(f"Unable to open PDF: {pdf_path}") from exc
    except RuntimeError as exc:
        raise PdfExtractionError(f"PyMuPDF failed while processing {pdf_path}: {exc}") from exc

    return ExtractionResult(
        table_png_path=table_png_path,
        words_json_path=words_json_path,
        metadata_json_path=metadata_json_path,
        word_count=len(words),
    )


def _extract_native_words(page: fitz.Page, crop_bbox: BoundingBox) -> list[NativeWord]:
    clip = fitz.Rect(crop_bbox.x0, crop_bbox.y0, crop_bbox.x1, crop_bbox.y1)
    raw_words = page.get_text("words", clip=clip)
    words: list[NativeWord] = []

    for raw_word in raw_words:
        x0, y0, x1, y1, text, block_no, line_no, word_no = raw_word[:8]
        words.append(
            NativeWord(
                text=str(text),
                bbox=BoundingBox(float(x0), float(y0), float(x1), float(y1)),
                block_no=int(block_no),
                line_no=int(line_no),
                word_no=int(word_no),
            )
        )

    return words


def _render_crop(page: fitz.Page, crop_bbox: BoundingBox, output_path: Path, dpi: int) -> None:
    clip = fitz.Rect(crop_bbox.x0, crop_bbox.y0, crop_bbox.x1, crop_bbox.y1)
    matrix = fitz.Matrix(dpi / 72, dpi / 72)
    pixmap = page.get_pixmap(matrix=matrix, clip=clip, alpha=False)
    pixmap.save(output_path)


def _validate_bbox_within_page(bbox: BoundingBox, page_rect: fitz.Rect) -> None:
    bbox.validate()
    if bbox.x0 < page_rect.x0 or bbox.y0 < page_rect.y0:
        raise PdfExtractionError("Bounding box starts outside the page")
    if bbox.x1 > page_rect.x1 or bbox.y1 > page_rect.y1:
        raise PdfExtractionError(
            f"Bounding box ends outside the page ({page_rect.width:.1f} x {page_rect.height:.1f} points)"
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
