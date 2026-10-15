"""Public, typed, in-memory API. Nothing is written to disk and nothing is printed.

    from tableproof.api import native_page, ocr_page, render

- :func:`native_page`: words from the PDF text layer → one reconstructed table per page.
- :func:`ocr_page`: an image of a page (e.g. rendered from a scan) → Tesseract → the same
  reconstruction. Word confidence is carried into each cell (the lowest word confidence in
  it); ``raw_text`` is what Tesseract read, ``text`` is the same unless the opt-in
  significance-marker pass changed it (off by default; never for financial statements).

Boxes are PDF points with the origin at the top-left of the page. Results are frozen
dataclasses with no document content beyond the cells themselves.
"""

from __future__ import annotations

import io
from dataclasses import dataclass
from pathlib import Path

from PIL import Image

from tableproof import __version__
from tableproof.models import BoundingBox, ReconstructionOptions, TableGrid, WordBox
from tableproof.ocr_adapter import correct_significance_markers
from tableproof.ocr_engine import TesseractOcrEngine
from tableproof.ocr_image_preprocessor import deskew, preprocess_image
from tableproof.pdf_extractor import extract_words, render_page
from tableproof.table_reconstructor import ReconstructionError, reconstruct_table_grid

BBox = tuple[float, float, float, float]
FINANCIAL_STATEMENT = ReconstructionOptions(profile="financial_statement")

__all__ = [
    "FINANCIAL_STATEMENT",
    "Cell",
    "Diagnostics",
    "ReconstructionOptions",
    "TableResult",
    "__version__",
    "native_page",
    "ocr_page",
    "render",
]


@dataclass(frozen=True)
class Cell:
    row: int
    column: int
    row_span: int
    column_span: int
    text: str
    raw_text: str
    bbox: BBox | None
    confidence: float  # geometric (native) or lowest OCR word confidence (OCR)
    is_header: bool


@dataclass(frozen=True)
class Diagnostics:
    """Content-free accounting for one table."""

    warnings: tuple[str, ...]
    words_total: int
    words_unassigned: int
    unresolved_lines: int
    no_word_loss: bool
    deskew_degrees: float = 0.0


@dataclass(frozen=True)
class TableResult:
    page: int
    engine: str  # "native" | "ocr"
    version: str
    cells: tuple[Cell, ...]
    confidence: float
    bbox: BBox | None
    diagnostics: Diagnostics


def render(pdf: Path | str | bytes, page: int, *, dpi: int = 300) -> Image.Image:
    """Render a page (for OCR of scans or for evidence crops)."""
    return render_page(pdf, page, dpi=dpi)


def native_page(
    pdf: Path | str | bytes,
    page: int,
    *,
    clip: BBox | None = None,
    options: ReconstructionOptions = FINANCIAL_STATEMENT,
) -> TableResult | None:
    """Reconstruct the table on a text-layer page. ``None`` when the page has no words."""
    words = extract_words(pdf, page, BoundingBox(*clip) if clip else None).words
    boxes = [WordBox(i, w.text, w.bbox) for i, w in enumerate(words)]
    grid = _reconstruct(boxes, options)
    if grid is None:
        return None
    raw = {i: w.text for i, w in enumerate(words)}
    return _result(grid, page, "native", raw, conf=None, total=len(words), skew=0.0)


def ocr_page(
    image: bytes | Image.Image,
    page: int,
    *,
    dpi: int,
    engine: TesseractOcrEngine | None = None,
    options: ReconstructionOptions = FINANCIAL_STATEMENT,
    correct_markers: bool = False,
    straighten: bool = True,
) -> TableResult | None:
    """OCR one page image rendered at ``dpi`` and reconstruct its table."""
    engine = engine or TesseractOcrEngine()
    picture = image if isinstance(image, Image.Image) else Image.open(io.BytesIO(image))
    picture = picture.convert("L")
    skew = 0.0
    if straighten:
        picture, skew = deskew(picture)
    prepared = preprocess_image(picture)
    factor = prepared.width / picture.width  # preprocessing upscales
    buffer = io.BytesIO()
    prepared.save(buffer, format="PNG")
    result = engine.recognise_bytes(buffer.getvalue(), page_number=page)
    words = [correct_significance_markers(w) for w in result.words] if correct_markers else list(
        result.words
    )
    # Back to the page image's pixels, then to PDF points.
    to_points = 72.0 / dpi / factor
    boxes = [
        WordBox(i, w.text, BoundingBox(w.x0 * to_points, w.y0 * to_points, w.x1 * to_points,
                                       w.y1 * to_points))
        for i, w in enumerate(words)
    ]
    grid = _reconstruct(boxes, options)
    if grid is None:
        return None
    raw = {i: (w.raw_text if w.raw_text is not None else w.text) for i, w in enumerate(words)}
    conf = {i: w.confidence for i, w in enumerate(words)}
    return _result(grid, page, "ocr", raw, conf=conf, total=len(words), skew=skew)


def _reconstruct(boxes: list[WordBox], options: ReconstructionOptions) -> TableGrid | None:
    if not boxes:
        return None
    try:
        return reconstruct_table_grid(boxes, options=options)
    except ReconstructionError:
        return None


def _result(
    grid: TableGrid,
    page: int,
    engine: str,
    raw: dict[int, str],
    *,
    conf: dict[int, float] | None,
    total: int,
    skew: float,
) -> TableResult:
    cells: list[Cell] = []
    for row in grid.rows:
        for cell in row.cells:
            if cell.covered_by is not None or not cell.text:
                continue
            ids = cell.source_word_indices or []
            word_conf = min((conf[i] for i in ids if i in conf), default=None) if conf else None
            box = cell.bounding_box
            cells.append(
                Cell(
                    row=cell.row_index,
                    column=cell.column_index,
                    row_span=cell.row_span,
                    column_span=cell.column_span,
                    text=cell.text,
                    raw_text=" ".join(raw[i] for i in ids if i in raw) or cell.text,
                    bbox=(box.x0, box.y0, box.x1, box.y1) if box else None,
                    confidence=round(min(cell.confidence, word_conf), 4)
                    if word_conf is not None else round(cell.confidence, 4),
                    is_header=cell.is_header,
                )
            )
    resolution = grid.metrics.get("continuation_resolution", {})
    nwl = resolution.get("no_word_loss", {})
    unassigned = len(nwl.get("lost_word_indices", [])) + len(
        nwl.get("hidden_by_spans_word_indices", [])
    )
    box = grid.bounding_box
    return TableResult(
        page=page,
        engine=engine,
        version=__version__,
        cells=tuple(cells),
        confidence=round(grid.confidence, 4),
        bbox=(box.x0, box.y0, box.x1, box.y1) if box else None,
        diagnostics=Diagnostics(
            warnings=tuple(sorted({w.code for w in grid.warnings})),
            words_total=total,
            words_unassigned=unassigned,
            unresolved_lines=len(resolution.get("unresolved_lines", [])),
            no_word_loss=bool(nwl.get("passed", False)),
            deskew_degrees=round(skew, 2),
        ),
    )
