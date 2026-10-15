"""Public in-memory API: native and OCR pages, confidence, raw text, deskew."""

from __future__ import annotations

import io
import shutil
from pathlib import Path

import pytest
from PIL import Image

from tableproof import __version__
from tableproof.api import native_page, ocr_page, render
from tableproof.ocr_image_preprocessor import estimate_skew

DATA = Path(__file__).resolve().parents[1] / "data" / "financial_statements"
SOFP = DATA / "fs01_sofp_two_period.pdf"
needs_tesseract = pytest.mark.skipif(shutil.which("tesseract") is None, reason="tesseract missing")


def _cells(result):
    return {c.text for c in result.cells}


def test_native_page_from_bytes_writes_nothing(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.chdir(tmp_path)
    result = native_page(SOFP.read_bytes(), 1)
    assert result is not None
    assert result.engine == "native"
    assert result.version == __version__
    assert "1,245,300" in _cells(result)
    assert result.diagnostics.no_word_loss
    assert all(c.raw_text == c.text for c in result.cells)
    assert all(c.bbox is not None and 0 <= c.bbox[0] < c.bbox[2] <= 600 for c in result.cells)
    assert list(tmp_path.iterdir()) == []


def test_estimate_skew_recovers_a_known_rotation() -> None:
    page = render(SOFP, 1, dpi=150).convert("L")
    for angle in (-1.5, 0.8):
        rotated = page.rotate(angle, resample=Image.Resampling.BICUBIC, fillcolor=255)
        assert estimate_skew(rotated) == pytest.approx(angle, abs=0.25)


@needs_tesseract
def test_ocr_page_carries_word_confidence_and_raw_text() -> None:
    image = render(SOFP, 1, dpi=300)
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    result = ocr_page(buffer.getvalue(), 1, dpi=300)
    assert result is not None
    assert result.engine == "ocr"
    amounts = [c for c in result.cells if c.text == "1,245,300"]
    assert amounts, "a clean 300 dpi render must OCR this amount"
    assert all(0 <= c.confidence <= 1 for c in result.cells)
    assert all(c.raw_text for c in result.cells)
    # Boxes are in PDF points (A4 is 595 x 842).
    assert all(c.bbox is not None and c.bbox[2] < 600 for c in result.cells)


@needs_tesseract
def test_ocr_correction_pass_is_off_by_default() -> None:
    from tableproof.ocr_models import OcrResult, OcrWordBox

    class StubEngine:
        def recognise_bytes(self, image: bytes, page_number: int = 1) -> OcrResult:
            words = [OcrWordBox("130,446\"", 100, 100, 200, 120, 0.9, 1, "stub", "1",
                                raw_text="130,446\""),
                     OcrWordBox("Revenue", 10, 100, 80, 120, 0.95, 1, "stub", "2",
                                raw_text="Revenue")]
            return OcrResult("stub", "0", "", 1, words, {})

    blank = Image.new("L", (300, 200), 255)
    result = ocr_page(blank, 1, dpi=72, engine=StubEngine(), straighten=False)  # type: ignore[arg-type]
    assert result is not None
    texts = _cells(result)
    assert any('130,446"' in t for t in texts)  # unchanged: the marker pass did not run
    changed = ocr_page(blank, 1, dpi=72, engine=StubEngine(), straighten=False,  # type: ignore[arg-type]
                       correct_markers=True)
    assert changed is not None
    cell = next(c for c in changed.cells if "130,446" in c.text)
    assert "130,446*" in cell.text
    assert '130,446"' in cell.raw_text  # raw text kept
