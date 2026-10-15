"""OCR regression on the financial-statement fixtures (rendered and degraded deterministically).

Gate: share of ground-truth amounts read verbatim (digits are what matter). Measured
2026-10-15 with Tesseract 5.5: clean 98.0%, hard (1.5 degrees, noise, 200 dpi, JPEG) 91.7%.
The gate leaves a small margin below those numbers.
"""

from __future__ import annotations

import io
import json
import random
import re
import shutil
from pathlib import Path

import pytest
from PIL import Image, ImageFilter

from tableproof.api import ocr_page, render

DATA = Path(__file__).resolve().parents[1] / "data" / "financial_statements"
AMOUNT = re.compile(r"^\(?\d[\d,]*(\.\d+)?\)?$")
pytestmark = pytest.mark.skipif(shutil.which("tesseract") is None, reason="tesseract missing")


def _hard(img: Image.Image, seed: int) -> tuple[Image.Image, int]:
    rnd = random.Random(seed)
    img = img.rotate(-1.5, resample=Image.Resampling.BICUBIC, fillcolor=255)
    px = img.load()
    for _ in range(int(img.width * img.height * 0.008)):
        px[rnd.randrange(img.width), rnd.randrange(img.height)] = rnd.choice([0, 255])
    img = img.filter(ImageFilter.GaussianBlur(0.6))
    img = img.resize((int(img.width / 1.5), int(img.height / 1.5)), Image.Resampling.BILINEAR)
    buffer = io.BytesIO()
    img.save(buffer, "JPEG", quality=55)
    return Image.open(io.BytesIO(buffer.getvalue())).convert("L"), 200


def _recall(variant: str) -> float:
    total = found = 0
    for gt in sorted(DATA.glob("*.gt.json")):
        for number, rows in enumerate(json.loads(gt.read_text())["pages"], start=1):
            page = render(gt.with_name(gt.name.replace(".gt.json", ".pdf")), number, dpi=300)
            image, dpi = (page.convert("L"), 300) if variant == "clean" else _hard(page.convert("L"), number)
            result = ocr_page(image, number, dpi=dpi)
            tokens = {t for c in (result.cells if result else ()) for t in c.text.split()}
            expected = [v for row in rows for v in row[1:] if AMOUNT.match(v or "")]
            total += len(expected)
            found += sum(1 for v in expected if v in tokens)
    return found / total


def test_clean_scans_read_amounts_verbatim() -> None:
    assert _recall("clean") >= 0.95


def test_degraded_scans_read_most_amounts_verbatim() -> None:
    assert _recall("hard") >= 0.88
