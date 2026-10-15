from __future__ import annotations

import json
from pathlib import Path

import fitz

import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from models import BoundingBox
from pdf_extractor import extract_table_area


def _create_sample_pdf(path: Path) -> None:
    document = fitz.open()
    page = document.new_page(width=612, height=792)
    page.insert_text((72, 96), "Revenue Q1 Q2", fontsize=12)
    page.insert_text((72, 122), "Product A 100 120", fontsize=12)
    page.insert_text((72, 148), "Product B 90 110", fontsize=12)
    document.save(path)
    document.close()


def test_extract_table_area_preserves_png_words_and_metadata(tmp_path: Path) -> None:
    pdf_path = tmp_path / "sample.pdf"
    output_dir = tmp_path / "outputs"
    _create_sample_pdf(pdf_path)

    result = extract_table_area(
        pdf_path=pdf_path,
        page_number=1,
        bbox=BoundingBox(60, 80, 260, 170),
        output_dir=output_dir,
        dpi=144,
    )

    assert result.table_png_path.exists()
    assert result.table_png_path.stat().st_size > 0
    assert result.words_json_path.exists()
    assert result.metadata_json_path.exists()

    words = json.loads(result.words_json_path.read_text(encoding="utf-8"))
    metadata = json.loads(result.metadata_json_path.read_text(encoding="utf-8"))

    assert "Revenue" in [word["text"] for word in words]
    assert metadata["page_number"] == 1
    assert metadata["crop_bbox"] == {"x0": 60, "y0": 80, "x1": 260, "y1": 170}
    assert metadata["word_count"] == len(words)
