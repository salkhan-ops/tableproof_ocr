"""The library never needs PyMuPDF (AGPL-3.0) at runtime; it is a dev-only fixture tool."""

from __future__ import annotations

import subprocess
import sys
import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
AGPL_OR_GPL = {"pymupdf", "fitz", "pymupdfb", "ghostscript", "poppler"}


def test_runtime_dependencies_are_permissive() -> None:
    project = tomllib.loads((ROOT / "pyproject.toml").read_text())["project"]
    names = {d.split(">")[0].split("<")[0].split("=")[0].strip().lower() for d in project["dependencies"]}
    assert not names & AGPL_OR_GPL


def test_library_works_with_pymupdf_blocked() -> None:
    pdf = sorted((ROOT / "data" / "raw_pdfs").glob("*.pdf"))[0]
    code = (
        "import sys; sys.modules['fitz'] = None; sys.modules['pymupdf'] = None\n"
        "from tableproof.pdf_extractor import extract_words, render_page\n"
        "from tableproof.table_reconstructor import reconstruct_table_grid\n"
        "from tableproof.models import WordBox\n"
        f"pw = extract_words({str(pdf)!r}, 1)\n"
        "assert pw.words\n"
        "img = render_page(" + repr(str(pdf)) + ", 1, dpi=72)\n"
        "grid = reconstruct_table_grid([WordBox(i, w.text, w.bbox) for i, w in enumerate(pw.words)])\n"
        "assert grid.row_count > 0\n"
    )
    subprocess.run([sys.executable, "-c", code], check=True)
