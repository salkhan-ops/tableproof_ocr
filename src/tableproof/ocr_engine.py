"""Tesseract OCR as a local subprocess. Nothing is downloaded; nothing leaves the machine.

- Tesseract's TSV is read with ``csv.QUOTE_NONE``: a recognised ``"`` is text, not a CSV quote.
  (With default quoting, one ``"`` swallowed the following rows: most of a page was lost.)
- Languages are explicit (``-l eng`` by default) and ``tessdata_dir`` pins the language data.
- Results carry words (text, raw text, box, confidence). The full TSV is document content and
  is kept only when ``keep_raw_response=True`` (benchmarks), never by default.
- Errors never include paths, Tesseract's stderr or recognised text.
"""

from __future__ import annotations

import csv
import json
import subprocess
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from tableproof.ocr_models import OcrResult, OcrWordBox

_TIMEOUT_S = 120


class OcrEngineError(RuntimeError):
    pass


def parse_tsv(tsv: str, page_number: int, engine: str = "tesseract") -> list[OcrWordBox]:
    """Words from Tesseract TSV output. Pure function (no I/O)."""
    reader = csv.DictReader(tsv.splitlines(), delimiter="\t", quoting=csv.QUOTE_NONE)
    words: list[OcrWordBox] = []
    for index, row in enumerate(reader):
        raw = row.get("text") or ""
        text = " ".join(raw.split())
        if not text:
            continue
        confidence = _float(row.get("conf"), -1.0)
        if confidence < 0:
            continue
        x, y = _float(row.get("left"), 0.0), _float(row.get("top"), 0.0)
        width, height = _float(row.get("width"), 0.0), _float(row.get("height"), 0.0)
        if width <= 0 or height <= 0:
            continue
        words.append(
            OcrWordBox(
                text=text,
                x0=x,
                y0=y,
                x1=x + width,
                y1=y + height,
                confidence=confidence / 100.0,
                page_number=page_number,
                source_engine=engine,
                source_token_id=f"tsv-{index}",
                line_hint=f"{row.get('block_num', '')}:{row.get('par_num', '')}:{row.get('line_num', '')}",
                rotation=0.0,
                raw_text=raw,
            )
        )
    return words


class TesseractOcrEngine:
    name = "tesseract"

    def __init__(
        self,
        executable: str = "tesseract",
        *,
        languages: Sequence[str] = ("eng",),
        tessdata_dir: Path | str | None = None,
        page_segmentation_mode: int = 6,
        keep_raw_response: bool = False,
    ) -> None:
        if not languages:
            raise ValueError("at least one OCR language is required")
        self.executable = executable
        self.languages = tuple(languages)
        self.tessdata_dir = Path(tessdata_dir) if tessdata_dir else None
        self.page_segmentation_mode = page_segmentation_mode
        self.keep_raw_response = keep_raw_response

    def version(self) -> str:
        try:
            result = subprocess.run(
                [self.executable, "--version"],
                check=True,
                text=True,
                capture_output=True,
                timeout=_TIMEOUT_S,
            )
        except (OSError, subprocess.SubprocessError) as exc:
            raise OcrEngineError("Tesseract is not available") from exc
        first = result.stdout.splitlines()[0] if result.stdout else "tesseract unknown"
        return first.replace("tesseract", "").strip()

    def _command(self, source: str) -> list[str]:
        command = [
            self.executable,
            source,
            "stdout",
            "--psm",
            str(self.page_segmentation_mode),
            "-l",
            "+".join(self.languages),
        ]
        if self.tessdata_dir is not None:
            command += ["--tessdata-dir", str(self.tessdata_dir)]
        return [*command, "tsv"]

    def _run(self, source: str, data: bytes | None) -> str:
        try:
            result = subprocess.run(
                self._command(source),
                check=True,
                input=data,
                capture_output=True,
                timeout=_TIMEOUT_S,
            )
        except subprocess.CalledProcessError as exc:
            raise OcrEngineError(f"Tesseract failed (exit status {exc.returncode})") from None
        except (OSError, subprocess.TimeoutExpired) as exc:
            raise OcrEngineError("Tesseract could not be run") from exc
        return result.stdout.decode("utf-8", errors="replace")

    def recognise(self, image_path: Path | str, page_number: int = 1) -> OcrResult:
        image_path = Path(image_path)
        if not image_path.exists():
            raise OcrEngineError("The OCR image does not exist")
        return self._result(self._run(str(image_path), None), str(image_path), page_number)

    def recognise_bytes(self, image: bytes, page_number: int = 1) -> OcrResult:
        """OCR an encoded image (PNG, JPEG, TIFF) passed on stdin. Nothing is written to disk."""
        return self._result(self._run("stdin", image), "", page_number)

    def _result(self, tsv: str, image_path: str, page_number: int) -> OcrResult:
        raw: dict[str, Any] = {"format": "tesseract_tsv", "languages": list(self.languages)}
        if self.keep_raw_response:
            raw["rows"] = list(
                csv.DictReader(tsv.splitlines(), delimiter="\t", quoting=csv.QUOTE_NONE)
            )
        return OcrResult(
            engine=self.name,
            engine_version=self.version(),
            image_path=image_path,
            page_number=page_number,
            words=parse_tsv(tsv, page_number, self.name),
            raw_response=raw,
        )

    def save_raw(self, result: OcrResult, path: Path | str) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(result.to_dict(), indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )


def _float(value: Any, default: float) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default
