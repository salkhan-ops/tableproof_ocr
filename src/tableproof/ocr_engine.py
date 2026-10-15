from __future__ import annotations

import csv
import json
import subprocess
from pathlib import Path
from typing import Any

from tableproof.ocr_models import OcrResult, OcrWordBox


class OcrEngineError(RuntimeError):
    pass


class TesseractOcrEngine:
    name = "tesseract"

    def __init__(self, executable: str = "tesseract") -> None:
        self.executable = executable

    def version(self) -> str:
        try:
            result = subprocess.run(
                [self.executable, "--version"],
                check=True,
                text=True,
                capture_output=True,
            )
        except (OSError, subprocess.CalledProcessError) as exc:
            raise OcrEngineError(f"Tesseract is not available: {exc}") from exc
        first = result.stdout.splitlines()[0] if result.stdout else "tesseract unknown"
        return first.replace("tesseract", "").strip()

    def recognise(self, image_path: Path | str, page_number: int = 1) -> OcrResult:
        image_path = Path(image_path)
        if not image_path.exists():
            raise OcrEngineError(f"OCR image does not exist: {image_path}")
        try:
            result = subprocess.run(
                [
                    self.executable,
                    str(image_path),
                    "stdout",
                    "--psm",
                    "6",
                    "tsv",
                ],
                check=True,
                text=True,
                capture_output=True,
            )
        except subprocess.CalledProcessError as exc:
            raise OcrEngineError(f"Tesseract failed for {image_path}: {exc.stderr}") from exc

        rows = list(csv.DictReader(result.stdout.splitlines(), delimiter="\t"))
        words: list[OcrWordBox] = []
        for index, row in enumerate(rows):
            text = " ".join(str(row.get("text", "")).split())
            if not text:
                continue
            confidence = _float(row.get("conf"), -1.0)
            if confidence < 0:
                continue
            x = _float(row.get("left"), 0.0)
            y = _float(row.get("top"), 0.0)
            width = _float(row.get("width"), 0.0)
            height = _float(row.get("height"), 0.0)
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
                    source_engine=self.name,
                    source_token_id=f"tsv-{index}",
                    line_hint=f"{row.get('block_num', '')}:{row.get('par_num', '')}:{row.get('line_num', '')}",
                    rotation=0.0,
                    raw_text=row.get("text", ""),
                )
            )
        return OcrResult(
            engine=self.name,
            engine_version=self.version(),
            image_path=str(image_path),
            page_number=page_number,
            words=words,
            raw_response={"format": "tesseract_tsv", "rows": rows, "stderr": result.stderr},
        )

    def save_raw(self, result: OcrResult, path: Path | str) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(result.to_dict(), indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _float(value: Any, default: float) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default
