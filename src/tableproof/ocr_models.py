from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

from tableproof.models import BoundingBox, WordBox


@dataclass(frozen=True)
class OcrWordBox:
    text: str
    x0: float
    y0: float
    x1: float
    y1: float
    confidence: float
    page_number: int
    source_engine: str
    source_token_id: str
    line_hint: str | None = None
    rotation: float | None = None
    raw_text: str | None = None

    @property
    def bounding_box(self) -> BoundingBox:
        return BoundingBox(self.x0, self.y0, self.x1, self.y1)

    def to_word_box(self, index: int) -> WordBox:
        return WordBox(index=index, text=self.text, bounding_box=self.bounding_box)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "OcrWordBox":
        return cls(
            text=str(payload.get("text", "")),
            x0=float(payload["x0"]),
            y0=float(payload["y0"]),
            x1=float(payload["x1"]),
            y1=float(payload["y1"]),
            confidence=float(payload.get("confidence", 0.0)),
            page_number=int(payload.get("page_number", 1)),
            source_engine=str(payload.get("source_engine", "")),
            source_token_id=str(payload.get("source_token_id", "")),
            line_hint=payload.get("line_hint"),
            rotation=float(payload["rotation"]) if payload.get("rotation") is not None else None,
            raw_text=payload.get("raw_text"),
        )


@dataclass(frozen=True)
class OcrResult:
    engine: str
    engine_version: str
    image_path: str
    page_number: int
    words: list[OcrWordBox]
    raw_response: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {
            "engine": self.engine,
            "engine_version": self.engine_version,
            "image_path": self.image_path,
            "page_number": self.page_number,
            "words": [word.to_dict() for word in self.words],
            "raw_response": self.raw_response,
        }
