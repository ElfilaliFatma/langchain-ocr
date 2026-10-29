"""Data models for OCR output and structured interpretation."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

from pydantic import BaseModel, Field, field_validator


@dataclass
class OcrWord:
    """One recognised text line with its confidence and bounding box."""

    text: str
    confidence: float
    box: tuple[float, float, float, float] = (0.0, 0.0, 0.0, 0.0)

    @property
    def x(self) -> float:
        return self.box[0]

    @property
    def y(self) -> float:
        return self.box[1]

    @property
    def height(self) -> float:
        return self.box[3] - self.box[1]


@dataclass
class OcrResult:
    """Raw OCR output plus timing and engine provenance."""

    text: str
    words: list[OcrWord] = field(default_factory=list)
    engine: str = "unknown"
    duration_ms: float = 0.0
    image_size: tuple[int, int] = (0, 0)

    @property
    def mean_confidence(self) -> float:
        if not self.words:
            return 0.0
        return sum(word.confidence for word in self.words) / len(self.words)

    @property
    def low_confidence_words(self) -> list[OcrWord]:
        """Lines below 0.6 confidence, the ones worth re-checking."""

        return [word for word in self.words if word.confidence < 0.6]


class ExtractedField(BaseModel):
    key: str
    value: str
    confidence: float = Field(default=0.5, ge=0.0, le=1.0)


class StructuredDocument(BaseModel):
    """What the LLM understands about the document."""

    document_type: str = "unknown"
    summary: str = ""
    fields: list[ExtractedField] = Field(default_factory=list)
    tables: list[list[list[str]]] = Field(default_factory=list)
    language: str = "unknown"
    warnings: list[str] = Field(default_factory=list)

    @field_validator("fields", mode="before")
    @classmethod
    def _coerce_fields(cls, value):
        """Accept {"key": "total", "value": "10"} and {"total": "10"} shapes."""

        if not value:
            return []
        if isinstance(value, dict):
            return [{"key": str(k), "value": str(v)} for k, v in value.items()]
        cleaned = []
        for item in value:
            if isinstance(item, dict) and "key" in item:
                cleaned.append(item)
            elif isinstance(item, dict) and len(item) == 1:
                key, val = next(iter(item.items()))
                cleaned.append({"key": str(key), "value": str(val)})
            elif isinstance(item, str) and ":" in item:
                key, val = item.split(":", 1)
                cleaned.append({"key": key.strip(), "value": val.strip()})
        return cleaned

    def as_dict(self) -> dict[str, str]:
        return {f.key: f.value for f in self.fields}


DocumentKind = Literal["invoice", "receipt", "id_card", "letter", "form", "other"]
