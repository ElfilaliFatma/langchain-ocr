"""Pluggable OCR backends.

Design: an abstract ``OcrEngine`` with two implementations. The rest of the
application only depends on the protocol, so swapping backends (or adding a
cloud OCR later) never touches the LangChain layer.

* ``RapidOcrEngine`` -- PP-OCR models via onnxruntime. Pure pip install, no
  system binary, good on screenshots and scans. Default.
* ``TesseractEngine`` -- requires the Tesseract binary; useful when it is
  already installed, and stronger on some printed fonts.
"""

from __future__ import annotations

import io
import shutil
import time
from abc import ABC, abstractmethod
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps

from .schema import OcrResult, OcrWord


class OcrEngine(ABC):
    """Protocol every backend implements."""

    name = "base"

    @abstractmethod
    def extract(self, image: Image.Image, language: str = "en") -> OcrResult:
        ...

    def extract_path(self, path: str | Path, language: str = "en") -> OcrResult:
        return self.extract(load_image(path), language)


class RapidOcrEngine(OcrEngine):
    name = "rapidocr"

    def __init__(self) -> None:
        self._engine = None

    def _lazy_engine(self):
        """Load ONNX models on first use.

        Importing RapidOCR loads several models (~15 MB) and takes a few
        seconds; deferring it keeps the CLI's ``--help`` and the tests fast.
        """

        if self._engine is None:
            from rapidocr_onnxruntime import RapidOCR

            self._engine = RapidOCR()
        return self._engine

    def extract(self, image: Image.Image, language: str = "en") -> OcrResult:
        started = time.perf_counter()
        engine = self._lazy_engine()
        array = np.array(image.convert("RGB"))
        raw, _ = engine(array)
        words = [
            OcrWord(
                text=str(item[1]).strip(),
                confidence=float(item[2]),
                box=box_from_points(item[0]),
            )
            for item in (raw or [])
            if str(item[1]).strip()
        ]
        duration = (time.perf_counter() - started) * 1000
        return OcrResult(
            text=join_lines(words),
            words=words,
            engine=self.name,
            duration_ms=round(duration, 1),
            image_size=image.size,
        )


class TesseractEngine(OcrEngine):
    name = "tesseract"

    LANGUAGE_MAP = {"en": "eng", "fr": "fra", "ar": "ara", "de": "deu", "es": "spa"}

    def __init__(self, binary: str | None = None) -> None:
        self.binary = binary or shutil.which("tesseract")
        if not self.binary:
            raise RuntimeError(
                "Tesseract is not installed. Use the rapidocr engine or install "
                "Tesseract and set TESSERACT_CMD."
            )

    def extract(self, image: Image.Image, language: str = "en") -> OcrResult:
        import pytesseract

        started = time.perf_counter()
        pytesseract.pytesseract.tesseract_cmd = self.binary
        lang = self.LANGUAGE_MAP.get(language, "eng")
        data = pytesseract.image_to_data(
            image, lang=lang, output_type=pytesseract.Output.DICT
        )
        words: list[OcrWord] = []
        line_buffer: dict[tuple[int, int, int], list[int]] = {}
        for index, text in enumerate(data["text"]):
            if not str(text).strip():
                continue
            confidence = float(data["conf"][index])
            if confidence < 0:
                continue
            key = (data["block_num"][index], data["par_num"][index], data["line_num"][index])
            line_buffer.setdefault(key, []).append(index)
            words.append(
                OcrWord(
                    text=str(text).strip(),
                    confidence=confidence / 100,
                    box=(
                        float(data["left"][index]),
                        float(data["top"][index]),
                        float(data["left"][index] + data["width"][index]),
                        float(data["top"][index] + data["height"][index]),
                    ),
                )
            )
        grouped = group_words_into_lines(words)
        duration = (time.perf_counter() - started) * 1000
        return OcrResult(
            text=join_lines(grouped),
            words=grouped,
            engine=self.name,
            duration_ms=round(duration, 1),
            image_size=image.size,
        )


def get_engine(name: str = "rapidocr", **kwargs) -> OcrEngine:
    normalized = (name or "rapidocr").lower()
    if normalized in {"rapidocr", "rapid", "onnx"}:
        return RapidOcrEngine()
    if normalized in {"tesseract", "tess"}:
        return TesseractEngine(**kwargs)
    raise ValueError(f"Unknown OCR engine '{name}'. Use 'rapidocr' or 'tesseract'.")


# ---------------------------------------------------------------------- helpers
def load_image(path: str | Path) -> Image.Image:
    """Open an image or the first page of a PDF."""

    path = Path(path)
    if path.suffix.lower() == ".pdf":
        return _first_pdf_page(path)
    return Image.open(path)


def _first_pdf_page(path: Path) -> Image.Image:
    try:
        import pypdfium2 as pdfium  # optional, better rendering
    except ImportError:
        pdfium = None
    if pdfium is not None:
        document = pdfium.PdfDocument(str(path))
        page = document[0]
        return page.render(scale=2).to_pil()
    raise RuntimeError(
        "PDF input needs pypdfium2 (pip install pypdfium2) or a PNG/JPEG export."
    )


def preprocess(image: Image.Image, *, grayscale: bool = True, upscale: float = 1.0,
               auto_contrast: bool = True) -> Image.Image:
    """Light preprocessing that measurably helps small or low-contrast scans."""

    result = image.convert("L") if grayscale else image.convert("RGB")
    if upscale and upscale != 1.0:
        width, height = result.size
        result = result.resize((int(width * upscale), int(height * upscale)))
    if auto_contrast:
        result = ImageOps.autocontrast(result)
    return result


def box_from_points(points) -> tuple[float, float, float, float]:
    xs = [float(point[0]) for point in points]
    ys = [float(point[1]) for point in points]
    return (min(xs), min(ys), max(xs), max(ys))


def group_words_into_lines(words: list[OcrWord]) -> list[OcrWord]:
    """Merge word boxes that share a visual line (Tesseract emits words)."""

    if not words:
        return []
    ordered = sorted(words, key=lambda w: (round(w.y / 10), w.x))
    lines: list[list[OcrWord]] = []
    for word in ordered:
        placed = False
        for line in lines:
            reference = line[0]
            tolerance = max(6.0, reference.height * 0.5)
            if abs(word.y - reference.y) <= tolerance:
                line.append(word)
                placed = True
                break
        if not placed:
            lines.append([word])
    merged = []
    for line in lines:
        line.sort(key=lambda w: w.x)
        merged.append(
            OcrWord(
                text=" ".join(w.text for w in line),
                confidence=sum(w.confidence for w in line) / len(line),
                box=(
                    min(w.box[0] for w in line),
                    min(w.box[1] for w in line),
                    max(w.box[2] for w in line),
                    max(w.box[3] for w in line),
                ),
            )
        )
    return merged


def join_lines(words: list[OcrWord]) -> str:
    return "\n".join(word.text for word in words)


def image_to_bytes(image: Image.Image, fmt: str = "PNG") -> bytes:
    buffer = io.BytesIO()
    image.save(buffer, format=fmt)
    return buffer.getvalue()
