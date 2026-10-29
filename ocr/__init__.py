"""OCR + LLM document understanding.

Two independent stages:

1. **Extraction** (``engines.py``) -- pixels to text. Pluggable: RapidOCR is
   the default because it is pure pip, Tesseract is supported when installed.
2. **Interpretation** (``pipeline.py``) -- text to structured meaning through
   LangChain chains (summary, key-value extraction, classification, Q&A).
"""

from .engines import OcrEngine, RapidOcrEngine, TesseractEngine, get_engine
from .pipeline import DocumentPipeline, build_default_pipeline
from .schema import ExtractedField, OcrResult, OcrWord, StructuredDocument

__all__ = [
    "DocumentPipeline",
    "ExtractedField",
    "OcrEngine",
    "OcrResult",
    "OcrWord",
    "RapidOcrEngine",
    "StructuredDocument",
    "TesseractEngine",
    "build_default_pipeline",
    "get_engine",
]

__version__ = "1.0.0"
