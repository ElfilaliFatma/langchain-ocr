from __future__ import annotations

import pytest
from PIL import Image, ImageDraw, ImageFont

from ocr.engines import (
    OcrEngine,
    get_engine,
    group_words_into_lines,
    join_lines,
    preprocess,
)
from ocr.schema import OcrResult, OcrWord


def make_image(text: str = "Invoice total: 170.28 EUR") -> Image.Image:
    image = Image.new("RGB", (700, 120), "white")
    draw = ImageDraw.Draw(image)
    try:
        font = ImageFont.truetype("arial.ttf", 34)
    except OSError:
        font = ImageFont.load_default()
    draw.text((20, 40), text, fill="black", font=font)
    return image


def test_get_engine_dispatch():
    assert get_engine("rapidocr").name == "rapidocr"
    assert get_engine("onnx").name == "rapidocr"
    with pytest.raises(ValueError):
        get_engine("magic")


def test_tesseract_reports_missing_binary(monkeypatch):
    import ocr.engines as engines

    monkeypatch.setattr(engines.shutil, "which", lambda name: None)
    with pytest.raises(RuntimeError, match="Tesseract is not installed"):
        engines.TesseractEngine()


def test_rapidocr_reads_rendered_text():
    engine = get_engine("rapidocr")
    result = engine.extract(make_image())
    assert isinstance(result, OcrResult)
    assert "170.28" in result.text
    assert result.mean_confidence > 0.5
    assert result.engine == "rapidocr"


def test_preprocess_changes_mode_and_size():
    image = Image.new("RGB", (100, 50), "white")
    processed = preprocess(image, upscale=2.0)
    assert processed.mode == "L"
    assert processed.size == (200, 100)


def test_group_words_into_lines_merges_same_baseline():
    words = [
        OcrWord("Invoice", 0.9, (10, 10, 80, 30)),
        OcrWord("total", 0.9, (85, 11, 130, 30)),
        OcrWord("170.28", 0.9, (10, 50, 90, 70)),
    ]
    lines = group_words_into_lines(words)
    assert [line.text for line in lines] == ["Invoice total", "170.28"]


def test_group_words_handles_empty_input():
    assert group_words_into_lines([]) == []


def test_join_lines_strips_nothing_but_adds_newlines():
    words = [OcrWord("a", 1.0), OcrWord("b", 1.0)]
    assert join_lines(words) == "a\nb"


def test_abstract_engine_cannot_be_instantiated():
    with pytest.raises(TypeError):
        OcrEngine()
