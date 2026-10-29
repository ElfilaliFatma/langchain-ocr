from __future__ import annotations
import importlib.util
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location(
    "evaluate_ocr", ROOT / "scripts" / "evaluate_ocr.py"
)
evaluate_ocr = importlib.util.module_from_spec(spec)
spec.loader.exec_module(evaluate_ocr)
def test_levenshtein_basics():
    assert evaluate_ocr.levenshtein("abc", "abc") == 0
    assert evaluate_ocr.levenshtein("", "abc") == 3
    assert evaluate_ocr.levenshtein("abc", "") == 3
    assert evaluate_ocr.levenshtein("kitten", "sitting") == 3
def test_error_rates_perfect_match():
    cer, wer = evaluate_ocr.error_rates("total 10", "total 10")
    assert cer == 0.0 and wer == 0.0
def test_error_rates_partial():
    cer, _ = evaluate_ocr.error_rates("t0tal 1O", "total 10")
    assert 0 < cer < 0.5
def test_field_recall_normalises_whitespace():
    recall = evaluate_ocr.field_recall("TOTAL\n  DUE  170.28", {"total": "total due 170.28"})
    assert recall["total"] is True
