from __future__ import annotations

from typing import Any, List, Optional

import pytest
from langchain_core.callbacks import CallbackManagerForLLMRun
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from PIL import Image
from pydantic import Field

from ocr.config import Settings
from ocr.pipeline import DocumentPipeline
from ocr.schema import OcrResult, OcrWord, StructuredDocument


class FakeEngine:
    name = "fake"

    def __init__(self, text: str, confidence: float = 0.95):
        self.text = text
        self.confidence = confidence
        self.calls = 0

    def extract(self, image: Image.Image, language: str = "en") -> OcrResult:
        self.calls += 1
        words = [
            OcrWord(line, self.confidence, (0, index * 20, 200, index * 20 + 18))
            for index, line in enumerate(self.text.splitlines())
            if line.strip()
        ]
        return OcrResult(text=self.text, words=words, engine=self.name,
                         duration_ms=12.0, image_size=image.size)


class ScriptedChatModel(BaseChatModel):
    responses: List[str]
    calls: List[List[BaseMessage]] = Field(default_factory=list)

    @property
    def _llm_type(self) -> str:
        return "scripted"

    def _generate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> ChatResult:
        self.calls.append(list(messages))
        text = self.responses.pop(0) if self.responses else "{}"
        return ChatResult(generations=[ChatGeneration(message=AIMessage(content=text))])


INVOICE_JSON = """{"document_type": "invoice", "language": "fr", "summary": "Facture ACME.",
"fields": [{"key": "total", "value": "170.28 EUR", "confidence": 0.9},
{"key": "due_date", "value": "2026-11-12"}], "warnings": []}"""


def pipeline_with(*responses: str, text: str = "TOTAL DUE 170.28 EUR") -> DocumentPipeline:
    return DocumentPipeline(
        FakeEngine(text), ScriptedChatModel(responses=list(responses)), Settings()
    )


def test_extract_returns_ocr_result():
    pipeline = pipeline_with()
    result = pipeline.extract(Image.new("RGB", (100, 100), "white"))
    assert result.text == "TOTAL DUE 170.28 EUR"
    assert result.mean_confidence == 0.95
def test_understand_builds_structured_document():
    pipeline = pipeline_with(INVOICE_JSON)
    outcome = pipeline.understand(pipeline.extract(Image.new("RGB", (10, 10))))
    assert outcome.error is None
    assert isinstance(outcome.document, StructuredDocument)
    assert outcome.document.document_type == "invoice"
    assert outcome.document.as_dict()["total"] == "170.28 EUR"
def test_fenced_json_is_accepted():
    pipeline = pipeline_with(f"```json\n{INVOICE_JSON}\n```")
    outcome = pipeline.understand(pipeline.extract(Image.new("RGB", (10, 10))))
    assert outcome.document is not None
def test_invalid_json_reports_error():
    pipeline = pipeline_with("I cannot read this document.")
    outcome = pipeline.understand(pipeline.extract(Image.new("RGB", (10, 10))))
    assert outcome.document is None
    assert "JSON" in outcome.error
def test_empty_ocr_text_skips_llm():
    pipeline = pipeline_with(text="")
    outcome = pipeline.understand(pipeline.extract(Image.new("RGB", (10, 10))))
    assert outcome.document is None
    assert "no text" in outcome.error.lower()
    assert pipeline.llm.calls == []
def test_low_confidence_adds_warning():
    pipeline = DocumentPipeline(
        FakeEngine("TOTAL 170.28", confidence=0.3),
        ScriptedChatModel(responses=[INVOICE_JSON]),
        Settings(min_confidence=0.6),
    )
    outcome = pipeline.understand(pipeline.extract(Image.new("RGB", (10, 10))))
    assert any("confidence" in warning.lower() for warning in outcome.document.warnings)
def test_llm_failure_is_reported_not_raised():
    class Exploding(ScriptedChatModel):
        def _generate(self, *args, **kwargs):
            raise RuntimeError("ollama timeout")

    pipeline = DocumentPipeline(FakeEngine("text"), Exploding(responses=[]), Settings())
    outcome = pipeline.understand(pipeline.extract(Image.new("RGB", (10, 10))))
    assert outcome.document is None
    assert "ollama timeout" in outcome.error
def test_field_coercion_from_flat_dict():
    pipeline = pipeline_with('{"fields": {"total": "10", "supplier": "ACME"}}')
    outcome = pipeline.understand(pipeline.extract(Image.new("RGB", (10, 10))))
    assert outcome.document.as_dict() == {"total": "10", "supplier": "ACME"}
def test_warnings_string_becomes_list():
    pipeline = pipeline_with('{"warnings": "blurry scan"}')
    outcome = pipeline.understand(pipeline.extract(Image.new("RGB", (10, 10))))
    assert outcome.document.warnings == ["blurry scan"]
def test_prompt_contains_ocr_text_and_answer_language():
    model = ScriptedChatModel(responses=[INVOICE_JSON])
    pipeline = DocumentPipeline(FakeEngine("TOTAL 170.28"), model, Settings())
    pipeline.understand(pipeline.extract(Image.new("RGB", (10, 10))), answer_language="fr")
    human = model.calls[-1][-1].content
    system = model.calls[-1][0].content
    assert "<ocr_text>\nTOTAL 170.28\n</ocr_text>" in human
    assert "French" in system
def test_q_and_a_uses_only_document_text():
    model = ScriptedChatModel(responses=["The total is 170.28 EUR."])
    pipeline = DocumentPipeline(FakeEngine("TOTAL 170.28 EUR"), model, Settings())
    ocr = pipeline.extract(Image.new("RGB", (10, 10)))
    answer = pipeline.ask(ocr, "What is the total?")
    assert "170.28" in answer
    assert "What is the total?" in model.calls[-1][-1].content
def test_ask_rejects_empty_question():
    pipeline = pipeline_with("x")
    with pytest.raises(ValueError):
        pipeline.ask(pipeline.extract(Image.new("RGB", (10, 10))), "  ")
def test_pipeline_without_llm_cannot_understand():
    pipeline = DocumentPipeline(FakeEngine("text"), None, Settings())
    with pytest.raises(RuntimeError):
        pipeline.understand(pipeline.extract(Image.new("RGB", (10, 10))))


def test_cleanup_uses_text_model_not_json_model():
    """Regression: the cleanup chain must not inherit JSON-constrained decoding.

    Sharing one JSON-mode model made the repair chain answer with
    {"ocr_text": "..."} instead of repaired prose.
    """

    json_model = ScriptedChatModel(responses=[INVOICE_JSON])
    text_model = ScriptedChatModel(responses=["Invoice number: INV-2026-0142"])
    pipeline = DocumentPipeline(
        FakeEngine("Invoicenumber:INV-2026-0142"), json_model, Settings(), text_llm=text_model
    )
    repaired = pipeline.clean_text(pipeline.extract(Image.new("RGB", (10, 10))))
    assert repaired == "Invoice number: INV-2026-0142"
    assert json_model.calls == []


def test_qa_uses_text_model_not_json_model():
    """Regression: Q&A must answer in prose, not JSON."""

    json_model = ScriptedChatModel(responses=['{"VAT amount": "28.38"}'])
    text_model = ScriptedChatModel(responses=["The VAT amount is 28.38 EUR."])
    pipeline = DocumentPipeline(
        FakeEngine("VAT 20% 28.38"), json_model, Settings(), text_llm=text_model
    )
    ocr = pipeline.extract(Image.new("RGB", (10, 10)))
    answer = pipeline.ask(ocr, "What is the VAT amount?")
    assert answer == "The VAT amount is 28.38 EUR."
    assert json_model.calls == []
