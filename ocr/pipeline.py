"""End-to-end pipeline: image -> OCR -> LangChain interpretation."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.output_parsers import StrOutputParser
from PIL import Image

from .config import Settings
from .engines import OcrEngine, get_engine, load_image, preprocess
from .parsing import extract_json
from .prompts import cleanup_prompt, qa_prompt, understanding_prompt
from .schema import OcrResult, StructuredDocument

LANGUAGE_NAMES = {"en": "English", "fr": "French", "ar": "Arabic", "es": "Spanish"}


def build_llm(settings: Settings, json_mode: bool = True) -> BaseChatModel:
    if settings.provider == "ollama":
        from langchain_ollama import ChatOllama

        return ChatOllama(
            model=settings.model,
            base_url=settings.base_url,
            temperature=settings.temperature,
            num_predict=settings.max_tokens,
            format="json" if json_mode else None,
        )
    if settings.provider == "openai":
        if not settings.api_key:
            raise RuntimeError("OPENAI_API_KEY is required when LLM_PROVIDER=openai")
        from langchain_openai import ChatOpenAI

        return ChatOpenAI(
            model=settings.model,
            api_key=settings.api_key,
            temperature=settings.temperature,
            max_tokens=settings.max_tokens,
            timeout=settings.timeout,
            model_kwargs={"response_format": {"type": "json_object"}} if json_mode else {},
        )
    raise ValueError(f"Unsupported provider '{settings.provider}'.")


@dataclass
class PipelineResult:
    ocr: OcrResult
    document: StructuredDocument | None
    error: str | None = None


class DocumentPipeline:
    """Orchestrates extraction and interpretation.

    The OCR engine and the LangChain model are injected, so the pipeline can be
    exercised with a fake engine and a scripted model in tests.
    """

    def __init__(
        self,
        engine: OcrEngine,
        llm: BaseChatModel | None,
        settings: Settings | None = None,
        text_llm: BaseChatModel | None = None,
    ):
        self.engine = engine
        self.llm = llm
        self.settings = settings or Settings()
        # The cleanup chain returns plain prose, so it must not inherit the
        # JSON-constrained decoding used by the structured-understanding chain.
        self.text_llm = text_llm or llm
        if llm is not None:
            self.understand_chain = understanding_prompt | llm | StrOutputParser()
            # Q&A and cleanup both answer in prose, so they must not inherit the
            # JSON-constrained decoding used by the structured chain.
            self.qa_chain = qa_prompt | self.text_llm | StrOutputParser()
            self.cleanup_chain = cleanup_prompt | self.text_llm | StrOutputParser()
        else:
            self.understand_chain = self.qa_chain = self.cleanup_chain = None

    # ------------------------------------------------------------------- stage 1
    def extract(
        self,
        source: str | Path | Image.Image,
        language: str = "en",
        enhance: bool = False,
        upscale: float = 1.0,
    ) -> OcrResult:
        image = source if isinstance(source, Image.Image) else load_image(source)
        if enhance or upscale != 1.0:
            image = preprocess(image, upscale=upscale)
        return self.engine.extract(image, language)

    # ------------------------------------------------------------------- stage 2
    def understand(
        self, ocr: OcrResult, answer_language: str = "en"
    ) -> PipelineResult:
        if not ocr.text.strip():
            return PipelineResult(ocr, None, "OCR produced no text.")
        if self.understand_chain is None:
            raise RuntimeError("This pipeline was built without an LLM.")
        if ocr.mean_confidence and ocr.mean_confidence < self.settings.min_confidence:
            # Still analyse, but tell the user the extraction is unreliable.
            pass
        try:
            raw = self.understand_chain.invoke(
                {
                    "text": ocr.text,
                    "answer_language": LANGUAGE_NAMES.get(answer_language, "English"),
                }
            )
        except Exception as exc:
            return PipelineResult(ocr, None, f"LLM call failed: {exc}")

        payload = extract_json(raw)
        if payload is None:
            return PipelineResult(ocr, None, "The model did not return valid JSON.")

        payload.setdefault("document_type", "unknown")
        warnings = payload.get("warnings") or []
        if isinstance(warnings, str):
            warnings = [warnings]
        payload["warnings"] = warnings
        if ocr.low_confidence_words:
            payload["warnings"].append(
                f"{len(ocr.low_confidence_words)} OCR line(s) below "
                f"{self.settings.min_confidence:.0%} confidence; verify values."
            )
        try:
            document = StructuredDocument(**payload)
        except Exception as exc:
            return PipelineResult(ocr, None, f"Schema validation failed: {exc}")
        return PipelineResult(ocr, document)

    def run(
        self,
        source: str | Path | Image.Image,
        language: str = "en",
        answer_language: str = "en",
        enhance: bool = False,
    ) -> PipelineResult:
        ocr = self.extract(source, language, enhance=enhance)
        return self.understand(ocr, answer_language)

    # ------------------------------------------------------------------- stage 3
    def ask(self, ocr: OcrResult, question: str, answer_language: str = "en") -> str:
        if self.qa_chain is None:
            raise RuntimeError("This pipeline was built without an LLM.")
        if not question.strip():
            raise ValueError("Question is empty.")
        return self.qa_chain.invoke(
            {
                "text": ocr.text,
                "question": question,
                "answer_language": LANGUAGE_NAMES.get(answer_language, "English"),
            }
        ).strip()

    def clean_text(self, ocr: OcrResult) -> str:
        """LLM post-correction of OCR artefacts (opt-in: it costs one call)."""

        if self.cleanup_chain is None:
            raise RuntimeError("This pipeline was built without an LLM.")
        return self.cleanup_chain.invoke({"text": ocr.text}).strip()


def build_default_pipeline(settings: Settings | None = None, with_llm: bool = True):
    settings = settings or Settings.from_env()
    engine = get_engine(settings.engine, binary=settings.tesseract_cmd)
    if not with_llm:
        return DocumentPipeline(engine, None, settings)
    return DocumentPipeline(
        engine,
        build_llm(settings, json_mode=True),
        settings,
        text_llm=build_llm(settings, json_mode=False),
    )
