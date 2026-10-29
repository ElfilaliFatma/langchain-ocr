"""Settings for the OCR project."""

from __future__ import annotations

import os
from dataclasses import dataclass

try:
    from dotenv import load_dotenv

    load_dotenv()
except Exception:  # pragma: no cover
    pass


@dataclass(frozen=True)
class Settings:
    provider: str = "ollama"
    model: str = "mistral"
    base_url: str = "http://localhost:11434"
    temperature: float = 0.0
    max_tokens: int = 900
    timeout: int = 300
    engine: str = "rapidocr"
    tesseract_cmd: str | None = None
    min_confidence: float = 0.6
    api_key: str | None = None

    @classmethod
    def from_env(cls) -> "Settings":
        provider = os.getenv("LLM_PROVIDER", "ollama").strip().lower()
        default_model = "mistral" if provider == "ollama" else "gpt-4o-mini"
        return cls(
            provider=provider,
            model=os.getenv("LLM_MODEL", default_model),
            base_url=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"),
            temperature=float(os.getenv("LLM_TEMPERATURE", "0.0")),
            max_tokens=int(os.getenv("LLM_MAX_TOKENS", "900")),
            timeout=int(os.getenv("LLM_TIMEOUT", "300")),
            engine=os.getenv("OCR_ENGINE", "rapidocr"),
            tesseract_cmd=os.getenv("TESSERACT_CMD") or None,
            min_confidence=float(os.getenv("OCR_MIN_CONFIDENCE", "0.6")),
            api_key=os.getenv("OPENAI_API_KEY") if provider == "openai" else None,
        )
