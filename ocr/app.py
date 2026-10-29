"""Streamlit interface for OCR and document understanding."""

from __future__ import annotations

import json

import streamlit as st
from PIL import Image

from ocr.config import Settings
from ocr.engines import get_engine, load_image
from ocr.pipeline import DocumentPipeline, build_default_pipeline

st.set_page_config(page_title="OCR + LangChain", page_icon="📄", layout="wide")


@st.cache_resource(show_spinner=False)
def get_ocr(engine_name: str):
    return get_engine(engine_name)


@st.cache_resource(show_spinner=False)
def get_pipeline(engine_name: str, provider: str, model: str):
    settings = Settings.from_env()
    settings = Settings(
        provider=provider,
        model=model,
        base_url=settings.base_url,
        temperature=0.0,
        max_tokens=settings.max_tokens,
        timeout=settings.timeout,
        engine=engine_name,
        tesseract_cmd=settings.tesseract_cmd,
        min_confidence=settings.min_confidence,
        api_key=settings.api_key,
    )
    return build_default_pipeline(settings)


def main() -> None:
    st.title("📄 Simple OCR with LangChain")
    st.caption("RapidOCR extraction → structured understanding → document Q&A")

    with st.sidebar:
        st.header("Settings")
        engine_name = st.selectbox("OCR engine", ["rapidocr", "tesseract"])
        provider = st.selectbox("Provider", ["ollama", "openai"])
        model = st.text_input("Model", "mistral")
        enhance = st.toggle("Preprocess (grayscale + contrast)", value=False)
        upscale = st.slider("Upscale factor", 1.0, 3.0, 1.0, 0.5)
        answer_lang = st.selectbox("Answer language", ["en", "fr", "ar"])

    uploaded = st.file_uploader("Upload an image or PDF", type=["png", "jpg", "jpeg", "bmp", "pdf"])
    if uploaded is None:
        st.info("Upload a document to start. Sample images live in `samples/`.")
        return

    image = load_image_from_upload(uploaded)
    left, right = st.columns([1, 1])
    with left:
        st.image(image, caption="Input", use_container_width=True)

    with st.spinner("Running OCR…"):
        pipeline = DocumentPipeline(get_ocr(engine_name), None, Settings.from_env())
        ocr = pipeline.extract(image, enhance=enhance, upscale=upscale)

    with right:
        st.metric("Lines", len(ocr.words))
        st.metric("Mean confidence", f"{ocr.mean_confidence:.2f}")
        st.metric("Duration", f"{ocr.duration_ms:.0f} ms")
        if ocr.low_confidence_words:
            st.warning(f"{len(ocr.low_confidence_words)} low-confidence line(s)")

    tabs = st.tabs(["Extracted text", "Structured", "Ask a question"])
    with tabs[0]:
        edited = st.text_area("OCR text (editable before analysis)", value=ocr.text, height=320)
        st.download_button("Download .txt", edited.encode("utf-8"), file_name="ocr.txt")
    with tabs[1]:
        if st.button("Analyse document", type="primary"):
            with st.spinner("Interpreting with the LLM…"):
                full = get_pipeline(engine_name, provider, model)
                ocr.text = edited
                outcome = full.understand(ocr, answer_lang)
            if outcome.error:
                st.error(outcome.error)
            else:
                document = outcome.document
                st.subheader(f"{document.document_type.title()} · {document.language}")
                st.write(document.summary)
                if document.fields:
                    st.dataframe([f.model_dump() for f in document.fields],
                                 use_container_width=True, hide_index=True)
                for warning in document.warnings:
                    st.warning(warning)
                st.download_button(
                    "Download JSON",
                    json.dumps(document.model_dump(), ensure_ascii=False, indent=2),
                    file_name="document.json",
                )
    with tabs[2]:
        question = st.text_input("Question", placeholder="What is the total amount?")
        if st.button("Ask") and question.strip():
            with st.spinner("Thinking…"):
                full = get_pipeline(engine_name, provider, model)
                ocr.text = edited
                st.write(full.ask(ocr, question, answer_lang))


def load_image_from_upload(uploaded) -> Image.Image:
    """Streamlit uploads are file-like; PDfs need a temp file."""

    if uploaded.name.lower().endswith(".pdf"):
        import tempfile

        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as handle:
            handle.write(uploaded.read())
            return load_image(handle.name)
    return Image.open(uploaded)


if __name__ == "__main__":
    main()
