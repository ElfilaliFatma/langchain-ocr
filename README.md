# Simple OCR with LangChain

Extract text from an image or scanned PDF, then let an LLM **understand** it:
classify the document, pull out key fields, summarise it, and answer questions
about its contents.

![pipeline](docs/architecture.md)

## Two-stage design

| Stage | Technology | Why |
| --- | --- | --- |
| Text extraction | RapidOCR (PP-OCR models on onnxruntime) | pip-only, no system binary, strong on screenshots |
| Alternative | Tesseract via pytesseract | already installed on many machines, good printed-text accuracy |
| Understanding | LangChain chain + structured JSON | turns raw text into fields, summary, warnings |
| Q&A | separate chain over the OCR text | grounded answers with "not in the document" behaviour |

The OCR engine is behind an abstract `OcrEngine`, so the LLM layer never knows
which backend produced the text.

## Quick start

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements-dev.txt
python scripts\make_samples.py          # regenerate the sample documents
ollama pull mistral
copy .env.example .env
```

### CLI

```powershell
python -m ocr.cli samples\invoice.png --json
python -m ocr.cli samples\receipt.png --text-only --enhance
python -m ocr.cli samples\invoice.png --ask "What is the total amount due?"
python -m ocr.cli --folder samples --out report.json
```

### Web UI

```powershell
streamlit run ocr/app.py
```

The UI shows the image, OCR metrics, editable text, the structured result and a
question box.

**Tip for small fonts:** PP-OCR squeezes wide lines into a 320-px recogniser
input and can then drop intra-line spaces. Pass `--upscale 2` (or enable
"Preprocess" in the UI) — measured flattened CER improves from 0.039 to 0.009
on the sample documents.

### Tests and evaluation

```powershell
pytest
python scripts\evaluate_ocr.py
```

## Layout

```
ocr/
  schema.py    OcrWord / OcrResult / StructuredDocument
  engines.py   OcrEngine protocol, RapidOCR, Tesseract, helpers
  prompts.py   understanding / Q&A / cleanup prompts
  parsing.py   tolerant JSON extraction
  pipeline.py  extract → understand → ask
  cli.py       command line
  app.py       Streamlit UI
scripts/
  make_samples.py   renders invoice.png / receipt.png + ground truth
  evaluate_ocr.py   CER, WER, field recall, latency
samples/
tests/
```

## How LangChain contributes

LangChain is the bridge between unstructured pixels and structured data. The
prompt template holds the JSON contract, the model is provider-agnostic, the
output parser plus a tolerant JSON extractor handle imperfect generations, and
the Pydantic schema validates the result. The same chain serves the CLI, the UI
and the Q&A feature with different prompts.

## Limitations

- OCR quality drops on handwriting, rotated scans and low-resolution photos.
- RapidOCR models are Latin-focused; Arabic and Chinese need a different model
  set (Tesseract with the right language data, or cloud OCR).
- Tables are returned as flat text; reconstructing columns is heuristic.
- The LLM can hallucinate a field if a value is illegible — mitigated by
  verbatim-copy instructions, confidence warnings and Pydantic validation,
  not eliminated.
- Only the first PDF page is rendered.

## Improvements

- Deskew and binarise with OpenCV; add automatic rotation detection.
- Detect tables with a layout model and pass cell coordinates to the LLM.
- Cache OCR results by image hash.
- Render all PDF pages and merge results.
- Add a human-in-the-loop review screen for low-confidence fields.
- Swap the LLM stage for `with_structured_output()` and compare reliability.

Full write-up: [`docs/REPORT.md`](docs/REPORT.md).
