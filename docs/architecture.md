# Architecture

```
            ┌──────────────┐
 image ───► │ preprocessing│  grayscale · autocontrast · upscale (optional)
            └──────┬───────┘
                   ▼
         ┌───────────────────┐
         │   OcrEngine       │  RapidOCR  |  Tesseract
         │  extract(image)   │
         └─────────┬─────────┘
                   ▼
              OcrResult
        (text, per-line boxes,
         confidence, timing)
                   │
        ┌──────────┼───────────────┐
        ▼          ▼               ▼
  understanding   Q&A          cleanup
     chain       chain          chain
        │
        ▼
  StructuredDocument
  (type, summary, fields,
   tables, warnings)
```

## Separation of concerns

| Layer | Module | Depends on |
| --- | --- | --- |
| Pixel handling | `engines.py` | PIL, numpy, onnxruntime |
| Semantics | `prompts.py`, `pipeline.py` | LangChain |
| Validation | `schema.py` | Pydantic |
| Surfaces | `cli.py`, `app.py` | everything |

The OCR stage never imports LangChain and the LLM stage never imports
onnxruntime. `tests/test_pipeline.py` proves the seam by using a fake engine
with a scripted model.
