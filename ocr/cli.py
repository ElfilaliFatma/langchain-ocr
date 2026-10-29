"""CLI: extract, understand, ask, or batch-process a folder."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .config import Settings
from .engines import get_engine
from .pipeline import DocumentPipeline, build_default_pipeline

IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp", ".pdf"}


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(prog="ocr", description="OCR + LLM document analysis.")
    parser.add_argument("image", nargs="?", type=Path, help="Image or PDF to analyse.")
    parser.add_argument("--folder", type=Path, help="Batch-process every image in a folder.")
    parser.add_argument("--engine", default="rapidocr", choices=["rapidocr", "tesseract"])
    parser.add_argument("--lang", default="en", help="Language of the document (OCR hint).")
    parser.add_argument("--answer-lang", default="en", help="Language of the LLM output.")
    parser.add_argument("--enhance", action="store_true", help="Preprocess (contrast, upscale).")
    parser.add_argument("--upscale", type=float, default=1.0)
    parser.add_argument("--text-only", action="store_true", help="Skip the LLM stage.")
    parser.add_argument("--ask", help="Ask a question about the document.")
    parser.add_argument("--json", action="store_true", help="Print JSON.")
    parser.add_argument("--out", type=Path, help="Write JSON to this file.")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    settings = Settings.from_env()

    if args.folder:
        engine = get_engine(args.engine)
        pipeline = DocumentPipeline(engine, None, settings)
        rows = []
        files = sorted(
            path for path in args.folder.iterdir()
            if path.suffix.lower() in IMAGE_SUFFIXES
        )
        for path in files:
            ocr = pipeline.extract(path, args.lang, enhance=args.enhance, upscale=args.upscale)
            rows.append(
                {
                    "file": path.name,
                    "chars": len(ocr.text),
                    "mean_confidence": round(ocr.mean_confidence, 3),
                    "low_confidence_lines": len(ocr.low_confidence_words),
                    "duration_ms": ocr.duration_ms,
                }
            )
            print(f"{path.name:<28} {len(ocr.text):>5} chars  "
                  f"conf={ocr.mean_confidence:.2f}  {ocr.duration_ms:.0f} ms")
        if args.out:
            args.out.write_text(json.dumps(rows, indent=2), encoding="utf-8")
        return 0

    if not args.image:
        print("Provide an image path or --folder.")
        return 2

    if args.text_only:
        pipeline = DocumentPipeline(get_engine(args.engine), None, settings)
    else:
        pipeline = build_default_pipeline(settings)
        pipeline.engine = get_engine(args.engine)

    ocr = pipeline.extract(args.image, args.lang, enhance=args.enhance, upscale=args.upscale)
    print(f"[{ocr.engine}] {ocr.image_size[0]}x{ocr.image_size[1]}px · "
          f"{len(ocr.words)} lines · confidence {ocr.mean_confidence:.2f} · "
          f"{ocr.duration_ms:.0f} ms")
    print("-" * 60)
    print(ocr.text)
    print("-" * 60)

    if ocr.low_confidence_words:
        print("Low confidence lines:")
        for word in ocr.low_confidence_words:
            print(f"  {word.confidence:.2f}  {word.text}")

    if args.text_only or not ocr.text.strip():
        return 0

    if args.ask:
        print("\nAnswer:")
        print(pipeline.ask(ocr, args.ask, args.answer_lang))
        return 0

    outcome = pipeline.understand(ocr, args.answer_lang)
    if outcome.error:
        print(f"[LLM stage failed] {outcome.error}")
        return 1
    document = outcome.document
    if args.json:
        print(document.model_dump_json(indent=2))
    else:
        print(f"\nType: {document.document_type} · language: {document.language}")
        print(f"Summary: {document.summary}")
        if document.fields:
            print("\nFields:")
            for field in document.fields:
                print(f"  {field.key:<20} {field.value}")
        for warning in document.warnings:
            print(f"  ! {warning}")
    if args.out:
        args.out.write_text(
            json.dumps(
                {"ocr": {"text": ocr.text, "confidence": ocr.mean_confidence},
                 "document": document.model_dump()},
                ensure_ascii=False, indent=2,
            ),
            encoding="utf-8",
        )
        print(f"\nSaved to {args.out}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
