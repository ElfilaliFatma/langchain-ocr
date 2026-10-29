"""Measure OCR quality against ground-truth text.

Metrics:

* character error rate (CER) and word error rate (WER) via Levenshtein distance,
* exact-field recall: how many expected invoice fields appear verbatim,
* throughput in milliseconds per image.

Levenshtein is implemented locally to keep the dependency list small.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from ocr.engines import get_engine, preprocess  # noqa: E402
from PIL import Image  # noqa: E402

SAMPLES = ROOT / "samples"


def levenshtein(source: str, target: str) -> int:
    if source == target:
        return 0
    if not source:
        return len(target)
    if not target:
        return len(source)
    previous = list(range(len(target) + 1))
    for i, source_char in enumerate(source, start=1):
        current = [i]
        for j, target_char in enumerate(target, start=1):
            current.append(
                min(
                    previous[j] + 1,
                    current[j - 1] + 1,
                    previous[j - 1] + (source_char != target_char),
                )
            )
        previous = current
    return previous[-1]


def error_rates(hypothesis: str, reference: str) -> tuple[float, float]:
    cer = levenshtein(hypothesis, reference) / max(len(reference), 1)
    hyp_words = hypothesis.split()
    ref_words = reference.split()
    # word-level distance approximated on joined tokens
    wer = levenshtein(" ".join(hyp_words), " ".join(ref_words)) / max(
        len(" ".join(ref_words)), 1
    )
    return round(cer, 4), round(wer, 4)


def flatten(text: str) -> str:
    """Collapse all whitespace to single spaces.

    PP-OCR emits one box per visual line, so a multi-column table arrives as
    ``Description`` / ``Qty`` / ``Unit`` / ``Total`` on separate lines while the
    ground truth has them on one. Comparing raw strings therefore measures line
    segmentation, not recognition. The flattened CER isolates recognition
    quality; the raw CER is kept to show the segmentation effect.
    """

    return " ".join(text.split())


def field_recall(ocr_text: str, expected: dict[str, str]) -> dict[str, bool]:
    normalized = " ".join(ocr_text.split()).lower()
    return {key: value.lower() in normalized for key, value in expected.items()}


INVOICE_FIELDS = {
    "supplier": "ACME SUPPLIES SARL",
    "invoice_number": "INV-2026-0142",
    "due_date": "2026-11-12",
    "total": "170.28",
}


def main() -> int:
    engine = get_engine("rapidocr")
    report = {"engine": engine.name, "samples": [], "variants": {}}

    for name, expected_fields in (
        ("invoice", INVOICE_FIELDS),
        ("receipt", {"receipt_number": "4471", "total": "7.50"}),
    ):
        image_path = SAMPLES / f"{name}.png"
        ground_truth = (SAMPLES / f"{name}_expected.txt").read_text(encoding="utf-8")
        image = Image.open(image_path)
        # Variant A: raw image. Variant B: 2x upscale + autocontrast, which
        # restores the intra-line spaces that PP-OCR drops on wide lines.
        variants = {"raw": image, "upscaled_x2": preprocess(image, upscale=2.0)}
        for variant_name, variant_image in variants.items():
            result = engine.extract(variant_image)
            cer, wer = error_rates(result.text, ground_truth)
            cer_flat = round(
                levenshtein(flatten(result.text), flatten(ground_truth))
                / max(len(flatten(ground_truth)), 1),
                4,
            )
            fields = field_recall(result.text, expected_fields)
            record = {
                "file": image_path.name,
                "variant": variant_name,
                "cer": cer,
                "cer_flat": cer_flat,
                "wer": wer,
                "mean_confidence": round(result.mean_confidence, 3),
                "duration_ms": result.duration_ms,
                "fields": fields,
                "field_recall": round(sum(fields.values()) / len(fields), 3),
                "text": result.text,
            }
            report["samples"].append(record)
            report["variants"].setdefault(variant_name, []).append(record["cer_flat"])
            print(f"{name:<8} {variant_name:<12} CER={cer:.3f} CERflat={cer_flat:.3f} "
                  f"WER={wer:.3f} "
                  f"conf={result.mean_confidence:.2f} "
                  f"fields={sum(fields.values())}/{len(fields)}")

    report["average_cer"] = round(
        sum(sample["cer"] for sample in report["samples"]) / len(report["samples"]), 4
    )
    report["average_cer_flat"] = round(
        sum(sample["cer_flat"] for sample in report["samples"]) / len(report["samples"]), 4
    )
    report["average_field_recall"] = round(
        sum(sample["field_recall"] for sample in report["samples"]) / len(report["samples"]), 3
    )
    report["variant_average_cer"] = {
        name: round(sum(values) / len(values), 4)
        for name, values in report["variants"].items()
    }
    output = ROOT / "docs" / "ocr_evaluation.json"
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in report.items() if k != "samples"}, indent=2))
    print(f"Report written to {output.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
