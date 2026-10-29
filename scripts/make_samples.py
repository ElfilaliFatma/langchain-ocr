"""Generate the sample document images used in the demo and the report.

Rendering with PIL keeps the repository text-friendly (no binary blobs in git)
and makes the expected text exactly known, which is what the OCR accuracy test
compares against.
"""

from __future__ import annotations

from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
SAMPLES = ROOT / "samples"

INVOICE_LINES = [
    "ACME SUPPLIES SARL",
    "12 rue de la Paix, 75002 Paris",
    "Invoice number: INV-2026-0142",
    "Invoice date: 2026-10-12",
    "Due date: 2026-11-12",
    "",
    "Description            Qty    Unit      Total",
    "Wireless keyboard       2    45.00      90.00",
    "USB-C hub              1    39.90      39.90",
    "Shipping                               12.00",
    "",
    "Subtotal                              141.90",
    "VAT 20%                                28.38",
    "TOTAL DUE                             170.28 EUR",
]

RECEIPT_LINES = [
    "CAFE DU COIN",
    "Receipt #4471",
    "2026-10-20  09:14",
    "",
    "Espresso            2.50",
    "Croissant           1.80",
    "Orange juice        3.20",
    "",
    "TOTAL               7.50 EUR",
    "Paid by card",
    "Thank you!",
]


def font(size: int) -> ImageFont.FreeTypeFont:
    for name in ("arial.ttf", "DejaVuSans.ttf", "calibri.ttf"):
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            continue
    return ImageFont.load_default()


def render(lines: list[str], path: Path, size=(900, 620), font_size=26, spacing=32) -> None:
    image = Image.new("RGB", size, "white")
    draw = ImageDraw.Draw(image)
    text_font = font(font_size)
    y = 30
    for line in lines:
        draw.text((40, y), line, fill="black", font=text_font)
        y += spacing
    image.save(path)
    print(f"wrote {path.name}")


def main() -> None:
    SAMPLES.mkdir(exist_ok=True)
    render(INVOICE_LINES, SAMPLES / "invoice.png")
    render(RECEIPT_LINES, SAMPLES / "receipt.png", size=(560, 460), font_size=24, spacing=30)
    (SAMPLES / "invoice_expected.txt").write_text("\n".join(INVOICE_LINES), encoding="utf-8")
    (SAMPLES / "receipt_expected.txt").write_text("\n".join(RECEIPT_LINES), encoding="utf-8")


if __name__ == "__main__":
    main()
