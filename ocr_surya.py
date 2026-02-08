#!/usr/bin/env python3
"""
Surya OCR Script for Historical Newspaper PDFs
===============================================
Uses Surya OCR (detection + layout + recognition) to extract text
from scanned historical newspaper pages with column-aware reading order.

Usage:
    # Process all PDFs in sample_data/
    python ocr_surya.py

    # Process a specific PDF
    python ocr_surya.py --input sample_data/1886-11-01_The-Liverpool-Echo_Monday_p01.pdf

    # Process a folder of PDFs
    python ocr_surya.py --input sample_data/

    # Use higher resolution for dense small text
    python ocr_surya.py --resolution 2048
"""

import argparse
import json
import os
import sys
import time
from glob import glob
from pathlib import Path

import pypdfium2 as pdfium
import torch
from PIL import Image
from pypdf import PdfReader
from surya.detection import DetectionPredictor
from surya.foundation import FoundationPredictor
from surya.layout import LayoutPredictor
from surya.recognition import RecognitionPredictor
from surya.settings import settings


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def get_pdf_page_count(pdf_path: str) -> int:
    """Return the number of pages in a PDF."""
    return len(PdfReader(pdf_path).pages)


def collect_pdfs(input_path: str) -> list[str]:
    """Resolve *input_path* to a list of PDF file paths."""
    p = Path(input_path)
    if p.is_file() and p.suffix.lower() == ".pdf":
        return [str(p)]
    if p.is_dir():
        pdfs = sorted(glob(str(p / "*.pdf")))
        if not pdfs:
            sys.exit(f"No PDF files found in {p}")
        return pdfs
    sys.exit(f"Invalid input: {input_path}")


def render_page_to_image(
    pdf_path: str, page_num: int, resolution: int = 1536
) -> Image.Image:
    """Render a single PDF page to a PIL Image using pypdfium2."""
    doc = pdfium.PdfDocument(pdf_path)
    page = doc[page_num - 1]  # 0-indexed
    width, height = page.get_size()
    longest = max(width, height)
    scale = resolution / longest
    bitmap = page.render(scale=scale)
    pil_image = bitmap.to_pil()
    doc.close()
    return pil_image


# ---------------------------------------------------------------------------
# Core OCR
# ---------------------------------------------------------------------------

def load_models():
    """Load all Surya predictors (models auto-download on first use)."""
    print("Loading Surya models ...")
    foundation = FoundationPredictor()
    det_predictor = DetectionPredictor()
    rec_predictor = RecognitionPredictor(foundation)
    layout_predictor = LayoutPredictor(
        FoundationPredictor(checkpoint=settings.LAYOUT_MODEL_CHECKPOINT)
    )
    print("Models loaded.\n")
    return {
        "detection": det_predictor,
        "recognition": rec_predictor,
        "layout": layout_predictor,
    }


def assemble_text_with_layout(rec, layout) -> str:
    """Use layout regions to group text lines by column, then read in order.

    Layout regions are sorted by their ``position`` field (Surya reading
    order).  Each recognised text line is assigned to the layout region
    whose bounding box contains its centre.  Lines that fall outside every
    region are collected at the end.
    """
    text_labels = {
        "text", "section-header", "title", "caption", "list-item",
        "page-header", "page-footer", "footnote",
    }
    text_regions = [
        b for b in layout.bboxes if b.label.lower() in text_labels
    ]
    # Sort regions by Surya's reading-order position
    text_regions.sort(key=lambda r: r.position)

    assigned = set()
    all_parts: list[str] = []

    for region in text_regions:
        rx1, ry1, rx2, ry2 = region.bbox
        region_lines = []
        for idx, line in enumerate(rec.text_lines):
            if idx in assigned:
                continue
            cx = (line.bbox[0] + line.bbox[2]) / 2
            cy = (line.bbox[1] + line.bbox[3]) / 2
            if rx1 <= cx <= rx2 and ry1 <= cy <= ry2:
                region_lines.append(line)
                assigned.add(idx)
        # Sort lines within a region top-to-bottom
        region_lines.sort(key=lambda l: l.bbox[1])
        text = "\n".join(l.text for l in region_lines)
        if text.strip():
            all_parts.append(text)

    # Collect orphan lines not matched to any region
    orphan_lines = [
        line for idx, line in enumerate(rec.text_lines) if idx not in assigned
    ]
    if orphan_lines:
        orphan_lines.sort(key=lambda l: (l.bbox[1], l.bbox[0]))
        all_parts.append("\n".join(l.text for l in orphan_lines))

    return "\n\n".join(all_parts)


def ocr_page(
    predictors: dict,
    pdf_path: str,
    page_num: int,
    resolution: int = 1536,
) -> dict:
    """Run Surya OCR on a single PDF page and return structured output."""
    image = render_page_to_image(pdf_path, page_num, resolution)

    # Layout detection — identifies columns, tables, figures, etc.
    layout_preds = predictors["layout"]([image])
    layout = layout_preds[0]

    has_table = any(
        b.label.lower() in ("table", "table_of_contents")
        for b in layout.bboxes
    )
    has_diagram = any(
        b.label.lower() in ("figure", "image", "picture")
        for b in layout.bboxes
    )

    # Text detection + recognition
    rec_preds = predictors["recognition"](
        [image], det_predictor=predictors["detection"]
    )
    rec = rec_preds[0]

    # Assemble text using layout-aware reading order
    natural_text = assemble_text_with_layout(rec, layout)

    return {
        "page": page_num,
        "primary_language": "en",
        "is_rotation_valid": True,
        "rotation_correction": 0,
        "is_table": has_table,
        "is_diagram": has_diagram,
        "natural_text": natural_text,
    }


def ocr_pdf(
    predictors: dict,
    pdf_path: str,
    resolution: int = 1536,
) -> list[dict]:
    """OCR every page of a PDF, returning a list of per-page results."""
    num_pages = get_pdf_page_count(pdf_path)
    print(f"  Pages: {num_pages}")

    results = []
    for page_num in range(1, num_pages + 1):
        t0 = time.time()
        page_result = ocr_page(predictors, pdf_path, page_num, resolution)
        elapsed = time.time() - t0
        print(f"  Page {page_num}/{num_pages} done ({elapsed:.1f}s)")
        results.append(page_result)

    return results


# ---------------------------------------------------------------------------
# Output writers
# ---------------------------------------------------------------------------

def save_results(pdf_path: str, pages: list[dict], output_dir: str):
    """Write OCR results to .json and .txt files in *output_dir*."""
    stem = Path(pdf_path).stem
    os.makedirs(output_dir, exist_ok=True)

    # --- Full structured JSON ---
    json_path = os.path.join(output_dir, f"{stem}.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(
            {"source_pdf": os.path.basename(pdf_path), "pages": pages},
            f,
            ensure_ascii=False,
            indent=2,
        )

    # --- Plain text (just the natural_text from each page) ---
    txt_path = os.path.join(output_dir, f"{stem}.txt")
    with open(txt_path, "w", encoding="utf-8") as f:
        for page in pages:
            f.write(f"===== Page {page.get('page', '?')} =====\n\n")
            f.write(page.get("natural_text", "") + "\n\n")

    print(f"  -> {json_path}")
    print(f"  -> {txt_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="OCR historical newspaper PDFs with Surya"
    )
    parser.add_argument(
        "--input",
        default="sample_data",
        help="Path to a PDF file or a directory of PDFs (default: sample_data/)",
    )
    parser.add_argument(
        "--output",
        default="ocr_output",
        help="Directory to write results (default: ocr_output/)",
    )
    parser.add_argument(
        "--resolution",
        type=int,
        default=1536,
        help="Longest-side pixel dimension for page rendering (default: 1536). "
             "Use 2048 for very dense small text.",
    )
    args = parser.parse_args()

    if torch.cuda.is_available():
        print(f"Using CUDA: {torch.cuda.get_device_name(0)}")
    else:
        print("WARNING: No GPU detected — inference will be very slow on CPU.")

    pdf_files = collect_pdfs(args.input)
    print(f"Found {len(pdf_files)} PDF(s) to process.\n")

    predictors = load_models()

    for i, pdf_path in enumerate(pdf_files, 1):
        print(f"[{i}/{len(pdf_files)}] {os.path.basename(pdf_path)}")
        pages = ocr_pdf(predictors, pdf_path, args.resolution)
        save_results(pdf_path, pages, args.output)
        print()

    print("All done! Results are in:", args.output)


if __name__ == "__main__":
    main()
