#!/usr/bin/env python3
"""
OCR Pipeline for Historical Newspaper PDFs
===========================================
Uses EasyOCR (CRAFT detection + CRNN recognition) to extract text
from scanned historical newspaper pages.

Usage:
    python ocr_pipeline.py
    python ocr_pipeline.py --input sample_data/1886-11-01_The-Liverpool-Echo_Monday_p01.pdf
    python ocr_pipeline.py --input sample_data/
    python ocr_pipeline.py --resolution 2048
"""

import argparse
import json
import os
import sys
import time
from glob import glob
from pathlib import Path

import easyocr
import numpy as np
import pypdfium2 as pdfium
from PIL import Image
from pypdf import PdfReader


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
) -> np.ndarray:
    """Render a single PDF page to a numpy array using pypdfium2."""
    doc = pdfium.PdfDocument(pdf_path)
    page = doc[page_num - 1]
    width, height = page.get_size()
    longest = max(width, height)
    scale = resolution / longest
    bitmap = page.render(scale=scale)
    pil_image = bitmap.to_pil()
    doc.close()
    return np.array(pil_image)


# ---------------------------------------------------------------------------
# Core OCR
# ---------------------------------------------------------------------------

def load_reader(languages: list[str], gpu: bool = True) -> easyocr.Reader:
    """Initialise the EasyOCR reader."""
    print(f"Loading EasyOCR (languages={languages}, gpu={gpu}) ...")
    reader = easyocr.Reader(languages, gpu=gpu)
    print("Reader loaded.\n")
    return reader


def ocr_page(
    reader: easyocr.Reader,
    pdf_path: str,
    page_num: int,
    resolution: int = 1536,
) -> dict:
    """Run EasyOCR on a single PDF page and return structured output."""
    img = render_page_to_image(pdf_path, page_num, resolution)

    # EasyOCR returns list of (bbox, text, confidence)
    results = reader.readtext(img, paragraph=True)

    natural_text = "\n".join(text for _, text, _ in results)

    return {
        "page": page_num,
        "primary_language": "en",
        "is_rotation_valid": True,
        "rotation_correction": 0,
        "is_table": False,
        "is_diagram": False,
        "natural_text": natural_text.strip(),
    }


def ocr_pdf(
    reader: easyocr.Reader,
    pdf_path: str,
    resolution: int = 1536,
) -> list[dict]:
    """OCR every page of a PDF, returning a list of per-page results."""
    num_pages = get_pdf_page_count(pdf_path)
    print(f"  Pages: {num_pages}")

    results = []
    for page_num in range(1, num_pages + 1):
        t0 = time.time()
        page_result = ocr_page(reader, pdf_path, page_num, resolution)
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

    json_path = os.path.join(output_dir, f"{stem}.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(
            {"source_pdf": os.path.basename(pdf_path), "pages": pages},
            f,
            ensure_ascii=False,
            indent=2,
        )

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
        description="OCR historical newspaper PDFs with EasyOCR"
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
        help="Longest-side pixel dimension for page rendering (default: 1536).",
    )
    parser.add_argument(
        "--languages",
        nargs="+",
        default=["en"],
        help="Language codes for OCR (default: en).",
    )
    args = parser.parse_args()

    gpu = False
    try:
        import torch
        gpu = torch.cuda.is_available()
        if gpu:
            print(f"Using CUDA: {torch.cuda.get_device_name(0)}")
        else:
            print("No GPU detected — using CPU.")
    except ImportError:
        print("PyTorch not found — using CPU.")

    pdf_files = collect_pdfs(args.input)
    print(f"Found {len(pdf_files)} PDF(s) to process.\n")

    reader = load_reader(args.languages, gpu=gpu)

    for i, pdf_path in enumerate(pdf_files, 1):
        print(f"[{i}/{len(pdf_files)}] {os.path.basename(pdf_path)}")
        pages = ocr_pdf(reader, pdf_path, args.resolution)
        save_results(pdf_path, pages, args.output)
        print()

    print("All done! Results are in:", args.output)


if __name__ == "__main__":
    main()
