#!/usr/bin/env python3
"""
GOT-OCR 2.0 Script for Historical Newspaper PDFs
=================================================
Uses GOT-OCR 2.0 (natively in transformers) to extract text from
scanned historical newspaper pages.

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
from io import BytesIO
from pathlib import Path

import torch
from PIL import Image
from pypdf import PdfReader
from transformers import AutoModelForImageTextToText, AutoProcessor


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
    """Render a single PDF page to a PIL Image using pypdf."""
    import pypdfium2 as pdfium

    doc = pdfium.PdfDocument(pdf_path)
    page = doc[page_num - 1]
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

MODEL_ID = "stepfun-ai/GOT-OCR-2.0-hf"


def load_model(device: torch.device):
    """Load GOT-OCR 2.0 model and processor."""
    print(f"Loading model: {MODEL_ID} ...")
    model = AutoModelForImageTextToText.from_pretrained(
        MODEL_ID, torch_dtype=torch.bfloat16, device_map=str(device)
    )
    processor = AutoProcessor.from_pretrained(MODEL_ID)
    print("Model loaded.\n")
    return model, processor


def ocr_page(
    model,
    processor,
    device: torch.device,
    pdf_path: str,
    page_num: int,
    resolution: int = 1536,
) -> dict:
    """Run GOT-OCR on a single PDF page and return structured output."""
    image = render_page_to_image(pdf_path, page_num, resolution)

    inputs = processor(image, return_tensors="pt").to(device)

    generate_ids = model.generate(
        **inputs,
        do_sample=False,
        tokenizer=processor.tokenizer,
        stop_strings="<|im_end|>",
        max_new_tokens=8192,
    )

    # Decode only the new tokens
    prompt_len = inputs["input_ids"].shape[1]
    raw_text = processor.decode(
        generate_ids[0, prompt_len:], skip_special_tokens=True
    )

    return {
        "page": page_num,
        "primary_language": "en",
        "is_rotation_valid": True,
        "rotation_correction": 0,
        "is_table": False,
        "is_diagram": False,
        "natural_text": raw_text.strip(),
    }


def ocr_pdf(
    model,
    processor,
    device: torch.device,
    pdf_path: str,
    resolution: int = 1536,
) -> list[dict]:
    """OCR every page of a PDF, returning a list of per-page results."""
    num_pages = get_pdf_page_count(pdf_path)
    print(f"  Pages: {num_pages}")

    results = []
    for page_num in range(1, num_pages + 1):
        t0 = time.time()
        page_result = ocr_page(
            model, processor, device, pdf_path, page_num, resolution
        )
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
        description="OCR historical newspaper PDFs with GOT-OCR 2.0"
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
        device = torch.device("cuda")
        print(f"Using CUDA: {torch.cuda.get_device_name(0)}")
    else:
        device = torch.device("cpu")
        print("WARNING: No GPU detected — inference will be very slow on CPU.")

    pdf_files = collect_pdfs(args.input)
    print(f"Found {len(pdf_files)} PDF(s) to process.\n")

    model, processor = load_model(device)

    for i, pdf_path in enumerate(pdf_files, 1):
        print(f"[{i}/{len(pdf_files)}] {os.path.basename(pdf_path)}")
        pages = ocr_pdf(model, processor, device, pdf_path, args.resolution)
        save_results(pdf_path, pages, args.output)
        print()

    print("All done! Results are in:", args.output)


if __name__ == "__main__":
    main()
