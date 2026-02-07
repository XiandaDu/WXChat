#!/usr/bin/env python3
"""
olmOCR Script for Historical Newspaper PDFs
============================================
Uses the olmOCR-2 vision-language model (allenai/olmOCR-2-7B-1025)
to extract text from scanned historical newspaper pages.

Usage:
    # Process all PDFs in sample_data/
    python ocr_olmocr.py

    # Process a specific PDF
    python ocr_olmocr.py --input sample_data/1886-11-01_The-Liverpool-Echo_Monday_p01.pdf

    # Process a folder of PDFs
    python ocr_olmocr.py --input sample_data/

    # Use higher resolution for dense small text
    python ocr_olmocr.py --resolution 1536

    # Use the full-precision model instead of FP8
    python ocr_olmocr.py --model allenai/olmOCR-2-7B-1025
"""

import argparse
import base64
import json
import os
import subprocess
import sys
import time
from glob import glob
from io import BytesIO
from pathlib import Path

import torch
from PIL import Image
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

from olmocr.data.renderpdf import render_pdf_to_base64png
from olmocr.prompts import build_no_anchoring_v4_yaml_prompt


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def get_pdf_page_count(pdf_path: str) -> int:
    """Return the number of pages in a PDF (requires poppler 'pdfinfo')."""
    result = subprocess.run(
        ["pdfinfo", pdf_path], capture_output=True, text=True
    )
    for line in result.stdout.splitlines():
        if line.startswith("Pages:"):
            return int(line.split(":")[1].strip())
    raise RuntimeError(f"Could not determine page count for {pdf_path}")


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


def parse_model_output(raw: str) -> dict:
    """Try to parse the YAML/JSON output from olmOCR."""
    # The model returns a JSON-like string; try to parse it
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        pass
    # Sometimes the output is wrapped in markdown code fences
    for fence in ("```json", "```yaml", "```"):
        if fence in raw:
            raw = raw.split(fence, 1)[1]
            if "```" in raw:
                raw = raw.split("```", 1)[0]
            try:
                return json.loads(raw.strip())
            except json.JSONDecodeError:
                pass
    return {"natural_text": raw}


# ---------------------------------------------------------------------------
# Core OCR
# ---------------------------------------------------------------------------

def load_model(model_name: str, device: torch.device):
    """Load the olmOCR model and processor."""
    print(f"Loading model: {model_name} ...")
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        model_name, torch_dtype=torch.bfloat16
    ).eval()
    model.to(device)

    # The processor always comes from the base Qwen2.5-VL repo
    processor = AutoProcessor.from_pretrained("Qwen/Qwen2.5-VL-7B-Instruct")
    print("Model loaded.\n")
    return model, processor


def ocr_page(
    model,
    processor,
    device: torch.device,
    pdf_path: str,
    page_num: int,
    resolution: int = 1288,
) -> dict:
    """Run olmOCR on a single PDF page and return structured output."""
    # 1. Render page to base64 PNG
    image_b64 = render_pdf_to_base64png(
        pdf_path, page_num, target_longest_image_dim=resolution
    )

    # 2. Build the non-anchoring prompt (best for scanned / historical docs)
    prompt_text = build_no_anchoring_v4_yaml_prompt()

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": prompt_text},
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/png;base64,{image_b64}"
                    },
                },
            ],
        }
    ]

    # 3. Tokenise
    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    pil_image = Image.open(BytesIO(base64.b64decode(image_b64)))
    inputs = processor(
        text=[text], images=[pil_image], padding=True, return_tensors="pt"
    )
    inputs = {k: v.to(device) for k, v in inputs.items()}

    # 4. Generate
    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            temperature=0.1,
            max_new_tokens=8192,  # large for dense newspaper pages
            num_return_sequences=1,
            do_sample=True,
        )

    # 5. Decode only the new tokens
    prompt_len = inputs["input_ids"].shape[1]
    new_tokens = output_ids[:, prompt_len:]
    raw_text = processor.tokenizer.batch_decode(
        new_tokens, skip_special_tokens=True
    )[0]

    result = parse_model_output(raw_text)
    result["page"] = page_num
    return result


def ocr_pdf(
    model,
    processor,
    device: torch.device,
    pdf_path: str,
    resolution: int = 1288,
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
            f.write(page.get("natural_text", page.get("raw_output", "")) + "\n\n")

    print(f"  -> {json_path}")
    print(f"  -> {txt_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="OCR historical newspaper PDFs with olmOCR-2"
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
        "--model",
        default="allenai/olmOCR-2-7B-1025-FP8",
        help="HuggingFace model ID (default: allenai/olmOCR-2-7B-1025-FP8)",
    )
    parser.add_argument(
        "--resolution",
        type=int,
        default=1288,
        help="Longest-side pixel dimension for page rendering (default: 1288). "
             "Use 1536-2048 for very dense small text.",
    )
    args = parser.parse_args()

    # Resolve device
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"Using CUDA: {torch.cuda.get_device_name(0)}")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
        print("Using Apple MPS (Metal Performance Shaders)")
    else:
        device = torch.device("cpu")
        print("WARNING: No GPU detected — inference will be very slow on CPU.")

    # Collect PDFs
    pdf_files = collect_pdfs(args.input)
    print(f"Found {len(pdf_files)} PDF(s) to process.\n")

    # Load model once
    model, processor = load_model(args.model, device)

    # Process each PDF
    for i, pdf_path in enumerate(pdf_files, 1):
        print(f"[{i}/{len(pdf_files)}] {os.path.basename(pdf_path)}")
        pages = ocr_pdf(model, processor, device, pdf_path, args.resolution)
        save_results(pdf_path, pages, args.output)
        print()

    print("All done! Results are in:", args.output)


if __name__ == "__main__":
    main()
