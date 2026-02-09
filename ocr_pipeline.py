#!/usr/bin/env python3
"""
OCR Pipeline for Historical Newspaper PDFs
===========================================
Column-first segmentation pipeline using Tesseract 5 for scanned
Victorian-era newspapers. Splits dense multi-column pages into
individual column strips, then OCRs each strip independently.

Usage:
    python ocr_pipeline.py
    python ocr_pipeline.py --input sample_data/1886-11-01_The-Liverpool-Echo_Monday_p01.pdf
    python ocr_pipeline.py --input sample_data/ --output ocr_output/ --debug
    python ocr_pipeline.py --resolution 400   # faster, lower quality
"""

import argparse
import json
import os
import re
import sys
import time
import warnings
from glob import glob
from pathlib import Path

warnings.filterwarnings("ignore", category=RuntimeWarning)

import cv2
import numpy as np
import pypdfium2 as pdfium
import pytesseract
from PIL import Image
from pypdf import PdfReader
from scipy.ndimage import uniform_filter1d
from scipy.signal import find_peaks
from skimage.filters import threshold_sauvola
from skimage.transform import rotate as sk_rotate


# ---------------------------------------------------------------------------
# PDF helpers
# ---------------------------------------------------------------------------

def get_pdf_page_count(pdf_path: str) -> int:
    return len(PdfReader(pdf_path).pages)


def collect_pdfs(input_path: str) -> list:
    p = Path(input_path)
    if p.is_file() and p.suffix.lower() == ".pdf":
        return [str(p)]
    if p.is_dir():
        pdfs = sorted(glob(str(p / "*.pdf")))
        if not pdfs:
            sys.exit(f"No PDF files found in {p}")
        return pdfs
    sys.exit(f"Invalid input: {input_path}")


# ---------------------------------------------------------------------------
# Stage 1 — Rendering
# ---------------------------------------------------------------------------

def compute_render_scale(
    page_width_pts: float,
    page_height_pts: float,
    target_dpi: int,
    max_dimension: int = 12000,
) -> float:
    """Compute pypdfium2 render scale from target DPI, capped by *max_dimension*."""
    scale = target_dpi / 72.0
    longest_pts = max(page_width_pts, page_height_pts)
    if longest_pts * scale > max_dimension:
        scale = max_dimension / longest_pts
    return scale


def render_page(pdf_path: str, page_num: int, target_dpi: int) -> np.ndarray:
    """Render a single PDF page to a numpy RGB array."""
    doc = pdfium.PdfDocument(pdf_path)
    page = doc[page_num - 1]
    w, h = page.get_size()
    scale = compute_render_scale(w, h, target_dpi)
    bitmap = page.render(scale=scale)
    img = bitmap.to_pil().convert("RGB")
    doc.close()
    return np.array(img)


# ---------------------------------------------------------------------------
# Stage 2 — Preprocessing
# ---------------------------------------------------------------------------

def estimate_skew_angle(binary: np.ndarray) -> float:
    """Estimate page skew via horizontal-projection variance maximisation."""
    # Down-sample for speed
    h, w = binary.shape
    factor = max(1, h // 2000)
    small = binary[::factor, ::factor]

    best_angle, best_score = 0.0, 0.0
    for angle in np.arange(-2.0, 2.05, 0.2):
        rotated = sk_rotate(small, angle, resize=False, order=0, preserve_range=True)
        proj = np.sum(rotated == 0, axis=1)
        score = float(np.var(proj))
        if score > best_score:
            best_score = score
            best_angle = angle

    for angle in np.arange(best_angle - 0.3, best_angle + 0.35, 0.05):
        rotated = sk_rotate(small, angle, resize=False, order=0, preserve_range=True)
        proj = np.sum(rotated == 0, axis=1)
        score = float(np.var(proj))
        if score > best_score:
            best_score = score
            best_angle = angle

    return best_angle


def preprocess_page(image: np.ndarray) -> tuple:
    """
    Full preprocessing pipeline.

    Returns (enhanced_gray, binary) where both are uint8 numpy arrays.
    enhanced_gray is the CLAHE-enhanced grayscale image (for Tesseract OCR).
    binary is the Sauvola-binarised image (for layout segmentation only).
    """
    # 1. Grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

    # 2. CLAHE contrast enhancement
    clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(16, 16))
    enhanced = clahe.apply(gray)

    # 3. Sauvola adaptive binarisation (for segmentation, not OCR)
    win = 51
    if min(enhanced.shape) < win:
        win = max(3, min(enhanced.shape) // 2 * 2 + 1)
    thresh = threshold_sauvola(enhanced, window_size=win, k=0.2)
    binary = ((enhanced > thresh).astype(np.uint8)) * 255

    # 4. Deskew
    skew = estimate_skew_angle(binary)
    if abs(skew) > 0.1:
        enhanced = sk_rotate(enhanced, skew, resize=False, order=1,
                             preserve_range=True).astype(np.uint8)
        binary = sk_rotate(binary, skew, resize=False, order=0,
                           preserve_range=True).astype(np.uint8)

    return enhanced, binary


# ---------------------------------------------------------------------------
# Stage 3 — Column segmentation
# ---------------------------------------------------------------------------

def find_content_bounds(binary: np.ndarray, margin: int = 20) -> tuple:
    """Return (top, bottom, left, right) of the main content area."""
    v_proj = np.sum(binary == 0, axis=0)
    h_proj = np.sum(binary == 0, axis=1)
    thresh_v = max(np.max(v_proj) * 0.005, 1)
    thresh_h = max(np.max(h_proj) * 0.005, 1)

    cols = np.where(v_proj > thresh_v)[0]
    rows = np.where(h_proj > thresh_h)[0]
    if len(cols) == 0 or len(rows) == 0:
        return 0, binary.shape[0], 0, binary.shape[1]

    left = max(0, int(cols[0]) - margin)
    right = min(binary.shape[1], int(cols[-1]) + margin)
    top = max(0, int(rows[0]) - margin)
    bottom = min(binary.shape[0], int(rows[-1]) + margin)
    return top, bottom, left, right


def find_header_boundary(binary: np.ndarray) -> int:
    """Find the y-coordinate where the masthead ends and columns begin."""
    height, width = binary.shape
    h_proj = np.sum(binary == 0, axis=1).astype(float)
    h_smooth = uniform_filter1d(h_proj, size=max(20, height // 200))

    search_limit = height // 4
    if search_limit < 10:
        return height // 12

    median_val = np.median(h_smooth[:search_limit])
    threshold = max(median_val * 0.08, 1)
    is_gap = h_smooth[:search_limit] < threshold

    # Find runs of gap rows
    gap_runs = []
    run_start = None
    for i in range(len(is_gap)):
        if is_gap[i] and run_start is None:
            run_start = i
        elif not is_gap[i] and run_start is not None:
            gap_runs.append((run_start, i, i - run_start))
            run_start = None
    if run_start is not None:
        gap_runs.append((run_start, len(is_gap), len(is_gap) - run_start))

    min_y = height // 20
    min_gap = max(10, height // 300)
    valid = [(s, e, l) for s, e, l in gap_runs if e > min_y and l >= min_gap]
    if valid:
        best = max(valid, key=lambda x: x[2])
        return best[1]

    return height // 12


def detect_ruled_lines(binary: np.ndarray, y_start: int) -> list:
    """Detect vertical ruled lines separating newspaper columns."""
    region = binary[y_start:, :]
    h, w = region.shape
    if h < 100 or w < 100:
        return []

    min_len = h // 3
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, min_len))
    inv = 255 - region
    lines_img = cv2.morphologyEx(inv, cv2.MORPH_OPEN, kernel)

    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
        lines_img, connectivity=8
    )

    positions = []
    for i in range(1, num_labels):
        x, y, bw, bh, area = stats[i]
        if bw <= 10 and bh >= min_len and bh / max(bw, 1) > 15:
            positions.append(int(centroids[i][0]))

    return sorted(positions)


def find_column_boundaries_by_projection(
    binary: np.ndarray,
    y_start: int,
    min_col_width_ratio: float = 0.06,
    max_columns: int = 8,
) -> list:
    """Find column gutters via vertical projection profile valley detection."""
    region = binary[y_start:, :]
    h, w = region.shape
    if h < 100 or w < 100:
        return []

    v_proj = np.sum(region == 0, axis=0).astype(float)
    v_smooth = uniform_filter1d(v_proj, size=max(15, w // 200))
    v_max = np.max(v_smooth)
    if v_max < 1:
        return []

    v_norm = v_smooth / v_max
    v_inv = 1.0 - v_norm

    min_dist = int(w * min_col_width_ratio)
    peaks, props = find_peaks(
        v_inv,
        height=0.55,
        distance=min_dist,
        prominence=0.15,
        width=2,
    )

    margin = w // 25
    peaks = [int(p) for p in peaks if margin < p < w - margin]

    if len(peaks) > max_columns - 1:
        prominences = props["prominences"]
        # Filter to only those in the valid range
        valid_idx = [i for i, p in enumerate(peaks) if margin < p < w - margin]
        if len(valid_idx) > max_columns - 1:
            scored = sorted(valid_idx, key=lambda i: prominences[i], reverse=True)
            valid_idx = scored[: max_columns - 1]
        peaks = sorted([peaks[i] for i in valid_idx])

    return peaks


def segment_columns(binary: np.ndarray, max_columns: int = 8) -> tuple:
    """
    Segment a newspaper page into header + column regions.

    Returns:
        header_region: dict with y_start, y_end, x, width
        columns: list of dicts with index, x, width, y_start, y_end
    """
    height, width = binary.shape

    # 1. Content bounds
    top, bottom, left, right = find_content_bounds(binary)

    # 2. Header boundary
    header_y = find_header_boundary(binary[top:bottom, left:right])
    header_y += top  # Convert back to full-image coordinates

    # 3. Ruled lines (work on content region)
    ruled = detect_ruled_lines(binary[top:bottom, left:right], header_y - top)
    ruled = [r + left for r in ruled]  # Convert to full-image x coords

    # 4. Projection boundaries
    proj_bounds = find_column_boundaries_by_projection(
        binary[top:bottom, left:right], header_y - top,
        min_col_width_ratio=0.06, max_columns=max_columns,
    )
    proj_bounds = [p + left for p in proj_bounds]

    # 5. Merge ruled lines + projection boundaries, deduplicate
    merged = sorted(set(ruled + proj_bounds))
    content_width = right - left
    tol = content_width * 0.02
    if merged:
        deduped = [merged[0]]
        for b in merged[1:]:
            if b - deduped[-1] > tol:
                deduped.append(b)
        boundaries = deduped
    else:
        # Fallback: 6 equal columns
        cw = content_width // 6
        boundaries = [left + cw * i for i in range(1, 6)]

    # 6. Build column dicts, filtering thin strips
    edges = [left] + boundaries + [right]
    columns = []
    min_width = content_width * 0.07  # Column must be at least 7% of page width
    for i in range(len(edges) - 1):
        x0, x1 = int(edges[i]), int(edges[i + 1])
        w = x1 - x0
        if w < min_width:
            continue
        pad = min(8, w // 20)
        columns.append({
            "index": len(columns),
            "x": x0 + pad,
            "width": w - 2 * pad,
            "y_start": header_y,
            "y_end": bottom,
        })

    # 7. Split oversized columns (likely merged columns)
    #    Typical Victorian newspaper has 6-7 roughly equal columns.
    if len(columns) >= 2:
        widths = [c["width"] for c in columns]
        median_w = sorted(widths)[len(widths) // 2]
        split_threshold = median_w * 1.7

        new_columns = []
        for col in columns:
            if col["width"] > split_threshold:
                # Try to find sub-boundaries within this oversized column
                sub_region = binary[col["y_start"]:col["y_end"],
                                    col["x"]:col["x"] + col["width"]]
                sub_bounds = find_column_boundaries_by_projection(
                    sub_region, 0,
                    min_col_width_ratio=0.15,
                    max_columns=4,
                )
                if sub_bounds:
                    sub_edges = [0] + sub_bounds + [col["width"]]
                    for j in range(len(sub_edges) - 1):
                        sw = sub_edges[j + 1] - sub_edges[j]
                        if sw >= min_width:
                            sp = min(8, sw // 20)
                            new_columns.append({
                                "index": 0,
                                "x": col["x"] + sub_edges[j] + sp,
                                "width": sw - 2 * sp,
                                "y_start": col["y_start"],
                                "y_end": col["y_end"],
                            })
                else:
                    new_columns.append(col)
            else:
                new_columns.append(col)

        # Re-index
        for idx, c in enumerate(new_columns):
            c["index"] = idx
        columns = new_columns

    header = {
        "x": left,
        "width": right - left,
        "y_start": top,
        "y_end": header_y,
    }

    return header, columns


def segment_blocks_in_column(binary_col: np.ndarray, min_gap: int = 25) -> list:
    """Split a single column into text blocks separated by whitespace gaps."""
    h, w = binary_col.shape
    if h < 30:
        return [(0, h)]

    h_proj = np.sum(binary_col == 0, axis=1).astype(float)
    h_smooth = uniform_filter1d(h_proj, size=5)
    peak = np.max(h_smooth)
    if peak < 1:
        return [(0, h)]

    threshold = peak * 0.02
    is_gap = h_smooth < threshold

    blocks = []
    block_start = 0
    in_gap = False

    for row in range(h):
        if is_gap[row] and not in_gap:
            in_gap = True
            gap_start = row
        elif not is_gap[row] and in_gap:
            in_gap = False
            if row - gap_start >= min_gap and gap_start > block_start + 20:
                blocks.append((block_start, gap_start))
                block_start = row

    if h > block_start + 20:
        blocks.append((block_start, h))

    return blocks if blocks else [(0, h)]


# ---------------------------------------------------------------------------
# Stage 4 — Tesseract OCR
# ---------------------------------------------------------------------------

def ocr_image_block(img_block: np.ndarray, psm: int = 6,
                    dpi: int = 300) -> dict:
    """
    Run Tesseract on a grayscale image block.

    Pass CLAHE-enhanced grayscale so Tesseract applies its own optimal
    binarisation internally. This consistently outperforms feeding
    pre-binarised images on degraded historical scans.

    Returns dict with 'text', 'confidence', 'word_count'.
    """
    if img_block.size == 0 or img_block.shape[0] < 10 or img_block.shape[1] < 10:
        return {"text": "", "confidence": 0.0, "word_count": 0}

    pil_img = Image.fromarray(img_block)

    config = (
        f"--dpi {dpi} "
        f"--psm {psm} "
        "--oem 1 "
    )

    try:
        data = pytesseract.image_to_data(
            pil_img, lang="eng", config=config,
            output_type=pytesseract.Output.DICT,
        )
    except pytesseract.TesseractError:
        return {"text": "", "confidence": 0.0, "word_count": 0}

    # Reconstruct text with line structure
    text_lines = {}
    confidences = []
    for i in range(len(data["text"])):
        txt = data["text"][i].strip()
        conf = int(data["conf"][i])
        if txt and conf > 0:
            confidences.append(conf)
            key = (data["block_num"][i], data["line_num"][i])
            text_lines.setdefault(key, []).append(txt)

    lines = [" ".join(text_lines[k]) for k in sorted(text_lines)]
    full_text = "\n".join(lines)
    avg_conf = sum(confidences) / len(confidences) if confidences else 0.0

    return {
        "text": full_text,
        "confidence": round(avg_conf, 1),
        "word_count": len(confidences),
    }


# ---------------------------------------------------------------------------
# Stage 5 — Post-processing
# ---------------------------------------------------------------------------

CORRECTIONS = [
    # I between lowercase → l
    (r"(?<=[a-z])I(?=[a-z])", "l"),
    # 1 at start followed by lowercase → l
    (r"\b1(?=[a-z]{2,})", "l"),
    # 0 between letters → o/O
    (r"(?<=[A-Z])0(?=[A-Z])", "O"),
    (r"(?<=[a-z])0(?=[a-z])", "o"),
    # rn → m (word-internal)
    (r"(?<=[a-z])rn(?=[aeiouy])", "m"),
    # Common Victorian misreads
    (r"\bTlie\b", "The"),
    (r"\btlie\b", "the"),
    (r"\bTbe\b", "The"),
    (r"\btbe\b", "the"),
    (r"\bwbich\b", "which"),
    (r"\bwliich\b", "which"),
    # Punctuation spacing
    (r"\s+([.,;:!?])", r"\1"),
    (r"([.,;:!?])(?=[A-Za-z])", r"\1 "),
    # Hyphenation at line breaks
    (r"-\n\s*", ""),
    # Multiple spaces
    (r"[ \t]{2,}", " "),
]

_COMPILED = [(re.compile(p), r) for p, r in CORRECTIONS]


def postprocess_text(text: str) -> str:
    """Apply regex-based corrections for common OCR errors."""
    for pat, repl in _COMPILED:
        text = pat.sub(repl, text)

    # Remove garbage lines (<30% alphabetic characters)
    lines = text.split("\n")
    cleaned = []
    for line in lines:
        s = line.strip()
        if not s:
            cleaned.append("")
            continue
        alpha = sum(1 for c in s if c.isalpha())
        if alpha / max(len(s), 1) >= 0.25:
            cleaned.append(line)

    text = "\n".join(cleaned)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


# ---------------------------------------------------------------------------
# Debug helpers
# ---------------------------------------------------------------------------

def save_debug_images(
    binary: np.ndarray,
    gray: np.ndarray,
    header: dict,
    columns: list,
    debug_dir: str,
    page_id: str,
):
    """Save annotated debug images showing column boundaries."""
    os.makedirs(debug_dir, exist_ok=True)

    # 1. Preprocessed binary
    cv2.imwrite(os.path.join(debug_dir, f"{page_id}_binary.png"), binary)

    # 2. Column overlay
    vis = cv2.cvtColor(binary, cv2.COLOR_GRAY2BGR)

    # Header line (blue)
    hy = header["y_end"]
    cv2.line(vis, (0, hy), (vis.shape[1], hy), (255, 0, 0), 3)

    # Column rectangles (green)
    for col in columns:
        x0 = col["x"]
        y0 = col["y_start"]
        x1 = x0 + col["width"]
        y1 = col["y_end"]
        cv2.rectangle(vis, (x0, y0), (x1, y1), (0, 255, 0), 2)
        # Label
        cv2.putText(vis, f"Col {col['index']}", (x0 + 5, y0 + 40),
                     cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 2)

    # Down-scale for reasonable file size
    scale = 2000 / max(vis.shape[:2])
    if scale < 1.0:
        vis = cv2.resize(vis, None, fx=scale, fy=scale,
                         interpolation=cv2.INTER_AREA)
    cv2.imwrite(os.path.join(debug_dir, f"{page_id}_columns.png"), vis)


# ---------------------------------------------------------------------------
# Main processing
# ---------------------------------------------------------------------------

def process_page(
    pdf_path: str,
    page_num: int,
    target_dpi: int,
    debug_dir=None,
) -> dict:
    """Process a single PDF page through the full pipeline."""
    t0 = time.time()
    stem = Path(pdf_path).stem
    page_id = f"{stem}_p{page_num:02d}"

    print(f"    Rendering at {target_dpi} DPI ...")
    image = render_page(pdf_path, page_num, target_dpi)
    print(f"    Image size: {image.shape[1]}x{image.shape[0]}")

    print("    Preprocessing ...")
    gray, binary = preprocess_page(image)

    print("    Segmenting columns ...")
    header, columns = segment_columns(binary)
    print(f"    Found {len(columns)} columns, header ends at y={header['y_end']}")

    if debug_dir:
        save_debug_images(binary, gray, header, columns, debug_dir, page_id)
        print(f"    Debug images saved to {debug_dir}/")

    # OCR header (sparse text mode, using enhanced grayscale)
    print("    OCR: header ...")
    hdr_gray = gray[header["y_start"]:header["y_end"],
                    header["x"]:header["x"] + header["width"]]
    header_result = ocr_image_block(hdr_gray, psm=11, dpi=target_dpi)
    header_result["text"] = postprocess_text(header_result["text"])

    # OCR each column (using enhanced grayscale, binary only for segmentation)
    column_results = []
    for col in columns:
        print(f"    OCR: column {col['index']} ...")
        col_bin = binary[col["y_start"]:col["y_end"],
                         col["x"]:col["x"] + col["width"]]
        col_gray = gray[col["y_start"]:col["y_end"],
                        col["x"]:col["x"] + col["width"]]

        # Sub-segment into blocks (uses binary for whitespace detection)
        blocks = segment_blocks_in_column(col_bin)

        block_results = []
        for b_start, b_end in blocks:
            blk_gray = col_gray[b_start:b_end, :]
            if blk_gray.shape[0] < 15 or blk_gray.shape[1] < 15:
                continue
            result = ocr_image_block(blk_gray, psm=4, dpi=target_dpi)
            result["text"] = postprocess_text(result["text"])
            if result["text"]:
                result["y_start"] = b_start + col["y_start"]
                result["y_end"] = b_end + col["y_start"]
                block_results.append(result)

        col_text = "\n\n".join(b["text"] for b in block_results if b["text"])
        col_conf = (
            sum(b["confidence"] for b in block_results) / len(block_results)
            if block_results else 0.0
        )

        column_results.append({
            "index": col["index"],
            "bounds": {
                "x": col["x"],
                "y": col["y_start"],
                "width": col["width"],
                "height": col["y_end"] - col["y_start"],
            },
            "blocks": [
                {
                    "text": b["text"],
                    "confidence": b["confidence"],
                    "word_count": b["word_count"],
                    "y_start": b["y_start"],
                    "y_end": b["y_end"],
                }
                for b in block_results
            ],
            "full_text": col_text,
            "confidence": round(col_conf, 1),
        })

    # Assemble reading-order text
    parts = []
    if header_result["text"]:
        parts.append(header_result["text"])
    for cr in column_results:
        if cr["full_text"]:
            parts.append(cr["full_text"])
    full_text = "\n\n".join(parts)

    all_confs = [header_result["confidence"]] + [c["confidence"] for c in column_results]
    avg_conf = sum(all_confs) / len(all_confs) if all_confs else 0.0

    elapsed = time.time() - t0
    print(f"    Done in {elapsed:.1f}s  (avg confidence: {avg_conf:.1f}%)")

    return {
        "source_pdf": os.path.basename(pdf_path),
        "page": page_num,
        "processing_time_seconds": round(elapsed, 1),
        "resolution_dpi": target_dpi,
        "image_size": {"width": image.shape[1], "height": image.shape[0]},
        "header": {
            "text": header_result["text"],
            "confidence": header_result["confidence"],
            "word_count": header_result["word_count"],
            "bounds": header,
        },
        "columns": column_results,
        "full_text": full_text,
        "average_confidence": round(avg_conf, 1),
    }


# ---------------------------------------------------------------------------
# Output writers
# ---------------------------------------------------------------------------

def save_results(pdf_path: str, pages: list, output_dir: str):
    """Write OCR results to JSON and TXT."""
    stem = Path(pdf_path).stem
    os.makedirs(output_dir, exist_ok=True)

    json_path = os.path.join(output_dir, f"{stem}.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(
            {"source_pdf": os.path.basename(pdf_path), "pages": pages},
            f, ensure_ascii=False, indent=2,
        )

    txt_path = os.path.join(output_dir, f"{stem}.txt")
    with open(txt_path, "w", encoding="utf-8") as f:
        for page in pages:
            f.write(f"===== Page {page['page']} =====\n\n")

            if page["header"]["text"]:
                f.write("[HEADER]\n")
                f.write(page["header"]["text"] + "\n\n")

            for col in page["columns"]:
                f.write(f"--- Column {col['index']} "
                        f"(confidence: {col['confidence']}%) ---\n")
                f.write(col["full_text"] + "\n\n")

    print(f"    -> {json_path}")
    print(f"    -> {txt_path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="OCR pipeline for historical newspaper PDFs (Tesseract 5)"
    )
    parser.add_argument(
        "--input", default="sample_data",
        help="Path to a PDF file or directory of PDFs (default: sample_data/)",
    )
    parser.add_argument(
        "--output", default="ocr_output",
        help="Directory for output files (default: ocr_output/)",
    )
    parser.add_argument(
        "--resolution", type=int, default=600,
        help="Target rendering DPI, 400-800 recommended (default: 600).",
    )
    parser.add_argument(
        "--debug", action="store_true",
        help="Save intermediate debug images (preprocessed, column boundaries).",
    )
    parser.add_argument(
        "--pages", default=None,
        help="Page range to process, e.g. '1-3' or '2'. Default: all pages.",
    )
    parser.add_argument(
        "--tesseract-path", default=None,
        help="Path to tesseract binary (auto-detected if not set).",
    )
    args = parser.parse_args()

    # Configure Tesseract path
    if args.tesseract_path:
        pytesseract.pytesseract.tesseract_cmd = args.tesseract_path

    debug_dir = os.path.join(args.output, "debug") if args.debug else None

    pdf_files = collect_pdfs(args.input)
    print(f"Found {len(pdf_files)} PDF(s) to process.\n")

    for i, pdf_path in enumerate(pdf_files, 1):
        print(f"[{i}/{len(pdf_files)}] {os.path.basename(pdf_path)}")
        num_pages = get_pdf_page_count(pdf_path)

        # Parse --pages
        if args.pages:
            if "-" in args.pages:
                a, b = args.pages.split("-", 1)
                page_range = range(int(a), int(b) + 1)
            else:
                page_range = [int(args.pages)]
        else:
            page_range = range(1, num_pages + 1)

        page_results = []
        for pn in page_range:
            if pn < 1 or pn > num_pages:
                print(f"  Skipping page {pn} (out of range)")
                continue
            print(f"  Page {pn}/{num_pages}")
            result = process_page(pdf_path, pn, args.resolution, debug_dir)
            page_results.append(result)

        save_results(pdf_path, page_results, args.output)
        print()

    print("All done! Results are in:", args.output)


if __name__ == "__main__":
    main()
