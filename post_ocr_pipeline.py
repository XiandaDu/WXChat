#!/usr/bin/env python3
"""
Post-OCR Correction Pipeline
=============================
Reads regex-cleaned CSV data (Date, Text) and applies a multi-stage
correction pipeline to improve OCR text quality.

Stages (in order):
    1. Unicode correction  — encoding fixes, normalization, control chars
    2. Regex correction    — pattern-based OCR error fixes (era-specific)
    3. GPT correction      — LLM-based contextual correction (optional)

Usage:
    # Run all stages on historical data
    python post_ocr_pipeline.py --input data/historical_regex.csv --era historical

    # Run all stages on modern data
    python post_ocr_pipeline.py --input data/modern_regex.csv --era modern

    # Skip GPT stage (regex + unicode only, no API key needed)
    python post_ocr_pipeline.py --input data/modern_regex.csv --era modern --skip-gpt

    # Process specific row range
    python post_ocr_pipeline.py --input data/historical_regex.csv --era historical --rows 0-100

    # Custom output path
    python post_ocr_pipeline.py --input data/modern_regex.csv --output data/modern_corrected.csv
"""

import argparse
import csv
import logging
import os
import sys
import time

from models.unicode_corrector import UnicodeCorrector
from models.regex_corrector import RegexCorrector

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# Handle large CSV fields from full-newspaper-day texts
csv.field_size_limit(sys.maxsize)


def parse_row_range(rows_str: str, max_rows: int) -> tuple[int, int]:
    """Parse a row range string like '0-100' or '50' into (start, end)."""
    if "-" in rows_str:
        parts = rows_str.split("-", 1)
        start = int(parts[0])
        end = min(int(parts[1]), max_rows)
    else:
        start = int(rows_str)
        end = start + 1
    return start, end


def count_rows(input_path: str) -> int:
    """Count data rows in a CSV file (excluding header)."""
    with open(input_path, "r", encoding="utf-8") as f:
        reader = csv.reader(f)
        next(reader)  # skip header
        return sum(1 for _ in reader)


def load_rows(input_path: str, start: int = 0, end: int | None = None) -> list[dict]:
    """Load rows from CSV as list of {date, text} dicts."""
    rows = []
    with open(input_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for i, row in enumerate(reader):
            if i < start:
                continue
            if end is not None and i >= end:
                break
            rows.append({"date": row["Date"], "text": row["Text"]})
    return rows


def save_rows(rows: list[dict], output_path: str):
    """Write corrected rows to CSV."""
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with open(output_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f, quoting=csv.QUOTE_ALL)
        writer.writerow(["Date", "Text"])
        for row in rows:
            writer.writerow([row["date"], row["text"]])


def build_pipeline(era: str, skip_regex: bool, skip_gpt: bool, gpt_model: str, api_key: str | None):
    """Build the ordered list of correction stages."""
    stages = []

    # Stage 1: Unicode correction
    unicode_corrector = UnicodeCorrector()
    stages.append(("unicode", unicode_corrector))

    # Stage 2: Regex correction (era-specific)
    if not skip_regex:
        regex_corrector = RegexCorrector(era=era)
        stages.append(("regex", regex_corrector))

    # Stage 3: GPT correction (optional)
    if not skip_gpt:
        try:
            from models.gpt_corrector import GPTCorrector
            gpt_corrector = GPTCorrector(
                model=gpt_model,
                api_key=api_key,
            )
            stages.append(("gpt", gpt_corrector))
        except ValueError as e:
            logger.warning(f"GPT stage disabled: {e}")
        except FileNotFoundError as e:
            logger.warning(f"GPT stage disabled: {e}")

    return stages


def run_pipeline(
    input_path: str,
    output_path: str,
    era: str,
    skip_regex: bool,
    skip_gpt: bool,
    gpt_model: str,
    api_key: str | None,
    row_start: int,
    row_end: int | None,
    checkpoint_interval: int,
):
    """Execute the full correction pipeline."""
    logger.info(f"Input:  {input_path}")
    logger.info(f"Output: {output_path}")
    logger.info(f"Era:    {era}")

    # Build pipeline
    stages = build_pipeline(era, skip_regex, skip_gpt, gpt_model, api_key)
    stage_names = [name for name, _ in stages]
    logger.info(f"Stages: {' -> '.join(stage_names)}")

    # Load data
    logger.info("Loading data...")
    rows = load_rows(input_path, row_start, row_end)
    total = len(rows)
    logger.info(f"Loaded {total} rows (range: {row_start}-{row_start + total})")

    if total == 0:
        logger.warning("No rows to process.")
        return

    # Process rows
    corrected_rows = []
    t_start = time.time()

    for idx, row in enumerate(rows):
        row_num = row_start + idx
        text = row["text"]
        original_len = len(text)

        # Run each stage
        for stage_name, corrector in stages:
            text = corrector.correct(text)

        corrected_len = len(text)
        delta = original_len - corrected_len
        corrected_rows.append({"date": row["date"], "text": text})

        # Progress logging
        if (idx + 1) % 10 == 0 or idx == total - 1:
            elapsed = time.time() - t_start
            rate = (idx + 1) / elapsed if elapsed > 0 else 0
            eta = (total - idx - 1) / rate if rate > 0 else 0
            logger.info(
                f"  [{idx+1}/{total}] row {row_num} | "
                f"{original_len:,} -> {corrected_len:,} chars (delta: {delta:+,}) | "
                f"{rate:.1f} rows/s | ETA: {eta:.0f}s"
            )

        # Periodic checkpoint saves
        if checkpoint_interval > 0 and (idx + 1) % checkpoint_interval == 0:
            logger.info(f"  Checkpoint: saving {len(corrected_rows)} rows...")
            save_rows(corrected_rows, output_path)

    # Final save
    elapsed = time.time() - t_start
    logger.info(f"Pipeline complete in {elapsed:.1f}s ({total} rows)")
    save_rows(corrected_rows, output_path)
    logger.info(f"Saved to {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Post-OCR correction pipeline for newspaper CSV data"
    )
    parser.add_argument(
        "--input", required=True,
        help="Path to input CSV file (Date, Text columns)",
    )
    parser.add_argument(
        "--output", default=None,
        help="Path to output CSV file. Default: input name with '_corrected' suffix.",
    )
    parser.add_argument(
        "--era", choices=["historical", "modern"], required=True,
        help="Era of the text: 'historical' (1880-1920) or 'modern' (1990+).",
    )
    parser.add_argument(
        "--skip-regex", action="store_true",
        help="Skip regex correction stage.",
    )
    parser.add_argument(
        "--skip-gpt", action="store_true",
        help="Skip GPT correction stage (only run unicode + regex).",
    )
    parser.add_argument(
        "--gpt-model", default="gpt-4o-mini",
        help="OpenAI model to use for GPT correction (default: gpt-4o-mini).",
    )
    parser.add_argument(
        "--api-key", default=None,
        help="OpenAI API key. Falls back to OPENAI_API_KEY env var.",
    )
    parser.add_argument(
        "--rows", default=None,
        help="Row range to process, e.g. '0-100' or '50'. Default: all rows.",
    )
    parser.add_argument(
        "--checkpoint", type=int, default=100,
        help="Save intermediate results every N rows (default: 100, 0 to disable).",
    )
    args = parser.parse_args()

    # Validate input
    if not os.path.isfile(args.input):
        sys.exit(f"Input file not found: {args.input}")

    # Default output path
    if args.output is None:
        base, ext = os.path.splitext(args.input)
        args.output = f"{base}_corrected{ext}"

    # Parse row range
    row_start = 0
    row_end = None
    if args.rows:
        total_rows = count_rows(args.input)
        row_start, row_end = parse_row_range(args.rows, total_rows)
        logger.info(f"Row range: {row_start}-{row_end} (of {total_rows} total)")

    run_pipeline(
        input_path=args.input,
        output_path=args.output,
        era=args.era,
        skip_regex=args.skip_regex,
        skip_gpt=args.skip_gpt,
        gpt_model=args.gpt_model,
        api_key=args.api_key,
        row_start=row_start,
        row_end=row_end,
        checkpoint_interval=args.checkpoint,
    )


if __name__ == "__main__":
    main()
