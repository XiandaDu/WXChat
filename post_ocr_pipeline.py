#!/usr/bin/env python3
from __future__ import annotations

"""
Post-OCR Correction Pipeline (Multi-threaded)
==============================================
Reads regex-cleaned CSV data (Date, Text) and applies a multi-stage
correction pipeline to improve OCR text quality.

Features:
    - Multi-threaded processing for parallel row correction
    - Crash recovery: automatically resumes from where you left off
    - Environment variables via .env file (OPENAI_API_KEY, HUGGINGFACE_TOKEN)
    - Periodic HuggingFace Hub snapshots (default: twice daily)

Stages (in order):
    1. Unicode correction  — encoding fixes, normalization, control chars
    2. Regex correction    — pattern-based OCR error fixes (era-specific)
    3. GPT correction      — LLM-based contextual correction (optional)

Usage:
    # Run all stages with 4 worker threads
    python post_ocr_pipeline.py --input data/historical_regex.csv --era historical --workers 4

    # Resume a crashed run (automatic — just re-run the same command)
    python post_ocr_pipeline.py --input data/historical_regex.csv --era historical

    # Start fresh (discard previous progress)
    python post_ocr_pipeline.py --input data/historical_regex.csv --era historical --fresh

    # Enable HuggingFace snapshots (twice daily)
    python post_ocr_pipeline.py --input data/historical_regex.csv --era historical --hf-repo user/dataset

    # Skip GPT stage (regex + unicode only)
    python post_ocr_pipeline.py --input data/modern_regex.csv --era modern --skip-gpt

    # Process specific row range
    python post_ocr_pipeline.py --input data/historical_regex.csv --era historical --rows 0-100
"""

import argparse
import csv
import hashlib
import json
import logging
import os
import shutil
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from contextlib import ExitStack
from datetime import datetime, timezone
from pathlib import Path

from dotenv import load_dotenv

# Load .env before importing models that read env vars
load_dotenv()

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


# ---------------------------------------------------------------------------
# Progress tracking for crash recovery
# ---------------------------------------------------------------------------

class ProgressTracker:
    """Tracks per-row processing progress for crash recovery.

    Uses an append-only JSONL file for completed row results (crash-safe)
    and a JSON metadata file for run configuration validation.
    """

    def __init__(self, output_path: str, cfg_hash: str):
        self.progress_dir = Path(f"{output_path}.progress")
        self.state_file = self.progress_dir / "state.json"
        self.results_file = self.progress_dir / "results.jsonl"
        self.cfg_hash = cfg_hash
        self._lock = threading.Lock()
        self._completed: dict[int, dict] = {}

    def load(self) -> dict[int, dict]:
        """Load previously completed results. Returns {row_idx: {date, text}}."""
        if not self.state_file.exists():
            return {}

        try:
            state = json.loads(self.state_file.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            logger.warning("Corrupted state file. Starting fresh.")
            self.reset()
            return {}

        if state.get("config_hash") != self.cfg_hash:
            logger.warning("Pipeline config changed since last run. Starting fresh.")
            self.reset()
            return {}

        completed: dict[int, dict] = {}
        if self.results_file.exists():
            with open(self.results_file, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        entry = json.loads(line)
                        completed[entry["row_idx"]] = {
                            "date": entry["date"],
                            "text": entry["text"],
                        }
                    except (json.JSONDecodeError, KeyError):
                        continue  # skip corrupted lines

        self._completed = completed
        return completed

    def save_state(self, total_rows: int, row_start: int, row_end: int | None):
        """Write run metadata."""
        self.progress_dir.mkdir(parents=True, exist_ok=True)
        state = {
            "config_hash": self.cfg_hash,
            "total_rows": total_rows,
            "row_start": row_start,
            "row_end": row_end,
            "last_updated": datetime.now(timezone.utc).isoformat(),
        }
        self.state_file.write_text(json.dumps(state, indent=2), encoding="utf-8")

    def record(self, row_idx: int, date: str, text: str):
        """Append a completed row result (thread-safe, crash-safe)."""
        with self._lock:
            entry = json.dumps({"row_idx": row_idx, "date": date, "text": text})
            with open(self.results_file, "a", encoding="utf-8") as f:
                f.write(entry + "\n")
            self._completed[row_idx] = {"date": date, "text": text}

    def is_done(self, row_idx: int) -> bool:
        return row_idx in self._completed

    @property
    def completed_count(self) -> int:
        return len(self._completed)

    def get_all_completed(self) -> dict[int, dict]:
        with self._lock:
            return dict(self._completed)

    def reset(self):
        """Wipe all progress data."""
        if self.progress_dir.exists():
            shutil.rmtree(self.progress_dir)
        self._completed = {}

    def cleanup(self):
        """Remove progress files after successful completion."""
        if self.progress_dir.exists():
            shutil.rmtree(self.progress_dir)


# ---------------------------------------------------------------------------
# HuggingFace snapshot scheduler
# ---------------------------------------------------------------------------

class HuggingFaceSnapshotter:
    """Periodically pushes the output file to a HuggingFace dataset repo.

    Uses a daemon timer thread to push snapshots at a configurable interval
    (default: every 12 hours = twice daily). Also supports on-demand pushes.
    HuggingFace's git history provides the full snapshot timeline.
    """

    def __init__(
        self,
        repo_id: str,
        token: str,
        output_path: str,
        interval_hours: float = 12.0,
    ):
        self.repo_id = repo_id
        self.token = token
        self.output_path = output_path
        self.interval = interval_hours * 3600
        self._timer: threading.Timer | None = None
        self._lock = threading.Lock()
        self._running = False

    def start(self):
        """Begin the periodic snapshot schedule."""
        self._running = True
        self._schedule_next()
        logger.info(
            f"HuggingFace snapshots enabled: repo={self.repo_id}, "
            f"interval={self.interval / 3600:.0f}h"
        )

    def stop(self):
        """Stop the scheduler."""
        self._running = False
        if self._timer:
            self._timer.cancel()

    def _schedule_next(self):
        if self._running:
            self._timer = threading.Timer(self.interval, self._tick)
            self._timer.daemon = True
            self._timer.start()

    def _tick(self):
        self.push()
        self._schedule_next()

    def push(self):
        """Upload the current output file to HuggingFace Hub."""
        with self._lock:
            if not os.path.isfile(self.output_path):
                logger.warning("Snapshot skipped: output file does not exist yet.")
                return

            try:
                from huggingface_hub import HfApi

                api = HfApi(token=self.token)
                filename = Path(self.output_path).name
                timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")

                # Ensure the dataset repo exists
                api.create_repo(
                    repo_id=self.repo_id,
                    repo_type="dataset",
                    exist_ok=True,
                )

                # Upload — HF git history serves as the snapshot timeline
                api.upload_file(
                    path_or_fileobj=self.output_path,
                    path_in_repo=filename,
                    repo_id=self.repo_id,
                    repo_type="dataset",
                    commit_message=f"Snapshot {timestamp}",
                )
                logger.info(f"Snapshot pushed: {self.repo_id}/{filename}")

            except Exception as e:
                logger.error(f"HuggingFace snapshot failed: {e}")


# ---------------------------------------------------------------------------
# Data I/O helpers
# ---------------------------------------------------------------------------

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


def load_rows(
    input_path: str, start: int = 0, end: int | None = None
) -> list[dict]:
    """Load rows from CSV as list of {date, text, row_idx} dicts."""
    rows = []
    with open(input_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for i, row in enumerate(reader):
            if i < start:
                continue
            if end is not None and i >= end:
                break
            rows.append({"date": row["Date"], "text": row["Text"], "row_idx": i})
    return rows


def save_rows(rows: list[dict], output_path: str):
    """Write rows to CSV, sorted by row_idx."""
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    sorted_rows = sorted(rows, key=lambda r: r.get("row_idx", 0))

    # Atomic write: write to temp file then rename
    tmp_path = output_path + ".tmp"
    with open(tmp_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f, quoting=csv.QUOTE_ALL)
        writer.writerow(["Date", "Text"])
        for row in sorted_rows:
            writer.writerow([row["date"], row["text"]])
    os.replace(tmp_path, output_path)


def compute_config_hash(
    input_path: str,
    era: str,
    skip_regex: bool,
    skip_gpt: bool,
    gpt_model: str,
    row_start: int,
    row_end: int | None,
) -> str:
    """Deterministic hash of pipeline configuration for resume validation."""
    key = (
        f"{os.path.abspath(input_path)}|{era}|{skip_regex}|"
        f"{skip_gpt}|{gpt_model}|{row_start}|{row_end}"
    )
    return hashlib.sha256(key.encode()).hexdigest()[:16]


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------

def build_pipeline(
    era: str,
    skip_regex: bool,
    skip_gpt: bool,
    gpt_model: str,
    api_key: str | None,
    rpm: int = 10000,
    executor: ThreadPoolExecutor | None = None,
):
    """Build the ordered list of correction stages."""
    stages = []

    # Stage 1: Unicode correction
    stages.append(("unicode", UnicodeCorrector()))

    # Stage 2: Regex correction (era-specific)
    if not skip_regex:
        stages.append(("regex", RegexCorrector(era=era)))

    # Stage 3: GPT correction (optional)
    if not skip_gpt:
        try:
            from models.gpt_corrector import GPTCorrector

            stages.append(("gpt", GPTCorrector(
                model=gpt_model,
                api_key=api_key,
                requests_per_minute=rpm,
                executor=executor,
            )))
        except ValueError as e:
            logger.warning(f"GPT stage disabled: {e}")
        except FileNotFoundError as e:
            logger.warning(f"GPT stage disabled: {e}")

    return stages


def process_row(row: dict, stages: list) -> dict:
    """Run a single row through all correction stages (called by worker threads)."""
    text = row["text"]
    original_len = len(text)

    for _, corrector in stages:
        text = corrector.correct(text)

    return {
        "row_idx": row["row_idx"],
        "date": row["date"],
        "text": text,
        "original_len": original_len,
    }


def _write_intermediate_output(
    all_rows: list[dict],
    tracker: ProgressTracker,
    output_path: str,
):
    """Write current progress to the output CSV."""
    completed = tracker.get_all_completed()
    rows_out = []
    for row in all_rows:
        idx = row["row_idx"]
        if idx in completed:
            rows_out.append({**completed[idx], "row_idx": idx})
    save_rows(rows_out, output_path)


def _write_final_output(
    all_rows: list[dict],
    completed: dict[int, dict],
    output_path: str,
):
    """Write the final output CSV with all rows."""
    rows_out = []
    for row in all_rows:
        idx = row["row_idx"]
        if idx in completed:
            rows_out.append({**completed[idx], "row_idx": idx})
        else:
            # Fallback: keep original text for any missing rows
            rows_out.append(row)
    save_rows(rows_out, output_path)


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
    num_workers: int,
    fresh: bool,
    hf_repo: str | None,
    hf_token: str | None,
    snapshot_interval: float,
    rpm: int = 10000,
):
    """Execute the multi-threaded correction pipeline with crash recovery."""
    logger.info(f"Input:   {input_path}")
    logger.info(f"Output:  {output_path}")
    logger.info(f"Era:     {era}")
    logger.info(f"Workers: {num_workers}")
    logger.info(f"RPM:     {rpm}")

    # A single shared thread pool handles both row-level and chunk-level
    # parallelism.  Row workers submit their chunks into the same pool,
    # so idle time waiting for API responses is filled by other rows'
    # chunks — maximising throughput up to the RPM limit.
    with ExitStack() as stack:
        chunk_pool = stack.enter_context(
            ThreadPoolExecutor(max_workers=num_workers)
        ) if not skip_gpt else None

        # Build correction stages (GPTCorrector receives the shared pool)
        stages = build_pipeline(
            era, skip_regex, skip_gpt, gpt_model, api_key,
            rpm=rpm, executor=chunk_pool,
        )
        stage_names = [name for name, _ in stages]
        logger.info(f"Stages:  {' -> '.join(stage_names)}")

        # Progress tracker for crash recovery
        cfg_hash = compute_config_hash(
            input_path, era, skip_regex, skip_gpt, gpt_model, row_start, row_end
        )
        tracker = ProgressTracker(output_path, cfg_hash)

        if fresh:
            tracker.reset()
            logger.info("Fresh start: cleared previous progress.")

        # Load existing progress (if any)
        completed = tracker.load()

        # Load input data
        logger.info("Loading data...")
        all_rows = load_rows(input_path, row_start, row_end)
        total = len(all_rows)
        logger.info(f"Loaded {total} rows (range: {row_start}-{row_start + total})")

        if total == 0:
            logger.warning("No rows to process.")
            return

        # Filter out already-completed rows
        pending_rows = [r for r in all_rows if r["row_idx"] not in completed]
        already_done = total - len(pending_rows)

        if already_done > 0:
            logger.info(
                f"Resuming: {already_done} rows already done, "
                f"{len(pending_rows)} remaining"
            )

        if not pending_rows:
            logger.info("All rows already processed. Writing final output.")
            _write_final_output(all_rows, completed, output_path)
            tracker.cleanup()
            return

        # Save state for future resume
        tracker.save_state(total, row_start, row_end)

        # HuggingFace snapshotter
        snapshotter = None
        if hf_repo:
            token = hf_token or os.environ.get("HUGGINGFACE_TOKEN")
            if not token:
                logger.warning(
                    "HuggingFace snapshots disabled: no token found. "
                    "Set HUGGINGFACE_TOKEN in .env or pass --hf-token."
                )
            else:
                snapshotter = HuggingFaceSnapshotter(
                    repo_id=hf_repo,
                    token=token,
                    output_path=output_path,
                    interval_hours=snapshot_interval,
                )
                snapshotter.start()

        # Process rows with thread pool
        t_start = time.time()
        done_count = already_done
        failed_count = 0
        checkpoint_counter = 0

        try:
            with ThreadPoolExecutor(max_workers=num_workers) as row_executor:
                # Submit all pending rows
                future_to_row = {
                    row_executor.submit(process_row, row, stages): row
                    for row in pending_rows
                }

                for future in as_completed(future_to_row):
                    row = future_to_row[future]
                    row_idx = row["row_idx"]

                    try:
                        result = future.result()
                        tracker.record(result["row_idx"], result["date"], result["text"])
                        done_count += 1

                        delta = result["original_len"] - len(result["text"])

                        # Progress logging every 10 rows or on the last row
                        if done_count % 10 == 0 or done_count == total:
                            elapsed = time.time() - t_start
                            processed = done_count - already_done
                            rate = processed / elapsed if elapsed > 0 else 0
                            remaining = total - done_count
                            eta = remaining / rate if rate > 0 else 0
                            logger.info(
                                f"  [{done_count}/{total}] row {row_idx} | "
                                f"{result['original_len']:,} -> {len(result['text']):,} "
                                f"(delta: {delta:+,}) | {rate:.1f} rows/s | ETA: {eta:.0f}s"
                            )

                    except Exception as e:
                        logger.error(
                            f"  Row {row_idx} ({row['date']}) failed: {e}. "
                            "Keeping original text."
                        )
                        tracker.record(row_idx, row["date"], row["text"])
                        done_count += 1
                        failed_count += 1

                    # Periodic checkpoint: write output CSV
                    checkpoint_counter += 1
                    if (
                        checkpoint_interval > 0
                        and checkpoint_counter % checkpoint_interval == 0
                    ):
                        logger.info(f"  Checkpoint: saving {done_count} rows...")
                        _write_intermediate_output(all_rows, tracker, output_path)

        except KeyboardInterrupt:
            logger.warning("Interrupted! Progress saved. Re-run the same command to resume.")
            _write_intermediate_output(all_rows, tracker, output_path)
            if snapshotter:
                snapshotter.stop()
            return

        # Final output
        elapsed = time.time() - t_start
        logger.info(
            f"Pipeline complete in {elapsed:.1f}s "
            f"({total} rows, {failed_count} failed)"
        )

        _write_final_output(all_rows, tracker.get_all_completed(), output_path)

        # Final HuggingFace snapshot
        if snapshotter:
            logger.info("Pushing final snapshot to HuggingFace...")
            snapshotter.push()
            snapshotter.stop()

        # Clean up progress files on success
        tracker.cleanup()
        logger.info(f"Saved to {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Post-OCR correction pipeline (multi-threaded, with crash recovery)"
    )
    parser.add_argument(
        "--input",
        required=True,
        help="Path to input CSV file (Date, Text columns)",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Path to output CSV file. Default: input name with '_corrected' suffix.",
    )
    parser.add_argument(
        "--era",
        choices=["historical", "modern"],
        required=True,
        help="Era of the text: 'historical' (1880-1920) or 'modern' (1990+).",
    )
    parser.add_argument(
        "--skip-regex",
        action="store_true",
        help="Skip regex correction stage.",
    )
    parser.add_argument(
        "--skip-gpt",
        action="store_true",
        help="Skip GPT correction stage (only run unicode + regex).",
    )
    parser.add_argument(
        "--gpt-model",
        default="gpt-4o-mini",
        help="OpenAI model to use for GPT correction (default: gpt-4o-mini).",
    )
    parser.add_argument(
        "--api-key",
        default=None,
        help="OpenAI API key. Falls back to OPENAI_API_KEY env var.",
    )
    parser.add_argument(
        "--rows",
        default=None,
        help="Row range to process, e.g. '0-100' or '50'. Default: all rows.",
    )
    parser.add_argument(
        "--checkpoint",
        type=int,
        default=100,
        help="Save intermediate results every N rows (default: 100, 0 to disable).",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=4,
        help="Number of worker threads (default: 4).",
    )
    parser.add_argument(
        "--fresh",
        action="store_true",
        help="Discard previous progress and start from scratch.",
    )
    parser.add_argument(
        "--hf-repo",
        default=None,
        help="HuggingFace dataset repo for snapshots, e.g. 'username/dataset-name'.",
    )
    parser.add_argument(
        "--hf-token",
        default=None,
        help="HuggingFace token. Falls back to HUGGINGFACE_TOKEN env var.",
    )
    parser.add_argument(
        "--snapshot-interval",
        type=float,
        default=12.0,
        help="Hours between HuggingFace snapshots (default: 12.0 = twice daily).",
    )
    parser.add_argument(
        "--rpm",
        type=int,
        default=10000,
        help="OpenAI API rate limit in requests per minute (default: 10000).",
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
        num_workers=args.workers,
        fresh=args.fresh,
        hf_repo=args.hf_repo,
        hf_token=args.hf_token,
        snapshot_interval=args.snapshot_interval,
        rpm=args.rpm,
    )


if __name__ == "__main__":
    main()
