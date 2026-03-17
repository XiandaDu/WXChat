from __future__ import annotations

"""
GPT Corrector
=============
Uses OpenAI's GPT API for contextual OCR post-correction.
Chunks large texts, sends each through the LLM with a correction
prompt, and reassembles the corrected output.
"""

import os
import time
import logging
import threading
from pathlib import Path

import tiktoken
from openai import OpenAI

logger = logging.getLogger(__name__)

PROMPTS_DIR = Path(__file__).resolve().parent.parent / "prompts"

# Approximate token-to-character ratio for English text
CHARS_PER_TOKEN = 4
# Leave room for the system prompt and response overhead
MAX_CHUNK_TOKENS = 3000
MAX_CHUNK_CHARS = MAX_CHUNK_TOKENS * CHARS_PER_TOKEN


class GPTCorrector:
    """Contextual OCR correction using OpenAI GPT models."""

    def __init__(
        self,
        model: str = "gpt-4o-mini",
        api_key: str | None = None,
        max_retries: int = 3,
        requests_per_minute: int = 30,
    ):
        """
        Args:
            model: OpenAI model name (default: gpt-4o-mini for cost efficiency).
            api_key: OpenAI API key. Falls back to OPENAI_API_KEY env var.
            max_retries: Number of retries on transient API errors.
            requests_per_minute: Rate limit for API calls.
        """
        self.model = model
        self.max_retries = max_retries
        self.min_interval = 60.0 / requests_per_minute
        self._last_request_time = 0.0

        key = api_key or os.environ.get("OPENAI_API_KEY")
        if not key:
            raise ValueError(
                "OpenAI API key required. Set OPENAI_API_KEY environment "
                "variable or pass api_key parameter."
            )
        self.client = OpenAI(api_key=key)
        self.system_prompt = self._load_prompt()
        self._rate_lock = threading.Lock()

        # Token encoder for accurate token counting
        try:
            self._enc = tiktoken.encoding_for_model(self.model)
        except KeyError:
            self._enc = tiktoken.get_encoding("cl100k_base")

    def _load_prompt(self) -> str:
        """Load the correction prompt from prompts/ folder."""
        prompt_file = PROMPTS_DIR / "correction.txt"
        if not prompt_file.exists():
            raise FileNotFoundError(
                f"Prompt file not found: {prompt_file}"
            )
        return prompt_file.read_text(encoding="utf-8").strip()

    def correct(self, text: str) -> str:
        """Correct the full text, chunking if necessary."""
        if not text.strip():
            return text

        chunks, separators = self._chunk_text(text)
        if len(chunks) == 1:
            return self._correct_chunk(chunks[0])

        corrected_chunks = []
        for i, chunk in enumerate(chunks):
            logger.info(f"  GPT correcting chunk {i+1}/{len(chunks)} "
                        f"({len(chunk)} chars)")
            corrected = self._correct_chunk(chunk)
            corrected_chunks.append(corrected)

        # Reassemble using the original separators between chunks
        result = corrected_chunks[0]
        for sep, chunk in zip(separators, corrected_chunks[1:]):
            result += sep + chunk
        return result

    def _chunk_text(self, text: str) -> tuple[list[str], list[str]]:
        """Split text into chunks that fit within the token limit.

        Returns (chunks, separators) where separators[i] is the string
        that originally appeared between chunks[i] and chunks[i+1].
        This allows faithful reassembly after correction.

        Splits on paragraph boundaries (double newlines) to preserve
        document structure. Falls back to single newlines if paragraphs
        are too large.
        """
        if len(text) <= MAX_CHUNK_CHARS:
            return [text], []

        paragraphs = text.split("\n\n")
        chunks: list[str] = []
        separators: list[str] = []
        current_chunk: list[str] = []
        current_len = 0

        for para in paragraphs:
            para_len = len(para) + 2  # +2 for the \n\n separator
            if current_len + para_len > MAX_CHUNK_CHARS and current_chunk:
                chunks.append("\n\n".join(current_chunk))
                separators.append("\n\n")
                current_chunk = []
                current_len = 0

            # If a single paragraph exceeds the limit, split by lines
            if para_len > MAX_CHUNK_CHARS:
                if current_chunk:
                    chunks.append("\n\n".join(current_chunk))
                    separators.append("\n\n")
                    current_chunk = []
                    current_len = 0
                line_chunks, line_seps = self._chunk_by_lines(para)
                if chunks and line_chunks:
                    separators.append("\n\n")
                chunks.extend(line_chunks)
                separators.extend(line_seps)
            else:
                current_chunk.append(para)
                current_len += para_len

        if current_chunk:
            if chunks:
                separators.append("\n\n")
            chunks.append("\n\n".join(current_chunk))

        return chunks, separators

    def _chunk_by_lines(self, text: str) -> tuple[list[str], list[str]]:
        """Split a large block by individual lines.

        Returns (chunks, separators) for faithful reassembly.
        Falls back to a sliding window over characters when a single
        line exceeds MAX_CHUNK_CHARS, breaking at sentence-ending
        punctuation or spaces.
        """
        lines = text.split("\n")
        chunks: list[str] = []
        separators: list[str] = []
        current_lines: list[str] = []
        current_len = 0

        for line in lines:
            line_len = len(line) + 1
            if current_len + line_len > MAX_CHUNK_CHARS and current_lines:
                chunks.append("\n".join(current_lines))
                separators.append("\n")
                current_lines = []
                current_len = 0

            # Single line exceeds the limit – split by character window
            if line_len > MAX_CHUNK_CHARS:
                if current_lines:
                    chunks.append("\n".join(current_lines))
                    separators.append("\n")
                    current_lines = []
                    current_len = 0
                char_chunks = self._chunk_by_chars(line)
                if chunks and char_chunks:
                    separators.append("\n")
                for j, cc in enumerate(char_chunks):
                    chunks.append(cc)
                    if j < len(char_chunks) - 1:
                        # Character-level splits had no separator; use empty string
                        separators.append("")
            else:
                current_lines.append(line)
                current_len += line_len

        if current_lines:
            if chunks:
                separators.append("\n")
            chunks.append("\n".join(current_lines))

        return chunks, separators

    def _chunk_by_chars(self, text: str) -> list[str]:
        """Sliding-window split for text with no line breaks.

        Tries to break at sentence-ending punctuation (.!?。！？)
        within the last 20% of the window. Falls back to the last
        space, then to a hard cut at MAX_CHUNK_CHARS.
        """
        chunks = []
        start = 0
        length = len(text)

        while start < length:
            end = start + MAX_CHUNK_CHARS
            if end >= length:
                chunks.append(text[start:])
                break

            # Search for a sentence boundary in the last 20% of the window
            search_start = start + int(MAX_CHUNK_CHARS * 0.8)
            best = -1
            for i in range(end - 1, search_start - 1, -1):
                if text[i] in ".!?。！？":
                    best = i + 1  # include the punctuation
                    break

            if best > start:
                chunks.append(text[start:best])
                start = best
                continue

            # Fall back to the last space in the window
            space_pos = text.rfind(" ", search_start, end)
            if space_pos > start:
                chunks.append(text[start:space_pos + 1])  # include the trailing space
                start = space_pos + 1
                continue

            # No good break point – hard cut
            chunks.append(text[start:end])
            start = end

        return chunks

    def _correct_chunk(self, chunk: str) -> str:
        """Send a single chunk to the GPT API for correction."""
        self._rate_limit()

        # Dynamically compute max_tokens based on actual input token count
        # to avoid truncating the response.  OCR text often has a higher
        # token-per-character ratio than normal English.
        input_tokens = len(self._enc.encode(chunk))
        dynamic_max_tokens = int(input_tokens * 1.5)
        # Ensure a reasonable minimum and cap to model limits
        dynamic_max_tokens = max(dynamic_max_tokens, 1024)
        dynamic_max_tokens = min(dynamic_max_tokens, 16384)

        for attempt in range(1, self.max_retries + 1):
            try:
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": self.system_prompt},
                        {"role": "user", "content": chunk},
                    ],
                    temperature=0.1,
                    max_tokens=dynamic_max_tokens,
                )
                result = response.choices[0].message.content
                if result is None:
                    logger.warning("  GPT returned None content, keeping original.")
                    return chunk
                return result.strip()

            except Exception as e:
                logger.warning(
                    f"  GPT API error (attempt {attempt}/{self.max_retries}): {e}"
                )
                if attempt < self.max_retries:
                    backoff = 2 ** attempt
                    logger.info(f"  Retrying in {backoff}s...")
                    time.sleep(backoff)
                else:
                    logger.error(
                        "  GPT correction failed after all retries. "
                        "Returning original text."
                    )
                    return chunk

    def _rate_limit(self):
        """Enforce rate limiting between API calls (thread-safe)."""
        with self._rate_lock:
            elapsed = time.time() - self._last_request_time
            if elapsed < self.min_interval:
                time.sleep(self.min_interval - elapsed)
            self._last_request_time = time.time()
