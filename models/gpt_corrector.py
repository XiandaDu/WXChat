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
from pathlib import Path

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

        chunks = self._chunk_text(text)
        if len(chunks) == 1:
            return self._correct_chunk(chunks[0])

        corrected_chunks = []
        for i, chunk in enumerate(chunks):
            logger.info(f"  GPT correcting chunk {i+1}/{len(chunks)} "
                        f"({len(chunk)} chars)")
            corrected = self._correct_chunk(chunk)
            corrected_chunks.append(corrected)

        return "\n\n".join(corrected_chunks)

    def _chunk_text(self, text: str) -> list[str]:
        """Split text into chunks that fit within the token limit.

        Splits on paragraph boundaries (double newlines) to preserve
        document structure. Falls back to single newlines if paragraphs
        are too large.
        """
        if len(text) <= MAX_CHUNK_CHARS:
            return [text]

        paragraphs = text.split("\n\n")
        chunks = []
        current_chunk = []
        current_len = 0

        for para in paragraphs:
            para_len = len(para) + 2  # +2 for the \n\n separator
            if current_len + para_len > MAX_CHUNK_CHARS and current_chunk:
                chunks.append("\n\n".join(current_chunk))
                current_chunk = []
                current_len = 0

            # If a single paragraph exceeds the limit, split by lines
            if para_len > MAX_CHUNK_CHARS:
                if current_chunk:
                    chunks.append("\n\n".join(current_chunk))
                    current_chunk = []
                    current_len = 0
                line_chunks = self._chunk_by_lines(para)
                chunks.extend(line_chunks)
            else:
                current_chunk.append(para)
                current_len += para_len

        if current_chunk:
            chunks.append("\n\n".join(current_chunk))

        return chunks

    def _chunk_by_lines(self, text: str) -> list[str]:
        """Split a large block by individual lines."""
        lines = text.split("\n")
        chunks = []
        current_lines = []
        current_len = 0

        for line in lines:
            line_len = len(line) + 1
            if current_len + line_len > MAX_CHUNK_CHARS and current_lines:
                chunks.append("\n".join(current_lines))
                current_lines = []
                current_len = 0
            current_lines.append(line)
            current_len += line_len

        if current_lines:
            chunks.append("\n".join(current_lines))

        return chunks

    def _correct_chunk(self, chunk: str) -> str:
        """Send a single chunk to the GPT API for correction."""
        self._rate_limit()

        for attempt in range(1, self.max_retries + 1):
            try:
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": self.system_prompt},
                        {"role": "user", "content": chunk},
                    ],
                    temperature=0.1,
                    max_tokens=MAX_CHUNK_TOKENS * 2,
                )
                return response.choices[0].message.content.strip()

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
        """Enforce rate limiting between API calls."""
        elapsed = time.time() - self._last_request_time
        if elapsed < self.min_interval:
            time.sleep(self.min_interval - elapsed)
        self._last_request_time = time.time()
