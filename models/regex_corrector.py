"""
Regex Corrector
===============
Pattern-based OCR error correction using compiled regular expressions.
Includes general OCR fixes plus era-specific rules for historical and modern text.
"""

import re


# ---------------------------------------------------------------------------
# General OCR character-confusion rules (apply to all text)
# ---------------------------------------------------------------------------
GENERAL_CORRECTIONS = [
    # --- Character confusion ---
    # 'I' between lowercase letters → 'l'
    (r"(?<=[a-z])I(?=[a-z])", "l"),
    # '1' at word start followed by lowercase → 'l'
    (r"\b1(?=[a-z]{2,})", "l"),
    # '|' between letters → 'l'
    (r"(?<=[a-zA-Z])\|(?=[a-zA-Z])", "l"),
    # '0' between uppercase → 'O'
    (r"(?<=[A-Z])0(?=[A-Z])", "O"),
    # '0' between lowercase → 'o'
    (r"(?<=[a-z])0(?=[a-z])", "o"),
    # 'rn' before vowels in word-internal position → 'm'
    (r"(?<=[a-z])rn(?=[aeiouy])", "m"),
    # 'vv' → 'w' (common in degraded scans)
    (r"(?<=[a-z])vv(?=[a-z])", "w"),
    # 'cl' misread as 'd' — reverse: 'd' often correct, skip
    # 'ii' at start of word → 'u' (common in historical)
    (r"\bii(?=[a-z])", "u"),

    # --- Common OCR misreads (whole words) ---
    (r"\btbe\b", "the"),
    (r"\bTbe\b", "The"),
    (r"\btlie\b", "the"),
    (r"\bTlie\b", "The"),
    (r"\btbe\b", "the"),
    (r"\bTbe\b", "The"),
    (r"\bwbich\b", "which"),
    (r"\bwliich\b", "which"),
    (r"\bwilich\b", "which"),
    (r"\btbat\b", "that"),
    (r"\bTbat\b", "That"),
    (r"\btbis\b", "this"),
    (r"\bTbis\b", "This"),
    (r"\bwitb\b", "with"),
    (r"\bbave\b", "have"),
    (r"\bHave\b", "Have"),
    (r"\bwben\b", "when"),
    (r"\bWben\b", "When"),
    (r"\btben\b", "then"),
    (r"\bTben\b", "Then"),
    (r"\btbey\b", "they"),
    (r"\bTbey\b", "They"),
    (r"\btbeir\b", "their"),
    (r"\bTbeir\b", "Their"),
    (r"\botber\b", "other"),
    (r"\bOtber\b", "Other"),
    (r"\bafter\b", "after"),
    (r"\bnurnber\b", "number"),
    (r"\bmernber\b", "member"),
    (r"\bSepternber\b", "September"),
    (r"\bNovernber\b", "November"),
    (r"\bDecernber\b", "December"),

    # --- Punctuation and spacing ---
    # Space before punctuation
    (r"\s+([.,;:!?])", r"\1"),
    # Missing space after punctuation (letter immediately after)
    (r"([.,;:!?])(?=[A-Za-z])", r"\1 "),
    # Hyphenation at line breaks (re-join words)
    (r"-\n\s*", ""),
    # Multiple spaces → single space
    (r"[ \t]{2,}", " "),
    # Multiple periods (not ellipsis) → single period
    (r"\.{4,}", "..."),
]

# ---------------------------------------------------------------------------
# Historical-era specific rules (Victorian newspapers 1880-1920)
# ---------------------------------------------------------------------------
HISTORICAL_CORRECTIONS = [
    # Heavy degradation patterns
    # Single letters separated by spaces/dots (fragmented text)
    (r"(?<= )[.:;,]{2,}(?= )", " "),
    # Sequences of single characters with dots/spaces (garbled text)
    (r"(?:^|\n)[\s.,:;|!'\-]{10,}(?:\n|$)", "\n"),
    # 'f' misread as long-s 'ſ' artifacts
    (r"\bf\s?he\b", "the"),
    (r"\bf\s?hat\b", "that"),
    (r"\bf\s?his\b", "this"),
    # 'ct' misread as 'el'
    (r"\bseledl(?=ed|ion|ing)", "select"),
    # Common Victorian abbreviations preserved
    # Double-quote artifacts from OCR
    (r'"{2,}', '"'),
    # Excessive dashes
    (r"-{3,}", "--"),
    # Remove lines that are mostly punctuation/symbols (< 20% alpha)
    # (handled in garbage_filter below)
]

# ---------------------------------------------------------------------------
# Modern-era specific rules (1990-present)
# ---------------------------------------------------------------------------
MODERN_CORRECTIONS = [
    # Score formatting: fix common sports score OCR errors
    (r"(\d)\s*-\s*(\d)", r"\1-\2"),
    # Fix "I" that should be "1" in numeric contexts
    (r"(?<=\d)I(?=\d)", "1"),
    (r"(?<=\s)I(?=\d{2,})", "1"),
    # Fix "O" that should be "0" in numeric contexts
    (r"(?<=\d)O(?=\d)", "0"),
    # Fix broken dollar amounts
    (r"\$\s+(\d)", r"$\1"),
    # Fix percentage signs
    (r"(\d)\s+%", r"\1%"),
    # Common modern newspaper OCR misreads
    (r"\bgovemment\b", "government"),
    (r"\bGovemment\b", "Government"),
    (r"\bmanagernent\b", "management"),
    (r"\bManagernent\b", "Management"),
    (r"\bdeveloprnent\b", "development"),
    (r"\bDeveloprnent\b", "Development"),
    (r"\benviromnent\b", "environment"),
    (r"\bEnviromnent\b", "Environment"),
    (r"\bgovernrnent\b", "government"),
    (r"\bGovernrnent\b", "Government"),
]


def _compile_rules(rules: list) -> list:
    """Compile a list of (pattern, replacement) tuples."""
    return [(re.compile(p), r) for p, r in rules]


_GENERAL_COMPILED = _compile_rules(GENERAL_CORRECTIONS)
_HISTORICAL_COMPILED = _compile_rules(HISTORICAL_CORRECTIONS)
_MODERN_COMPILED = _compile_rules(MODERN_CORRECTIONS)


class RegexCorrector:
    """Apply regex-based OCR error corrections."""

    def __init__(self, era: str = "general"):
        """
        Args:
            era: One of "general", "historical", "modern".
                 Controls which additional era-specific rules to apply.
        """
        self.era = era
        self._rules = list(_GENERAL_COMPILED)
        if era == "historical":
            self._rules.extend(_HISTORICAL_COMPILED)
        elif era == "modern":
            self._rules.extend(_MODERN_COMPILED)

    def correct(self, text: str) -> str:
        """Apply all regex corrections to the text."""
        for pat, repl in self._rules:
            text = pat.sub(repl, text)
        text = self._filter_garbage_lines(text)
        text = self._collapse_blank_lines(text)
        return text.strip()

    def _filter_garbage_lines(self, text: str) -> str:
        """Remove lines that are mostly non-alphabetic (OCR garbage)."""
        threshold = 0.20 if self.era == "historical" else 0.25
        lines = text.split("\n")
        cleaned = []
        for line in lines:
            s = line.strip()
            if not s:
                cleaned.append("")
                continue
            alpha_count = sum(1 for c in s if c.isalpha())
            alpha_ratio = alpha_count / len(s)
            # Keep lines with enough alpha content, or very short lines
            # that might be dates, numbers, etc.
            if alpha_ratio >= threshold or len(s) <= 5:
                cleaned.append(line)
        return "\n".join(cleaned)

    def _collapse_blank_lines(self, text: str) -> str:
        """Collapse 3+ consecutive blank lines into 2."""
        return re.sub(r"\n{3,}", "\n\n", text)
