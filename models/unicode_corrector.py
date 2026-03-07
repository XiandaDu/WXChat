"""
Unicode Corrector
=================
Fixes encoding issues, normalizes Unicode, removes control characters,
and handles common mojibake patterns from OCR output.
"""

import re
import unicodedata


# Common mojibake patterns: (broken sequence → correct character)
MOJIBAKE_MAP = {
    "â\x80\x99": "\u2019",   # '
    "â\x80\x9c": "\u201c",   # "
    "â\x80\x9d": "\u201d",   # "
    "â\x80\x93": "\u2013",   # –
    "â\x80\x94": "\u2014",   # —
    "â\x80\xa6": "\u2026",   # …
    "Ã©": "é",
    "Ã¨": "è",
    "Ã ": "à",
    "Ã¢": "â",
    "Ã®": "î",
    "Ã´": "ô",
    "Ã¼": "ü",
    "Ã¶": "ö",
    "Ã¤": "ä",
    "Ã±": "ñ",
    "Ã§": "ç",
    "Â£": "£",
    "Â§": "§",
    "Â°": "°",
    "Â½": "½",
    "Â¼": "¼",
    "Â¾": "¾",
    "ï¬\x81": "fi",
    "ï¬\x82": "fl",
    "ï¬\x80": "ff",
    "ï¬\x83": "ffi",
    "ï¬\x84": "ffl",
}

# Fancy Unicode → ASCII equivalents for OCR text normalization
UNICODE_TO_ASCII = {
    "\u2018": "'",   # '
    "\u2019": "'",   # '
    "\u201a": "'",   # ‚
    "\u201c": '"',   # "
    "\u201d": '"',   # "
    "\u201e": '"',   # „
    "\u2013": "-",   # –
    "\u2014": "--",  # —
    "\u2026": "...", # …
    "\u00a0": " ",   # non-breaking space
    "\u2002": " ",   # en space
    "\u2003": " ",   # em space
    "\u2009": " ",   # thin space
    "\u200a": " ",   # hair space
    "\u200b": "",    # zero-width space
    "\u200c": "",    # zero-width non-joiner
    "\u200d": "",    # zero-width joiner
    "\ufeff": "",    # BOM
    "\u00ad": "",    # soft hyphen
    "\ufffd": "",    # replacement character
    "\u00b7": ".",   # middle dot
    "\u2022": "-",   # bullet
    "\u00ab": '"',   # «
    "\u00bb": '"',   # »
}

# Ligature decomposition
LIGATURES = {
    "\ufb01": "fi",
    "\ufb02": "fl",
    "\ufb00": "ff",
    "\ufb03": "ffi",
    "\ufb04": "ffl",
    "\u0132": "IJ",
    "\u0133": "ij",
    "\u0152": "OE",
    "\u0153": "oe",
    "\u00c6": "AE",
    "\u00e6": "ae",
}

# Control characters to strip (keep \n, \r, \t)
_CONTROL_RE = re.compile(
    r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\x9f]"
)

# Repeated replacement character or other garbage Unicode
_GARBAGE_UNICODE_RE = re.compile(
    r"[\ufffd\ufffe\uffff]{2,}"
)


class UnicodeCorrector:
    """Normalize and fix Unicode/encoding issues in OCR text."""

    def __init__(self, normalize_form: str = "NFC"):
        self.normalize_form = normalize_form

    def correct(self, text: str) -> str:
        """Run all Unicode corrections on the input text."""
        text = self._fix_mojibake(text)
        text = self._decompose_ligatures(text)
        text = self._normalize_unicode_chars(text)
        text = self._strip_control_chars(text)
        text = self._remove_garbage_unicode(text)
        text = self._normalize_form(text)
        text = self._fix_whitespace(text)
        return text

    def _fix_mojibake(self, text: str) -> str:
        """Fix common UTF-8 mojibake patterns."""
        for broken, fixed in MOJIBAKE_MAP.items():
            text = text.replace(broken, fixed)
        return text

    def _decompose_ligatures(self, text: str) -> str:
        """Replace typographic ligatures with their component letters."""
        for lig, decomposed in LIGATURES.items():
            text = text.replace(lig, decomposed)
        return text

    def _normalize_unicode_chars(self, text: str) -> str:
        """Replace fancy Unicode punctuation with ASCII equivalents."""
        for fancy, plain in UNICODE_TO_ASCII.items():
            text = text.replace(fancy, plain)
        return text

    def _strip_control_chars(self, text: str) -> str:
        """Remove non-printable control characters (preserving newlines/tabs)."""
        return _CONTROL_RE.sub("", text)

    def _remove_garbage_unicode(self, text: str) -> str:
        """Remove sequences of Unicode replacement/garbage characters."""
        return _GARBAGE_UNICODE_RE.sub("", text)

    def _normalize_form(self, text: str) -> str:
        """Apply Unicode normalization (NFC by default)."""
        return unicodedata.normalize(self.normalize_form, text)

    def _fix_whitespace(self, text: str) -> str:
        """Normalize various whitespace characters to standard space."""
        # Replace any remaining Unicode whitespace variants with regular space
        text = re.sub(r"[^\S\n\r\t]+", " ", text)
        return text
