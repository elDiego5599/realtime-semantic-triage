"""Text sanitization utilities for security advisories."""

import re

_CVE_RE = re.compile(r"\bCVE-\d{4}-\d{4,}\b", re.IGNORECASE)
_GHSA_RE = re.compile(
    r"\bGHSA-[A-Za-z0-9]{4}-[A-Za-z0-9]{4}-[A-Za-z0-9]{4}\b",
    re.IGNORECASE,
)
_URL_RE = re.compile(r"https?://[^\s]*[^\s.,;:!?)\]}'\"]", re.IGNORECASE)
_EMPTY_BRACKETS_RE = re.compile(r"\(\s*\)|\[\s*\]")
_HORIZONTAL_WHITESPACE_RE = re.compile(r"[ \t]+")
_EXCESS_NEWLINES_RE = re.compile(r"\n{3,}")


def sanitize(text: str) -> str:
    """Remove identifiers and URLs, and normalize advisory text whitespace."""
    result = _URL_RE.sub("", text)
    result = _CVE_RE.sub("", result)
    result = _GHSA_RE.sub("", result)
    result = _EMPTY_BRACKETS_RE.sub("", result)

    lines = []
    for line in result.split("\n"):
        line = _HORIZONTAL_WHITESPACE_RE.sub(" ", line)
        lines.append(line.strip())

    result = "\n".join(lines)
    result = _EXCESS_NEWLINES_RE.sub("\n\n", result)
    return result.strip()


def find_identifiers(text: str) -> set[str]:
    """Return unique CVE and GHSA identifiers found in text, uppercased."""
    identifiers = {match.group(0).upper() for match in _CVE_RE.finditer(text)}
    identifiers.update(match.group(0).upper() for match in _GHSA_RE.finditer(text))
    return identifiers
