"""Text sanitization utilities for security advisories."""

import re

_SENTINEL = "\x00"
_DASH = "-\u2010\u2011\u2012\u2013\u2014\u2015"

_CVE_RE = re.compile(
    rf"(?<![A-Za-z0-9])CVE[{_DASH}][0-9]{{4}}[{_DASH}][0-9]{{4,}}(?![0-9])",
    re.IGNORECASE,
)
_GHSA_RE = re.compile(
    rf"(?<![A-Za-z0-9])GHSA(?:[{_DASH}][A-Za-z0-9]{{4}}){{3}}(?![A-Za-z0-9])",
    re.IGNORECASE,
)
_URL_RE = re.compile(
    r"""https?://(?:[^\s<>"'()`]|\([^\s<>"'()`]*\))*""",
    re.IGNORECASE,
)
_MARKDOWN_LINK_RE = re.compile(rf"\[([^\[\]]*)\]\({_SENTINEL}\)")
_EMPTY_PAIR_RES = (
    re.compile(rf"\([\s,;:{_SENTINEL}]*{_SENTINEL}[\s,;:{_SENTINEL}]*\)"),
    re.compile(rf"\[[\s,;:{_SENTINEL}]*{_SENTINEL}[\s,;:{_SENTINEL}]*\]"),
    re.compile(rf"<[\s,;:{_SENTINEL}]*{_SENTINEL}[\s,;:{_SENTINEL}]*>"),
    re.compile(rf"(?<![^\s{_SENTINEL}(\[<])`{_SENTINEL}+`(?![^\s{_SENTINEL})\]>.,;:!?])"),
    re.compile(
        rf"(?<![^\s{_SENTINEL}(\[<])\*\*"
        rf"[\s,;:{_SENTINEL}]*{_SENTINEL}[\s,;:{_SENTINEL}]*"
        rf"\*\*(?![^\s{_SENTINEL})\]>.,;:!?])"
    ),
)
_LABEL = (
    r"(?:(?:fix|patch|upstream|github|gitlab|nvd|security|original) )?"
    r"(?:fix|patch|commit|url|link|references?|origin|repository|repo|advisory"
    r"|cve|cve ids?|ghsa|ghsa id|pr|pull request|issue|report|source|patchwork)"
)
# Residue that only exists because a removed identifier or URL left it behind.
# Every pattern requires a sentinel, so text without removals is never touched.
_RESIDUE_RES = (
    # "A, B and C" or "A / B" where every item was removed.
    re.compile(
        rf"{_SENTINEL}(?:[ \t]*(?:[,/|&;]|(?<![A-Za-z])(?:and|or)(?![A-Za-z])))+[ \t]*{_SENTINEL}",
        re.IGNORECASE,
    ),
    # Template sentence whose only content was the identifier.
    re.compile(
        rf"\bThis (?:vulnerability|issue|advisory) corresponds to[ \t]*{_SENTINEL}+[ \t]*(?:\.|$)",
        re.IGNORECASE | re.MULTILINE,
    ),
    # A label line such as "CVE ID: <id>" or "- Fix commit: <url>" left without a value.
    re.compile(
        rf"^[ \t]*(?:[-*+][ \t]+)?(?:\*\*)?{_LABEL}(?:\*\*)?:(?:\*\*)?[ \t]*"
        rf"{_SENTINEL}[{_SENTINEL} \t.,;]*$",
        re.IGNORECASE | re.MULTILINE,
    ),
    # A list item or line left with only markers and punctuation.
    re.compile(
        rf"^(?=[^\n]*{_SENTINEL})(?:[ \t]*(?:[-*+>]|#+|\d+\.|\[\d+\]))?"
        rf"[{_SENTINEL} \t`'\".,;:/|\\-]*$",
        re.MULTILINE,
    ),
)
_UNICODE_DASH_RE = re.compile(r"[\u2010-\u2015]")
_HORIZONTAL_WHITESPACE_RE = re.compile(r"[ \t]+")
_EXCESS_NEWLINES_RE = re.compile(r"\n{3,}")


def _trim_url(url: str) -> str:
    """Drop trailing punctuation and unmatched closing brackets from a URL."""
    while url:
        last = url[-1]
        if last in ".,;:!?*}":
            url = url[:-1]
        elif last == "]" and url.count("]") > url.count("["):
            url = url[:-1]
        else:
            break
    return url


def _replace_url(match: re.Match[str]) -> str:
    matched = match.group(0)
    trimmed = _trim_url(matched)
    return _SENTINEL + matched[len(trimmed) :]


def _clean_empty_markup(text: str) -> str:
    """Remove bracket pairs and other residue that a removal left empty, innermost first."""
    while True:
        updated = text
        for pattern in (*_EMPTY_PAIR_RES, *_RESIDUE_RES):
            updated = pattern.sub(_SENTINEL, updated)
        if updated == text:
            return text
        text = updated


def _canonical_identifier(identifier: str) -> str:
    return _UNICODE_DASH_RE.sub("-", identifier).upper()


def sanitize(text: str) -> str:
    """Remove identifiers and URLs, and normalize advisory text whitespace."""
    result = text.replace("\r\n", "\n")
    while True:
        updated = _URL_RE.sub(_replace_url, result)
        updated = _CVE_RE.sub(_SENTINEL, updated)
        updated = _GHSA_RE.sub(_SENTINEL, updated)
        updated = _MARKDOWN_LINK_RE.sub(r"\1", updated)
        updated = _clean_empty_markup(updated)
        updated = updated.replace(_SENTINEL, "")
        if updated == result:
            break
        result = updated

    lines = []
    for line in result.split("\n"):
        line = _HORIZONTAL_WHITESPACE_RE.sub(" ", line)
        lines.append(line.strip())

    result = "\n".join(lines)
    result = _EXCESS_NEWLINES_RE.sub("\n\n", result)
    return result.strip()


def find_identifiers(text: str) -> set[str]:
    """Return unique CVE and GHSA identifiers found in text, uppercased."""
    return {
        _canonical_identifier(match.group(0))
        for pattern in (_CVE_RE, _GHSA_RE)
        for match in pattern.finditer(text)
    }
