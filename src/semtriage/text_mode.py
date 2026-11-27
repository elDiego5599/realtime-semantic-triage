"""Text selection for embedding: full sanitized text, a truncated text or a lead."""

import re

from semtriage.models import Event, Source
from semtriage.sanitize import sanitize

TEXT_MODES: tuple[str, ...] = ("full", "truncate", "lead")

DEFAULT_MAX_CHARS = 1000

_FENCE_RE = re.compile(r"^\s*(`{3,}|~{3,})(.*)$")
_HEADING_RE = re.compile(r"^(?:#{1,6}\s+(?P<hash>.*?)[\s#]*|\*\*(?P<bold>[^*]+)\*\*:?)$")
_SUMMARY_HEADINGS = frozenset({"summary", "executive summary"})


def truncate_text(text: str, max_chars: int) -> str:
    """Return text cut to at most `max_chars` characters at a whitespace boundary.

    A single word longer than the limit is cut inside the word. Trailing
    whitespace is stripped.
    """
    if max_chars < 1:
        raise ValueError(f"max_chars must be positive, got {max_chars}")
    if len(text) <= max_chars:
        return text
    cut = text[:max_chars]
    if not text[max_chars].isspace():
        tail = re.search(r"\s\S*$", cut)
        if tail is not None and tail.start() > 0:  # a leading space alone is no boundary
            cut = cut[: tail.start()]
    return cut.rstrip()


def _heading(line: str) -> str | None:
    """Return the lowercased heading text of a heading line, else None.

    Markdown `#` headings always count. A line that is only bold text counts
    only when it names a summary, since bold sub-labels are common inside sections.
    """
    match = _HEADING_RE.match(line.strip())
    if match is None:
        return None
    if match.group("hash") is not None:
        return match.group("hash").strip().rstrip(":").strip().lower()
    name = match.group("bold").strip().rstrip(":").strip().lower()
    return name if name in _SUMMARY_HEADINGS else None


def _without_code_blocks(lines: list[str]) -> list[str]:
    """Drop fenced code blocks (``` or ~~~); an unclosed fence drops the rest.

    A block closes on a run of the same character at least as long as the opening
    run with nothing after it. A line whose marker repeats later on the same line
    (inline code such as ```x```) does not open a block.
    """
    kept: list[str] = []
    fence: str | None = None
    for line in lines:
        match = _FENCE_RE.match(line)
        if fence is None:
            if match is None or match.group(1) in match.group(2):
                kept.append(line)
            else:
                fence = match.group(1)
        elif match is not None and match.group(1).startswith(fence) and not match.group(2).strip():
            fence = None
    return kept


def _paragraphs(lines: list[str]) -> list[str]:
    """Group non-blank lines into paragraphs separated by blank lines."""
    paragraphs: list[list[str]] = [[]]
    for line in lines:
        if line.strip():
            paragraphs[-1].append(line)
        elif paragraphs[-1]:
            paragraphs.append([])
    return ["\n".join(paragraph) for paragraph in paragraphs if paragraph]


def lead_text(text: str) -> str:
    """Return the title line plus the lead of a GHSA-style markdown write-up.

    The first line is the title (the advisory summary) unless it is a heading.
    Code blocks are dropped. The lead is the section under a `Summary` heading
    when it has text, otherwise the first paragraph that is not a heading
    (headings end paragraphs). Setext headings (underlined with `---`) are not
    recognised and stay as text.
    Later sections are dropped. When the Summary section is missing or has no
    text, the lead is the first paragraph found anywhere after the title, which
    may come from any section.
    """
    lines = _without_code_blocks(text.split("\n"))
    title = ""
    if lines and _heading(lines[0]) is None:
        title = lines[0].strip()
        lines = lines[1:]
    lead = ""
    for index, line in enumerate(lines):
        if _heading(line) in _SUMMARY_HEADINGS:
            section: list[str] = []
            for following in lines[index + 1 :]:
                if _heading(following) is not None:
                    break
                section.append(following)
            lead = "\n\n".join(_paragraphs(section))
            if lead:
                break
    if not lead:
        body = ["" if _heading(line) is not None else line for line in lines]
        paragraphs = _paragraphs(body)
        lead = paragraphs[0] if paragraphs else ""
    return "\n\n".join(part for part in (title, lead) if part)


def _check_text_mode(text_mode: str, max_chars: int) -> None:
    if text_mode not in TEXT_MODES:
        raise ValueError(f"text_mode must be one of {TEXT_MODES}, got {text_mode!r}")
    if max_chars < 1:
        raise ValueError(f"max_chars must be positive, got {max_chars}")


def text_mode_label(text_mode: str, max_chars: int = DEFAULT_MAX_CHARS) -> str:
    """Return the label stored in the embedding cache, for example `lead:1000`."""
    _check_text_mode(text_mode, max_chars)
    return "full" if text_mode == "full" else f"{text_mode}:{max_chars}"


def parse_text_mode_label(label: str) -> tuple[str, int]:
    """Parse a `text_mode_label` back into a text mode and its character cap."""
    if label == "full":
        return "full", DEFAULT_MAX_CHARS
    mode, sep, limit = label.partition(":")
    if mode in TEXT_MODES and mode != "full" and sep and limit.isdigit() and int(limit) > 0:
        return mode, int(limit)
    raise ValueError(f"bad text mode label {label!r}")


def build_text(event: Event, text_mode: str, *, max_chars: int = DEFAULT_MAX_CHARS) -> str:
    """Return the text to embed for an event.

    The event text is sanitized first, exactly as in `full` mode, and only then
    shortened: shortening only removes whole lines or a tail cut at whitespace,
    so it cannot rebuild an identifier or URL that sanitizing removed, while
    shortening first could cut an identifier or URL into a fragment the
    sanitizer patterns no longer match. `truncate` caps any text at `max_chars`;
    `lead` keeps the title and lead of GHSA write-ups (see `lead_text`, or the
    whole text when that is empty) and the plain description of NVD events,
    then caps at `max_chars`.
    """
    _check_text_mode(text_mode, max_chars)
    text = sanitize(event.text)
    if text_mode == "full":
        return text
    if text_mode == "lead" and event.source == Source.GHSA:
        text = lead_text(text) or text
    return truncate_text(text, max_chars)


__all__ = [
    "DEFAULT_MAX_CHARS",
    "TEXT_MODES",
    "build_text",
    "lead_text",
    "parse_text_mode_label",
    "text_mode_label",
    "truncate_text",
]
