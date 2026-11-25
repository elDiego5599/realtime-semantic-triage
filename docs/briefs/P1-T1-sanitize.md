# Brief P1-T1: project scaffold and text sanitizer

## Context

Repository `realtime-semantic-triage`, a Python project that deduplicates security advisories (NVD and GitHub advisories) by meaning. The CVE ID is the ground-truth label, so every CVE and GHSA identifier and every URL must be removed from advisory text before it is embedded. Otherwise the model would match duplicates by identifier instead of by content.

## Task

1. Create the Python project scaffold.
2. Implement `src/semtriage/sanitize.py` so that every test in `tests/test_sanitize.py` passes.

## Files to create

- `pyproject.toml`
  - project name `semtriage`, version `0.1.0`, `requires-python = ">=3.12"`, no runtime dependencies.
  - build backend `hatchling`, src layout (`src/semtriage`).
  - dev dependency group (`[dependency-groups] dev`) with `pytest` and `ruff`.
  - `[tool.pytest.ini_options]`: `testpaths = ["tests"]`, `addopts = "-x -q"`.
  - `[tool.ruff]`: `line-length = 100`, `target-version = "py312"`, lint rules `E`, `F`, `I`, `UP`, `B`.
- `src/semtriage/__init__.py` (empty docstring only).
- `src/semtriage/sanitize.py`.
- `.python-version` containing `3.12`.

Do not modify `tests/test_sanitize.py`. Do not create any other file.

## Interface (exact)

```python
def sanitize(text: str) -> str: ...
def find_identifiers(text: str) -> set[str]: ...
```

## Behavior

- CVE identifier: `CVE-` followed by a 4 digit year, `-`, and 4 or more digits, case-insensitive, on word boundaries. `CVE-2026-1` is not an identifier.
- GHSA identifier: `GHSA-` followed by three groups of 4 alphanumeric characters separated by `-`, case-insensitive, on word boundaries.
- URL: `http://` or `https://` followed by non-whitespace characters. Trailing punctuation `.,;:!?)]'"` is not part of the URL.
- `sanitize`:
  1. Remove URLs, then CVE identifiers, then GHSA identifiers.
  2. Remove pairs of `()` or `[]` that contain only whitespace after step 1.
  3. Within each line, collapse runs of spaces and tabs into one space and strip the line.
  4. Collapse 3 or more consecutive newlines into exactly 2.
  5. Strip the whole result.
  - Text with nothing to remove must come back unchanged (see the tests).
- `find_identifiers`: returns every CVE and GHSA identifier found anywhere in the text, including inside URLs, uppercased, as a set.
- Compile regular expressions once at module level. Standard library only. Type hints and a short module docstring. No print statements.

## Commands

```
uv sync
uv run pytest
uv run ruff check .
uv run ruff format --check .
```

## Done criteria

- `uv run pytest` passes every test.
- `uv run ruff check .` and `uv run ruff format --check .` report no issues.
- Only the files listed above were created. Do not run any git command.
- Finish with a short summary of what you did and the final test output.
