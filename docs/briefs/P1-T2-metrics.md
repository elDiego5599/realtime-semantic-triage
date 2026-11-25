# Brief P1-T2: clustering metrics (pairwise and B-cubed)

## Context

Repository `realtime-semantic-triage` (Python, `uv`, src layout, package `semtriage`). Deduplication of security advisories is evaluated as clustering: each advisory has a gold cluster label (its CVE ID) and a predicted cluster label. This task adds the two metrics used for every experiment.

## Task

Implement `src/semtriage/metrics.py` so that every test in `tests/test_metrics.py` passes.

Do not modify any test. Do not create or modify any other file. Keep `uv.lock`. Do not change `pyproject.toml`: in particular, never loosen the ruff or pytest configuration to make checks pass. Do not run any git command.

## Interface (exact)

```python
from collections.abc import Hashable, Mapping
from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class Scores:
    precision: float
    recall: float
    f1: float


def pairwise_scores(pred: Mapping[str, Hashable], gold: Mapping[str, Hashable]) -> Scores: ...
def bcubed_scores(pred: Mapping[str, Hashable], gold: Mapping[str, Hashable]) -> Scores: ...
```

Both mappings go from item ID to cluster label. Labels are any hashable value, including `None`.

## Behavior

- Validation (both functions): raise `ValueError` if the mappings are empty or if their key sets differ.
- F1 is the harmonic mean of precision and recall, and 0.0 when both are 0.
- Pairwise: a pair is an unordered pair of distinct items in the same cluster.
  - precision = true positive pairs / predicted pairs, recall = true positive pairs / gold pairs.
  - If there are no predicted pairs, precision is 1.0. If there are no gold pairs, recall is 1.0.
  - Compute from the contingency table of (predicted label, gold label) counts using `n * (n - 1) // 2`, never by enumerating pairs. Must run in linear time.
- B-cubed: for each item, precision is |predicted cluster of the item intersected with its gold cluster| divided by |predicted cluster|, and recall is the same intersection divided by |gold cluster|. Overall precision and recall are the averages over all items. Compute from the same contingency table, in linear time.
- Standard library only (`collections.Counter` is fine). Type hints, a module docstring and one-line docstrings per public name. No print statements.

## Commands

```
uv sync
uv run pytest
uv run ruff check .
uv run ruff format --check .
```

## Done criteria

- `uv run pytest` passes every test (both `tests/test_sanitize.py` and `tests/test_metrics.py`).
- `uv run ruff check .` and `uv run ruff format --check .` report no issues.
- Only `src/semtriage/metrics.py` was created.
- Finish with a short summary and the final test output.
