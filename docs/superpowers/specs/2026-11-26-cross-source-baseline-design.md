# Cross-source candidate pairs and singleton reference for the dedup baseline

## Intent

In the recorded dataset a duplicate is always one NVD event plus one GHSA event for the same CVE. The first real baseline run compared every pair of events. Same-source pairs can never be true duplicates, but single-link clustering chains them into giant clusters, so pairwise F1 was about 0 at every threshold and the best B-cubed F1 (0.901) was below the trivial all-singletons answer (about 0.975). The report did not show that reference, so the gap was easy to miss.

## Goals

- Build candidate edges only between events whose `source` differs (cross-source mode), and make it the default.
- Keep the all-pairs behavior reachable and unchanged.
- Show the all-singletons reference in the report next to the threshold rows.
- Keep the row-block computation memory-bounded, and much faster on the real data (about 124M cross-source products instead of about 1.1G).

## Non-goals

- No change to the metrics module or its conventions.
- No new clustering method (single link stays).

## Approaches

1. Filter the all-pairs edges with a mask on group codes. Simple, but it still computes every dot product, so it saves no time.
2. Compute only the cross-group blocks: for each pair of groups, multiply row blocks of the first group by the rows of the second. Work is proportional to cross-group pairs. Chosen.
3. Sort rows by group and reuse the upper-triangle loop with skipped blocks. Same cost as 2 but harder to read and to map back to original indices.

## Design

- `sweep(ids, vectors, gold, thresholds, *, groups=None)` and `threshold_clusters(vectors, threshold, *, groups=None)`. `groups` is a sequence of hashable group keys, one per row. `None` means all pairs (current behavior). When given, an edge exists only between rows whose groups differ. A length mismatch raises `ValueError`.
- Edge collection with groups: group keys are mapped to codes; for every pair of distinct groups (a, b), rows of a are processed in blocks of 1024 against all rows of b; each kept edge is stored with left = smaller original index, right = larger. Edges are then sorted by similarity as today. Labels stay the smallest index in each component, so results do not depend on edge order. Since same-source events are never linked directly, cross-source mode assumes at most one advisory per source per CVE; this holds in the recorded dataset.
- `run_baseline(events, embedder, thresholds, *, pairs="cross-source")`, with `pairs` in `{"cross-source", "all"}`; any other value raises `ValueError`. Cross-source passes `groups=[event.source for event in kept]`.
- CLI: `--pairs {cross-source,all}`, default `cross-source`.
- Reference: a `ReferenceRow(name, n_clusters, pairwise, bcubed)` dataclass and `singletons_reference(ids, gold)`, which scores the prediction where every event is its own cluster with the existing metric functions.
- `BaselineReport` gains `pairs: str | None = None` and `reference: ReferenceRow | None = None`; `run_baseline` always sets both.
- `format_report` prints `Pairs: <mode>` when set, and the reference as the first table row, labeled `singletons` in the threshold column.

## Undefined values

When no pair is predicted (every cluster is a singleton, which is exactly when the cluster count equals the event count), pairwise precision is undefined. The metrics module returns 1.0 for it by convention; the report shows it as `n/a`, for the reference row and for any threshold row with no predicted pairs. Pairwise recall is then 0 (when gold has pairs) and pairwise F1 is shown as 0.000. The numeric values in the dataclasses are unchanged.

## Left to the builder

Variable names, helper layout, and how group keys are encoded to codes.
