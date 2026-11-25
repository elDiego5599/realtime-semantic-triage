# ADR 0001: Data source is security advisories (NVD and GitHub reviewed advisories)

- Status: accepted
- Date: 2026-11-25

## Context

Semantic deduplication needs a real stream where the same incident shows up several times with different wording, plus a ground-truth label that does not require manual annotation. NVD, GitHub Security Advisories (GHSA) and OSV were evaluated with live queries covering 2026-11-18 to 2026-11-25.

## Measurements

| Source | Endpoint | Volume (7 days) | Rate limit without key | With key |
|---|---|---|---|---|
| NVD | `services.nvd.nist.gov/rest/json/cves/2.0` (`pubStartDate`, `lastModStartDate`) | 4,298 CVEs published (about 600 to 1,000 per weekday) | 5 requests per 30 s | 50 requests per 30 s (free key) |
| GHSA reviewed | `api.github.com/advisories?type=reviewed` | 158 advisories, 156 with a CVE (about 70 per weekday, about 8 per weekend day) | 60 requests per hour | 5,000 requests per hour (`gh` token) |
| GHSA unreviewed | `?type=unreviewed` | 100% with a CVE | | |
| OSV | `api.osv.dev/v1/vulns/{id}` and `modified_id.csv` on GCS (167 MB, refreshed every few minutes) | Aggregator of the sources above plus Linux distributions | no published limit | |

Random sample of 40 reviewed GHSA advisories that carry a CVE:

- 38 of 40 (95%) exist in NVD and 37 of 40 in OSV. Nearly every GHSA advisory has an NVD counterpart, so labels come for free.
- Text similarity (difflib character ratio), GHSA vs NVD: median 0.07. GHSA ships a long Markdown write-up, NVD a short paragraph. These are hard pairs, which is the point.
- OSV CVE record vs NVD: median 0.94 (36 of 37 at or above 0.9). It is a near copy of NVD.
- Unreviewed GHSA records are mirrors of NVD as well.

## Decision

1. Primary sources are NVD and reviewed GHSA. The CVE ID is the incident (cluster) label.
2. OSV is not used as a third text source. Its CVE and GHSA records copy NVD or GHSA, so it would add trivial duplicates that inflate F1. Distribution records (Debian, Ubuntu) may be added later if they turn out to rewrite the text; that check is pending.
3. Unreviewed GHSA is excluded for the same reason.
4. Label leakage: CVE and GHSA identifiers, advisory URLs and the `aliases` and `related` fields are stripped from the text before embedding. The CVE lives only in the evaluation payload.
5. Historical backfill plus live poller: both APIs support date-range queries, so the labeled dataset is built from months of history ordered by `published`, without waiting days for recording. The live poller feeds bronze storage for the demo and for replay.

## Consequences

- Clusters are small (2 to 3 events) and NVD contributes many singletons as distractors. Report B-cubed alongside pairwise F1, since singletons dominate.
- NVD has an analysis backlog: only 79 of the first 2,000 CVEs of the week carried CPE configurations. Vendor and product filters must come from the GHSA ecosystem and package and from the issuing CNA (`sourceIdentifier`), not from CPE.
- Request a free NVD API key; use the `gh` token for GHSA.
- Pending for phase 1: measure the publication lag between GHSA and NVD for each CVE, since it defines arrival order during replay.
