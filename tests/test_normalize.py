import copy
import json
from datetime import UTC, datetime
from pathlib import Path

import pytest
from pydantic import ValidationError

from semtriage.models import Event, Source
from semtriage.normalize import normalize_ghsa, normalize_nvd
from semtriage.sanitize import find_identifiers

FIXTURES = Path(__file__).parent / "fixtures"


@pytest.fixture
def nvd_item() -> dict:
    return json.loads((FIXTURES / "nvd_cve.json").read_text())


@pytest.fixture
def ghsa_item() -> dict:
    return json.loads((FIXTURES / "ghsa_advisory.json").read_text())


# NVD


def test_nvd_basic_fields(nvd_item):
    event = normalize_nvd(nvd_item)
    assert isinstance(event, Event)
    assert event.source is Source.NVD
    assert event.source_id == "CVE-2026-64670"
    assert event.event_id == "nvd:CVE-2026-64670"
    assert event.cve_ids == ("CVE-2026-64670",)
    assert event.cna == "security-advisories@github.com"
    assert event.ecosystems == ()
    assert event.packages == ()


def test_nvd_timestamps_are_utc_aware(nvd_item):
    event = normalize_nvd(nvd_item)
    assert event.published == datetime(2026, 11, 11, 5, 19, 51, 527000, tzinfo=UTC)
    assert event.modified == datetime(2026, 11, 12, 20, 18, 49, 697000, tzinfo=UTC)


def test_nvd_text_is_english_description_sanitized(nvd_item):
    event = normalize_nvd(nvd_item)
    assert event.text.startswith("Waline is a comment system with backend support.")
    assert event.text.endswith("there are no publicly available patches.")


def test_nvd_picks_english_description(nvd_item):
    descriptions = nvd_item["cve"]["descriptions"]
    descriptions.insert(0, {"lang": "es", "value": "Texto en otro idioma."})
    assert normalize_nvd(nvd_item).text.startswith("Waline")


def test_nvd_severity_prefers_newest_cvss(nvd_item):
    assert normalize_nvd(nvd_item).severity == "CRITICAL"
    metrics = nvd_item["cve"]["metrics"]
    metrics["cvssMetricV40"] = [{"cvssData": {"baseSeverity": "MEDIUM"}}]
    assert normalize_nvd(nvd_item).severity == "MEDIUM"


def test_nvd_severity_v2_lives_on_metric(nvd_item):
    nvd_item["cve"]["metrics"] = {"cvssMetricV2": [{"baseSeverity": "LOW", "cvssData": {}}]}
    assert normalize_nvd(nvd_item).severity == "LOW"


def test_nvd_severity_missing(nvd_item):
    nvd_item["cve"]["metrics"] = {}
    assert normalize_nvd(nvd_item).severity is None
    del nvd_item["cve"]["metrics"]
    assert normalize_nvd(nvd_item).severity is None


def test_nvd_cwes_sorted_unique_and_real_only(nvd_item):
    assert normalize_nvd(nvd_item).cwe_ids == ("CWE-116", "CWE-79")
    nvd_item["cve"]["weaknesses"].append(
        {"description": [{"lang": "en", "value": "NVD-CWE-noinfo"}, {"value": "CWE-79"}]}
    )
    assert normalize_nvd(nvd_item).cwe_ids == ("CWE-116", "CWE-79")
    del nvd_item["cve"]["weaknesses"]
    assert normalize_nvd(nvd_item).cwe_ids == ()


def test_nvd_rejected_is_skipped(nvd_item):
    nvd_item["cve"]["vulnStatus"] = "Rejected"
    assert normalize_nvd(nvd_item) is None


def test_nvd_empty_text_is_skipped(nvd_item):
    nvd_item["cve"]["descriptions"] = [{"lang": "en", "value": "CVE-2026-64670"}]
    assert normalize_nvd(nvd_item) is None


# GHSA


def test_ghsa_basic_fields(ghsa_item):
    event = normalize_ghsa(ghsa_item)
    assert isinstance(event, Event)
    assert event.source is Source.GHSA
    assert event.source_id == "GHSA-crj3-q72p-ggqm"
    assert event.event_id == "ghsa:GHSA-crj3-q72p-ggqm"
    assert event.cve_ids == ("CVE-2026-64670",)
    assert event.cwe_ids == ("CWE-116", "CWE-79")
    assert event.ecosystems == ("npm",)
    assert event.packages == ("@waline/client",)
    assert event.severity == "CRITICAL"
    assert event.cna is None


def test_ghsa_timestamps_are_utc_aware(ghsa_item):
    event = normalize_ghsa(ghsa_item)
    assert event.published == datetime(2026, 11, 25, 15, 26, 23, tzinfo=UTC)
    assert event.modified == datetime(2026, 11, 25, 15, 26, 25, tzinfo=UTC)


def test_ghsa_text_is_summary_then_description(ghsa_item):
    text = normalize_ghsa(ghsa_item).text
    first, _, rest = text.partition("\n\n")
    assert first == (
        "@waline/client vulnerable to stored XSS through unsanitized comment edit previews"
    )
    assert rest.startswith("### Summary")


def test_ghsa_cve_ids_merge_cve_id_and_identifiers(ghsa_item):
    ghsa_item["identifiers"].append({"type": "CVE", "value": "cve-2026-00001"})
    assert normalize_ghsa(ghsa_item).cve_ids == ("CVE-2026-00001", "CVE-2026-64670")
    ghsa_item["cve_id"] = None
    ghsa_item["identifiers"] = [{"type": "GHSA", "value": "GHSA-crj3-q72p-ggqm"}]
    assert normalize_ghsa(ghsa_item).cve_ids == ()


def test_ghsa_multiple_packages_sorted_unique(ghsa_item):
    vuln = ghsa_item["vulnerabilities"][0]
    extra = copy.deepcopy(vuln)
    extra["package"] = {"ecosystem": "pip", "name": "waline-py"}
    ghsa_item["vulnerabilities"] += [copy.deepcopy(vuln), extra, {"package": None}]
    event = normalize_ghsa(ghsa_item)
    assert event.ecosystems == ("npm", "pip")
    assert event.packages == ("@waline/client", "waline-py")


@pytest.mark.parametrize(
    "raw, expected",
    [
        ("low", "LOW"),
        ("medium", "MEDIUM"),
        ("moderate", "MEDIUM"),
        ("HIGH", "HIGH"),
        ("unknown", None),
    ],
)
def test_ghsa_severity_mapping(ghsa_item, raw, expected):
    ghsa_item["severity"] = raw
    assert normalize_ghsa(ghsa_item).severity == expected


def test_ghsa_withdrawn_is_skipped(ghsa_item):
    ghsa_item["withdrawn_at"] = "2026-11-26T00:00:00Z"
    assert normalize_ghsa(ghsa_item) is None


# Shared


@pytest.mark.parametrize("name", ["nvd", "ghsa"])
def test_text_has_no_identifier_leakage(name, nvd_item, ghsa_item):
    event = normalize_nvd(nvd_item) if name == "nvd" else normalize_ghsa(ghsa_item)
    assert find_identifiers(event.text) == set()
    assert "http" not in event.text


def test_same_incident_shares_label_across_sources(nvd_item, ghsa_item):
    assert normalize_nvd(nvd_item).cve_ids == normalize_ghsa(ghsa_item).cve_ids


def test_inputs_are_not_mutated(nvd_item, ghsa_item):
    nvd_before, ghsa_before = copy.deepcopy(nvd_item), copy.deepcopy(ghsa_item)
    normalize_nvd(nvd_item)
    normalize_ghsa(ghsa_item)
    assert nvd_item == nvd_before
    assert ghsa_item == ghsa_before


def test_event_is_frozen_and_strict(nvd_item):
    event = normalize_nvd(nvd_item)
    with pytest.raises(ValidationError):
        event.text = "changed"
    with pytest.raises(ValidationError):
        Event(**event.model_dump(), unexpected="x")


def test_event_rejects_naive_datetimes(nvd_item):
    data = normalize_nvd(nvd_item).model_dump()
    data["published"] = datetime(2026, 11, 11)
    with pytest.raises(ValidationError):
        Event(**data)


def test_event_round_trips_through_json(ghsa_item):
    event = normalize_ghsa(ghsa_item)
    assert Event.model_validate_json(event.model_dump_json()) == event
