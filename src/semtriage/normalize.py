"""Normalizers that turn raw NVD and GHSA records into Events."""

import re
from collections.abc import Iterable
from datetime import UTC, datetime
from typing import get_args

from semtriage.models import Event, Severity, Source
from semtriage.sanitize import sanitize

_NVD_SEVERITY_KEYS = ("cvssMetricV40", "cvssMetricV31", "cvssMetricV30", "cvssMetricV2")
_VALID_SEVERITIES = frozenset(get_args(Severity))
_GHSA_SEVERITIES: dict[str, Severity] = {
    "low": "LOW",
    "medium": "MEDIUM",
    "moderate": "MEDIUM",
    "high": "HIGH",
    "critical": "CRITICAL",
}


def _parse_utc(value: str) -> datetime:
    parsed = datetime.fromisoformat(value)
    if parsed.tzinfo is None:
        return parsed.replace(tzinfo=UTC)
    return parsed.astimezone(UTC)


def _sorted_unique(values: Iterable[str]) -> tuple[str, ...]:
    return tuple(sorted(set(values)))


def product_key_from_package(name: str) -> str:
    """Return the lowercased package name without its namespace."""
    return re.split(r"[/:]", name)[-1].lower()


def _cpe_product(criteria: str) -> str | None:
    parts = re.split(r"(?<!\\):", criteria)
    if len(parts) < 5:
        return None
    product = parts[4].replace("\\", "").lower()
    return None if product in {"", "*"} else product


def _cpe_products(nodes: list[dict]) -> Iterable[str]:
    for node in nodes:
        for match in node.get("cpeMatch") or []:
            product = _cpe_product(match.get("criteria") or "")
            if product is not None:
                yield product
        yield from _cpe_products(node.get("children") or [])


def product_keys_from_cpes(configurations: list[dict]) -> tuple[str, ...]:
    """Return sorted, unique, lowercased product keys from CPE configurations."""
    products = []
    for configuration in configurations or []:
        products.extend(_cpe_products(configuration.get("nodes") or []))
    return _sorted_unique(products)


def _first_english_description(descriptions: list[dict]) -> str:
    for description in descriptions:
        if description.get("lang") == "en":
            return description.get("value") or ""
    return ""


def _nvd_cwe_ids(weaknesses: list[dict]) -> tuple[str, ...]:
    values = []
    for weakness in weaknesses:
        for description in weakness.get("description") or []:
            value = description.get("value")
            if isinstance(value, str) and value.startswith("CWE-"):
                values.append(value)
    return _sorted_unique(values)


def _coerce_severity(raw: object) -> Severity | None:
    if not isinstance(raw, str):
        return None
    value = raw.upper()
    return value if value in _VALID_SEVERITIES else None


def _nvd_severity(metrics: dict) -> Severity | None:
    for key in _NVD_SEVERITY_KEYS:
        entries = metrics.get(key) or []
        if not entries:
            continue
        entry = entries[0]
        if key == "cvssMetricV2":
            return _coerce_severity(entry.get("baseSeverity"))
        return _coerce_severity((entry.get("cvssData") or {}).get("baseSeverity"))
    return None


def _ghsa_severity(raw: object) -> Severity | None:
    if not isinstance(raw, str):
        return None
    return _GHSA_SEVERITIES.get(raw.lower())


def normalize_nvd(item: dict) -> Event | None:
    """Normalize one NVD API vulnerability item into an Event."""
    cve = item["cve"]
    if cve.get("vulnStatus") == "Rejected":
        return None

    text = sanitize(_first_english_description(cve.get("descriptions") or []))
    if not text:
        return None

    source_id = cve["id"]
    return Event(
        event_id=f"{Source.NVD.value}:{source_id}",
        source=Source.NVD,
        source_id=source_id,
        published=_parse_utc(cve["published"]),
        modified=_parse_utc(cve["lastModified"]),
        text=text,
        cve_ids=_sorted_unique([source_id.upper()]),
        cwe_ids=_nvd_cwe_ids(cve.get("weaknesses") or []),
        ecosystems=(),
        packages=(),
        products=product_keys_from_cpes(cve.get("configurations") or []),
        cna=cve.get("sourceIdentifier"),
        severity=_nvd_severity(cve.get("metrics") or {}),
    )


def normalize_ghsa(advisory: dict) -> Event | None:
    """Normalize one GitHub global security advisory into an Event."""
    if advisory.get("withdrawn_at"):
        return None

    summary = advisory.get("summary") or ""
    description = advisory.get("description") or ""
    text = sanitize(f"{summary}\n\n{description}")
    if not text:
        return None

    cve_values = []
    primary_cve = advisory.get("cve_id")
    if primary_cve:
        cve_values.append(primary_cve)
    for identifier in advisory.get("identifiers") or []:
        if identifier.get("type") == "CVE" and identifier.get("value"):
            cve_values.append(identifier["value"])

    ecosystems = []
    packages = []
    for vulnerability in advisory.get("vulnerabilities") or []:
        package = vulnerability.get("package")
        if package is None:
            continue
        ecosystems.append(package["ecosystem"])
        packages.append(package["name"])

    source_id = advisory["ghsa_id"]
    return Event(
        event_id=f"{Source.GHSA.value}:{source_id}",
        source=Source.GHSA,
        source_id=source_id,
        published=_parse_utc(advisory["published_at"]),
        modified=_parse_utc(advisory["updated_at"]),
        text=text,
        cve_ids=_sorted_unique(value.upper() for value in cve_values),
        cwe_ids=_sorted_unique(cwe["cwe_id"] for cwe in advisory.get("cwes") or []),
        ecosystems=_sorted_unique(ecosystems),
        packages=_sorted_unique(packages),
        products=_sorted_unique(product_key_from_package(name) for name in packages),
        cna=None,
        severity=_ghsa_severity(advisory.get("severity")),
    )
