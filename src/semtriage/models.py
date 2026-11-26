"""Domain model for normalized security advisory events."""

from enum import StrEnum
from typing import Literal

from pydantic import AwareDatetime, BaseModel, ConfigDict

Severity = Literal["LOW", "MEDIUM", "HIGH", "CRITICAL"]


class Source(StrEnum):
    """Origin feed an advisory was ingested from."""

    NVD = "nvd"
    GHSA = "ghsa"


class Event(BaseModel):
    """One normalized security advisory, independent of its source feed."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    event_id: str  # f"{source.value}:{source_id}"
    source: Source
    source_id: str
    published: AwareDatetime
    modified: AwareDatetime
    text: str
    cve_ids: tuple[str, ...]
    cwe_ids: tuple[str, ...]
    ecosystems: tuple[str, ...]
    packages: tuple[str, ...]
    cna: str | None
    severity: Severity | None
