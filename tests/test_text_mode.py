from datetime import UTC, datetime

import numpy as np
import pytest

from semtriage.baseline import (
    format_report,
    main,
    read_embeddings,
    read_text_mode,
    run_baseline,
    texts_sha256,
)
from semtriage.embed import HashingEmbedder
from semtriage.models import Event, Source
from semtriage.sanitize import sanitize
from semtriage.text_mode import (
    DEFAULT_MAX_CHARS,
    TEXT_MODES,
    build_text,
    lead_text,
    parse_text_mode_label,
    text_mode_label,
    truncate_text,
)

WHEN = datetime(2024, 5, 1, tzinfo=UTC)

GHSA_TEXT = (
    "Widget parser crashes on nested frames\n"
    "\n"
    "### Summary\n"
    "A crafted frame makes the widget parser recurse without bound.\n"
    "\n"
    "Second summary paragraph stays.\n"
    "\n"
    "### Details\n"
    "The recursion happens in parse_frame.\n"
    "\n"
    "### PoC\n"
    "```python\n"
    "# not a heading\n"
    "payload = build()\n"
    "```\n"
    "\n"
    "### Impact\n"
    "Denial of service."
)


def make_event(source_id: str, text: str, cve_ids: tuple[str, ...], source: Source) -> Event:
    return Event(
        event_id=f"{source.value}:{source_id}",
        source=source,
        source_id=source_id,
        published=WHEN,
        modified=WHEN,
        text=text,
        cve_ids=cve_ids,
        cwe_ids=(),
        ecosystems=(),
        packages=(),
        cna=None,
        severity=None,
    )


def sample_events() -> list[Event]:
    return [
        make_event(
            "CVE-2024-0001", "widget parser recursion " * 80, ("CVE-2024-0001",), Source.NVD
        ),
        make_event("GHSA-aaaa-bbbb-cccc", GHSA_TEXT, ("CVE-2024-0001",), Source.GHSA),
        make_event("CVE-2024-0002", "gadget login bypass " * 5, ("CVE-2024-0002",), Source.NVD),
        make_event(
            "GHSA-dddd-eeee-ffff",
            "Gadget login bypass\n\n### Impact\nLogin can be skipped.",
            ("CVE-2024-0002",),
            Source.GHSA,
        ),
    ]


def test_constants():
    assert TEXT_MODES == ("full", "truncate", "lead")
    assert DEFAULT_MAX_CHARS == 1000


# truncate_text


def test_truncate_keeps_short_text():
    assert truncate_text("short text", 100) == "short text"
    assert truncate_text("exactly ten", 11) == "exactly ten"


def test_truncate_cuts_at_word_boundary():
    assert truncate_text("alpha beta gamma", 13) == "alpha beta"
    assert truncate_text("alpha beta gamma", 10) == "alpha beta"
    assert truncate_text("alpha beta gamma", 11) == "alpha beta"


def test_truncate_treats_newlines_as_boundaries_and_strips():
    assert truncate_text("alpha\n\nbeta gamma", 9) == "alpha"


def test_truncate_hard_cuts_a_single_long_word():
    assert truncate_text("abcdefghij", 4) == "abcd"


def test_truncate_never_exceeds_limit():
    text = "lorem ipsum dolor sit amet " * 50
    for limit in (1, 7, 50, 333):
        assert len(truncate_text(text, limit)) <= limit


def test_truncate_rejects_non_positive_limit():
    with pytest.raises(ValueError):
        truncate_text("text", 0)


# lead_text


def test_lead_keeps_title_and_summary_section_only():
    assert lead_text(GHSA_TEXT) == (
        "Widget parser crashes on nested frames\n"
        "\n"
        "A crafted frame makes the widget parser recurse without bound.\n"
        "\n"
        "Second summary paragraph stays."
    )


def test_lead_without_summary_uses_first_paragraph():
    text = (
        "Title line\n\n### Impact\nFirst paragraph line one.\nLine two.\n\n"
        "Second paragraph.\n\n### Patches\nUpgrade."
    )
    assert lead_text(text) == "Title line\n\nFirst paragraph line one.\nLine two."


def test_lead_drops_code_blocks_before_choosing_the_paragraph():
    text = "Title\n\n```sh\n# comment\ncurl x\n```\n\nReal first paragraph.\n\nMore."
    assert lead_text(text) == "Title\n\nReal first paragraph."


def test_lead_drops_tilde_and_unclosed_code_blocks():
    text = "Title\n\n### Summary\nKeep.\n~~~\ncode\n~~~\nAlso keep.\n```\nnever closed"
    assert lead_text(text) == "Title\n\nKeep.\nAlso keep."


def test_lead_summary_heading_variants():
    for heading in ("## Summary", "### Summary:", "#### Executive Summary", "**Summary**"):
        text = f"Title\n\n### Impact\nImpact text.\n\n{heading}\nThe summary.\n\n## Details\nX"
        assert lead_text(text) == "Title\n\nThe summary.", heading


def test_lead_title_only():
    assert lead_text("Just a title") == "Just a title"


def test_lead_text_starting_with_heading_has_no_title():
    assert lead_text("### Summary\nBody.\n\n### Impact\nX") == "Body."


def test_lead_empty_summary_section_falls_back_to_first_paragraph():
    text = "Title\n\n### Summary\n\n### Impact\nImpact text."
    assert lead_text(text) == "Title\n\nImpact text."


# build_text


def test_build_text_full_is_sanitized_text_unchanged():
    event = sample_events()[1]
    assert build_text(event, "full") == sanitize(event.text)
    assert build_text(event, "full", max_chars=10) == sanitize(event.text)


def test_build_text_truncate():
    event = sample_events()[0]
    text = build_text(event, "truncate", max_chars=30)
    assert text == "widget parser recursion widget"
    assert build_text(sample_events()[1], "truncate", max_chars=6) == "Widget"


def test_build_text_lead_ghsa_then_capped():
    event = sample_events()[1]
    assert build_text(event, "lead", max_chars=1000) == lead_text(sanitize(event.text))
    assert build_text(event, "lead", max_chars=45) == (
        "Widget parser crashes on nested frames\n\nA"
    )


def test_build_text_lead_nvd_keeps_capped_description():
    event = make_event(
        "CVE-2024-0003",
        "### Summary\nNVD text is never split into sections.\n\nSecond.",
        ("CVE-2024-0003",),
        Source.NVD,
    )
    assert build_text(event, "lead", max_chars=1000) == sanitize(event.text)
    assert build_text(event, "lead", max_chars=11) == "### Summary"


def test_build_text_sanitizes_before_shortening():
    raw = "Title\n\n### Summary\nSee https://example.test/advisory/CVE-2024-0009 now."
    event = make_event("GHSA-gggg-hhhh-iiii", raw, ("CVE-2024-0009",), Source.GHSA)
    for mode in TEXT_MODES:
        text = build_text(event, mode, max_chars=40)
        assert "CVE" not in text and "http" not in text, mode


def test_build_text_rejects_unknown_mode_and_bad_limit():
    event = sample_events()[0]
    with pytest.raises(ValueError):
        build_text(event, "short")
    with pytest.raises(ValueError):
        build_text(event, "truncate", max_chars=0)


# run_baseline and cache


def test_run_baseline_full_matches_default():
    default = run_baseline(sample_events(), HashingEmbedder(), (0.5,))
    full = run_baseline(sample_events(), HashingEmbedder(), (0.5,), text_mode="full")
    assert format_report(default) == format_report(full)
    assert "Text:" not in format_report(full)


def test_run_baseline_reports_non_default_text_mode():
    report = run_baseline(
        sample_events(), HashingEmbedder(), (0.5,), text_mode="lead", max_chars=200
    )
    assert "Text: lead (max 200 chars)" in format_report(report).splitlines()


def test_run_baseline_embeds_shortened_texts():
    seen: list[str] = []

    class Recorder(HashingEmbedder):
        def embed(self, texts):
            seen.extend(texts)
            return super().embed(texts)

    run_baseline(sample_events(), Recorder(), (0.5,), text_mode="truncate", max_chars=30)
    assert all(len(text) <= 30 for text in seen)
    assert seen[0] == "widget parser recursion widget"


@pytest.mark.parametrize(
    ("saved", "loaded"),
    [
        (("full", 1000), ("lead", 1000)),
        (("lead", 1000), ("truncate", 1000)),
        (("truncate", 1000), ("full", 1000)),
        (("truncate", 1000), ("truncate", 40)),
    ],
)
def test_cache_refuses_other_text_mode(tmp_path, saved, loaded):
    path = tmp_path / "vectors.npz"
    run_baseline(
        sample_events(),
        HashingEmbedder(),
        (0.5,),
        text_mode=saved[0],
        max_chars=saved[1],
        save_embeddings=path,
    )
    with pytest.raises(ValueError, match="text mode differs: cache built with "):
        run_baseline(
            sample_events(),
            None,
            (0.5,),
            text_mode=loaded[0],
            max_chars=loaded[1],
            load_embeddings=path,
        )
    # The text hash alone, without the stored mode label, also refuses the cache.
    events = sample_events()
    ids = [event.event_id for event in events]
    texts = [build_text(event, loaded[0], max_chars=loaded[1]) for event in events]
    with pytest.raises(ValueError, match="texts differ"):
        read_embeddings(path, ids, texts)


def test_cache_same_text_mode_loads(tmp_path):
    path = tmp_path / "vectors.npz"
    first = run_baseline(
        sample_events(), HashingEmbedder(), (0.5,), text_mode="lead", save_embeddings=path
    )
    second = run_baseline(sample_events(), None, (0.5,), text_mode="lead", load_embeddings=path)
    assert format_report(first) == format_report(second)
    with np.load(path) as data:
        assert str(data["text_mode"]) == "lead:1000"


def test_read_embeddings_names_the_stored_text_mode(tmp_path):
    path = tmp_path / "vectors.npz"
    run_baseline(sample_events(), HashingEmbedder(), (0.5,), text_mode="lead", save_embeddings=path)
    events = sample_events()
    ids = [event.event_id for event in events]
    texts = [build_text(event, "full") for event in events]
    with pytest.raises(
        ValueError, match="text mode differs: cache built with lead:1000, this run uses full"
    ):
        read_embeddings(path, ids, texts, text_mode="full")


# CLI


def write_events(path, events):
    path.write_text("".join(event.model_dump_json() + "\n" for event in events))


def test_main_default_text_mode_is_full(tmp_path, capsys):
    events = tmp_path / "events.jsonl"
    write_events(events, sample_events())
    assert main(["--events", str(events), "--thresholds", "0.5"]) == 0
    default = capsys.readouterr().out
    assert main(["--events", str(events), "--thresholds", "0.5", "--text-mode", "full"]) == 0
    assert capsys.readouterr().out == default
    assert "Text:" not in default


@pytest.mark.parametrize("mode", ["truncate", "lead"])
def test_main_text_mode_and_max_chars(tmp_path, capsys, mode):
    events = tmp_path / "events.jsonl"
    write_events(events, sample_events())
    argv = ["--events", str(events), "--thresholds", "0.5", "--text-mode", mode]
    assert main([*argv, "--max-chars", "300"]) == 0
    assert f"Text: {mode} (max 300 chars)" in capsys.readouterr().out
    assert main(argv) == 0
    assert f"Text: {mode} (max 1000 chars)" in capsys.readouterr().out


@pytest.mark.parametrize(
    "extra",
    [
        ["--text-mode", "short"],
        ["--text-mode", "truncate", "--max-chars", "0"],
        ["--text-mode", "truncate", "--max-chars", "-5"],
        ["--text-mode", "lead", "--max-chars", "ten"],
        ["--max-chars", "500"],
        ["--text-mode", "full", "--max-chars", "500"],
    ],
)
def test_main_rejects_bad_text_options(tmp_path, extra):
    events = tmp_path / "events.jsonl"
    write_events(events, sample_events())
    with pytest.raises(SystemExit) as excinfo:
        main(["--events", str(events), *extra])
    assert excinfo.value.code == 2


# review fixes


def test_lead_bold_sub_label_inside_summary_is_kept():
    text = "T\n\n### Summary\n**Affected versions**\nBelow 1.2\n\n### Impact\nI"
    assert lead_text(text) == "T\n\n**Affected versions**\nBelow 1.2"


def test_lead_one_line_fence_does_not_open_a_block():
    text = "T\n\n### Summary\nIntro.\n```x```\nAfter."
    assert lead_text(text) == "T\n\nIntro.\n```x```\nAfter."


def test_lead_longer_fence_closes_only_on_same_or_longer_run():
    text = "T\n\n### Summary\n````md\n```\ninner\n```\n````\nAfter."
    assert lead_text(text) == "T\n\nAfter."


def test_lead_closing_fence_must_use_same_character():
    text = "T\n\n### Summary\n```\nA\n~~~\nB\n```\nAfter."
    assert lead_text(text) == "T\n\nAfter."


def test_lead_indented_fence_is_dropped():
    text = "T\n\n### Summary\n   ```\n   code\n   ```\nAfter."
    assert lead_text(text) == "T\n\nAfter."


def test_lead_fallback_splits_paragraphs_at_headings():
    assert lead_text("T\nPara A\n### Details\nPara B") == "T\n\nPara A"


def test_truncate_cuts_at_any_whitespace():
    assert truncate_text("alpha beta", 7) == "alpha"


def _write_unlabeled_full_cache(path, events):
    ids = [event.event_id for event in events]
    texts = [build_text(event, "full") for event in events]
    with path.open("wb") as target:
        np.savez(
            target,
            event_ids=np.asarray(ids, dtype=str),
            vectors=HashingEmbedder().embed(texts),
            embedder=np.asarray("hashing-256", dtype=str),
            text_sha256=np.asarray(texts_sha256(texts), dtype=str),
        )


def test_old_cache_without_text_mode_key_still_loads(tmp_path):
    events = sample_events()
    path = tmp_path / "old.npz"
    _write_unlabeled_full_cache(path, events)
    report = run_baseline(events, None, (0.5,), load_embeddings=path)
    assert report.n_events == len(events)


def test_old_cache_without_text_mode_key_counts_as_full(tmp_path):
    events = sample_events()
    path = tmp_path / "old.npz"
    _write_unlabeled_full_cache(path, events)
    with pytest.raises(
        ValueError, match="text mode differs: cache built with full, this run uses lead:1000"
    ):
        run_baseline(events, None, (0.5,), text_mode="lead", load_embeddings=path)


def test_lead_mode_falls_back_to_full_text_when_lead_is_empty():
    text = "### Impact\n```\nonly code\n```"
    event = make_event("GHSA-jjjj-kkkk-llll", text, ("CVE-2024-0010",), Source.GHSA)
    assert lead_text(sanitize(text)) == ""
    assert build_text(event, "lead", max_chars=10) == "### Impact"


def test_text_mode_label_round_trip():
    for mode, limit in (("full", DEFAULT_MAX_CHARS), ("truncate", 40), ("lead", 1000)):
        assert parse_text_mode_label(text_mode_label(mode, limit)) == (
            mode,
            DEFAULT_MAX_CHARS if mode == "full" else limit,
        )


@pytest.mark.parametrize("label", ["", "lead", "lead:", "lead:x", "lead:0", "full:10", "short:5"])
def test_parse_text_mode_label_rejects_bad_labels(label):
    with pytest.raises(ValueError, match="text mode"):
        parse_text_mode_label(label)


def test_read_text_mode(tmp_path):
    path = tmp_path / "vectors.npz"
    run_baseline(
        sample_events(),
        HashingEmbedder(),
        (0.5,),
        text_mode="truncate",
        max_chars=40,
        save_embeddings=path,
    )
    assert read_text_mode(path) == "truncate:40"
    old = tmp_path / "old.npz"
    with old.open("wb") as target:
        np.savez(target, event_ids=np.asarray(["a"], dtype=str))
    assert read_text_mode(old) == "full"
    garbage = tmp_path / "garbage.npz"
    garbage.write_bytes(b"not a zip")
    with pytest.raises(ValueError, match="cannot read"):
        read_text_mode(garbage)


def test_truncate_keeps_word_when_only_space_is_leading() -> None:
    assert truncate_text(" abcdef", 4) == " abc"
