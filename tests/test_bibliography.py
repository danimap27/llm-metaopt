"""Tests that the bibliography stays compilable and traceable.

BibTeX parses a percent sign inside an entry as a field name, so a comment line
inside ``@article{...}`` breaks the build with "You're missing a field name".
Provenance therefore lives in ``refs_sources.md``, and these tests keep it that
way.
"""

from __future__ import annotations

import pathlib
import re

import pytest

PAPER = pathlib.Path(__file__).resolve().parents[1] / "paper"
BIB = PAPER / "refs.bib"
SOURCES = PAPER / "refs_sources.md"
ENTRY_PATTERN = re.compile(r"^@\w+\{([^,]+),")


def _entries() -> dict[str, list[str]]:
    """Entry key to the lines inside its braces."""
    entries: dict[str, list[str]] = {}
    current: str | None = None
    depth = 0
    for line in BIB.read_text(encoding="utf-8").splitlines():
        match = ENTRY_PATTERN.match(line)
        if match and current is None:
            current = match.group(1)
            entries[current] = []
            depth = line.count("{") - line.count("}")
            continue
        if current is not None:
            entries[current].append(line)
            depth += line.count("{") - line.count("}")
            if depth <= 0:
                current = None
    return entries


@pytest.mark.skipif(not BIB.exists(), reason="no bibliography yet")
def test_no_percent_comments_inside_entries():
    for key, lines in _entries().items():
        for line in lines:
            stripped = line.strip()
            if stripped.startswith("%"):
                pytest.fail(f"entry {key!r} contains a percent comment, which breaks BibTeX: {stripped!r}")


@pytest.mark.skipif(not BIB.exists(), reason="no bibliography yet")
def test_every_entry_declares_the_expected_fields():
    for key, lines in _entries().items():
        body = "\n".join(lines)
        assert re.search(r"\btitle\s*=", body), f"{key} has no title"
        assert re.search(r"\bauthor\s*=", body), f"{key} has no author"
        assert re.search(r"\byear\s*=", body), f"{key} has no year"


@pytest.mark.skipif(not (BIB.exists() and SOURCES.exists()), reason="no provenance file yet")
def test_provenance_covers_every_entry():
    keys = set(_entries())
    documented = set(re.findall(r"^\| `([^`]+)` \|", SOURCES.read_text(encoding="utf-8"), flags=re.MULTILINE))
    missing = sorted(keys - documented)
    assert not missing, f"entries without a verification source: {missing}"
