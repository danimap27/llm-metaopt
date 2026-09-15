#!/usr/bin/env python3
"""Verify bibliographic entries and emit BibTeX from authoritative metadata.

The paper rule is strict: only entries that resolve to a public record are
allowed in ``paper/refs.bib``. This script queries the arXiv API and Crossref
for every identifier, prints the BibTeX that the authoritative metadata
supports, and reports anything that could not be resolved, so nothing is
invented and every entry can be traced to a source.

Usage
-----
    python scripts/verify_references.py --ids 1803.11173 2309.03409
    python scripts/verify_references.py --dois 10.1038/s41586-023-06924-6
    python scripts/verify_references.py --ids-file paper/refs_ids.txt

Options
-------
    --ids          arXiv identifiers (new style or old style)
    --dois         Digital Object Identifiers for Crossref
    --search       arXiv full-text search queries, one entry per flag
    --out          write the BibTeX to this file instead of stdout
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import unicodedata
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from typing import Any, Dict, List, Optional

ARXIV_API = "http://export.arxiv.org/api/query"
CROSSREF_API = "https://api.crossref.org/works/"
ATOM = "{http://www.w3.org/2005/Atom}"
USER_AGENT = "llm-metaopt-bib-verifier/1.0 (mailto:danimp2002@gmail.com)"


def _fetch(url: str, timeout: float = 30.0) -> bytes:
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return response.read()


def _ascii(text: str) -> str:
    normalized = unicodedata.normalize("NFKD", text)
    return normalized.encode("ascii", "ignore").decode("ascii")


def _bibtex_key(surname: str, year: str, title: str) -> str:
    surname = re.sub(r"[^A-Za-z]", "", _ascii(surname).lower()) or "anon"
    words = [word for word in re.findall(r"[A-Za-z]+", _ascii(title)) if len(word) > 3]
    word = words[0].lower() if words else "work"
    return f"{surname}{year}{word}"


def fetch_arxiv(arxiv_id: str) -> Optional[Dict[str, Any]]:
    """Return authoritative metadata for one arXiv identifier."""
    query = urllib.parse.urlencode({"id_list": arxiv_id})
    payload = _fetch(f"{ARXIV_API}?{query}")
    root = ET.fromstring(payload)
    entry = root.find(f"{ATOM}entry")
    if entry is None:
        return None
    title = " ".join((entry.findtext(f"{ATOM}title") or "").split())
    if not title or title.lower().startswith("error"):
        return None
    authors = []
    for author in entry.findall(f"{ATOM}author"):
        name = author.findtext(f"{ATOM}name") or ""
        authors.append(name)
    published = entry.findtext(f"{ATOM}published") or ""
    year = published[:4] if published else ""
    journal_ref = (entry.findtext("{http://arxiv.org/schemas/atom}journal_ref") or "").strip()
    doi = (entry.findtext("{http://arxiv.org/schemas/atom}doi") or "").strip()
    primary = ""
    category = entry.find("{http://arxiv.org/schemas/atom}primary_category")
    if category is not None:
        primary = category.get("term", "")
    canonical = (entry.findtext(f"{ATOM}id") or "").strip()
    return {
        "arxiv_id": arxiv_id,
        "title": title,
        "authors": authors,
        "year": year,
        "journal_ref": journal_ref,
        "doi": doi,
        "primary_class": primary,
        "abs_url": canonical or f"https://arxiv.org/abs/{arxiv_id}",
    }


def bibtex_from_arxiv(meta: Dict[str, Any]) -> str:
    key = _bibtex_key(meta["authors"][0].split()[-1] if meta["authors"] else "anon", meta["year"], meta["title"])
    author_field = " and ".join(meta["authors"])
    lines = [f"@misc{{{key},"]
    lines.append(f"  author       = {{{author_field}}},")
    lines.append(f"  title        = {{{meta['title']}}},")
    lines.append(f"  year         = {{{meta['year']}}},")
    lines.append(f"  eprint       = {{{meta['arxiv_id']}}},")
    lines.append("  archivePrefix = {arXiv},")
    if meta["primary_class"]:
        lines.append(f"  primaryClass = {{{meta['primary_class']}}},")
    if meta["doi"]:
        lines.append(f"  doi          = {{{meta['doi']}}},")
    if meta["journal_ref"]:
        lines.append(f"  note         = {{Published as: {meta['journal_ref']}}},")
    lines.append(f"  % VERIFIED: {meta['abs_url']}")
    lines.append("}")
    return "\n".join(lines)


def fetch_crossref(doi: str) -> Optional[Dict[str, Any]]:
    payload = _fetch(CROSSREF_API + urllib.parse.quote(doi))
    message = json.loads(payload)["message"]
    title = (message.get("title") or [""])[0]
    if not title:
        return None
    authors = []
    for author in message.get("author", []) or []:
        given = author.get("given", "")
        family = author.get("family", "")
        authors.append(f"{given} {family}".strip() or author.get("name", ""))
    year = ""
    for field in ("published-print", "published-online", "issued", "created"):
        parts = (message.get(field) or {}).get("date-parts")
        if parts and parts[0] and parts[0][0]:
            year = str(parts[0][0])
            break
    return {
        "doi": doi,
        "title": title,
        "authors": authors,
        "year": year,
        "container": (message.get("container-title") or [""])[0],
        "volume": message.get("volume", ""),
        "issue": message.get("issue", ""),
        "page": message.get("page", ""),
        "publisher": message.get("publisher", ""),
        "url": message.get("URL", f"https://doi.org/{doi}"),
    }


def bibtex_from_crossref(meta: Dict[str, Any]) -> str:
    surname = meta["authors"][0].split()[-1] if meta["authors"] else "anon"
    key = _bibtex_key(surname, meta["year"], meta["title"])
    lines = [f"@article{{{key},"]
    lines.append(f"  author    = {{ {' and '.join(meta['authors'])} }},")
    lines.append(f"  title     = {{{meta['title']}}},")
    if meta["container"]:
        lines.append(f"  journal   = {{{meta['container']}}},")
    lines.append(f"  year      = {{{meta['year']}}},")
    for field in ("volume", "issue", "page"):
        if meta[field]:
            lines.append(f"  {field:<9} = {{{meta[field]}}},")
    lines.append(f"  doi       = {{{meta['doi']}}},")
    lines.append(f"  % VERIFIED: https://api.crossref.org/works/{meta['doi']}")
    lines.append("}")
    return "\n".join(lines)


def search_arxiv(query: str, max_results: int = 3) -> List[Dict[str, Any]]:
    params = urllib.parse.urlencode(
        {"search_query": query, "start": 0, "max_results": max_results, "sortBy": "relevance"}
    )
    payload = _fetch(f"{ARXIV_API}?{params}")
    root = ET.fromstring(payload)
    results = []
    for entry in root.findall(f"{ATOM}entry"):
        title = " ".join((entry.findtext(f"{ATOM}title") or "").split())
        identifier = (entry.findtext(f"{ATOM}id") or "").strip()
        published = entry.findtext(f"{ATOM}published") or ""
        authors = [author.findtext(f"{ATOM}name") or "" for author in entry.findall(f"{ATOM}author")]
        results.append(
            {
                "arxiv_id": identifier.rsplit("/", 1)[-1],
                "title": title,
                "authors": authors,
                "year": published[:4],
                "abs_url": identifier,
            }
        )
    return results


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Verify arXiv and Crossref references for paper/refs.bib")
    parser.add_argument("--ids", nargs="*", default=[], help="arXiv identifiers")
    parser.add_argument("--dois", nargs="*", default=[], help="DOIs to verify through Crossref")
    parser.add_argument("--search", nargs="*", default=[], help="arXiv search queries")
    parser.add_argument("--ids-file", default=None, help="file with one arXiv identifier per line")
    parser.add_argument("--out", default=None, help="write the BibTeX here instead of stdout")
    args = parser.parse_args(argv)

    identifiers = list(args.ids)
    if args.ids_file:
        with open(args.ids_file, "r", encoding="utf-8") as handle:
            identifiers += [line.strip() for line in handle if line.strip() and not line.startswith("#")]

    chunks: List[str] = []
    failures: List[str] = []

    for arxiv_id in identifiers:
        try:
            meta = fetch_arxiv(arxiv_id)
        except Exception as exc:  # noqa: BLE001 - reported, never silently dropped
            failures.append(f"{arxiv_id}: {exc!r}")
            continue
        if meta is None:
            failures.append(f"{arxiv_id}: not resolved by the arXiv API")
            continue
        chunks.append(bibtex_from_arxiv(meta))

    for doi in args.dois:
        try:
            meta = fetch_crossref(doi)
        except Exception as exc:  # noqa: BLE001
            failures.append(f"{doi}: {exc!r}")
            continue
        if meta is None:
            failures.append(f"{doi}: not resolved by Crossref")
            continue
        chunks.append(bibtex_from_crossref(meta))

    for query in args.search:
        try:
            results = search_arxiv(query)
        except Exception as exc:  # noqa: BLE001
            failures.append(f"search {query!r}: {exc!r}")
            continue
        for result in results:
            chunks.append(
                f"% SEARCH HIT for {query!r}: {result['abs_url']}\n"
                f"%   {result['title']} ({result['year']}) - {', '.join(result['authors'][:3])}"
            )

    output = "\n\n".join(chunks) if chunks else "% no entries resolved\n"
    if failures:
        output += "\n\n% UNRESOLVED (do not cite)\n" + "\n".join(f"%   {item}" for item in failures)

    if args.out:
        with open(args.out, "w", encoding="utf-8") as handle:
            handle.write(output + "\n")
        print(f"wrote {args.out} ({len(chunks)} entries, {len(failures)} unresolved)", file=sys.stderr)
    else:
        print(output)
    return 0 if chunks else 1


if __name__ == "__main__":
    raise SystemExit(main())
