#!/usr/bin/env python3
"""Manuscript lint that enforces the writing rules of this project.

Rules, all of them checked outside LaTeX comments and outside math mode:

1. No semicolons in prose.
2. No em dashes and no double hyphen in prose.
3. No bold inside the main sections (between the introduction and the conclusion).
4. No itemized or enumerated lists inside the main sections.
5. No acronyms in the abstract (a run of two or more capitals outside math).
6. Every ``\\cite`` key must exist in the bibliography file.
7. Every ``\\input`` path must exist relative to the manuscript directory.

Usage
-----
    python scripts/lint_manuscript.py paper/main.tex
    python scripts/lint_manuscript.py paper/main.tex --bib paper/refs.bib
"""

from __future__ import annotations

import argparse
import pathlib
import re
import sys
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

MAIN_SECTION_START = "Introduction"
MAIN_SECTION_END = "Conclusion"

CITE_PATTERN = re.compile(r"\\cite[a-z]*\{([^}]*)\}")
INPUT_PATTERN = re.compile(r"\\input\{([^}]*)\}")
SECTION_PATTERN = re.compile(r"\\section\*?\{([^}]*)\}")
ACRONYM_PATTERN = re.compile(r"\b[A-Z]{2,}\b")
MATH_PATTERN = re.compile(r"\$[^$]*\$")

# Model names that look like acronyms but are not, so the abstract may name them.
ACRONYM_ALLOWLIST = ("XY",)


def strip_comment(line: str) -> str:
    """Remove a LaTeX comment, keeping the part before an unescaped percent sign."""
    index = 0
    while True:
        index = line.find("%", index)
        if index == -1:
            return line
        if index == 0 or line[index - 1] != "\\":
            return line[:index]
        index += 1


def strip_math(line: str) -> str:
    """Remove inline math so that operators inside equations never trip the rules."""
    return MATH_PATTERN.sub(" ", line)


def bib_keys(path: pathlib.Path) -> List[str]:
    """Keys declared in a BibTeX file."""
    if not path.exists():
        return []
    return re.findall(r"@\w+\{([^,]+),", path.read_text(encoding="utf-8"))


def lint(text: str, bib_path: Optional[pathlib.Path] = None, tex_path: Optional[pathlib.Path] = None) -> List[str]:
    """Return one violation per line as ``file:line: rule: message``."""
    violations: List[str] = []
    lines = text.splitlines()
    location = str(tex_path) if tex_path else "<manuscript>"

    def report(number: int, rule: str, message: str) -> None:
        violations.append(f"{location}:{number}: {rule}: {message}")

    in_abstract = False
    in_main_section = False
    in_algorithm = False
    for number, raw_line in enumerate(lines, start=1):
        line = strip_comment(raw_line)
        if "\\begin{abstract}" in line:
            in_abstract = True
        if "\\end{abstract}" in line:
            in_abstract = False

        section = SECTION_PATTERN.search(line)
        if section:
            title = section.group(1)
            if MAIN_SECTION_START.lower() in title.lower():
                in_main_section = True
            elif MAIN_SECTION_END.lower() in title.lower():
                in_main_section = False

        if line.lstrip().startswith('\\' + "begin{algorithm}"):
            in_algorithm = True
        if line.lstrip().startswith('\\' + "end{algorithm}"):
            in_algorithm = False

        prose = strip_math(line)
        if not prose.strip():
            continue

        if ";" in prose:
            report(number, "no-semicolons", "a semicolon appears in prose")
        if "\u2014" in prose or "--" in prose:
            report(number, "no-em-dashes", "an em dash or a double hyphen appears in prose")
        if in_main_section and not in_algorithm and "\\textbf" in prose:
            report(number, "no-bold-in-main-sections", "bold text inside a main section")
        if in_main_section and not in_algorithm and ("\\begin{itemize}" in prose or "\\begin{enumerate}" in prose):
            report(number, "no-lists-in-main-sections", "a list inside a main section")
        if in_abstract:
            for acronym in ACRONYM_PATTERN.findall(prose):
                if acronym in ACRONYM_ALLOWLIST:
                    continue
                report(number, "no-acronyms-in-abstract", f"acronym {acronym!r} in the abstract")

        citation = CITE_PATTERN.search(line)
        if citation:
            keys = [key.strip() for key in citation.group(1).split(",") if key.strip()]
            if bib_path is not None:
                known = bib_keys(bib_path)
                for key in keys:
                    if key not in known:
                        report(number, "unknown-citation", f"cite key {key!r} is not in {bib_path.name}")

        inclusion = INPUT_PATTERN.search(line)
        if inclusion:
            target = inclusion.group(1)
            if not target.endswith(".tex"):
                target = target + ".tex"
            base = tex_path.parent if tex_path else pathlib.Path(".")
            candidate = (base / target).resolve()
            if not candidate.exists():
                report(number, "missing-input", f"the included file {target!r} does not exist")

    return violations


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Lint the manuscript against the project writing rules")
    parser.add_argument("manuscript", nargs="?", default="paper/main.tex")
    parser.add_argument("--bib", default=None, help="bibliography file (defaults to a sibling refs.bib)")
    args = parser.parse_args(argv)

    tex_path = pathlib.Path(args.manuscript)
    if not tex_path.exists():
        print(f"lint_manuscript: {tex_path} does not exist", file=sys.stderr)
        return 2
    bib_path = pathlib.Path(args.bib) if args.bib else tex_path.parent / "refs.bib"
    if not bib_path.exists():
        bib_path = None

    violations = lint(tex_path.read_text(encoding="utf-8"), bib_path=bib_path, tex_path=tex_path)
    if not violations:
        print("clean")
        return 0
    for violation in violations:
        print(violation)
    print(f"{len(violations)} violation(s)")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
