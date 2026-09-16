"""Tests for the manuscript lint."""

from __future__ import annotations

import pathlib

from scripts.lint_manuscript import lint

CLEAN = r"""
\documentclass{article}
\begin{document}
\begin{abstract}
Variational quantum algorithms are limited by non-convex optimization on noisy intermediate-scale hardware.
\end{abstract}
\section{Introduction}
Hybrid control loops run a local optimizer and a supervisor model in step.
The supervisor reads telemetry and reports one diagnostic label.
\section{Conclusion}
Nothing here either.
\end{document}
"""

DIRTY = r"""
\documentclass{article}
\begin{document}
\begin{abstract}
The VQA benchmark uses NISQ hardware.
\end{abstract}
\section{Introduction}
This sentence has a semicolon; it should not.
This sentence has an em dash \u2014 and a double hyphen -- too.
We \textbf{stress} one word and then list:
\begin{itemize}
\item one
\end{itemize}
We cite \cite{missing2026} and include \input{tables/missing.tex}.
\section{Conclusion}
Fine.
\end{document}
"""


def test_clean_manuscript_passes():
    assert lint(CLEAN) == []


def test_each_rule_fires_on_a_dirty_manuscript(tmp_path: pathlib.Path):
    bib = tmp_path / "refs.bib"
    bib.write_text("@misc{other2026,\n  title = {Another title},\n}\n", encoding="utf-8")
    violations = lint(DIRTY, bib_path=bib)
    rules = {violation.split(":")[2].strip() for violation in violations}
    assert "no-semicolons" in rules
    assert "no-em-dashes" in rules
    assert "no-bold-in-main-sections" in rules
    assert "no-lists-in-main-sections" in rules
    assert "no-acronyms-in-abstract" in rules
    assert "unknown-citation" in rules
    assert "missing-input" in rules


def test_bold_outside_main_sections_is_allowed():
    text = "\\section{Abstract}\nSome \\textbf{bold} text before the introduction.\n\\section{Introduction}\nplain.\n"
    assert lint(text) == []


def test_bibliography_lookup(tmp_path: pathlib.Path):
    bib = tmp_path / "refs.bib"
    bib.write_text("@misc{known2026,\n  title = {A title},\n}\n", encoding="utf-8")
    text = "\\section{Introduction}\nWe cite \\cite{known2026} here.\n"
    assert lint(text, bib_path=bib) == []
    assert any("unknown-citation" in violation for violation in lint("\\cite{nope2026}\n", bib_path=bib))
