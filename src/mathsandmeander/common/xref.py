"""Numbered equations, figures and tables with clickable references, shared by every notebook.

Each kind has its own counter, numbered within the subsection of the latest hand-numbered heading,
e.g. (1.4.2), Figure 1.4.1 and Table 1.4.1; before any numbered heading the numbers are plain
(1), (2), ... Everything is found by scanning the notebook source in order, as ``bib.cited_keys``
does, so it needs no shared state.

**Equations** (markdown). KaTeX has no ``\\label``/``\\eqref``, so the numbers are written into the
text. In a ``mo.md(r\"\"\"...\"\"\")`` cell put an empty anchor line directly above the ``$$`` block::

    <span id="eq-cont-comp"></span>
    $$V(t) = V_{0}e^{rt} \\tag{?}$$

**Figures and tables** (code). Make the captioned output the last expression of the cell; its number
is worked out when the cell runs::

    xref.figure(__file__, "call-pnl", fig, r"European call: holder and writer P\\&L.")
    xref.table(__file__, "quotes", df, "Option quotes.")

**References** in markdown (or in a caption): ``[(?)](#eq-cont-comp)``, ``[Figure ?](#fig-call-pnl)``,
``[Table ?](#tab-quotes)``. Then run ``uv run python -m mathsandmeander.common.xref [notebook.py ...]``
(no arguments: every notebook), which sets every ``\\tag{...}`` and reference number. It changes
nothing and reports the line instead if an anchor or link is malformed (see ``lint``), e.g.
``#ref-eq-...`` (the bibliography prefix), ``id= "eq-..."``, an empty link ``[(?)]()`` or an unclosed ``$$``.

**Sections** are not numbered by this module, but ``lint`` checks them: a link ``[1.1. Basics](#sec-name)``
(e.g. in a contents list) needs exactly one anchor ``<span id="sec-name"></span>`` on the line above the heading.
"""

import re
import sys
from functools import lru_cache
from pathlib import Path

import marimo as mo

NOTEBOOKS = Path(__file__).resolve().parents[3] / "Notebooks"

KINDS = ("equation", "figure", "table")
PREFIX = {
    "equation": "eq",
    "figure": "fig",
    "table": "tab",
}  # anchor id is f"{prefix}-{label}"
WORD = {"figure": "Figure", "table": "Table"}

_LABEL = r"[\w:.\-]+"
# Hand-numbered markdown headings: "## 2.", "### 1.4.", "#### 1.1.3." (not "### I. Preface")
_HEADING = re.compile(r"^[ \t]*#{2,4}[ \t]+(\d+(?:\.\d+)*)\.?(?=\s)", re.MULTILINE)
# A labelled equation: the anchor line, then the $$...$$ block (groups: open, label, body, close)
_ANCHOR = re.compile(
    rf'(<span id="eq-({_LABEL})"></span>\s*\$\$)(.*?)(\$\$)', re.DOTALL
)
_TAG = re.compile(r"\\tag\{[^}]*\}")
# xref.figure(__file__, "label", ...) / xref.table(__file__, "label", ...)
_CALL = re.compile(rf"""\bxref\.(figure|table)\(\s*__file__\s*,\s*["']({_LABEL})["']""")
_EQ_REF = re.compile(rf"\[\(([^)\]]*)\)\]\(#eq-({_LABEL})\)")  # [(1.4.2)](#eq-name)
_OUT_REF = re.compile(
    rf"\[(\w+) [^\]]*\]\(#(fig|tab)-({_LABEL})\)"
)  # [Figure 1.4.1](#fig-name)
_KIND_OF = {"fig": "figure", "tab": "table"}
# Loose patterns for lint: anything that looks like an anchor or a link, well-formed or not
_LOOSE_ANCHOR = re.compile(rf"""\bid\s*=\s*["']eq-({_LABEL})""")
_LOOSE_EQ_LINK = re.compile(rf"\(#eq-({_LABEL})\)")
_LOOSE_OUT_LINK = re.compile(rf"\(#(?:fig|tab)-({_LABEL})\)")
_BIB_PREFIX_LINK = re.compile(rf"#ref-(eq|fig|tab)-({_LABEL})")
# An xref-style link whose target was never filled in: [(?)](), [Figure ?](), [Table 1](#). Only these
# shapes, since the lint also scans code, where e.g. legs[0]() is ordinary Python.
_EMPTY_LINK = re.compile(r"\[(?:\([^)\]]*\)|(?:Figure|Table) [^\]]*)\]\(\s*#?\s*\)")
# Section anchors above headings (<span id="sec-name"></span>) and the links to them, e.g. in a contents list
_SEC_ANCHOR = re.compile(rf'<span id="sec-({_LABEL})"></span>')
_SEC_LINK = re.compile(rf"\]\(#sec-({_LABEL})\)")


def _items(text: str, kind: str) -> list[tuple[int, str]]:
    """``(position, label)`` of every item of ``kind`` in the source."""
    if kind == "equation":
        return [(m.start(), m.group(2)) for m in _ANCHOR.finditer(text)]
    return [(m.start(), m.group(2)) for m in _CALL.finditer(text) if m.group(1) == kind]


def _events(text: str, kind: str) -> list[tuple[str, str]]:
    """``("heading", "1.1.3")`` and ``("item", "cont-comp")`` for one kind, in source order."""
    found = [(m.start(), "heading", m.group(1)) for m in _HEADING.finditer(text)]
    found += [(pos, "item", label) for pos, label in _items(text, kind)]
    return [(event, value) for _, event, value in sorted(found)]


def number_items(events: list[tuple[str, str]]) -> dict[str, str]:
    """Number of each item, e.g. ``{"cont-comp": "1.1.2", "geom": "1"}``, from ``_events``."""
    numbers: dict[str, str] = {}
    prefix, count = "", 0  # no numbered heading yet: plain (1), (2), ...
    for event, value in events:
        if event == "heading":
            # Subsection of the heading: "1.1.3" -> "1.1"; a section heading "2" -> "2.0", as LaTeX
            # numbers items that come before a section's first subsection
            subsection = ".".join((value.split(".") + ["0"])[:2])
            if subsection != prefix:
                prefix, count = subsection, 0
        else:
            count += 1
            numbers[value] = f"{prefix}.{count}" if prefix else str(count)
    return numbers


def numbers(text: str) -> dict[str, dict[str, str]]:
    """``{kind: {label: number}}`` for every kind. Raises ``ValueError`` for a repeated label."""
    result = {}
    for kind in KINDS:
        labels = [label for _, label in _items(text, kind)]
        repeated = sorted({label for label in labels if labels.count(label) > 1})
        if repeated:
            raise ValueError(f"{kind} labels used more than once: {repeated}")
        result[kind] = number_items(_events(text, kind))
    return result


def lint(text: str) -> list[str]:
    """Anchors and links that look like cross-references but that ``renumber`` would skip.

    The patterns above only see well-formed markup, so a typo would otherwise be ignored silently:
    the item goes unnumbered, or the link keeps a stale number and does not jump. Each problem is
    reported as ``"line n: ..."``.
    """

    def line(pos: int) -> int:
        return text.count("\n", 0, pos) + 1

    problems = []
    for m in _BIB_PREFIX_LINK.finditer(text):
        problems.append(
            f"line {line(m.start())}: link '#ref-{m.group(1)}-{m.group(2)}' uses the bibliography "
            f"prefix; write '#{m.group(1)}-{m.group(2)}'"
        )

    anchor_ids = {m.start() + len("<span ") for m in _ANCHOR.finditer(text)}
    for m in _LOOSE_ANCHOR.finditer(text):
        if m.start() not in anchor_ids:
            problems.append(
                f"line {line(m.start())}: anchor 'eq-{m.group(1)}' is not recognised; write "
                f'<span id="eq-{m.group(1)}"></span> on the line directly above the $$ block, and '
                "check that the $$ block before it is closed"
            )

    for loose, strict, form in (
        (_LOOSE_EQ_LINK, _EQ_REF, "[(?)](#eq-label)"),
        (_LOOSE_OUT_LINK, _OUT_REF, "[Figure ?](#fig-label) or [Table ?](#tab-label)"),
    ):
        ends = {m.end() for m in strict.finditer(text)}
        for m in loose.finditer(text):
            if m.end() not in ends:
                problems.append(
                    f"line {line(m.start())}: link '{m.group(0)}' is not in the form {form}"
                )

    for m in _EMPTY_LINK.finditer(text):
        problems.append(
            f"line {line(m.start())}: link '{m.group(0)}' has no target; write e.g. [(?)](#eq-label)"
        )

    sections = [m.group(1) for m in _SEC_ANCHOR.finditer(text)]
    for label in sorted({s for s in sections if sections.count(s) > 1}):
        problems.append(f"section anchor 'sec-{label}' is used more than once")
    for m in _SEC_LINK.finditer(text):
        if m.group(1) not in sections:
            problems.append(
                f"line {line(m.start())}: link '#sec-{m.group(1)}' has no matching "
                f'<span id="sec-{m.group(1)}"></span> above a heading'
            )

    if text.count("$$") % 2:
        problems.append("odd number of $$ delimiters: a display equation is not closed")
    return problems


def renumber(text: str) -> str:
    """``text`` with every equation's ``\\tag`` and every reference's number set.

    Raises ``ValueError`` for malformed anchors or links (see ``lint``), a repeated label, a
    reference to an unknown label, or a reference whose word does not match its kind
    (``[Table 1](#fig-x)``).
    """
    problems = lint(text)
    if problems:
        raise ValueError("malformed cross-references:\n  " + "\n  ".join(problems))
    found = numbers(text)
    unknown = sorted(
        {m.group(2) for m in _EQ_REF.finditer(text)} - found["equation"].keys()
    )
    for m in _OUT_REF.finditer(text):
        word, kind, label = m.group(1), _KIND_OF[m.group(2)], m.group(3)
        if word != WORD[kind]:
            raise ValueError(f"{m.group(0)!r} should start with {WORD[kind]!r}")
        if label not in found[kind]:
            unknown.append(f"{PREFIX[kind]}-{label}")
    if unknown:
        raise ValueError(f"references to unknown labels: {unknown}")

    def set_tag(m: re.Match) -> str:
        opening, label, body, closing = m.groups()
        tag = rf"\tag{{{found['equation'][label]}}}"
        if _TAG.search(body):
            body = _TAG.sub(lambda _: tag, body, count=1)
        else:  # insert before the closing $$, keeping its line break/indentation
            stripped = body.rstrip()
            body = f"{stripped} {tag}{body[len(stripped) :]}"
        return f"{opening}{body}{closing}"

    def set_out_ref(m: re.Match) -> str:
        kind, label = _KIND_OF[m.group(2)], m.group(3)
        return f"[{WORD[kind]} {found[kind][label]}](#{m.group(2)}-{label})"

    text = _ANCHOR.sub(set_tag, text)
    text = _EQ_REF.sub(
        lambda m: f"[({found['equation'][m.group(2)]})](#eq-{m.group(2)})", text
    )
    return _OUT_REF.sub(set_out_ref, text)


def _read(path: Path) -> str:
    with path.open(encoding="utf-8", newline="") as f:  # keep the file's line endings
        return f.read()


@lru_cache(maxsize=16)
def _numbers_of(path: Path, mtime: float) -> dict[str, dict[str, str]]:
    return numbers(_read(path))


def _captioned(kind: str, notebook: str | Path, label: str, output, caption: str):
    path = Path(notebook).resolve()
    found = _numbers_of(path, path.stat().st_mtime)[kind]
    if label not in found:
        raise KeyError(
            f"{kind} {label!r} not found in {path.name}: the label must be a string literal, "
            f'as in xref.{kind}(__file__, "{label}", ...)'
        )
    head = f"**{WORD[kind]} {found[label]}:**"
    text = f"{head} {caption}" if caption else head
    return mo.vstack(
        [
            mo.Html(f'<span id="{PREFIX[kind]}-{label}"></span>'),
            output,
            mo.Html(f'<div class="xref-caption">{mo.md(text).text}</div>'),
        ]
    )


def figure(notebook: str | Path, label: str, output, caption: str = ""):
    """``output`` (a plot, widget, layout, ...) with anchor ``fig-label`` and a numbered caption below.

    Call it as the last expression of a cell: ``xref.figure(__file__, "label", fig, "Caption.")``.
    """
    return _captioned("figure", notebook, label, output, caption)


def table(notebook: str | Path, label: str, output, caption: str = ""):
    """Like ``figure``, for a table (e.g. a DataFrame): anchor ``tab-label``, caption "Table n: ..."."""
    return _captioned("table", notebook, label, output, caption)


def notebooks() -> list[Path]:
    """Every notebook under ``Notebooks/`` (not the theme folder)."""
    return sorted(p for p in NOTEBOOKS.rglob("*.py") if "_theme" not in p.parts)


def main(argv: list[str] | None = None) -> int:
    """Renumber the given notebooks in place (all notebooks if none are given)."""
    paths = [Path(a) for a in argv] if argv else notebooks()
    status = 0
    for path in paths:
        source = _read(path)
        try:
            updated = renumber(source)
        except ValueError as e:  # leave the file untouched and report what to fix
            print(f"{path}: {e}", file=sys.stderr)
            status = 1
            continue
        if updated != source:
            with path.open("w", encoding="utf-8", newline="") as f:
                f.write(updated)
            print(f"renumbered {path}")
    return status


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
