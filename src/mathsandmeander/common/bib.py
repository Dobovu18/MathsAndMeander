"""BibTeX citations for marimo notebooks, shared by every notebook.

All references live in ``Notebooks/references.bib``. In a markdown cell type the citation
yourself and add an invisible marker after it, e.g. ``Hull (2014) <!-- @JCH14 -->`` (several
sources: ``<!-- @JCH14 @DW11 -->``); in code, ``bib.cite("JCH14")`` returns the author-year
text. End the notebook with a cell ``bib.bibliography(__file__)``. The reference list is built
by scanning the notebook's own source for these markers, so it needs no shared state.
"""

import re
from functools import lru_cache
from pathlib import Path

import marimo as mo
from pybtex.database import BibliographyData, Entry, parse_file
from pybtex.plugin import find_plugin

BIB_FILE = Path(__file__).resolve().parents[3] / "Notebooks" / "references.bib"

_CITE = re.compile(r"""\bcite\(\s*["']([^"']+)["']\s*\)""")
_LINK = re.compile(r"\]\(#ref-([\w:.\-/]+)\)")  # [Hull (2014)](#ref-JCH14)
_MARKER = re.compile(r"<!--\s*((?:@[\w:.\-/]+\s*)+)-->")  # <!-- @key1 @key2 -->


@lru_cache(maxsize=4)
def _load(path: Path, mtime: float) -> BibliographyData:
    return parse_file(path, bib_format="bibtex")


def load(path: Path = BIB_FILE) -> BibliographyData:
    """Parse the bibliography (re-parsed when the file changes)."""
    return _load(path, path.stat().st_mtime)


def _entry(key: str, path: Path) -> Entry:
    entries = load(path).entries
    if key not in entries:
        raise KeyError(f"{key!r} is not in {path.name}")
    return entries[key]


def _surname(person) -> str:
    return " ".join(person.prelast_names + person.last_names).replace("{", "").replace("}", "")


def _author_year(entry: Entry) -> str:
    names = [_surname(p) for p in entry.persons.get("author", [])]
    year = entry.fields.get("year", "n.d.")
    if not names:
        return year
    if len(names) == 1:
        who = names[0]
    elif len(names) == 2:
        who = f"{names[0]} & {names[1]}"
    else:
        who = f"{names[0]} et al."
    return f"{who}, {year}"


def cite(key: str, path: Path = BIB_FILE) -> str:
    """Inline author-year citation, e.g. ``(Black & Scholes, 1973)``."""
    return f"({_author_year(_entry(key, path))})"


def cited_keys(notebook: str | Path) -> list[str]:
    """Keys cited in the notebook source (``](#ref-key)``, ``cite(...)`` or ``<!-- @key -->``), by first use."""
    text = Path(notebook).read_text(encoding="utf-8")
    found = [(m.start(), [m.group(1)]) for m in _CITE.finditer(text)]
    found += [
        (m.start(), re.findall(r"@([\w:.\-/]+)", m.group(1))) for m in _MARKER.finditer(text)
    ]
    found += [(m.start(), [m.group(1)]) for m in _LINK.finditer(text)]
    return list(dict.fromkeys(k for _, keys in sorted(found) for k in keys))


def references(keys: list[str], path: Path = BIB_FILE) -> list[tuple[str, str]]:
    """``(key, markdown)`` for each of ``keys``, sorted by author and year."""
    if not keys:
        return []
    style = find_plugin("pybtex.style.formatting", "plain")()
    backend = find_plugin("pybtex.backends", "markdown")()
    bib = BibliographyData(entries={k: _entry(k, path) for k in keys})
    items = [
        (_author_year(bib.entries[e.key]), e.key, e.text.render(backend).replace("\n", " "))
        for e in style.format_entries(bib.entries.values())
    ]
    return [(key, text) for _, key, text in sorted(items)]


def references_markdown(keys: list[str], path: Path = BIB_FILE) -> str:
    """Markdown reference list for ``keys``, sorted by author and year."""
    return "\n\n".join(text for _, text in references(keys, path))


def references_html(keys: list[str], path: Path = BIB_FILE) -> str:
    """HTML reference list; entry ``key`` has ``id="ref-key"`` for ``](#ref-key)`` links."""
    return "".join(
        f'<div id="ref-{key}" class="bib-entry">{mo.md(text).text}</div>'
        for key, text in references(keys, path)
    )


def bibliography(notebook: str | Path, path: Path = BIB_FILE):
    """Final cell of every notebook: ``bib.bibliography(__file__)``.

    Shows the references cited in ``notebook``; nothing if it cites none.
    """
    body = references_html(cited_keys(notebook), path)
    if not body:
        return mo.md("")
    return mo.vstack([mo.md("## References"), mo.Html(body)])
