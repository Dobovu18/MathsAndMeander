from pathlib import Path

import pytest

from mathsandmeander.common import bib

NOTEBOOKS = Path(__file__).resolve().parents[1] / "Notebooks"

BIB = """
@article{one2000,
  author = {Alpha, Ann and Beta, Bob},
  title = {A Title},
  journal = {A Journal},
  year = {2000},
}
@article{two1999,
  author = {Gamma, Gus and Delta, Di and Eps, Ed},
  title = {Another},
  journal = {A Journal},
  year = {1999},
}
"""


@pytest.fixture
def bibfile(tmp_path):
    path = tmp_path / "refs.bib"
    path.write_text(BIB, encoding="utf-8")
    return path


def test_cite_formats_author_year(bibfile):
    assert bib.cite("one2000", bibfile) == "(Alpha & Beta, 2000)"
    assert bib.cite("two1999", bibfile) == "(Gamma et al., 1999)"


def test_cite_unknown_key_raises(bibfile):
    with pytest.raises(KeyError):
        bib.cite("missing", bibfile)


def test_cited_keys_in_order_of_first_use(tmp_path):
    nb = tmp_path / "nb.py"
    nb.write_text('bib.cite("b")\nbib.cite( \'a\' )\nbib.cite("b")\n', encoding="utf-8")
    assert bib.cited_keys(nb) == ["b", "a"]


def test_cited_keys_include_markers(tmp_path):
    nb = tmp_path / "nb.py"
    source = 'mo.md("Foo (2000) <!-- @b @a -->, bar <!--@c-->")\nbib.cite("d")\n'
    nb.write_text(source, encoding="utf-8")
    assert bib.cited_keys(nb) == ["b", "a", "c", "d"]


def test_references_sorted_by_author(bibfile):
    text = bib.references_markdown(["two1999", "one2000"], bibfile)
    assert text.index("Alpha") < text.index("Gamma")
    assert "A title" in text


def test_references_empty_without_citations(bibfile):
    assert bib.references_markdown([], bibfile) == ""


def test_project_bibliography_parses_and_cites():
    for key in bib.load().entries:
        assert bib.cite(key).startswith("(")


@pytest.mark.parametrize(
    "notebook",
    [p for p in NOTEBOOKS.rglob("*.py") if "_theme" not in p.parts],
    ids=lambda p: p.name,
)
def test_every_notebook_ends_with_bibliography(notebook):
    source = notebook.read_text(encoding="utf-8")
    assert "bib.bibliography(__file__)" in source
    assert all(k in bib.load().entries for k in bib.cited_keys(notebook))


def test_link_citations_found_and_anchored(tmp_path, bibfile):
    nb = tmp_path / "nb.py"
    nb.write_text('mo.md("see [Alpha (2000)](#ref-one2000) and [x](#top)")\n', encoding="utf-8")
    assert bib.cited_keys(nb) == ["one2000"]
    assert 'id="ref-one2000"' in bib.references_html(["one2000"], bibfile)
