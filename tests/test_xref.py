import marimo as mo
import pytest

from mathsandmeander.common import xref


def eq(label: str, body: str = "x", tag: str = r" \tag{?}") -> str:
    return f'<span id="eq-{label}"></span>\n$${body}{tag}$$\n'


def fig(label: str) -> str:
    return f'xref.figure(__file__, "{label}", plot)\n'


def tab(label: str) -> str:
    return f"xref.table(__file__, '{label}', df)\n"


# Equations


def test_numbered_within_subsection_and_reset():
    text = "### 1.1. A\n" + eq("a") + eq("b") + "### 1.2. B\n" + eq("c")
    assert xref.numbers(text)["equation"] == {"a": "1.1.1", "b": "1.1.2", "c": "1.2.1"}


def test_subsubsection_counts_in_its_subsection():
    text = (
        "### 1.1. A\n"
        + eq("a")
        + "#### 1.1.3. Deeper\n"
        + eq("b")
        + "## 2. Next\n### 2.1. C\n"
        + eq("c")
    )
    assert xref.numbers(text)["equation"] == {"a": "1.1.1", "b": "1.1.2", "c": "2.1.1"}


def test_item_before_first_subsection_is_numbered_like_latex():
    text = "## 2. Section\n" + eq("a") + "### 2.1. A\n" + eq("b")
    assert xref.numbers(text)["equation"] == {"a": "2.0.1", "b": "2.1.1"}


def test_plain_numbers_without_numbered_headings():
    text = "### Preface\n" + eq("a") + eq("b") + eq("c")
    assert xref.numbers(text)["equation"] == {"a": "1", "b": "2", "c": "3"}


def test_plain_numbers_before_first_numbered_heading():
    text = "### I. Preface\n" + eq("a") + "### 1.1. A\n" + eq("b")
    assert xref.numbers(text)["equation"] == {"a": "1", "b": "1.1.1"}


def test_unlabelled_equations_are_not_numbered():
    text = "### 1.1. A\n$$y$$\n" + eq("a")
    assert xref.numbers(text)["equation"] == {"a": "1.1.1"}


def test_renumber_sets_tags_and_references():
    text = (
        "see [(?)](#eq-b)\n### 1.1. A\n"
        + eq("a", tag=r" \tag{9.9}")
        + eq("b")
        + "by [(7)](#eq-a)"
    )
    out = xref.renumber(text)
    assert r"x \tag{1.1.1}$$" in out and r"x \tag{1.1.2}$$" in out
    assert "see [(1.1.2)](#eq-b)" in out and "by [(1.1.1)](#eq-a)" in out


def test_renumber_inserts_missing_tag_before_closing():
    single = xref.renumber("### 1.1. A\n" + eq("a", body="x = 1", tag=""))
    assert r"$$x = 1 \tag{1.1.1}$$" in single
    multi = xref.renumber(
        '### 1.1. A\n<span id="eq-a"></span>\n    $$\n    x = 1\n    $$\n'
    )
    assert "    x = 1 \\tag{1.1.1}\n    $$" in multi


def test_renumber_is_idempotent():
    text = "### 1.1. A\n" + eq("a") + fig("p") + "[(?)](#eq-a) [Figure ?](#fig-p)"
    once = xref.renumber(text)
    assert xref.renumber(once) == once


def test_duplicate_label_raises():
    with pytest.raises(ValueError, match="more than once"):
        xref.renumber(eq("a") + eq("a"))


def test_unknown_reference_raises():
    with pytest.raises(ValueError, match="unknown"):
        xref.renumber(eq("a") + "[(1)](#eq-missing)")


# Figures and tables


def test_counters_are_independent():
    text = (
        "### 1.1. A\n"
        + eq("e")
        + fig("f1")
        + tab("t")
        + fig("f2")
        + "### 1.2. B\n"
        + fig("f3")
    )
    found = xref.numbers(text)
    assert found["equation"] == {"e": "1.1.1"}
    assert found["figure"] == {"f1": "1.1.1", "f2": "1.1.2", "f3": "1.2.1"}
    assert found["table"] == {"t": "1.1.1"}


def test_figures_before_numbered_heading_are_plain():
    text = fig("a") + fig("b") + "### 1.1. A\n" + fig("c")
    assert xref.numbers(text)["figure"] == {"a": "1", "b": "2", "c": "1.1.1"}


def test_same_label_in_different_kinds_is_allowed():
    assert xref.numbers(eq("x") + fig("x") + tab("x"))["table"] == {"x": "1"}


def test_renumber_sets_figure_and_table_references():
    text = (
        "[Figure ?](#fig-p) and [Table 9](#tab-q)\n### 2.1. A\n" + fig("p") + tab("q")
    )
    assert xref.renumber(text).startswith(
        "[Figure 2.1.1](#fig-p) and [Table 2.1.1](#tab-q)"
    )


def test_reference_word_must_match_kind():
    with pytest.raises(ValueError, match="should start with 'Figure'"):
        xref.renumber(fig("p") + "[Table ?](#fig-p)")


def test_duplicate_figure_and_unknown_table_raise():
    with pytest.raises(ValueError, match="more than once"):
        xref.renumber(fig("p") + fig("p"))
    with pytest.raises(ValueError, match="tab-missing"):
        xref.renumber("[Table ?](#tab-missing)")


def test_figure_output_has_anchor_and_caption(tmp_path):
    nb = tmp_path / "nb.py"
    nb.write_text("### 1.1. A\n" + fig("a") + tab("b"), encoding="utf-8")
    html = xref.figure(nb, "a", mo.md("plot"), "A *caption*.").text
    assert 'id="fig-a"' in html and "Figure 1.1.1:" in html and "caption" in html
    assert 'id="tab-b"' in xref.table(nb, "b", mo.md("df")).text
    with pytest.raises(KeyError, match="string literal"):
        xref.figure(nb, "missing", mo.md("plot"))


# Lint: markup that looks like a cross-reference but would be skipped silently


def test_lint_accepts_well_formed_markup():
    text = (
        "* [1.1. A](#sec-a)\n"
        '<span id="sec-a"></span>\n### 1.1. A\n'
        + eq("a")
        + fig("p")
        + tab("t")
        + "[(?)](#eq-a) [Figure ?](#fig-p) [Table ?](#tab-t)\n"
        + "legs[0]() and d['k']()\n"  # code with ]() is not a link
    )
    assert xref.lint(text) == []


def test_lint_flags_bibliography_prefix_link_with_its_line():
    problems = xref.lint(eq("a") + "see [(1)](#ref-eq-a)\n")
    assert len(problems) == 1
    assert (
        problems[0].startswith("line 3:")
        and "bibliography prefix" in problems[0]
        and "'#eq-a'" in problems[0]
    )


@pytest.mark.parametrize(
    "anchor",
    [
        '<span id= "eq-a"></span>\n$$x$$',  # stray space
        '<span id="eq-a"><span>\n$$x$$',  # opening tag where the closing one should be
        '<span id="eq-a"></span>\ntext in between\n$$x$$',  # not directly above the $$ block
    ],
    ids=["space", "unclosed-span", "text-between"],
)
def test_lint_flags_unrecognised_anchor(anchor):
    problems = xref.lint(anchor)
    assert len(problems) == 1 and "anchor 'eq-a' is not recognised" in problems[0]


def test_lint_flags_anchor_swallowed_by_unclosed_display_maths():
    # The first block has no closing $$, so _ANCHOR pairs it with the opening $$ of the second
    problems = xref.lint('<span id="eq-a"></span>\n$$x\n\n' + eq("b"))
    assert any("anchor 'eq-b' is not recognised" in p for p in problems)
    assert any("odd number of $$" in p for p in problems)


@pytest.mark.parametrize(
    "link", ["[(1.7.1)(#eq-a)]", "[1.7.1](#eq-a)", "[Figure](#fig-a)"]
)
def test_lint_flags_malformed_link(link):
    problems = xref.lint(eq("a") + fig("a") + link)
    assert len(problems) == 1 and "is not in the form" in problems[0]


@pytest.mark.parametrize(
    "link", ["[(?)]()", "[(2.1.1)]( )", "[Figure ?]()", "[Table 1](#)"]
)
def test_lint_flags_empty_link(link):
    problems = xref.lint(eq("a") + "see " + link + "\n")
    assert (
        len(problems) == 1
        and "has no target" in problems[0]
        and problems[0].startswith("line 3:")
    )


def test_lint_flags_section_link_without_anchor():
    problems = xref.lint(
        '* [1. A](#sec-a)\n* [2. B](#sec-b)\n<span id="sec-a"></span>\n## 1. A\n'
    )
    assert (
        len(problems) == 1
        and "'#sec-b' has no matching" in problems[0]
        and problems[0].startswith("line 2:")
    )


def test_lint_flags_repeated_section_anchor():
    problems = xref.lint(
        '<span id="sec-a"></span>\n## 1. A\n<span id="sec-a"></span>\n## 2. B\n'
    )
    assert problems == ["section anchor 'sec-a' is used more than once"]


def test_renumber_refuses_malformed_markup():
    with pytest.raises(ValueError, match="malformed cross-references"):
        xref.renumber(eq("a") + "[(1)](#ref-eq-a)")


def test_main_reports_and_leaves_file_untouched(tmp_path, capsys):
    notebook = tmp_path / "nb.py"
    source = eq("a") + "[(9)](#ref-eq-a)\n"
    notebook.write_text(source, encoding="utf-8")
    assert xref.main([str(notebook)]) == 1
    assert notebook.read_text(encoding="utf-8") == source
    assert "bibliography prefix" in capsys.readouterr().err


# Every notebook


@pytest.mark.parametrize("notebook", xref.notebooks(), ids=lambda p: p.name)
def test_every_notebook_numbers_up_to_date(notebook):
    source = xref._read(notebook)
    assert xref.renumber(source) == source, (
        f"run: uv run python -m mathsandmeander.common.xref {notebook}"
    )
