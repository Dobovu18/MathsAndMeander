import re
from pathlib import Path

import matplotlib as mpl

from mathsandmeander.common import theme

ROOT = Path(__file__).resolve().parent.parent
HEX = re.compile(r"#[0-9a-fA-F]{6}\b")

PALETTE = {
    value.lower()
    for name, value in vars(theme).items()
    if name.isupper() and isinstance(value, str) and HEX.fullmatch(value)
}


def _hexes(path: Path) -> set[str]:
    return {m.lower() for m in HEX.findall(path.read_text(encoding="utf-8"))}


def test_apply_sets_blackboard_style():
    theme.apply()
    assert mpl.rcParams["figure.facecolor"] == theme.BACKGROUND
    assert mpl.rcParams["axes.facecolor"] == theme.BACKGROUND
    assert mpl.rcParams["text.color"] == theme.FOREGROUND
    cycle = [c["color"] for c in mpl.rcParams["axes.prop_cycle"]]
    assert cycle == theme.CHALK


def test_chalk_palette_has_required_colours():
    for colour in (theme.YELLOW, theme.BLUE, theme.PINK, theme.GREEN):
        assert colour in theme.CHALK


def test_css_uses_only_theme_colours():
    assert _hexes(ROOT / "Notebooks" / "_theme" / "blackboard.css") <= PALETTE


def test_vscode_settings_use_only_theme_colours():
    assert _hexes(ROOT / ".vscode" / "settings.json") <= PALETTE


def test_css_and_vscode_share_core_colours():
    css = _hexes(ROOT / "Notebooks" / "_theme" / "blackboard.css")
    vscode = _hexes(ROOT / ".vscode" / "settings.json")
    core = {theme.BACKGROUND, theme.FOREGROUND, theme.SURFACE, theme.BORDER}
    assert core <= css
    assert core <= vscode


def test_notebooks_use_shared_css():
    for nb in (ROOT / "Notebooks").glob("*/*.py"):
        assert 'css_file="../_theme/blackboard.css"' in nb.read_text(encoding="utf-8"), nb


def test_ticks_are_white_and_text_is_latex_styled():
    theme.apply()
    assert mpl.rcParams["xtick.color"] == theme.FOREGROUND
    assert mpl.rcParams["ytick.color"] == theme.FOREGROUND
    assert mpl.rcParams["mathtext.fontset"] == "cm"
    assert mpl.rcParams["text.usetex"] is False
    theme.apply(usetex=True)
    assert mpl.rcParams["text.usetex"] is True
    theme.apply()


def test_macros_file_defines_R_and_C():
    table = theme.macro_table()
    assert table["R"] == r"\mathbb{R}"
    assert table["C"] == r"\mathbb{C}"
    # KaTeX predefines \R, \N, \Z and \newcommand aborts on an existing name,
    # so the marimo/KaTeX text must use \gdef, never \newcommand.
    assert r"\gdef\R{\mathbb{R}}" in theme.macros()
    assert r"\gdef\C{\mathbb{C}}" in theme.macros()
    assert r"\newcommand" not in theme.macros()
    assert r"\newcommand{\R}{\mathbb{R}}" in theme.latex_preamble()


def test_tex_expands_macros_but_not_longer_commands():
    assert theme.tex(r"$f:\R\to\C$") == r"$f:\mathbb{R}\to\mathbb{C}$"
    assert theme.tex(r"$A \Rightarrow B$") == r"$A \Rightarrow B$"
    assert theme.tex(r"$\R^2$") == r"$\mathbb{R}^2$"


def test_expanded_macros_render_in_mathtext_and_reach_latex_preamble():
    import io

    import matplotlib.pyplot as plt

    theme.apply()
    fig, ax = plt.subplots()
    ax.set_title(theme.tex(r"$f:\R\to\C$"))
    fig.savefig(io.BytesIO(), format="png")
    plt.close(fig)
    assert r"\newcommand{\R}{\mathbb{R}}" in mpl.rcParams["text.latex.preamble"]


def test_md_expands_macros_but_not_code():
    source = "Math $f:\\R\\to\\C$ and `\\R` and\n```\n\\C\n```\nthen $\\Z$"
    html = theme.md(source).text
    assert r"\mathbb{R}" in html
    assert r"\mathbb{C}" in html
    assert r"\mathbb{Z}" in html
    assert r"<code>\R</code>" in html  # inline code untouched
    assert "\\C\n" in html  # fenced code untouched


def test_style_plotly_uses_theme_colours():
    import plotly.graph_objects as go

    fig = theme.style_plotly(go.Figure(go.Scatter(x=[0, 1], y=[0, 1])))
    assert fig.layout.paper_bgcolor == theme.BACKGROUND
    assert fig.layout.plot_bgcolor == theme.BACKGROUND
    assert list(fig.layout.colorway) == theme.CHALK


def test_style_altair_uses_theme_colours():
    import altair as alt
    import pandas as pd

    chart = alt.Chart(pd.DataFrame({"x": [0, 1], "y": [0, 1]})).mark_point().encode(x="x", y="y")
    config = theme.style_altair(chart).to_dict()["config"]
    assert config["background"] == theme.BACKGROUND
    assert config["axis"]["labelColor"] == theme.FOREGROUND
    assert config["range"]["category"] == theme.CHALK


def test_tex_ignores_escaped_backslash():
    assert theme.tex("a \\\\C b") == "a \\\\C b"  # \\C is a line break then C
