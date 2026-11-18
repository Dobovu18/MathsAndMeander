"""Blackboard theme shared by notebooks, the browser app, VS Code and plots.

This module is the single source of truth for the palette. The hex values are
repeated in ``Notebooks/_theme/blackboard.css`` and ``.vscode/settings.json``;
``tests/test_theme.py`` fails if they drift apart.
"""

import re
from pathlib import Path

import matplotlib as mpl
from cycler import cycler

MACROS_FILE = Path(__file__).resolve().parents[3] / "Notebooks" / "_theme" / "macros.tex"
_NEWCOMMAND = re.compile(r"\\newcommand\{\\([A-Za-z]+)\}\{(.*)\}")
_CODE = re.compile(r"(```.*?```|`[^`\n]+`)", re.DOTALL)

BACKGROUND = "#202021"  # dark grey board
SURFACE = "#34373a"  # cards, popovers, code blocks, cell editors
BORDER = "#4a4e52"
GRID = "#5a5e62"
FOREGROUND = "#f2f2f2"  # white chalk
MUTED = "#b4b8bc"

YELLOW = "#f4e285"
BLUE = "#7cb7e8"
PINK = "#f4a6c6"
GREEN = "#a5d6a7"
ORANGE = "#f8b878"
PURPLE = "#c3a6e8"

CHALK = [YELLOW, BLUE, PINK, GREEN, ORANGE, PURPLE]


def macro_table() -> dict[str, str]:
    """Return the custom LaTeX macros in ``macros.tex`` as ``{name: body}``."""
    table = {}
    for line in MACROS_FILE.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("%"):
            continue
        match = _NEWCOMMAND.fullmatch(line)
        if match is None:
            raise ValueError(f"Unsupported line in {MACROS_FILE.name}: {line!r}")
        table[match.group(1)] = match.group(2)
    return table


def macros() -> str:
    r"""Return the macros as ``\gdef`` lines, for ``mo.latex(theme.macros())``.

    ``\gdef`` rather than ``\newcommand``: KaTeX already defines ``\R``, ``\N``
    and ``\Z``, and ``\newcommand`` aborts the whole block on an existing name,
    so none of the later macros would be defined.
    """
    return "\n".join(
        rf"\gdef\{name}{{{body}}}" for name, body in macro_table().items()
    )


def latex_preamble() -> str:
    r"""Return the macros as ``\newcommand`` lines for a real LaTeX preamble.

    ``\newcommand`` fails loudly if a name clashes with an existing command.
    """
    return "\n".join(
        rf"\newcommand{{\{name}}}{{{body}}}" for name, body in macro_table().items()
    )


def tex(text: str) -> str:
    r"""Expand the custom macros in ``text`` so matplotlib's mathtext understands them.

    Mathtext has no ``\newcommand``, so wrap plot labels: ``theme.tex(r"$f:\R\to\C$")``.
    """
    for name, body in macro_table().items():
        text = re.sub(rf"(?<!\\)\\{name}(?![A-Za-z])", lambda _, body=body: body, text)
    return text


def md(text: str):
    r"""Like ``mo.md``, but expands the custom macros first.

    KaTeX's macro state is not reliably shared between cells (it fails in the VS
    Code marimo extension), so macros are expanded in Python instead. Inline
    code and fenced code blocks are left untouched.
    """
    import marimo as mo

    parts = _CODE.split(text)  # odd entries are the code spans
    return mo.md("".join(p if i % 2 else tex(p) for i, p in enumerate(parts)))


def apply(usetex: bool = False) -> None:
    """Set matplotlib's global style to the blackboard theme.

    Text is typeset in LaTeX style. By default this uses matplotlib's built-in
    mathtext with Computer Modern fonts (fast, no TeX install needed). Pass
    ``usetex=True`` to typeset with a real LaTeX installation instead (slower,
    and every label is compiled on each redraw). Custom macros from
    ``Notebooks/_theme/macros.tex`` work directly with ``usetex=True``; in
    mathtext mode wrap labels in :func:`tex`.
    """
    mpl.rcParams.update(
        {
            "figure.facecolor": BACKGROUND,
            "figure.edgecolor": BACKGROUND,
            "axes.facecolor": BACKGROUND,
            "axes.edgecolor": MUTED,
            "axes.labelcolor": FOREGROUND,
            "axes.titlecolor": FOREGROUND,
            "axes.prop_cycle": cycler(color=CHALK),
            "axes.grid": True,
            "grid.color": GRID,
            "grid.alpha": 0.5,
            "grid.linewidth": 0.6,
            "xtick.color": FOREGROUND,
            "ytick.color": FOREGROUND,
            "text.color": FOREGROUND,
            "lines.linewidth": 2.0,
            "legend.facecolor": SURFACE,
            "legend.edgecolor": BORDER,
            "legend.labelcolor": FOREGROUND,
            "savefig.facecolor": BACKGROUND,
            "savefig.edgecolor": BACKGROUND,
            "font.family": "serif",
            "font.serif": ["Computer Modern Roman"] if usetex else ["cmr10", "DejaVu Serif"],
            "mathtext.fontset": "cm",
            "axes.formatter.use_mathtext": True,
            "text.usetex": usetex,
            "text.latex.preamble": "\\usepackage{amssymb}\n" + latex_preamble(),
            "figure.autolayout": True,
        }
    )


def style_plotly(fig):
    """Apply the blackboard theme to a plotly figure and return it."""
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor=BACKGROUND,
        plot_bgcolor=BACKGROUND,
        font={"color": FOREGROUND},
        colorway=CHALK,
    )
    fig.update_xaxes(gridcolor=GRID, zerolinecolor=MUTED)
    fig.update_yaxes(gridcolor=GRID, zerolinecolor=MUTED)
    return fig


def style_altair(chart):
    """Apply the blackboard theme to a (top-level) altair chart and return it."""
    return (
        chart.configure(background=BACKGROUND)
        .configure_axis(
            labelColor=FOREGROUND,
            titleColor=FOREGROUND,
            gridColor=GRID,
            domainColor=MUTED,
            tickColor=MUTED,
        )
        .configure_legend(labelColor=FOREGROUND, titleColor=FOREGROUND)
        .configure_title(color=FOREGROUND)
        .configure_range(category=CHALK)
    )
