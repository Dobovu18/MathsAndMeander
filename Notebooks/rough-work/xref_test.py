import marimo

__generated_with = "0.25.1"
app = marimo.App(css_file="../_theme/blackboard.css")

with app.setup:
    import marimo as mo
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd

    from mathsandmeander.common import bib, theme, xref

    theme.apply()


@app.cell
def _():
    mo.md(r"""
    # Cross-references test

    A spike for numbered equations, figures and tables (see `mathsandmeander.common.xref`). Check that:

    1. each labelled equation shows its number on the right, and each figure and table has a
       numbered caption below it;
    2. the empty anchors above equations and outputs leave no odd gaps;
    3. the slider in [Figure 1.2.1](#fig-call-payoff) still works inside the captioned figure;
    4. clicking a reference such as [(1.2.1)](#eq-call), [Figure 1.2.1](#fig-call-payoff),
       [Table 1.2.1](#tab-payoff-sample) or [Table 1.2.2](#tab-moneyness) (all further down) scrolls to
       it, in both the browser app and the VS Code editor;
    5. clicking a contents entry below scrolls to its section, and clicking
       [Hull (2014)](#ref-JCH14) scrolls to the reference list at the end.

    Contents:
    * [1. Growth](#sec-growth)
        * [1.1. Continuous compounding](#sec-compounding)
        * [1.2. Options](#sec-options)

    ### Preface (unnumbered)

    Before any numbered heading, labelled equations and figures get plain numbers:

    <span id="eq-geom"></span>
    $$\sum_{k=0}^{\infty} x^{k} = \frac{1}{1-x}, \qquad |x| < 1. \tag{1}$$

    And this one is not labelled, so it has no number:

    $$e^{i\pi} + 1 = 0$$
    """)
    return


@app.cell
def _():
    _x = 0.8
    _n = np.arange(1, 21)
    _fig, _ax = plt.subplots(figsize=(6, 3))
    _ax.plot(_n, (1 - _x**_n) / (1 - _x), "o-", label="partial sum")
    _ax.axhline(1 / (1 - _x), color=theme.PINK, lw=1, label="limit")
    _ax.set_xlabel(r"number of terms $n$")
    _ax.legend()
    xref.figure(__file__, "partial-sums", _fig, r"Partial sums of [(1)](#eq-geom) for $x = 0.8$.")
    return


@app.cell
def _():
    mo.md(r"""
    <span id="sec-growth"></span>
    ## 1. Growth

    <span id="sec-compounding"></span>
    ### 1.1. Continuous compounding

    A deposit growing at rate $r$ satisfies

    <span id="eq-ode"></span>
    $$\frac{\mathrm{d}V}{V} = r\,\mathrm{d}t, \tag{1.1.1}$$

    whose solution is

    <span id="eq-exp"></span>
    $$V(t) = V_{0}e^{rt}. \tag{1.1.2}$$

    Integrating [(1.1.1)](#eq-ode) gives [(1.1.2)](#eq-exp). The geometric series [(1)](#eq-geom) and
    its [Figure 1](#fig-partial-sums) are in the preface.
    """)
    return


@app.cell
def _():
    mo.md(r"""
    #### 1.1.1. Discounting

    A sub-subsection is numbered within its subsection, so this is the third equation of 1.1:

    <span id="eq-discount"></span>
    $$V_{0} = V(T)e^{-rT}. \tag{1.1.3}$$

    It is [(1.1.2)](#eq-exp) solved for $V_{0}$. [Figure 1.1.1](#fig-growth) shows the growth.
    """)
    return


@app.cell
def _():
    _t = np.linspace(0, 10, 200)
    _fig, _ax = plt.subplots(figsize=(6, 3))
    for _r in (0.02, 0.05, 0.08):
        _ax.plot(_t, np.exp(_r * _t), label=rf"$r = {_r}$")
    _ax.set_xlabel(r"$t$ (years)")
    _ax.set_ylabel(r"$V(t)/V_{0}$")
    _ax.legend()
    xref.figure(__file__, "growth", _fig, r"Continuous compounding, [(1.1.2)](#eq-exp), for three rates.")
    return


@app.cell
def _():
    filler = "\n\n".join(
        f"Filler paragraph {i}, so that the page scrolls between the items and their references."
        for i in range(1, 31)
    )
    mo.md(filler)
    return


@app.cell
def _():
    mo.md(r"""
    <span id="sec-options"></span>
    ### 1.2. Options

    The counters restart in each subsection. The payoff of a call is

    <span id="eq-call"></span>
    $$\max(S_{T} - K, 0). \tag{1.2.1}$$

    [Figure 1.2.1](#fig-call-payoff) plots it, and [Table 1.2.1](#tab-payoff-sample) evaluates it on
    simulated prices.
    """)
    return


@app.cell
def _():
    strike = mo.ui.slider(start=50, stop=150, step=5, value=100, label="Strike $K$", show_value=True)
    return (strike,)


@app.cell
def _(strike):
    _s = np.linspace(0, 200, 400)
    _fig, _ax = plt.subplots(figsize=(6, 3))
    _ax.plot(_s, np.maximum(_s - strike.value, 0))
    _ax.axvline(strike.value, color=theme.PINK, lw=1)
    _ax.set_xlabel(r"$S_{T}$")
    _ax.set_ylabel("payoff")
    xref.figure(__file__, "call-payoff", mo.vstack([strike, _fig]), r"Call payoff [(1.2.1)](#eq-call); drag the strike.")
    return


@app.cell
def _():
    _rng = np.random.default_rng(0)
    _prices = 100 * np.exp(0.2 * _rng.standard_normal(5))
    _table = pd.DataFrame({"S_T": _prices.round(2), "payoff (K = 100)": np.maximum(_prices - 100, 0).round(2)})
    xref.table(__file__, "payoff-sample", _table, "Call payoffs at five simulated prices (seed 0).")
    return


@app.cell(hide_code=True)
def _():
    # A markdown table labelled with xref.table, as Table 2.1.1 in options-part1.py
    xref.table(__file__, "moneyness", mo.md(r"""
    | Option | ITM | ATM | OTM |
    | ------ | --- | --- | --- |
    | Call | $S > K$ | $S = K$ | $S < K$ |
    | Put | $S < K$ | $S = K$ | $S > K$ |
    """), r"Moneyness of a call and a put (a markdown table) [Hull (2014)](#ref-JCH14).")
    return


@app.cell
def _():
    mo.md(r"""
    References back up the page: [(1.1.1)](#eq-ode), [(1.1.3)](#eq-discount), [(1)](#eq-geom),
    [Figure 1](#fig-partial-sums), [Figure 1.1.1](#fig-growth), [Figure 1.2.1](#fig-call-payoff),
    [Table 1.2.1](#tab-payoff-sample), [Table 1.2.2](#tab-moneyness), and the sections
    [1. Growth](#sec-growth) and [1.2. Options](#sec-options).
    """)
    return


@app.cell
def _():
    bib.bibliography(__file__)
    return


if __name__ == "__main__":
    app.run()
