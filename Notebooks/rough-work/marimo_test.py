import marimo

__generated_with = "0.25.1"
app = marimo.App(css_file="../_theme/blackboard.css")

with app.setup:
    # Initialization code that runs before all other cells

    from math import factorial

    import altair as alt
    import anywidget
    import marimo as mo
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd
    import panel as pn
    import plotly.graph_objects as go
    import traitlets

    from mathsandmeander.common import bib, theme
    from mathsandmeander.common.widgets import PlaySlider

    theme.apply()#usetex=True)
    mo.latex(theme.macros())


@app.cell
def _():
    mo.md(r"""
    # Marimo template

    This is a **markdown** cell. You can use *italics*, **bold**, `inline code`,
    and [links](https://marimo.io), plus lists:

    - Plain text works as expected.
    - Maths is written in LaTeX between dollar signs.

    ## Maths examples

    An inline equation: the area of a circle is $A = \pi r^2$, and
    $e^{i\pi} + 1 = 0$ is Euler's identity.

    A display equation on its own line, the Gaussian integral:

    $$
    \int_{-\infty}^{\infty} e^{-x^2}\,dx = \sqrt{\pi}
    $$

    And the Fourier transform of a function $f$:

    $$
    \hat{f}(\xi) = \int_{-\infty}^{\infty} f(x)\, e^{-2\pi i x \xi}\,dx
    $$

    Custom macros from `Notebooks/_theme/macros.tex` work everywhere:
    $f : \R \to \mathbb{C}$ is written with `\R` and `\mathbb{C}`.

    ## Code examples

    Code displayed in a markdown cell
    ```python
    print('hello world')
    ```

    Code displayed in a coding cell
    """)
    return


@app.cell
def _():
    frequency = mo.ui.slider(
        start=1, stop=10, step=0.01, value=2, label="Frequency (Hz)", show_value=True
    )
    amplitude = mo.ui.slider(
        start=0.1, stop=1, step=0.01, value=0.5, label="Amplitude", show_value=True
    )
    phase = mo.ui.slider(
        start=0, stop=2 * np.pi, step=0.01, value=0, label="Phase (rad)", show_value=True
    )

    frequency, amplitude, phase
    return amplitude, frequency, phase


@app.cell
def _(amplitude, frequency, phase):
    t = np.linspace(0, 1, 1000)
    sin_wave = amplitude.value * np.sin(2 * np.pi * frequency.value * t + phase.value)
    cos_wave = amplitude.value * np.cos(2 * np.pi * frequency.value * t + phase.value)

    fig, ax = plt.subplots(figsize=(8, 3.5))
    ax.plot(t, sin_wave, label = r"$\sin(2\pi f t + \phi)$")
    ax.plot(t, cos_wave, label = r"$\cos(2\pi f t + \phi)$")
    ax.set_xlabel(r"$t\ (\mathrm{s})$")
    ax.set_ylabel("Amplitude")
    ax.set_title(rf"Waves: $f = {frequency.value}\ \mathrm{{Hz}}$, $\phi = {phase.value}\ \mathrm{{rad}}$")
    ax.grid(alpha=0.3)
    ax.axhline(0, color=theme.FOREGROUND, alpha=0.5, lw=1)
    ax.legend()
    fig
    return


@app.cell
def _():
    mo.md(r"""
    ## Interactive components

    Each marimo `mo.ui` element is defined in the **controls** cell below, then shown
    next to a small piece of maths in its own cell, so only that example recomputes
    when you change a control. (An element's `.value` cannot be read in the cell that
    defines it.) Covered: `slider`, `range_slider`, `number`, `text`, `text_area`,
    `checkbox`, `switch`, `radio`, `dropdown`, `multiselect`, `button`, `run_button`,
    `date_range`, `refresh`, `array`, `dictionary`, `batch` + `form`, `file` and
    `table`. `date` and `datetime` work like `date_range`. A second controls cell
    adds `code_editor`, `file_browser`, `matrix`, `data_editor`, `tabs`, `matplotlib`,
    `altair_chart`, `plotly`, `data_explorer`, `dataframe`, `anywidget` and `panel`.
    Only `microphone` (needs a microphone) and `chat` (needs an LLM) are left out.
    A custom `PlaySlider` widget (in `mathsandmeander.common.widgets`) adds a slider with a
    play button, shown last with a travelling wave.
    """)
    return


@app.cell
def _():
    # --- controls: every component is defined here --------------------------------
    functions = {  # name -> (f, f', antiderivative of f)
        "sin": (np.sin, np.cos, lambda x: -np.cos(x)),
        "cos": (np.cos, lambda x: -np.sin(x), np.sin),
        "exp": (np.exp, np.exp, np.exp),
    }

    # Calculus
    func = mo.ui.radio(options=list(functions), value="sin", inline=True, label="f(x)")
    x0 = mo.ui.number(start=-3, stop=3, step=0.1, value=1.0, label="Tangent point x0")
    show_tangent = mo.ui.checkbox(value=True, label="Tangent line")
    show_deriv = mo.ui.switch(value=False, label="Derivative f'")
    interval = mo.ui.range_slider(
        start=0, stop=3, step=0.05, value=[0.5, 2.0], show_value=True, label="Interval [a, b]"
    )
    rule = mo.ui.dropdown(
        options=["left", "midpoint", "trapezoid"], value="midpoint", label="Rule"
    )
    n_sub = mo.ui.slider(start=1, stop=50, value=8, show_value=True, label="Subintervals n")
    shade = mo.ui.switch(value=True, label="Shade area")

    # Approximation
    taylor_order = mo.ui.slider(
        start=1, stop=15, step=2, value=5, show_value=True, label="Taylor degree"
    )
    legendre = mo.ui.multiselect(
        options={f"P{n}": n for n in range(6)}, value=["P1", "P2", "P3"], label="Legendre"
    )

    # Algebra
    quad = mo.ui.dictionary(
        {
            "a": mo.ui.slider(-3, 3, 0.1, value=1, show_value=True, label="a"),
            "b": mo.ui.slider(-3, 3, 0.1, value=0, show_value=True, label="b"),
            "c": mo.ui.slider(-3, 3, 0.1, value=-1, show_value=True, label="c"),
        },
        label="Quadratic a x^2 + b x + c",
    )
    poly = mo.ui.array(
        [mo.ui.number(-3, 3, step=0.5, value=v) for v in (-2, 0, 1)],
        label="Polynomial coefficients c0, c1, c2",
    )
    complex_form = (
        mo.md("Re z: {re} &nbsp; Im z: {im}")
        .batch(
            re=mo.ui.number(-3, 3, step=0.1, value=1.0),
            im=mo.ui.number(-3, 3, step=0.1, value=1.0),
        )
        .form(submit_button_label="Plot z")
    )

    # Data and fitting
    data_text = mo.ui.text_area(
        value="0, 1.1\n1, 2.9\n2, 5.2\n3, 7.1\n4, 8.8", rows=6, label="Data: x, y per line"
    )
    data_file = mo.ui.file(filetypes=[".csv", ".txt"], label="or upload a CSV")
    fit_degree = mo.ui.number(start=1, stop=5, step=1, value=1, label="Polynomial degree")

    # Estimating pi
    fraction = mo.ui.text(value="355/113", label="Fraction p/q")
    resample = mo.ui.button(value=0, on_click=lambda count: count + 1, label="Draw new sample")
    run_mc = mo.ui.run_button(label="Run 10^6 points")

    # Decay, animation and series
    span = mo.ui.date_range(value=("2024-01-01", "2025-01-01"), label="Time span")
    half_life = mo.ui.number(start=1, stop=5000, step=1, value=365, label="Half-life (days)")
    # no default_interval, so auto-refresh starts off; pick an interval to start it
    ticker = mo.ui.refresh(options=["0.5s", "1s", "2s"], label="Rotate")
    terms = mo.ui.table(
        [{"n": n, "1/n^2": round(1 / n**2, 4)} for n in range(1, 11)],
        selection="multi",
        initial_selection=[0, 1, 2],
        pagination=False,
        label="Terms of the series",
    )
    return (
        complex_form,
        data_file,
        data_text,
        fit_degree,
        fraction,
        func,
        functions,
        half_life,
        interval,
        legendre,
        n_sub,
        poly,
        quad,
        resample,
        rule,
        run_mc,
        shade,
        show_deriv,
        show_tangent,
        span,
        taylor_order,
        terms,
        ticker,
        x0,
    )


@app.cell
def _(func, functions, show_deriv, show_tangent, x0):
    # radio, number, checkbox, switch: the tangent line to a curve
    _f, _df, _ = functions[func.value]
    _x = np.linspace(-3, 3, 400)
    _fig, _ax = plt.subplots(figsize=(7, 3.2))
    _ax.plot(_x, _f(_x), label=r"$f(x)$")
    if show_deriv.value:
        _ax.plot(_x, _df(_x), label=r"$f'(x)$")
    if show_tangent.value:
        _slope, _y0 = _df(x0.value), _f(x0.value)
        _ax.plot(_x, _y0 + _slope * (_x - x0.value), "--", color=theme.PINK,
                 label=rf"tangent, slope ${_slope:.2f}$")
        _ax.plot([x0.value], [_y0], "o", color=theme.PINK)
    _ax.set_ylim(-3, 3)
    _ax.axhline(0, color=theme.FOREGROUND, alpha=0.5, lw=1)
    _ax.legend()
    mo.vstack([
        mo.md("### Tangent lines: `radio`, `number`, `checkbox`, `switch`\n"
              "Choose $f$ and a point $x_0$; the tangent has slope $f'(x_0)$."),
        mo.hstack([func, x0, show_tangent, show_deriv]),
        _fig,
    ])
    return


@app.cell
def _(func, functions, interval, n_sub, rule, shade):
    # range_slider, dropdown, slider, switch: numerical integration
    _f, _, _F = functions[func.value]
    _a, _b = interval.value
    _edges = np.linspace(_a, _b, n_sub.value + 1)
    _dx = _edges[1] - _edges[0]
    _mid = (_edges[:-1] + _edges[1:]) / 2
    _approx = {
        "left": _dx * _f(_edges[:-1]).sum(),
        "midpoint": _dx * _f(_mid).sum(),
        "trapezoid": _dx * (_f(_edges).sum() - (_f(_edges[0]) + _f(_edges[-1])) / 2),
    }[rule.value]
    _exact = _F(_b) - _F(_a)
    _x = np.linspace(_a, _b, 300)
    _fig, _ax = plt.subplots(figsize=(7, 3.2))
    _ax.plot(_x, _f(_x))
    if shade.value and rule.value == "trapezoid":
        _ax.fill_between(_edges, _f(_edges), color=theme.BLUE, alpha=0.3)
    elif shade.value:
        _heights = _f(_edges[:-1]) if rule.value == "left" else _f(_mid)
        _ax.bar(_edges[:-1], _heights, width=_dx, align="edge", color=theme.BLUE, alpha=0.3)
    _ax.axhline(0, color=theme.FOREGROUND, alpha=0.5, lw=1)
    mo.vstack([
        mo.md(r"### Riemann sums: `range_slider`, `dropdown`, `slider`, `switch`" "\n"
              r"Approximate $\int_a^b f(x)\,dx$ on the interval chosen with the "
              "range slider (the function comes from the radio above)."),
        mo.hstack([interval, rule, n_sub, shade]),
        _fig,
        mo.md(f"Approximation **{_approx:.5f}**, exact **{_exact:.5f}**, "
              f"error {abs(_approx - _exact):.2e}"),
    ])
    return


@app.cell
def _(legendre, taylor_order):
    # slider, multiselect: approximating functions
    _x = np.linspace(-2 * np.pi, 2 * np.pi, 500)
    _taylor = sum(
        (-1) ** k * _x ** (2 * k + 1) / factorial(2 * k + 1)
        for k in range((taylor_order.value + 1) // 2)
    )
    _fig, (_ax1, _ax2) = plt.subplots(1, 2, figsize=(9, 3.2))
    _ax1.plot(_x, np.sin(_x), label=r"$\sin x$")
    _ax1.plot(_x, _taylor, label=rf"Taylor, degree ${taylor_order.value}$")
    _ax1.set_ylim(-2, 2)
    _ax1.legend()
    _t = np.linspace(-1, 1, 300)
    for _n in legendre.value:
        _ax2.plot(_t, np.polynomial.legendre.Legendre.basis(_n)(_t), label=rf"$P_{_n}$")
    if legendre.value:
        _ax2.legend()
    mo.vstack([
        mo.md("### Approximation: `slider`, `multiselect`\n"
              "Left: the Taylor polynomial of $\\sin x$ at 0. "
              "Right: Legendre polynomials on $[-1, 1]$."),
        mo.hstack([taylor_order, legendre]),
        _fig,
    ])
    return


@app.cell
def _(poly, quad):
    # dictionary, array: named and indexed parameters
    _a, _b, _c = quad.value["a"], quad.value["b"], quad.value["c"]
    _roots = np.roots([_a, _b, _c]) if (_a, _b, _c) != (0, 0, 0) else np.array([])
    _x = np.linspace(-4, 4, 400)
    _fig, _ax = plt.subplots(figsize=(7, 3.2))
    _ax.plot(_x, _a * _x**2 + _b * _x + _c)
    _real = _roots[np.isreal(_roots)].real
    _ax.plot(_real, np.zeros_like(_real), "o", color=theme.PINK)
    _ax.set_ylim(-8, 8)
    _ax.axhline(0, color=theme.FOREGROUND, alpha=0.5, lw=1)
    _poly_roots = np.polynomial.Polynomial(poly.value).roots()
    mo.vstack([
        mo.md("### Parameters: `dictionary`, `array`\n"
              "A `dictionary` holds named parameters (the quadratic); an `array` holds an "
              "indexed list (polynomial coefficients)."),
        mo.hstack([quad, poly]),
        _fig,
        mo.md(f"Discriminant $b^2 - 4ac = {_b**2 - 4 * _a * _c:.2f}$. "
              f"Quadratic roots: {np.round(_roots, 3)}. "
              f"Roots of the polynomial: {np.round(_poly_roots, 3)}."),
    ])
    return


@app.cell
def _(complex_form):
    # batch + form: inputs applied only when you press the submit button
    _v = complex_form.value or {"re": 1.0, "im": 1.0}
    _z = complex(_v["re"], _v["im"])
    _fig, _ax = plt.subplots(figsize=(4.5, 4.5))
    for _w, _label, _colour in [(_z, r"$z$", theme.YELLOW), (_z**2, r"$z^2$", theme.GREEN)]:
        _ax.annotate("", xy=(_w.real, _w.imag), xytext=(0, 0),
                     arrowprops={"arrowstyle": "->", "color": _colour, "lw": 2})
        _ax.text(_w.real, _w.imag, f" {_label}", color=_colour)
    _lim = max(2.0, 1.1 * abs(_z**2))
    _ax.set_xlim(-_lim, _lim)
    _ax.set_ylim(-_lim, _lim)
    _ax.set_aspect("equal")
    _ax.set_xlabel(r"$\mathrm{Re}$")
    _ax.set_ylabel(r"$\mathrm{Im}$")
    mo.vstack([
        mo.md("### Forms: `batch` + `form`\n"
              "Edit both parts of $z$, then press the button: squaring doubles the angle."),
        complex_form,
        _fig,
        mo.md(f"$|z| = {abs(_z):.3f}$, $\\arg z = {np.angle(_z):.3f}$ rad, "
              f"$z^2 = {_z**2:.3f}$"),
    ])
    return


@app.cell
def _(data_file, data_text, fit_degree):
    # text_area, file, number: least-squares fitting
    from io import StringIO as _StringIO

    _text = data_file.contents().decode() if data_file.value else data_text.value
    try:
        _data = np.loadtxt(_StringIO(_text), delimiter=",", ndmin=2)
        _x, _y = _data[:, 0], _data[:, 1]
        _coef = np.polyfit(_x, _y, int(fit_degree.value))
        _pred = np.polyval(_coef, _x)
        _r2 = 1 - ((_y - _pred) ** 2).sum() / ((_y - _y.mean()) ** 2).sum()
        _fig, _ax = plt.subplots(figsize=(7, 3.2))
        _ax.plot(_x, _y, "o", label="data")
        _xs = np.linspace(_x.min(), _x.max(), 200)
        _ax.plot(_xs, np.polyval(_coef, _xs), label="fit")
        _ax.legend()
        _body = [_fig, mo.md(f"Coefficients (highest power first): {np.round(_coef, 3)}, "
                             f"$R^2 = {_r2:.4f}$")]
    except (ValueError, IndexError) as _err:
        _body = [mo.md(f"**Could not read the data:** {_err}")]
    mo.vstack([
        mo.md("### Fitting data: `text_area`, `file`, `number`\n"
              "Type points or upload a CSV (an uploaded file replaces the text)."),
        mo.hstack([data_text, mo.vstack([data_file, fit_degree])]),
        *_body,
    ])
    return


@app.cell
def _(fraction, resample, run_mc):
    # text, button, run_button: three ways to estimate pi
    from fractions import Fraction as _Fraction
    from math import pi as _pi

    try:
        _q = float(_Fraction(fraction.value))
        _frac_md = f"{fraction.value} = {_q:.8f}, error {abs(_q - _pi):.2e}"
    except (ValueError, ZeroDivisionError):
        _frac_md = "Enter a fraction such as `22/7`."
    _rng = np.random.default_rng(resample.value)  # seed = number of button clicks
    _pts = _rng.uniform(-1, 1, size=(2000, 2))
    _inside = (_pts**2).sum(axis=1) <= 1
    _fig, _ax = plt.subplots(figsize=(4.2, 4.2))
    _ax.plot(*_pts[_inside].T, ".", color=theme.GREEN, ms=3)
    _ax.plot(*_pts[~_inside].T, ".", color=theme.PINK, ms=3)
    _ax.set_aspect("equal")
    if run_mc.value:
        _big = np.random.default_rng(0).uniform(-1, 1, size=(1_000_000, 2))
        _big_md = f"$10^6$ points: $\\pi \\approx {4 * ((_big**2).sum(axis=1) <= 1).mean():.5f}$"
    else:
        _big_md = "Press the run button for $10^6$ points (it only computes when pressed)."
    mo.vstack([
        mo.md("### Estimating $\\pi$: `text`, `button`, `run_button`\n"
              "A fraction you type, a Monte Carlo sample redrawn by the button, and a "
              "heavy run gated by the run button."),
        mo.hstack([fraction, resample, run_mc]),
        mo.md(_frac_md),
        _fig,
        mo.md(f"2000 points: $\\pi \\approx {4 * _inside.mean():.4f}$. {_big_md}"),
    ])
    return


@app.cell
def _(half_life, span):
    # date_range, number: radioactive decay over a chosen time span
    _start, _end = span.value
    _days = (_end - _start).days
    _t_half = half_life.value or 1
    _remaining = 0.5 ** (_days / _t_half)
    _t = np.linspace(0, max(_days, 1) * 1.5, 300)
    _fig, _ax = plt.subplots(figsize=(7, 3))
    _ax.plot(_t, 0.5 ** (_t / _t_half), label=r"$N(t)/N_0 = 2^{-t/T}$")
    _ax.plot([_days], [_remaining], "o", color=theme.PINK)
    _ax.set_xlabel(r"$t$ (days)")
    _ax.legend()
    mo.vstack([
        mo.md("### Dates: `date_range`, `number`\n"
              "How much of a radioactive sample remains after the chosen time span?"),
        mo.hstack([span, half_life]),
        _fig,
        mo.md(f"{_days} days: **{100 * _remaining:.1f}%** remains."),
    ])
    return


@app.cell
def _(ticker):
    # refresh: a rotating phasor driven by a timer
    from re import search as _search

    # .value is "" before the first tick, then e.g. "1s (3)" with a tick counter
    _tick = _search(r"\((\d+)\)", ticker.value)
    _theta = 0.4 * (int(_tick.group(1)) if _tick else 0)
    _s = np.linspace(0, 2 * np.pi, 200)
    _fig, _ax = plt.subplots(figsize=(3.6, 3.6))
    _ax.plot(np.cos(_s), np.sin(_s), color=theme.GRID)
    _ax.plot([0, np.cos(_theta)], [0, np.sin(_theta)], "-o", color=theme.YELLOW)
    _ax.plot([np.cos(_theta)] * 2, [0, np.sin(_theta)], "--", color=theme.BLUE)
    _ax.set_aspect("equal")
    mo.vstack([
        mo.md("### Timers: `refresh`\n"
              r"Re-runs this cell on a timer: the phasor $e^{i\theta}$ turns."),
        ticker,
        _fig,
    ])
    return


@app.cell
def _(terms):
    # table: pick the terms of a series
    _chosen = [row["n"] for row in terms.value]
    _n = np.arange(1, 11)
    _fig, _ax = plt.subplots(figsize=(7, 3))
    _ax.bar(_n, 1 / _n**2, color=[theme.YELLOW if k in _chosen else theme.GRID for k in _n])
    _ax.set_xticks(_n)
    mo.vstack([
        mo.md("### Selecting rows: `table`\n"
              "Tick rows to include terms of $\\sum 1/n^2$."),
        terms,
        _fig,
        mo.md(f"Partial sum of the selected terms: **{sum(1 / k**2 for k in _chosen):.4f}** "
              f"(the full series is $\\pi^2/6 \\approx {np.pi**2 / 6:.4f}$)."),
    ])
    return


@app.cell
def _():
    # --- controls, part 2: components that need extra libraries --------------------

    expr_editor = mo.ui.code_editor(
        value="np.sin(18 * x) * np.exp(-x**2 / 4)", language="python", min_height=60, label="f(x) ="
    )
    file_pick = mo.ui.file_browser(
        initial_path=mo.notebook_dir().parent,
        filetypes=[".csv", ".txt"],
        multiple=False,
        ignore_empty_dirs=True,
        label="Pick a data file",
    )
    transform = mo.ui.matrix(
        [[1.0, 0.5], [0.0, 1.0]], min_value=-3, max_value=3, step=0.1, label="Matrix A"
    )
    points_editor = mo.ui.data_editor(
        {"x": [0.0, 1.0, 2.0, 3.0, 4.0], "y": [1.1, 2.9, 5.2, 7.1, 8.8]},
        label="Edit the points",
    )
    series_tabs = mo.ui.tabs(
        {
            "sin": mo.md(r"$$\sin x = \sum_{k\ge 0} \frac{(-1)^k x^{2k+1}}{(2k+1)!}$$"),
            "cos": mo.md(r"$$\cos x = \sum_{k\ge 0} \frac{(-1)^k x^{2k}}{(2k)!}$$"),
            "exp": mo.md(r"$$e^x = \sum_{k\ge 0} \frac{x^k}{k!}$$"),
        },
        label="Power series",
    )

    # interactive matplotlib: box-select points on a noisy line
    _rng = np.random.default_rng(1)
    scatter_x = _rng.uniform(0, 10, 60)
    scatter_y = 0.8 * scatter_x + _rng.normal(0, 1.5, 60)
    _, _ax = plt.subplots(figsize=(6, 3.2))
    _ax.scatter(scatter_x, scatter_y, color=theme.BLUE, s=14)
    mpl_select = mo.ui.matplotlib(_ax)

    # altair: brush a noisy parabola
    _x = _rng.uniform(-3, 3, 80)
    _chart = (
        alt.Chart(pd.DataFrame({"x": _x, "y": _x**2 + _rng.normal(0, 1, 80)}))
        .mark_point(color=theme.YELLOW)
        .encode(x="x", y="y")
        .properties(width=380, height=220)
    )
    alt_chart = mo.ui.altair_chart(theme.style_altair(_chart), chart_selection="interval")

    # plotly: select an arc of the unit circle
    circle_angles = np.linspace(0, 2 * np.pi, 120)
    _circle = go.Figure(
        go.Scatter(
            x=np.cos(circle_angles),
            y=np.sin(circle_angles),
            mode="markers",
            marker={"color": theme.GREEN, "size": 6},
        )
    )
    _circle.update_layout(width=380, height=380, dragmode="select", yaxis_scaleanchor="x")
    plotly_chart = mo.ui.plotly(theme.style_plotly(_circle))

    # dataframe and data_explorer: growth rates of common sequences
    _n = np.arange(1, 21)
    _growth = pd.DataFrame(
        {
            "n": _n,
            "n_squared": _n**2,
            "n_cubed": _n**3,
            "two_to_n": 2.0**_n,
            "n_factorial": [float(factorial(int(k))) for k in _n],
        }
    )
    explorer = mo.ui.data_explorer(_growth)
    frame = mo.ui.dataframe(_growth)

    # anywidget: a custom angle dial
    class _AngleWidget(anywidget.AnyWidget):
        _esm = """
        function render({ model, el }) {
          const input = document.createElement("input");
          input.type = "range";
          input.min = 0;
          input.max = 360;
          input.value = model.get("degrees");
          const label = document.createElement("span");
          label.textContent = " " + input.value + " degrees";
          input.addEventListener("input", () => {
            model.set("degrees", Number(input.value));
            model.save_changes();
            label.textContent = " " + input.value + " degrees";
          });
          el.append(input, label);
        }
        export default { render };
        """
        degrees = traitlets.Float(45.0).tag(sync=True)

    angle = mo.ui.anywidget(_AngleWidget())

    # panel: a Panel widget bound to a Panel function (reactive on the Panel side)
    pn.extension(theme="dark")
    _exponent = pn.widgets.FloatSlider(name="Exponent p", start=-2, end=4, step=0.25, value=2)
    _derivative = pn.bind(
        lambda p: pn.pane.Markdown(f"d/dx of x^{p:g} at x = 2 is **{p * 2 ** (p - 1):.4f}**"),
        _exponent,
    )
    panel_ui = mo.ui.panel(pn.Column(_exponent, _derivative))
    return (
        alt_chart,
        angle,
        circle_angles,
        explorer,
        expr_editor,
        file_pick,
        frame,
        mpl_select,
        panel_ui,
        plotly_chart,
        points_editor,
        scatter_x,
        scatter_y,
        series_tabs,
        transform,
    )


@app.cell
def _(expr_editor):
    # code_editor: plot a function you type (evaluated with eval, so only run code you trust)
    _x = np.linspace(-2 * np.pi, 2 * np.pi, 400)
    try:
        _y = np.broadcast_to(
            eval(expr_editor.value, {"__builtins__": {}}, {"np": np, "x": _x}), _x.shape
        )
        _fig, _ax = plt.subplots(figsize=(7, 3))
        _ax.plot(_x, _y)
        _ax.set_xlabel(r"$x$")
        _ax.set_ylabel(r"$f(x)$")
        _body = _fig
    except (
        SyntaxError, NameError, TypeError, ValueError, ArithmeticError,
        AttributeError, IndexError, KeyError,
    ) as _err:
        _body = mo.md(f"**Could not evaluate the expression:** `{_err}`")
    mo.vstack([
        mo.md("### Typing code: `code_editor`\n"
              "Write an expression in `x` (a NumPy array) and `np`, for example a wave packet in a Gaussian envelope."),
        expr_editor,
        _body,
    ])
    return


@app.cell
def _(file_pick):
    # file_browser: choose a data file from disk
    _chosen = file_pick.value
    if _chosen:
        try:
            _data = np.loadtxt(_chosen[0].path, delimiter=",", ndmin=2)
            _fig, _ax = plt.subplots(figsize=(7, 3))
            _ax.plot(_data[:, 0], _data[:, 1], "o-")
            _body = [_fig, mo.md(f"Mean of y: **{_data[:, 1].mean():.4f}**, "
                                 f"standard deviation: **{_data[:, 1].std(ddof=1):.4f}**")]
        except (ValueError, IndexError, OSError) as _err:
            _body = [mo.md(f"**Could not read `{_chosen[0].path.name}` as x, y columns:** {_err}")]
    else:
        _body = [mo.md("No file selected. Pick a CSV with two columns, x and y.")]
    mo.vstack([
        mo.md("### Browsing files: `file_browser`\n"
              "Choose a two-column data file under `Notebooks/` to plot it and summarise y."),
        file_pick,
        *_body,
    ])
    return


@app.cell
def _(transform):
    # matrix: a linear map of the plane
    _A = np.asarray(transform.value, dtype=float)
    _fig, _ax = plt.subplots(figsize=(4.5, 4.5))
    for _c in np.linspace(-2, 2, 9):
        for _line in (np.array([[_c, _c], [-2, 2]]), np.array([[-2, 2], [_c, _c]])):
            _t = _A @ _line
            _ax.plot(_t[0], _t[1], color=theme.GRID, lw=1)
    _square = np.array([[0, 1, 1, 0, 0], [0, 0, 1, 1, 0]], dtype=float)
    _ax.fill(*(_A @ _square), color=theme.YELLOW, alpha=0.5)
    for _col, _colour in ((0, theme.PINK), (1, theme.GREEN)):
        _ax.annotate("", xy=tuple(_A[:, _col]), xytext=(0, 0),
                     arrowprops={"arrowstyle": "->", "color": _colour, "lw": 2})
    _ax.set_xlim(-6, 6)
    _ax.set_ylim(-6, 6)
    _ax.set_aspect("equal")
    mo.vstack([
        mo.md("### Matrices: `matrix`\n"
              "Edit the entries of $A$ to see how it maps the unit square and the grid."),
        transform,
        _fig,
        mo.md(f"$\\det A = {np.linalg.det(_A):.3f}$, "
              f"eigenvalues of $A$: {np.round(np.linalg.eigvals(_A), 3)}"),
    ])
    return


@app.cell
def _(points_editor):
    # data_editor: edit the data and the regression line follows
    _d = points_editor.value
    _x = np.asarray(_d["x"], dtype=float)
    _y = np.asarray(_d["y"], dtype=float)
    _ok = np.isfinite(_x) & np.isfinite(_y)
    if _ok.sum() >= 2:
        _slope, _intercept = np.polyfit(_x[_ok], _y[_ok], 1)
        _fig, _ax = plt.subplots(figsize=(7, 3))
        _ax.plot(_x[_ok], _y[_ok], "o")
        _xs = np.linspace(_x[_ok].min(), _x[_ok].max(), 50)
        _ax.plot(_xs, _slope * _xs + _intercept)
        _body = [_fig, mo.md(f"$y \\approx {_slope:.3f}\\,x {_intercept:+.3f}$")]
    else:
        _body = [mo.md("Need at least two complete points.")]
    mo.vstack([
        mo.md("### Editing data: `data_editor`\n"
              "Change any cell in the table; the least-squares line updates."),
        points_editor,
        *_body,
    ])
    return


@app.cell
def _(series_tabs):
    # tabs: pick a power series and compare its partial sums with the function
    _exact, _term = {
        "sin": (np.sin, lambda x, k: (-1) ** k * x ** (2 * k + 1) / factorial(2 * k + 1)),
        "cos": (np.cos, lambda x, k: (-1) ** k * x ** (2 * k) / factorial(2 * k)),
        "exp": (np.exp, lambda x, k: x**k / factorial(k)),
    }[series_tabs.value]
    _x = np.linspace(-3, 3, 300)
    _fig, _ax = plt.subplots(figsize=(7, 3.2))
    _ax.plot(_x, _exact(_x), color=theme.FOREGROUND, lw=3, alpha=0.6, label="exact")
    _partial = 0
    for _k in range(4):
        _partial = _partial + _term(_x, _k)
        _ax.plot(_x, _partial, label=rf"${_k + 1}$ terms")
    _ax.set_ylim(-4, 4)
    _ax.legend()
    mo.vstack([
        mo.md("### Tabs: `tabs`\n"
              "The selected tab chooses which series is compared with its function."),
        series_tabs,
        _fig,
    ])
    return


@app.cell
def _(mpl_select, scatter_x, scatter_y):
    # matplotlib: drag a box on the plot to select points
    _mask = mpl_select.value.get_mask(scatter_x, scatter_y)
    if _mask.sum() >= 2:
        _slope, _intercept = np.polyfit(scatter_x[_mask], scatter_y[_mask], 1)
        _result = (f"{int(_mask.sum())} points selected: "
                   f"$y \\approx {_slope:.3f}\\,x {_intercept:+.3f}$")
    else:
        _result = "Drag a box over at least two points."
    mo.vstack([
        mo.md("### Selecting on a plot: `matplotlib`\n"
              "The figure itself is the control; fit a line to the points you box."),
        mpl_select,
        mo.md(_result),
    ])
    return


@app.cell
def _(alt_chart):
    # altair_chart: brush part of a scatter plot
    _sel = alt_chart.value if len(alt_chart.value) else alt_chart.dataframe
    if len(_sel) >= 3:
        _a, _b, _c = np.polyfit(_sel["x"], _sel["y"], 2)
        _result = (f"{len(_sel)} points: $y \\approx {_a:.2f}x^2 {_b:+.2f}x {_c:+.2f}$ "
                   "(all points until you brush)")
    else:
        _result = "Brush at least three points."
    mo.vstack([
        mo.md("### Selecting in a chart: `altair_chart`\n"
              "Drag a rectangle over the noisy parabola to fit a quadratic to those points."),
        alt_chart,
        mo.md(_result),
    ])
    return


@app.cell
def _(circle_angles, plotly_chart):
    # plotly: select an arc of the unit circle
    _idx = list(plotly_chart.indices)
    if _idx:
        _lo, _hi = circle_angles[min(_idx)], circle_angles[max(_idx)]
        _result = (f"Selected angles from ${_lo:.2f}$ to ${_hi:.2f}$ rad, "
                   f"so the arc length is about ${_hi - _lo:.2f}$.")
    else:
        _result = "Box-select some points on the circle."
    mo.vstack([
        mo.md("### Selecting in plotly: `plotly`\n"
              "Select part of the unit circle; arc length equals the angle swept."),
        plotly_chart,
        mo.md(_result),
    ])
    return


@app.cell
def _(explorer):
    # data_explorer: build charts of the growth-rate table interactively
    mo.vstack([
        mo.md("### Exploring data: `data_explorer`\n"
              "Pick columns for the axes (try `n` against `two_to_n`, then `n_factorial`) "
              "to compare growth rates."),
        explorer,
    ])
    return


@app.cell
def _(frame):
    # dataframe: filter and transform the table with the GUI, then plot the result
    _df = frame.value
    _cols = [c for c in _df.columns if c != "n" and _df[c].dtype.kind in "if"]
    if "n" in _df.columns and _cols and len(_df):
        _fig, _ax = plt.subplots(figsize=(7, 3.2))
        for _c in _cols:
            _ax.semilogy(_df["n"], _df[_c], label=_c.replace("_", " "))
        _ax.legend()
        _body = [_fig]
    else:
        _body = [mo.md("Keep the `n` column and at least one other numeric column.")]
    mo.vstack([
        mo.md("### Transforming data: `dataframe`\n"
              "Use the transform builder (filter rows, select columns) and the plot follows."),
        frame,
        *_body,
    ])
    return


@app.cell
def _(angle):
    # anywidget: a custom widget drives a unit-circle diagram
    _deg = angle.value["degrees"]
    _th = np.radians(_deg)
    _s = np.linspace(0, 2 * np.pi, 200)
    _fig, _ax = plt.subplots(figsize=(3.6, 3.6))
    _ax.plot(np.cos(_s), np.sin(_s), color=theme.GRID)
    _ax.plot([0, np.cos(_th)], [0, np.sin(_th)], "-o", color=theme.YELLOW)
    _ax.plot([np.cos(_th)] * 2, [0, np.sin(_th)], "--", color=theme.BLUE)
    _ax.set_aspect("equal")
    mo.vstack([
        mo.md("### Custom widgets: `anywidget`\n"
              "A small JavaScript widget sets the angle $\\theta$."),
        angle,
        _fig,
        mo.md(f"$\\cos\\theta = {np.cos(_th):.3f}$, $\\sin\\theta = {np.sin(_th):.3f}$"),
    ])
    return


@app.cell
def _(panel_ui):
    # panel: a Panel widget and a function bound to it, shown through marimo
    mo.vstack([
        mo.md("### Panel widgets: `panel`\n"
              "Panel's own reactivity updates the derivative $p\\,x^{p-1}$ at $x = 2$."),
        panel_ui,
    ])
    return


@app.cell
def _():
    # A slider with a play button (custom anywidget). It is displayed here, in a cell
    # that does not read its value, so it is not redrawn while it animates.
    wave_player = mo.ui.anywidget(
        # interval (ms) is the time between updates; a matplotlib redraw takes ~0.3 s
        PlaySlider(
            start=0, stop=4, step=0.01, speed=1.0, loop=True, label="t (s)", interval=250
        )
    )
    mo.vstack([
        mo.md("### Play slider: a travelling wave\n"
              "Press Play (or drag the slider) to move time forward. The wave "
              "$y(x, t) = \\sin(kx - \\omega t)$ travels to the right at speed "
              "$c = \\omega / k$."),
        wave_player,
    ])
    return (wave_player,)


@app.cell
def _(wave_player):
    # The wave at the time chosen by the play slider
    _t = wave_player.value["value"]
    _k = _omega = np.pi  # wavelength 2, so the speed is c = omega / k = 1
    _x = np.linspace(0, 6, 600)
    _crests = (np.pi / 2 + _omega * _t + 2 * np.pi * np.arange(-10, 10)) / _k
    _crests = _crests[(_crests >= 0) & (_crests <= 6)]
    _fig, _ax = plt.subplots(figsize=(7, 3.2))
    _ax.plot(_x, np.sin(_k * _x), "--", color=theme.GRID, label=r"$t = 0$")
    _ax.plot(_x, np.sin(_k * _x - _omega * _t), color=theme.YELLOW, label=r"$y(x, t)$")
    _ax.plot(_crests, np.ones_like(_crests), "o", color=theme.PINK, label="crests")
    _ax.set_ylim(-1.3, 1.3)
    _ax.set_xlabel(r"$x$")
    _ax.set_title(rf"$y = \sin(kx - \omega t)$ at $t = {_t:.2f}$ s")
    _ax.legend(loc="lower right")
    _view = mo.vstack([
        _fig,
        mo.md(f"Each crest has moved {_omega * _t / _k:.2f} to the right since $t = 0$."),
    ])
    plt.close(_fig)  # the figure is already rendered; free it so frames do not pile up
    _view
    return


@app.cell(hide_code=True)
def _():
    bib.bibliography(__file__)
    return


if __name__ == "__main__":
    app.run()
