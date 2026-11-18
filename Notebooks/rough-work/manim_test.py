import marimo

__generated_with = "0.25.1"
app = marimo.App(css_file="../_theme/blackboard.css")

with app.setup:
    # Initialization code that runs before all other cells

    import manim as mn
    import marimo as mo
    import numpy as np
    from manim_slides.slide.manim import Slide

    from mathsandmeander.common import bib, manim_tools, theme

    theme.apply()#usetex=True)
    mo.latex(theme.macros())


@app.cell
def _():
    mo.md(r"""
    # Manim in marimo

    [manim](https://www.manim.community) renders maths animations to video. Each example
    below defines a `Scene` (the manim code is in the cell), renders it with
    `mathsandmeander.common.manim_tools`, and shows the result in the notebook.

    - Everything renders automatically when the notebook runs. The first render of a scene
      takes seconds to a minute (it also compiles the LaTeX); after that it is served from a
      cache in the gitignored `.manim_media/` folder, keyed on the scene's code, the quality
      and the theme. Delete that folder to force a re-render.
    - **Low** quality (480p, 15 fps) is the quickest. Changing the quality below re-renders
      every example at that quality, and high quality can take minutes.
    - Scenes use the blackboard background and palette from `theme.py`, and the custom
      LaTeX macros from `macros.tex` (the equation example ends with `\R`).
    """)
    return


@app.cell
def _():
    quality = mo.ui.dropdown(
        options=["low", "medium", "high"], value="low", label="Render quality"
    )
    quality
    return (quality,)


@app.cell
def _(quality):
    # Animating equations: TransformMatchingTex morphs one line of maths into the next
    class CompleteTheSquare(mn.Scene):
        def construct(self):
            title = mn.Tex("Completing the square").to_edge(mn.UP)
            title.set_color(manim_tools.YELLOW)
            lines = [
                mn.MathTex(r"x^2 + 6x + 5 = 0"),
                mn.MathTex(r"x^2 + 6x = -5"),
                mn.MathTex(r"x^2 + 6x + 9 = -5 + 9"),
                mn.MathTex(r"(x + 3)^2 = 4"),
                mn.MathTex(r"x + 3 = \pm 2"),
                mn.MathTex(r"x \in \{-1, -5\} \subset \R"),  # \R comes from macros.tex
            ]
            self.play(mn.Write(title))
            current = lines[0]
            self.play(mn.Write(current))
            self.wait(0.5)
            for next_line in lines[1:]:
                self.play(mn.TransformMatchingTex(current, next_line), run_time=1.2)
                current = next_line
                self.wait(0.4)
            self.play(mn.Circumscribe(current, color=manim_tools.PINK))
            self.wait(0.5)

    _video = manim_tools.render_video(CompleteTheSquare, quality.value)
    mo.vstack([
        mo.md(
            "### Animating equations\n"
            "`MathTex` objects are written, then `TransformMatchingTex` morphs each line "
            "into the next, moving the matching terms."
        ),
        mo.video(_video.read_bytes(), loop=True, muted=True, autoplay=True, width=720),
    ])
    return


@app.cell
def _(quality):
    # Animating graphs: a ValueTracker drives a curve, a label and a dot together
    class AnimatedGraph(mn.Scene):
        def construct(self):
            axes = mn.Axes(
                x_range=[0, 2 * np.pi, np.pi / 2],
                y_range=[-2, 2, 1],
                x_length=11,
                y_length=5,
                tips=False,
                axis_config={"color": manim_tools.FOREGROUND},
            )
            labels = axes.get_axis_labels(x_label="x", y_label="y")
            amplitude = mn.ValueTracker(1)
            curve = mn.always_redraw(
                lambda: axes.plot(
                    lambda x: amplitude.get_value() * np.sin(x), color=manim_tools.YELLOW
                )
            )
            caption = mn.always_redraw(
                lambda: mn.MathTex(rf"y = {amplitude.get_value():.1f}\sin x")
                .to_corner(mn.UR)
                .set_color(manim_tools.BLUE)
            )
            peak = mn.always_redraw(
                lambda: mn.Dot(
                    axes.c2p(np.pi / 2, amplitude.get_value()), color=manim_tools.PINK
                )
            )
            self.play(mn.Create(axes), mn.Write(labels))
            self.play(mn.Create(curve), mn.FadeIn(caption))
            self.add(peak)
            self.play(amplitude.animate.set_value(2), run_time=2)
            self.play(amplitude.animate.set_value(0.5), run_time=2)
            self.play(amplitude.animate.set_value(1), run_time=1)
            self.wait(0.5)

    _video = manim_tools.render_video(AnimatedGraph, quality.value)
    mo.vstack([
        mo.md(
            "### Animating graphs\n"
            "`always_redraw` rebuilds the curve, label and peak marker every frame while "
            "a `ValueTracker` changes the amplitude of $y = A\\sin x$."
        ),
        mo.video(_video.read_bytes(), loop=True, muted=True, autoplay=True, width=720),
    ])
    return


@app.cell
def _(quality):
    # Displaying geometry: render a single frame (a still image) of a static scene
    class PythagorasFigure(mn.Scene):
        def construct(self):
            c, p, q = np.array([0, 0, 0]), np.array([3, 0, 0]), np.array([0, 4, 0])
            outward = np.array([4, 3, 0])  # perpendicular to the hypotenuse, length 5
            triangle = mn.Polygon(c, p, q, color=manim_tools.FOREGROUND, stroke_width=3)
            square_a = mn.Polygon(
                c, p, p + [0, -3, 0], c + [0, -3, 0],
                color=manim_tools.BLUE, fill_opacity=0.4,
            )
            square_b = mn.Polygon(
                c, q, q + [-4, 0, 0], c + [-4, 0, 0],
                color=manim_tools.PINK, fill_opacity=0.4,
            )
            square_c = mn.Polygon(
                p, q, q + outward, p + outward,
                color=manim_tools.YELLOW, fill_opacity=0.4,
            )
            right_angle = mn.RightAngle(
                mn.Line(c, p), mn.Line(c, q), length=0.4, quadrant=(1, 1),
                color=manim_tools.FOREGROUND,
            )
            areas = mn.VGroup(
                mn.MathTex(r"a^2 = 9").move_to(square_a.get_center()),
                mn.MathTex(r"b^2 = 16").move_to(square_b.get_center()),
                mn.MathTex(r"c^2 = 25").move_to(square_c.get_center()),
            )
            figure = mn.VGroup(square_a, square_b, square_c, triangle, right_angle, areas)
            figure.scale_to_fit_height(6.2).move_to(mn.DOWN * 0.3)
            equation = mn.MathTex(r"a^2 + b^2 = c^2", r"\;\Rightarrow\; 9 + 16 = 25")
            self.add(figure, equation.to_corner(mn.UL))

    _png = manim_tools.render_image(PythagorasFigure, quality.value)
    mo.vstack([
        mo.md(
            "### Displaying geometry\n"
            "Polygons, a right-angle marker and `MathTex` labels are added to a scene "
            "without animating; `render_image` saves just the last frame as a png."
        ),
        mo.image(_png.read_bytes(), width=720),
    ])
    return


@app.cell
def _(quality):
    # manim-slides: a Slide is a Scene that pauses at every next_slide()
    class UnitCircleSlides(Slide):
        def construct(self):
            title = mn.Tex("Euler's formula").scale(1.4).set_color(manim_tools.YELLOW)
            formula = mn.MathTex(r"e^{i\theta} = \cos\theta + i\sin\theta")
            formula.next_to(title, mn.DOWN)
            self.play(mn.Write(title))
            self.play(mn.Write(formula))
            self.next_slide()  # slide 1 ends here

            self.play(mn.FadeOut(title), formula.animate.to_edge(mn.UP))
            plane = mn.ComplexPlane(x_range=[-2, 2], y_range=[-2, 2], x_length=4, y_length=4)
            plane.add_coordinates()
            radius = plane.c2p(1, 0)[0] - plane.c2p(0, 0)[0]
            circle = mn.Circle(radius=radius, color=manim_tools.BLUE).move_to(plane.c2p(0, 0))
            theta = mn.ValueTracker(0)

            def point():
                return plane.c2p(np.cos(theta.get_value()), np.sin(theta.get_value()))

            spoke = mn.always_redraw(
                lambda: mn.Line(plane.c2p(0, 0), point(), color=manim_tools.PINK)
            )
            marker = mn.always_redraw(lambda: mn.Dot(point(), color=manim_tools.YELLOW))
            self.play(mn.Create(plane), mn.Create(circle))
            self.add(spoke, marker)
            self.play(theta.animate.set_value(np.pi / 3), run_time=2)
            self.next_slide()  # slide 2 ends here

            self.play(theta.animate.set_value(np.pi), run_time=2)
            identity = mn.MathTex(r"e^{i\pi} + 1 = 0").to_edge(mn.DOWN, buff=1.0)
            identity.set_color(manim_tools.GREEN)
            self.play(mn.Write(identity))
            self.next_slide()  # slide 3 ends here

    slides = manim_tools.render_slides(UnitCircleSlides, quality.value)
    mo.md(
        "### Slides: `manim-slides`\n"
        "A `Slide` is a scene with `self.next_slide()` markers. manim-slides renders one "
        "clip per slide; use the arrows on the slide to step through them. Each slide is a "
        "plain video (no iframe, no CDN), like the examples above."
    )
    return (slides,)


@app.cell
def _(slides):
    # Arrows on the slide (and the left/right keys) cycle through the clips
    manim_tools.slide_show(slides)
    return


@app.cell(hide_code=True)
def _():
    bib.bibliography(__file__)
    return


if __name__ == "__main__":
    app.run()
