import manim as mn
import pytest
from manim_slides.slide.manim import Slide

from mathsandmeander.common import manim_tools, theme

RENDERS = {"scene": 0, "slides": 0}


class CircleScene(mn.Scene):
    def construct(self):
        RENDERS["scene"] += 1
        self.play(mn.Create(mn.Circle(color=manim_tools.BLUE)), run_time=0.3)


class SlowerCircleScene(mn.Scene):
    def construct(self):
        RENDERS["scene"] += 1
        self.play(mn.Create(mn.Circle(color=manim_tools.BLUE)), run_time=0.4)


class CircleSlides(Slide):
    def construct(self):
        RENDERS["slides"] += 1
        self.play(mn.Create(mn.Circle(color=manim_tools.PINK)), run_time=0.3)
        self.next_slide()
        self.play(mn.Create(mn.Square(color=manim_tools.GREEN)), run_time=0.3)
        self.next_slide()


@pytest.fixture
def media_dir(tmp_path, monkeypatch):
    monkeypatch.setattr(manim_tools, "MEDIA_DIR", tmp_path)
    RENDERS.update(scene=0, slides=0)
    return tmp_path


def test_palette_matches_theme():
    assert manim_tools.YELLOW == mn.ManimColor(theme.YELLOW)
    assert manim_tools.BLUE == mn.ManimColor(theme.BLUE)
    assert manim_tools.PINK == mn.ManimColor(theme.PINK)
    assert manim_tools.GREEN == mn.ManimColor(theme.GREEN)
    assert manim_tools.FOREGROUND == mn.ManimColor(theme.FOREGROUND)


def test_tex_template_knows_the_macros():
    assert r"\newcommand{\R}{\mathbb{R}}" in manim_tools.tex_template().preamble


def test_slides_compat_shim_maps_removed_manim_options():
    manim_tools._ensure_slides_compat()
    manim_tools._ensure_slides_compat()  # idempotent
    assert mn.config["write_to_movie"] is True
    assert mn.config["flush_cache"] is False
    original = mn.config["max_files_cached"]
    mn.config["max_files_cached"] = float("inf")
    assert mn.config["max_files_cached"] == -1
    mn.config["max_files_cached"] = original


def test_fingerprint_tracks_scene_code_and_quality():
    same = manim_tools._fingerprint(CircleScene, "low")
    assert same == manim_tools._fingerprint(CircleScene, "low")
    assert same != manim_tools._fingerprint(CircleScene, "medium")
    assert same != manim_tools._fingerprint(SlowerCircleScene, "low")


def test_video_and_image_are_rendered_once_then_cached(media_dir):
    video = manim_tools.render_video(CircleScene, "low")
    assert video.suffix == ".mp4" and video.stat().st_size > 0
    image = manim_tools.render_image(CircleScene, "low")
    assert image.suffix == ".png" and image.stat().st_size > 0
    assert RENDERS["scene"] == 2  # one render for each of the two outputs

    assert manim_tools.render_video(CircleScene, "low") == video
    assert manim_tools.render_image(CircleScene, "low") == image
    assert RENDERS["scene"] == 2  # both came from the cache


def test_slides_are_rendered_once_then_cached(media_dir):
    slides = manim_tools.render_slides(CircleSlides, "low")
    assert len(slides.clips) == len(slides.loops) == 2
    assert all(clip.suffix == ".mp4" and clip.exists() for clip in slides.clips)
    assert "reveal" in slides.html
    assert "controls: true," in slides.html  # on-screen navigation arrows
    assert "loop: true," in slides.html  # last slide wraps to the first
    assert slides.html.count("data:video/mp4;base64") == 2  # one clip per slide
    assert RENDERS["slides"] == 1

    again = manim_tools.render_slides(CircleSlides, "low")
    assert again == slides
    assert manim_tools.render_slides_html(CircleSlides, "low") == slides.html
    assert RENDERS["slides"] == 1  # served from the cache


def test_slide_show_embeds_every_slide(media_dir):
    slides = manim_tools.render_slides(CircleSlides, "low")
    show = manim_tools.slide_show(slides)
    assert show.kinds == ["video", "video"]
    assert show.loops == slides.loops
    assert all(source.startswith("data:video/mp4;base64,") for source in show.sources)
    assert show.index == 0 and show.width == 720
    assert manim_tools.slide_show(slides, width=500).width == 500
