"""Render manim scenes so marimo notebooks can display them.

Rendering uses manim's Python API inside ``manim.tempconfig`` so the global manim
configuration is left untouched. Scenes get the blackboard background and a TeX
template that knows the macros in ``Notebooks/_theme/macros.tex``. Output goes to the
gitignored ``.manim_media/`` folder at the repository root.

Renders are cached by a fingerprint of the scene's code, the quality and the theme, so
a notebook can render its scenes on every load and only pay for a scene the first time
(or after its code changes). Delete ``.manim_media/`` to force everything to re-render.
"""

import base64
import hashlib
import inspect
import logging
import shutil
from dataclasses import dataclass
from pathlib import Path

import manim as mn

from mathsandmeander.common import theme

MEDIA_DIR = Path(__file__).resolve().parents[3] / ".manim_media"
QUALITIES = {"low": "low_quality", "medium": "medium_quality", "high": "high_quality"}

# The blackboard palette as manim colours, so scenes never hard-code colours.
YELLOW, BLUE, PINK, GREEN, FOREGROUND = (
    mn.ManimColor(colour)
    for colour in (theme.YELLOW, theme.BLUE, theme.PINK, theme.GREEN, theme.FOREGROUND)
)


def tex_template() -> mn.TexTemplate:
    """A TeX template that defines the custom macros (``\\R``, ``\\C``, ...)."""
    template = mn.TexTemplate()
    template.add_to_preamble("\\usepackage{amssymb}\n" + theme.latex_preamble())
    return template


def _settings(quality: str, **extra) -> dict:
    return {
        "quality": QUALITIES[quality],
        "media_dir": str(MEDIA_DIR),
        "background_color": theme.BACKGROUND,
        "tex_template": tex_template(),
        "preview": False,
        "progress_bar": "none",
        "verbosity": "WARNING",
        **extra,
    }


def _hash_code(code, digest) -> None:
    """Feed a code object (bytecode, names, constants, nested code) into ``digest``."""
    digest.update(code.co_code)
    digest.update(repr(code.co_names).encode())
    for const in code.co_consts:
        if hasattr(const, "co_code"):
            _hash_code(const, digest)
        else:
            digest.update(repr(const).encode())


def _fingerprint(scene_cls: type, quality: str) -> str:
    """A short hash that changes when the scene's code, the quality or the theme does.

    Values a scene captures from outside (closures) are not included.
    """
    digest = hashlib.sha256()
    theme_state = (theme.BACKGROUND, theme.FOREGROUND, theme.CHALK, theme.latex_preamble())
    digest.update(repr((mn.__version__, quality, theme_state)).encode())
    for name, member in sorted(vars(scene_cls).items()):
        if inspect.isfunction(member):
            digest.update(name.encode())
            _hash_code(member.__code__, digest)
    return digest.hexdigest()[:16]


def _cached_path(scene_cls: type, quality: str, suffix: str) -> Path:
    folder = MEDIA_DIR / "cache"
    folder.mkdir(parents=True, exist_ok=True)
    return folder / f"{scene_cls.__name__}-{quality}-{_fingerprint(scene_cls, quality)}{suffix}"


def render_video(scene_cls: type[mn.Scene], quality: str = "low") -> Path:
    """Render a scene to an mp4 file (cached) and return its path."""
    target = _cached_path(scene_cls, quality, ".mp4")
    if not target.exists():
        with mn.tempconfig(_settings(quality)):
            scene = scene_cls()
            scene.render()
            shutil.copy(scene.renderer.file_writer.movie_file_path, target)
    return target


def render_image(scene_cls: type[mn.Scene], quality: str = "low") -> Path:
    """Render only the last frame of a scene to a png file (cached) and return its path."""
    target = _cached_path(scene_cls, quality, ".png")
    if not target.exists():
        with mn.tempconfig(_settings(quality, save_last_frame=True)):
            scene = scene_cls()
            scene.render()
            shutil.copy(scene.renderer.file_writer.image_file_path, target)
    return target


def _ensure_slides_compat() -> None:
    """Let manim-slides 5.7 run on manim 0.22.

    manim-slides sets config options that manim 0.22 removed or changed:
    ``write_to_movie`` (movies are now written unless ``dry_run`` is set),
    ``flush_cache`` (removed) and ``max_files_cached = float("inf")`` (now ``-1``).
    """
    config_cls = type(mn.config)
    missing = {
        "write_to_movie": property(
            lambda self: not self.dry_run,
            lambda self, value: setattr(self, "dry_run", not value),
        ),
        "flush_cache": property(lambda self: False, lambda self, value: None),
    }
    for name, prop in missing.items():
        if not hasattr(config_cls, name):
            setattr(config_cls, name, prop)

    name = "max_files_cached"
    cached = getattr(config_cls, name)

    def set_max_files_cached(self, value):
        cached.fset(self, -1 if value == float("inf") else value)

    if cached.fset.__name__ != set_max_files_cached.__name__:  # not patched yet
        setattr(config_cls, name, property(cached.fget, set_max_files_cached))


@dataclass(frozen=True)
class RenderedSlides:
    """A rendered ``manim_slides.Slide``: one clip per slide plus a RevealJS page."""

    clips: list[Path]  # one file per slide, in order (mp4, or png for a still slide)
    loops: list[bool]  # whether each slide's clip loops
    html: str  # self-contained RevealJS page (videos embedded, RevealJS from a CDN)


def render_slides(slide_cls: type, quality: str = "low") -> RenderedSlides:
    """Render a ``manim_slides.Slide`` subclass (cached).

    The RevealJS page has on-screen navigation arrows and loops from the last slide
    back to the first.
    """
    from manim_slides.config import PresentationConfig
    from manim_slides.convert import JsBool, RevealJS

    _ensure_slides_compat()
    logging.getLogger("manim-slides").setLevel(logging.WARNING)
    folder = MEDIA_DIR / "slides"
    name = slide_cls.__name__
    config_file, page = folder / f"{name}.json", folder / f"{name}.html"
    stamp = folder / f"{name}.fingerprint"
    # Part of the cache key, so changing the page settings re-converts cached slides.
    page_settings = {"controls": JsBool.true, "loop": JsBool.true}
    fingerprint = _fingerprint(slide_cls, quality) + repr(sorted(page_settings.items()))

    up_to_date = (
        stamp.exists()
        and stamp.read_text(encoding="utf-8") == fingerprint
        and config_file.exists()
        and page.exists()
    )
    if not up_to_date:
        with mn.tempconfig(_settings(quality)):
            scene = slide_cls(output_folder=folder)
            scene.render()
        presentation = PresentationConfig.from_file(config_file)
        RevealJS(
            presentation_configs=[presentation], one_file=True, **page_settings
        ).convert_to(page)
        stamp.write_text(fingerprint, encoding="utf-8")

    presentation = PresentationConfig.from_file(config_file)
    return RenderedSlides(
        clips=[Path(slide.file) for slide in presentation.slides],
        loops=[bool(slide.loop) for slide in presentation.slides],
        html=page.read_text(encoding="utf-8"),
    )


def render_slides_html(slide_cls: type, quality: str = "low") -> str:
    """Render a slide scene and return just its RevealJS page, for ``mo.iframe``."""
    return render_slides(slide_cls, quality).html


def slide_show(slides: RenderedSlides, width: int = 720):
    """Show rendered slides one at a time, with arrows on the slide itself.

    Returns a :class:`~mathsandmeander.common.widgets.SlideShow` widget; just display it
    (``manim_tools.slide_show(slides)`` as the last expression of a cell). The arrows
    cycle through the slides and run in the browser, so it behaves the same in the
    browser app and in the VS Code editor (no iframe, no CDN, no kernel round trip).
    """
    from mathsandmeander.common.widgets import SlideShow

    sources, kinds = [], []
    for clip in slides.clips:
        kind, mime = ("video", "video/mp4") if clip.suffix == ".mp4" else ("image", "image/png")
        sources.append(f"data:{mime};base64,{base64.b64encode(clip.read_bytes()).decode()}")
        kinds.append(kind)
    return SlideShow(sources=sources, kinds=kinds, loops=slides.loops, width=width)
