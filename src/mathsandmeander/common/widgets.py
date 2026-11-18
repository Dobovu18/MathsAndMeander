"""Reusable marimo (anywidget) controls.

Wrap a widget with ``mo.ui.anywidget`` and read ``.value`` in another cell. Display the
wrapped widget in a cell that does *not* read ``.value``, otherwise the widget is
redrawn (and its timer restarted) on every change.
"""

import anywidget
import traitlets

_ESM = """
function render({ model, el }) {
  const root = document.createElement("div");
  root.className = "play-slider";
  const button = document.createElement("button");
  button.type = "button";
  const range = document.createElement("input");
  range.type = "range";
  const readout = document.createElement("span");
  readout.className = "play-slider-readout";

  const sync = () => {
    range.min = model.get("start");
    range.max = model.get("stop");
    range.step = model.get("step");
    range.value = model.get("value");
    button.textContent = model.get("playing") ? "Pause" : "Play";
    const label = model.get("label");
    readout.textContent = (label ? label + " = " : "") + Number(model.get("value")).toFixed(2);
  };

  range.addEventListener("input", () => {
    model.set("value", Number(range.value));
    model.save_changes();
  });

  button.addEventListener("click", () => {
    const atEnd = model.get("value") >= model.get("stop");
    if (!model.get("playing") && !model.get("loop") && atEnd) {
      model.set("value", model.get("start"));
    }
    model.set("playing", !model.get("playing"));
    model.save_changes();
  });

  // Advance by (speed * elapsed time) so the animation runs in real time even when
  // the Python side redraws slower than the timer ticks.
  const interval = model.get("interval");
  let last = null;
  const timer = setInterval(() => {
    if (!model.get("playing")) {
      last = null;
      return;
    }
    const now = performance.now();
    const dt = last === null ? interval : now - last;
    last = now;
    const lo = model.get("start");
    const hi = model.get("stop");
    let v = model.get("value") + (model.get("speed") * dt) / 1000;
    if (v > hi) {
      if (model.get("loop")) {
        v = lo + ((v - lo) % (hi - lo || 1));
      } else {
        v = hi;
        model.set("playing", false);
      }
    }
    model.set("value", v);
    model.save_changes();
  }, interval);

  for (const name of ["value", "start", "stop", "step", "playing", "label"]) {
    model.on("change:" + name, sync);
  }
  sync();
  root.append(button, range, readout);
  el.append(root);
  return () => clearInterval(timer);
}
export default { render };
"""

_CSS = """
.play-slider {
  display: flex;
  align-items: center;
  gap: 0.75rem;
  color: var(--foreground, inherit);
}
.play-slider button {
  min-width: 4.5rem;
  padding: 0.25rem 0.75rem;
  color: var(--foreground, inherit);
  background: var(--muted, transparent);
  border: 1px solid var(--border, currentColor);
  border-radius: var(--radius, 6px);
  cursor: pointer;
}
.play-slider input[type="range"] {
  flex: 1;
  accent-color: var(--primary, auto);
}
.play-slider-readout {
  min-width: 7rem;
  font-variant-numeric: tabular-nums;
}
"""


class PlaySlider(anywidget.AnyWidget):
    """A slider with a play/pause button that advances ``value`` on a timer.

    ``speed`` is in value units per second. With ``loop=True`` the value wraps from
    ``stop`` back to ``start``; otherwise playback stops at ``stop``. ``interval`` (ms)
    is how often the timer ticks, fixed when the widget is first shown.
    """

    _esm = _ESM
    _css = _CSS

    value = traitlets.Float(0.0).tag(sync=True)
    start = traitlets.Float(0.0).tag(sync=True)
    stop = traitlets.Float(1.0).tag(sync=True)
    step = traitlets.Float(0.01).tag(sync=True)
    speed = traitlets.Float(1.0).tag(sync=True)
    playing = traitlets.Bool(False).tag(sync=True)
    loop = traitlets.Bool(True).tag(sync=True)
    label = traitlets.Unicode("").tag(sync=True)
    interval = traitlets.Int(100).tag(sync=True)


_SLIDE_ESM = """
function render({ model, el }) {
  const root = document.createElement("div");
  root.className = "slide-show";
  root.tabIndex = 0;
  const label = document.createElement("div");
  label.className = "slide-show-label";
  const stage = document.createElement("div");
  stage.className = "slide-show-stage";
  const previous = document.createElement("button");
  previous.type = "button";
  previous.className = "slide-show-arrow slide-show-previous";
  previous.textContent = "<";
  previous.title = "Previous slide";
  const following = document.createElement("button");
  following.type = "button";
  following.className = "slide-show-arrow slide-show-next";
  following.textContent = ">";
  following.title = "Next slide";

  const count = () => model.get("sources").length;
  const wrap = (i) => (count() ? ((i % count()) + count()) % count() : 0);

  const show = () => {
    const i = wrap(model.get("index"));
    stage.replaceChildren();
    if (count()) {
      const src = model.get("sources")[i];
      let media;
      if (model.get("kinds")[i] === "video") {
        media = document.createElement("video");
        media.muted = true;
        media.autoplay = true;
        media.loop = Boolean(model.get("loops")[i]);
        media.playsInline = true;
        media.src = src;
        media.play().catch(() => {});
      } else {
        media = document.createElement("img");
        media.src = src;
      }
      media.className = "slide-show-media";
      stage.append(media);
    }
    stage.append(previous, following);
    label.textContent = "Slide " + (i + 1) + " of " + count();
    root.style.maxWidth = model.get("width") + "px";
  };

  const go = (delta) => {
    model.set("index", wrap(wrap(model.get("index")) + delta));
    model.save_changes();
  };
  previous.addEventListener("click", () => go(-1));
  following.addEventListener("click", () => go(1));
  root.addEventListener("keydown", (event) => {
    if (event.key === "ArrowLeft") { go(-1); event.preventDefault(); }
    if (event.key === "ArrowRight") { go(1); event.preventDefault(); }
  });

  for (const name of ["sources", "kinds", "loops", "index", "width"]) {
    model.on("change:" + name, show);
  }
  show();
  root.append(label, stage);
  el.append(root);
}
export default { render };
"""

_SLIDE_CSS = """
.slide-show {
  color: var(--foreground, inherit);
  outline: none;
}
.slide-show-label {
  margin-bottom: 0.25rem;
  font-weight: 600;
}
.slide-show-stage {
  position: relative;
  line-height: 0;
}
.slide-show-media {
  display: block;
  width: 100%;
}
.slide-show-arrow {
  position: absolute;
  top: 50%;
  transform: translateY(-50%);
  padding: 0.5rem 0.75rem;
  line-height: 1;
  font-size: 1.25rem;
  color: var(--foreground, inherit);
  background: var(--muted, transparent);
  border: 1px solid var(--border, currentColor);
  border-radius: var(--radius, 6px);
  opacity: 0.35;
  cursor: pointer;
}
.slide-show-arrow:hover,
.slide-show-arrow:focus-visible {
  opacity: 0.95;
}
.slide-show-previous { left: 0.5rem; }
.slide-show-next { right: 0.5rem; }
"""


class SlideShow(anywidget.AnyWidget):
    """Slides shown one at a time with arrows on the slide itself.

    ``sources`` are URLs or ``data:`` URIs, ``kinds`` says whether each is a ``"video"``
    or an ``"image"`` and ``loops`` whether its video loops. The arrow buttons (and the
    left/right arrow keys while the widget has focus) cycle through the slides, from the
    last back to the first. Navigation runs in the browser, so it needs no kernel round
    trip, no iframe and no CDN. ``index`` is the slide being shown (0-based).
    """

    _esm = _SLIDE_ESM
    _css = _SLIDE_CSS

    sources = traitlets.List(traitlets.Unicode()).tag(sync=True)
    kinds = traitlets.List(traitlets.Unicode()).tag(sync=True)
    loops = traitlets.List(traitlets.Bool()).tag(sync=True)
    index = traitlets.Int(0).tag(sync=True)
    width = traitlets.Int(720).tag(sync=True)
