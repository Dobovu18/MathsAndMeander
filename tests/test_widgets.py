from mathsandmeander.common.widgets import PlaySlider, SlideShow


def test_play_slider_defaults():
    slider = PlaySlider()
    assert slider.value == 0.0
    assert slider.playing is False
    assert slider.loop is True
    assert slider.interval == 100


def test_play_slider_configuration_and_synced_traits():
    slider = PlaySlider(start=0, stop=4, step=0.02, speed=2.0, loop=False, label="t")
    assert (slider.start, slider.stop, slider.step, slider.speed) == (0.0, 4.0, 0.02, 2.0)
    assert slider.loop is False
    synced = set(slider.trait_names(sync=True))
    assert {"value", "start", "stop", "step", "speed", "playing", "loop", "label"} <= synced


def test_play_slider_ships_front_end_code():
    assert "setInterval" in PlaySlider._esm
    assert "play-slider" in PlaySlider._css


def test_slide_show_defaults_and_synced_traits():
    show = SlideShow(sources=["a", "b"], kinds=["video", "image"], loops=[True, False])
    assert show.index == 0 and show.width == 720
    synced = set(show.trait_names(sync=True))
    assert {"sources", "kinds", "loops", "index", "width"} <= synced


def test_slide_show_ships_front_end_code_with_cycling_arrows():
    assert "ArrowLeft" in SlideShow._esm and "ArrowRight" in SlideShow._esm
    assert "% count()" in SlideShow._esm  # indices wrap around
    assert "slide-show-arrow" in SlideShow._css
