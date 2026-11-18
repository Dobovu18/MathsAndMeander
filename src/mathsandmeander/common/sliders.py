"""Helpers for choosing slider settings."""

import math


def slider_step(min_val: float, max_val: float, target_steps: int = 20) -> float:
    """Return a readable slider increment for the range ``[min_val, max_val]``.

    The increment is ``1`` or ``5`` times a power of 10 (..., 0.5, 1, 5, 10, 50, ...). Of
    those, the one whose number of increments is closest (on a log scale) to
    ``target_steps`` is chosen, so the range always gets between about
    ``target_steps / 2.3`` and ``target_steps * 2.3`` increments.
    """
    span = max_val - min_val
    if not (math.isfinite(span) and span > 0):
        raise ValueError(f"need finite min_val < max_val, got {min_val}, {max_val}")
    if target_steps < 1:
        raise ValueError("target_steps must be at least 1")

    ideal = span / target_steps
    exponent = math.floor(math.log10(ideal))
    # Candidates around the ideal step; build them from strings to avoid 5 * 10**-2 error.
    candidates = [
        float(f"{mantissa}e{power}")
        for power in (exponent - 1, exponent, exponent + 1)
        for mantissa in (1, 5)
    ]
    return min(candidates, key=lambda step: abs(math.log(step / ideal)))
