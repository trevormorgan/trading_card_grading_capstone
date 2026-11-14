"""Measure how well the picture of a card is centred between its borders.

The image has to show one card, cropped to its edges (as ``card-grading-split-scan`` saves
them), and the card needs a plain border of one colour around its picture.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from PIL import Image

# The outermost part of the image is ignored: it holds the cut edge and scanner shadow
EDGE_SKIP_FRACTION = 0.01
# The border colour is taken from a ring this wide, just inside the ignored edge
BORDER_SAMPLE_FRACTION = 0.02
# A border is not looked for beyond this share of the card's width or height
MAX_BORDER_FRACTION = 0.25
# How different from the border colour (0-255) a pixel must be to count as part of the picture
COLOR_TOLERANCE = 28
# The picture starts where this share of a line of pixels differs from the border colour ...
PICTURE_FRACTION = 0.5
# ... for this many lines in a row
MIN_PICTURE_LINES = 3
# If more than this share of the sampled ring differs from its own colour, the card has no plain border
MAX_BORDER_NOISE = 0.25


@dataclass(frozen=True)
class Centering:
    """The widths, in pixels, of the four borders of a card."""

    left: int
    right: int
    top: int
    bottom: int

    @staticmethod
    def _shares(first: int, second: int) -> tuple[float, float]:
        total = first + second
        return (50.0, 50.0) if total == 0 else (100 * first / total, 100 * second / total)

    @property
    def left_right(self) -> str:
        """The left and right borders as shares of their sum, for example ``'45/55'``."""
        return "{:.0f}/{:.0f}".format(*self._shares(self.left, self.right))

    @property
    def top_bottom(self) -> str:
        return "{:.0f}/{:.0f}".format(*self._shares(self.top, self.bottom))

    @property
    def worst(self) -> float:
        """The larger share of the more off-centre direction: 50 is perfectly centred, 100 has no border on one side."""
        return max(*self._shares(self.left, self.right), *self._shares(self.top, self.bottom))


def _border_width(profile: np.ndarray, skip: int) -> int | None:
    """Where the picture starts in a profile of 'share of pixels that differ from the border colour'."""
    run = 0
    for position in range(skip, int(len(profile) * MAX_BORDER_FRACTION)):
        run = run + 1 if profile[position] >= PICTURE_FRACTION else 0
        if run == MIN_PICTURE_LINES:
            return position - MIN_PICTURE_LINES + 1
    return None


def measure_centering(image: Image.Image) -> Centering | None:
    """Measure the borders of a card. Returns ``None`` if the card has no plain border that can be measured."""
    pixels = np.asarray(image.convert("RGB")).astype(np.int16)
    height, width = pixels.shape[:2]
    if min(height, width) < 50:
        return None
    skip_x, skip_y = max(1, round(width * EDGE_SKIP_FRACTION)), max(1, round(height * EDGE_SKIP_FRACTION))
    ring_x = skip_x + max(1, round(width * BORDER_SAMPLE_FRACTION))
    ring_y = skip_y + max(1, round(height * BORDER_SAMPLE_FRACTION))

    inner = pixels[skip_y:height - skip_y, skip_x:width - skip_x]
    ring = np.concatenate([inner[:ring_y - skip_y].reshape(-1, 3), inner[-(ring_y - skip_y):].reshape(-1, 3),
                           inner[:, :ring_x - skip_x].reshape(-1, 3), inner[:, -(ring_x - skip_x):].reshape(-1, 3)])
    border_color = np.median(ring, axis=0)
    if (np.abs(ring - border_color).max(axis=1) > COLOR_TOLERANCE).mean() > MAX_BORDER_NOISE:
        return None

    differs = np.abs(pixels - border_color).max(axis=2) > COLOR_TOLERANCE
    # the middle of each side is used, so that rounded or worn corners do not count
    columns = differs[int(height * 0.2):int(height * 0.8)].mean(axis=0)
    rows = differs[:, int(width * 0.2):int(width * 0.8)].mean(axis=1)
    borders = (_border_width(columns, skip_x), _border_width(columns[::-1], skip_x),
               _border_width(rows, skip_y), _border_width(rows[::-1], skip_y))
    if any(border is None for border in borders):
        return None
    return Centering(*borders)
