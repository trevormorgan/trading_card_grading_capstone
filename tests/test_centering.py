import io

import numpy as np
import pytest
from PIL import Image

from card_grading.centering import Centering, measure_centering

BORDER_COLOR = (245, 245, 238)


def make_card(left, right, top, bottom, size=(500, 700), picture=(180, 60, 50)):
    card = Image.new("RGB", size, BORDER_COLOR)
    card.paste(picture, (left, top, size[0] - right, size[1] - bottom))
    return card


def test_centred_card():
    centering = measure_centering(make_card(40, 40, 50, 50))
    assert centering == Centering(left=40, right=40, top=50, bottom=50)
    assert (centering.left_right, centering.top_bottom, centering.worst) == ("50/50", "50/50", 50.0)


def test_off_centre_card():
    centering = measure_centering(make_card(20, 60, 45, 55))
    assert (centering.left, centering.right, centering.top, centering.bottom) == (20, 60, 45, 55)
    assert (centering.left_right, centering.top_bottom) == ("25/75", "45/55")
    assert centering.worst == pytest.approx(75.0)


def test_measurement_survives_jpeg_compression_and_a_dark_edge():
    card = make_card(30, 50, 40, 40)
    card.paste((20, 20, 20), (0, 0, 2, 700))            # a sliver of scanner background along the left edge
    buffer = io.BytesIO()
    card.save(buffer, format="JPEG", quality=80)
    centering = measure_centering(Image.open(buffer))
    assert abs(centering.left - 30) <= 2 and abs(centering.right - 50) <= 2
    assert abs(centering.top - 40) <= 2 and abs(centering.bottom - 40) <= 2


def test_picture_with_patches_of_the_border_colour_is_still_measured():
    card = make_card(40, 40, 50, 50)
    card.paste(BORDER_COLOR, (200, 300, 300, 420))      # e.g. a white jersey in the photo
    assert measure_centering(card) == Centering(left=40, right=40, top=50, bottom=50)


def test_card_without_a_plain_border_cannot_be_measured():
    noise = np.random.default_rng(0).integers(0, 255, (700, 500, 3), dtype=np.uint8)
    assert measure_centering(Image.fromarray(noise)) is None
    assert measure_centering(Image.new("RGB", (500, 700), BORDER_COLOR)) is None   # no picture found
    assert measure_centering(Image.new("RGB", (20, 30), BORDER_COLOR)) is None     # too small
