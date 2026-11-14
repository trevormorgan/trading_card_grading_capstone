import pytest
from PIL import Image

pytest.importorskip("cv2")

from card_grading import scans  # noqa: E402

BACKGROUND = (25, 25, 28)
COLORS = [(200, 40, 40), (40, 160, 60), (50, 80, 200), (210, 180, 40), (160, 60, 180), (40, 170, 180)]


def make_card(color, size=(250, 350), border=20):
    card = Image.new("RGB", size, (245, 245, 240))
    card.paste(color, (border, border, size[0] - border, size[1] - border))
    return card


def make_scan(placements, size=(1000, 900)):
    """A dark scan with cards pasted on it: placements are (card, (x, y), rotation in degrees)."""
    scan = Image.new("RGB", size, BACKGROUND)
    for card, position, angle in placements:
        scan.paste(card.rotate(angle, expand=True, fillcolor=BACKGROUND, resample=Image.BICUBIC), position)
    return scan


def color_distance(pixel, color):
    return max(abs(a - b) for a, b in zip(pixel, color, strict=True))


def six_card_scan():
    positions = [(60, 50), (380, 60), (700, 45), (65, 480), (385, 470), (695, 485)]
    angles = [3, -4, 0, 5, -2, 1]
    return make_scan([(make_card(color), position, angle)
                      for color, position, angle in zip(COLORS, positions, angles, strict=True)])


def test_cards_are_found_straightened_and_numbered_row_by_row():
    scan = six_card_scan()
    scan.paste((250, 250, 250), (500, 440, 506, 446))   # a speck of dust

    cards, skipped = scans.split_scan(scan)

    assert (len(cards), skipped) == (6, 0)
    for card, color in zip(cards, COLORS, strict=True):
        assert abs(card.width - 250) <= 6 and abs(card.height - 350) <= 6
        assert color_distance(card.getpixel((card.width // 2, card.height // 2)), color) < 30
        # straightened: the border runs along all four edges
        for point in ((card.width // 2, 6), (card.width // 2, card.height - 7), (6, card.height // 2),
                      (card.width - 7, card.height // 2)):
            assert color_distance(card.getpixel(point), (245, 245, 240)) < 40


def test_a_card_lying_sideways_is_turned_upright():
    (card,), skipped = scans.split_scan(make_scan([(make_card(COLORS[0]), (300, 300), 90)]))
    assert skipped == 0
    assert card.height > card.width


def test_touching_cards_are_left_out_and_reported():
    scan = make_scan([(make_card(COLORS[0]), (50, 50), 0), (make_card(COLORS[1]), (300, 50), 0),   # no gap
                      (make_card(COLORS[2]), (650, 50), 0), (make_card(COLORS[3]), (100, 480), 0),
                      (make_card(COLORS[4]), (500, 480), 0)])
    cards, skipped = scans.split_scan(scan)
    assert (len(cards), skipped) == (3, 1)


def test_stated_resolution_is_used_to_check_the_size():
    scan = make_scan([(make_card(COLORS[0], size=(750, 1050), border=60), (100, 100), 0),
                      (make_card(COLORS[1], size=(375, 525), border=30), (1100, 100), 0)], size=(1700, 1300))
    scan.info["dpi"] = (300, 300)
    cards, skipped = scans.split_scan(scan)   # only the first is 2.5 x 3.5 inches at 300 dpi
    assert (len(cards), skipped) == (1, 1)
    assert abs(cards[0].width - 750) <= 8


def test_scan_without_cards_gives_nothing():
    assert scans.split_scan(Image.new("RGB", (600, 500), BACKGROUND)) == ([], 0)


def test_command_saves_numbered_card_images(tmp_path, capsys):
    scan_dir, output_dir = tmp_path / "scans", tmp_path / "cards"
    scan_dir.mkdir()
    six_card_scan().save(scan_dir / "page1.png")
    (scan_dir / "notes.txt").write_text("not a scan")

    scans.main([str(scan_dir), "--output_dir", str(output_dir)])

    assert sorted(path.name for path in output_dir.iterdir()) == [f"page1_{n:02d}.jpg" for n in range(1, 7)]
    assert "page1.png: 6 card(s)" in capsys.readouterr().out
