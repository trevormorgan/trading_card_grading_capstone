import csv

import numpy as np
import pytest
from PIL import Image

from card_grading import assess


def make_card(path, left=40, right=40):
    path.parent.mkdir(parents=True, exist_ok=True)
    card = Image.new("RGB", (500, 700), (245, 245, 238))
    card.paste((180, 60, 50), (left, 50, 500 - right, 650))
    card.save(path)


def read_rows(path):
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def test_centring_is_measured_without_a_model(tmp_path):
    make_card(tmp_path / "cards" / "page1_01.jpg", left=20, right=60)
    make_card(tmp_path / "cards" / "box2" / "page7_03.png")
    noise = np.random.default_rng(0).integers(0, 255, (700, 500, 3), dtype=np.uint8)
    Image.fromarray(noise).save(tmp_path / "cards" / "page1_02.jpg")
    (tmp_path / "cards" / "notes.txt").write_text("not an image")

    assess.main(["--images", str(tmp_path / "cards"), "--output", str(tmp_path / "out" / "assessment.csv")])

    rows = {row["file"]: row for row in read_rows(tmp_path / "out" / "assessment.csv")}
    assert sorted(rows) == ["box2/page7_03.png", "page1_01.jpg", "page1_02.jpg"]
    assert rows["page1_01.jpg"]["centering_left_right"] == "25/75"
    assert rows["page1_01.jpg"]["centering_worst"] == "75"
    assert rows["box2/page7_03.png"]["centering_left_right"] == "50/50"
    assert rows["page1_02.jpg"]["centering_left_right"] == ""   # no plain border to measure
    assert "grade" not in rows["page1_01.jpg"]


def test_grade_rows():
    probabilities = np.zeros((1, 10))
    probabilities[0, [6, 7, 8]] = [0.2, 0.5, 0.3]            # grades 7, 8 and 9
    (row,) = assess.grade_rows(probabilities)
    assert (row["grade"], row["confidence"], row["within_1"], row["expected_grade"]) == (8, "0.50", "1.00", "8.1")
    assert row["p9"] == "0.300"


def test_grades_are_estimated_with_a_model(tmp_path):
    pytest.importorskip("tensorflow")
    from card_grading import training

    make_card(tmp_path / "cards" / "a.jpg")
    make_card(tmp_path / "cards" / "b.png")
    _, model = training.build_model((64, 48), droprate=0.1, weights=None)
    model.save(tmp_path / "model.keras")

    assess.main(["--images", str(tmp_path / "cards"), "--output", str(tmp_path / "assessment.csv"),
                 "--model", str(tmp_path / "model.keras")])

    rows = read_rows(tmp_path / "assessment.csv")
    assert [row["file"] for row in rows] == ["a.jpg", "b.png"]
    for row in rows:
        probabilities = [float(row[f"p{grade}"]) for grade in range(1, 11)]
        assert sum(probabilities) == pytest.approx(1.0, abs=0.01)
        assert int(row["grade"]) == int(np.argmax(probabilities)) + 1
        assert float(row["confidence"]) == pytest.approx(max(probabilities), abs=0.01)
        assert row["centering_left_right"] == "50/50"
