"""Assess a folder of card images: centring, and the grade a trained model estimates.

USAGE:
    card-grading-assess --images ./cards --output assessment.csv
    card-grading-assess --images ./cards --output assessment.csv --model logs/<run>/best_model.keras

Every image should show one card (see ``card-grading-split-scan``). The result is a
spreadsheet (CSV) with one row per image:

- ``centering_left_right`` / ``centering_top_bottom``: the borders as shares, e.g. ``45/55``
  (empty when the card has no plain border to measure), and ``centering_worst``, the larger
  share of the more off-centre direction (50 is perfectly centred)
- with ``--model``: ``grade`` (the most likely PSA grade), ``confidence`` (how likely the model
  thinks that grade is), ``within_1`` (how likely the true grade is within one of it),
  ``expected_grade`` (the average over all grades) and ``p1`` ... ``p10`` (each grade's probability)

A model is only as good as the images it was trained on: one trained on photos of cards in
PSA cases has not seen loose cards, so treat its grades for those as rough hints.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np
from PIL import Image, UnidentifiedImageError

from card_grading.centering import measure_centering
from card_grading.data import GRADES

IMAGE_SUFFIXES = (".jpg", ".jpeg", ".png")
CENTERING_COLUMNS = ["centering_left_right", "centering_top_bottom", "centering_worst"]
GRADE_COLUMNS = ["grade", "confidence", "within_1", "expected_grade"] + [f"p{grade}" for grade in GRADES]


def find_images(folder: Path) -> list[Path]:
    """The image files in a folder and its subfolders, sorted by path."""
    return sorted(path for path in folder.rglob("*") if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES)


def centering_row(path: Path) -> dict:
    try:
        with Image.open(path) as image:
            centering = measure_centering(image)
    except (UnidentifiedImageError, OSError):
        centering = None
    if centering is None:
        return {}
    return {"centering_left_right": centering.left_right, "centering_top_bottom": centering.top_bottom,
            "centering_worst": f"{centering.worst:.0f}"}


def grade_rows(probabilities: np.ndarray) -> list[dict]:
    """Turn predicted grade probabilities (one row per image) into the grade columns."""
    grades = np.array(GRADES)
    rows = []
    for row in probabilities:
        best = int(np.argmax(row))
        within_1 = float(row[max(best - 1, 0):best + 2].sum())
        rows.append({"grade": int(grades[best]), "confidence": f"{row[best]:.2f}", "within_1": f"{within_1:.2f}",
                     "expected_grade": f"{float(row @ grades):.1f}",
                     **{f"p{grade}": f"{probability:.3f}" for grade, probability in zip(grades, row, strict=True)}})
    return rows


def predict_grades(model_path: Path, paths: list[Path], batch_size: int = 32) -> np.ndarray:
    """Grade probabilities for the images, averaged over each image and its mirror image."""
    # imported here so that measuring centring alone does not need TensorFlow
    from tensorflow import keras

    from card_grading import training

    model = keras.models.load_model(model_path)
    dataset = training.make_dataset([str(path) for path in paths], np.ones(len(paths), dtype=int),
                                    tuple(model.input_shape[1:3]), batch_size)
    return training.predict_probabilities(model, dataset, flip=True)


def assess(images_dir: Path, output: Path, model_path: Path | None = None) -> int:
    """Write the assessment of every image in ``images_dir`` to ``output``. Returns the number of images."""
    paths = find_images(images_dir)
    rows = [{"file": str(path.relative_to(images_dir))} | centering_row(path) for path in paths]
    columns = ["file"] + CENTERING_COLUMNS
    if model_path is not None and paths:
        for row, grades in zip(rows, grade_rows(predict_grades(model_path, paths)), strict=True):
            row |= grades
        columns += GRADE_COLUMNS
    output.parent.mkdir(parents=True, exist_ok=True)
    with open(output, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=columns, restval="")
        writer.writeheader()
        writer.writerows(rows)
    return len(rows)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Measure the centring of card images and estimate their grades")
    parser.add_argument("-i", "--images", type=Path, required=True, help="Folder of card images, one card per image")
    parser.add_argument("-o", "--output", type=Path, required=True, help="CSV file to write")
    parser.add_argument("-m", "--model", type=Path, default=None,
                        help="A trained model (best_model.keras of a training run) to estimate grades with")
    args = parser.parse_args(argv)
    if not args.images.is_dir():
        raise SystemExit(f"Not a folder: {args.images}")
    count = assess(args.images, args.output, args.model)
    print(f"{count} image(s) assessed, results in {args.output}")


if __name__ == "__main__":
    main()
