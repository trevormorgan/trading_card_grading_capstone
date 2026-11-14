"""Cut a scan of several cards into one straightened image per card.

USAGE:
    card-grading-split-scan ./scans --output_dir ./cards

Lay the cards on the scanner, or photograph them from straight above, on a plain background
that contrasts with the edges of the cards (black paper behind white-bordered cards, for
example) and leave a gap between the cards. Every card that is found is saved as
``<scan name>_<number>.jpg``, numbered row by row from the top left, turned upright
(portrait) and cropped to its edges.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

SCAN_SUFFIXES = (".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp")

# A standard card is 2.5 x 3.5 inches
CARD_SHORT_INCHES, CARD_LONG_INCHES = 2.5, 3.5
CARD_ASPECT = CARD_SHORT_INCHES / CARD_LONG_INCHES
ASPECT_TOLERANCE = 0.10   # accepted difference between a region's short/long ratio and a card's
SIZE_TOLERANCE = 0.15     # accepted difference from the real card size, when the scan states its resolution

# Regions smaller than this share of the scan are dust or specks, not cards
MIN_CARD_AREA_FRACTION = 0.01
# A card fills the rectangle around it; lower values mean an odd shape (a torn or cut-off card)
MIN_RECTANGLE_FILL = 0.85
# Without a stated resolution: a region this many times larger than the typical card is several cards touching
MAX_AREA_RATIO = 1.6

# The background colour is taken from a frame this wide around the edge of the scan
BACKGROUND_FRAME_FRACTION = 0.02
# How different from the background (0-255) a pixel must be to count as part of a card
MIN_CONTRAST = 25


def card_mask(scan: np.ndarray) -> np.ndarray:
    """A mask (0/255) of the pixels of an RGB scan that do not look like its background."""
    height, width = scan.shape[:2]
    margin = max(2, int(min(height, width) * BACKGROUND_FRAME_FRACTION))
    frame = np.concatenate([scan[:margin].reshape(-1, 3), scan[-margin:].reshape(-1, 3),
                            scan[:, :margin].reshape(-1, 3), scan[:, -margin:].reshape(-1, 3)])
    background = np.median(frame, axis=0)

    distance = np.abs(scan.astype(np.int16) - background).max(axis=2).astype(np.uint8)
    distance = cv2.GaussianBlur(distance, (5, 5), 0)
    threshold, _ = cv2.threshold(distance, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    mask = (distance > max(threshold, MIN_CONTRAST)).astype(np.uint8) * 255

    # remove specks and thin bridges between cards
    size = max(3, int(min(height, width) * 0.005)) | 1
    return cv2.morphologyEx(mask, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (size, size)))


def find_cards(scan: np.ndarray, dpi: float | None = None) -> tuple[list[tuple], int]:
    """Find the cards on an RGB scan.

    Returns the rotated rectangles around the cards (as OpenCV gives them: centre, size, angle),
    row by row from the top left, and the number of card-sized regions that were left out
    because they are not shaped or sized like one card.
    """
    contours = cv2.findContours(card_mask(scan), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[-2]
    min_area = MIN_CARD_AREA_FRACTION * scan.shape[0] * scan.shape[1]

    candidates = []
    skipped = 0
    for contour in contours:
        area = cv2.contourArea(contour)
        if area < min_area:
            continue
        rectangle = cv2.minAreaRect(contour)
        short, long = sorted(rectangle[1])
        card_shaped = (abs(short / long - CARD_ASPECT) <= ASPECT_TOLERANCE
                       and area / (short * long) >= MIN_RECTANGLE_FILL)
        if card_shaped and dpi:
            card_shaped = abs(long / (CARD_LONG_INCHES * dpi) - 1) <= SIZE_TOLERANCE
        if card_shaped:
            candidates.append(rectangle)
        else:
            skipped += 1

    if not dpi and candidates:
        # two cards side by side have nearly the proportions of one card, so compare the sizes
        typical_area = float(np.median([rect[1][0] * rect[1][1] for rect in candidates]))
        cards = [rect for rect in candidates if rect[1][0] * rect[1][1] <= MAX_AREA_RATIO * typical_area]
        skipped += len(candidates) - len(cards)
    else:
        cards = candidates
    return _in_reading_order(cards), skipped


def _in_reading_order(rectangles: list[tuple]) -> list[tuple]:
    """Sort rectangles row by row from the top, and from left to right within a row."""
    if not rectangles:
        return []
    row_height = float(np.median([max(rect[1]) for rect in rectangles]))
    rows: list[list[tuple]] = []
    for rect in sorted(rectangles, key=lambda rect: rect[0][1]):
        if rows and rect[0][1] - rows[-1][0][0][1] < row_height / 2:
            rows[-1].append(rect)
        else:
            rows.append([rect])
    return [rect for row in rows for rect in sorted(row, key=lambda rect: rect[0][0])]


def extract_card(scan: np.ndarray, rectangle: tuple) -> np.ndarray:
    """Cut a rotated rectangle out of the scan, straightened and turned upright (portrait)."""
    corners = cv2.boxPoints(rectangle)
    sums, differences = corners.sum(axis=1), np.diff(corners, axis=1).ravel()
    top_left, bottom_right = corners[np.argmin(sums)], corners[np.argmax(sums)]
    top_right, bottom_left = corners[np.argmin(differences)], corners[np.argmax(differences)]

    width = int(round(max(np.linalg.norm(top_right - top_left), np.linalg.norm(bottom_right - bottom_left))))
    height = int(round(max(np.linalg.norm(bottom_left - top_left), np.linalg.norm(bottom_right - top_right))))
    target = np.float32([[0, 0], [width - 1, 0], [width - 1, height - 1], [0, height - 1]])
    transform = cv2.getPerspectiveTransform(np.float32([top_left, top_right, bottom_right, bottom_left]), target)
    card = cv2.warpPerspective(scan, transform, (width, height), flags=cv2.INTER_CUBIC)
    return cv2.rotate(card, cv2.ROTATE_90_CLOCKWISE) if width > height else card


def split_scan(image: Image.Image) -> tuple[list[Image.Image], int]:
    """The cards on a scan as separate images, and the number of card-sized regions that were left out."""
    dpi = image.info.get("dpi", (None,))[0]
    # a resolution of 72 or 96 is what software writes when it does not know: no use as a size check
    dpi = float(dpi) if dpi and float(dpi) >= 150 else None
    scan = np.asarray(image.convert("RGB"))
    rectangles, skipped = find_cards(scan, dpi)
    return [Image.fromarray(extract_card(scan, rectangle)) for rectangle in rectangles], skipped


def split_scan_file(path: Path, output_dir: Path) -> tuple[int, int]:
    """Save every card on a scan into ``output_dir``. Returns the number saved and the number left out."""
    with Image.open(path) as image:
        cards, skipped = split_scan(image)
    output_dir.mkdir(parents=True, exist_ok=True)
    for number, card in enumerate(cards, start=1):
        card.save(output_dir / f"{path.stem}_{number:02d}.jpg", quality=95)
    return len(cards), skipped


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Cut scans of several cards into one image per card")
    parser.add_argument("scans", type=Path, help="A scan, or a folder of scans")
    parser.add_argument("-o", "--output_dir", type=Path, required=True, help="Folder the card images are saved to")
    args = parser.parse_args(argv)

    paths = ([args.scans] if args.scans.is_file()
             else sorted(path for path in args.scans.iterdir() if path.suffix.lower() in SCAN_SUFFIXES))
    if not paths:
        raise SystemExit(f"No scans found in {args.scans}")
    total = 0
    for path in paths:
        saved, skipped = split_scan_file(path, args.output_dir)
        total += saved
        note = (f" ({skipped} region(s) left out: not the shape or size of one card - "
                f"are cards touching or cut off?)" if skipped else "")
        print(f"{path.name}: {saved} card(s){note}")
    print(f"{total} card image(s) saved to {args.output_dir}")


if __name__ == "__main__":
    main()
