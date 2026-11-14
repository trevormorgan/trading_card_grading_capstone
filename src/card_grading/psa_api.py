"""Download PSA's own scans of graded cards, with their grades, through the PSA Public API.

USAGE:
    export PSA_API_TOKEN=...   # from https://www.psacard.com/publicapi (needs a PSA account)
    card-grading-fetch-psa --start_cert 80000000 --count 5000 --output_dir ./psa_scans
    card-grading-fetch-psa --certs_file my_certs.txt --output_dir ./psa_scans

For every cert number the card's details are looked up first. Only cards of the wanted
category (baseball by default) with a whole-number grade from 1 to 10 are downloaded, into
``<output_dir>/psa<grade>/card<cert>_PSA<grade>.jpg``, which is the layout ``card-grading-crop``
reads. What happened to every cert number is recorded in ``<output_dir>/psa_metadata.csv``.

A token alone is not enough: PSA only answers accounts it has approved for the API (others get
"Access to this API is limited to approved customers."). The API also allows a limited number
of calls per day. The download stops cleanly when the limit is reached; run the same command
again later and it carries on with the cert numbers that were not looked up yet.

Check PSA's API agreement before using the images, in particular for training models.
"""

from __future__ import annotations

import argparse
import csv
import io
import os
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import requests
from PIL import Image

from card_grading.data import GRADES, IMAGE_SUFFIX

API_URL = "https://api.psacard.com/publicapi"
TOKEN_VARIABLE = "PSA_API_TOKEN"
METADATA_FILE = "psa_metadata.csv"
METADATA_COLUMNS = ["cert_number", "status", "category", "year", "brand", "subject", "variety", "card_number",
                    "grade_text", "grade", "front_file", "back_file", "front_url", "back_url"]

FREE_TIER_CALLS_PER_DAY = 100
REQUEST_TIMEOUT_SECONDS = 30
SECONDS_BETWEEN_CALLS = 1.0
MAX_ATTEMPTS = 3

STATUS_SAVED = "saved"

_GRADE_PATTERN = re.compile(r"(?<![\d.])(\d{1,2}(?:\.\d)?)(?![\d.])")
_QUALIFIER_PATTERN = re.compile(r"\((?:OC|ST|PD|OF|MK|MC)\)|\b(?:OC|ST|PD|OF|MK|MC)\b")


class QuotaExceeded(Exception):
    """No more API calls can be made for now: the daily limit, or the ``--max_calls`` budget, is used up."""


class PsaApiError(Exception):
    """The API refused a request (for example because of a wrong token)."""


@dataclass(frozen=True)
class CertImages:
    """The scan urls PSA has for one cert. Either can be missing."""

    front: str | None = None
    back: str | None = None


def parse_grade(grade_text: str | None) -> int | None:
    """The whole-number grade (1-10) in a grade text such as ``'NM-MT 8'`` or ``'GEM MT 10'``.

    Returns ``None`` for texts without a number (``'AUTHENTIC'``), for half grades (``'NM-MT+ 8.5'``)
    and for grades with a qualifier (``'MINT 9 (OC)'``): a qualifier means the card has a flaw that
    its number does not reflect, which would make it a misleading training example.
    """
    if not grade_text or _QUALIFIER_PATTERN.search(grade_text):
        return None
    numbers = _GRADE_PATTERN.findall(grade_text)
    if not numbers:
        return None
    grade = float(numbers[-1])
    return int(grade) if grade.is_integer() and int(grade) in GRADES else None


def parse_images(response: Any) -> CertImages:
    """Pick the front and back scan urls out of a ``GetImagesByCertNumber`` response.

    PSA's API description does not specify this response, so it is read leniently: a list of
    entries (possibly wrapped in an object), each with an image url and usually a flag saying
    whether it is the front. Without flags the first image is taken to be the front.
    """
    entries = response
    if isinstance(response, dict):
        entries = next((value for value in response.values() if isinstance(value, list)), [])
    if not isinstance(entries, list):
        return CertImages()

    front = back = None
    unflagged: list[str] = []
    for entry in entries:
        if not isinstance(entry, dict):
            continue
        fields = {str(key).lower(): value for key, value in entry.items()}
        url = next((fields[key] for key in ("imageurl", "url", "image") if isinstance(fields.get(key), str)), None)
        if not url:
            continue
        is_front = fields.get("isfrontimage", fields.get("isfront"))
        if is_front is True:
            front = front or url
        elif is_front is False:
            back = back or url
        else:
            unflagged.append(url)
    for url in unflagged:
        if front is None:
            front = url
        elif back is None:
            back = url
    return CertImages(front=front, back=back)


class PsaClient:
    """A small client for the PSA Public API that keeps count of the calls it makes."""

    def __init__(self, token: str, session: requests.Session | None = None,
                 max_calls: int = FREE_TIER_CALLS_PER_DAY, seconds_between_calls: float = SECONDS_BETWEEN_CALLS):
        self.session = session or requests.Session()
        self.max_calls = max_calls
        self.seconds_between_calls = seconds_between_calls
        self.calls_made = 0
        # the token is only ever sent to the API itself, never to the hosts the images are on
        self._headers = {"Authorization": f"bearer {token}"}

    def _call(self, method: str, cert_number: int) -> Any:
        """One API call. Network errors and server errors are retried a few times."""
        if self.calls_made >= self.max_calls:
            raise QuotaExceeded(f"the budget of {self.max_calls} API calls for this run is used up")
        url = f"{API_URL}/cert/{method}/{cert_number}"
        for attempt in range(1, MAX_ATTEMPTS + 1):
            if self.calls_made:
                time.sleep(self.seconds_between_calls)
            self.calls_made += 1
            try:
                response = self.session.get(url, headers=self._headers, timeout=REQUEST_TIMEOUT_SECONDS)
            except requests.RequestException as error:
                if attempt == MAX_ATTEMPTS:
                    raise PsaApiError(f"{method} failed for cert {cert_number}: {error}") from error
                continue
            if response.status_code == 429:
                raise QuotaExceeded(f"PSA says the call limit is reached: {response.text.strip()[:200]}")
            if response.status_code in (401, 403):
                # e.g. "Access to this API is limited to approved customers." for an account
                # that has a token but has not been given API access by PSA
                raise PsaApiError(f"PSA refused the request (HTTP {response.status_code}): "
                                  f"{response.text.strip()[:200]}")
            if response.status_code >= 500 and attempt < MAX_ATTEMPTS:
                continue
            if response.status_code >= 400:
                # PSA answers 500 for a wrong token as well as for errors on its side
                raise PsaApiError(f"{method} failed for cert {cert_number}: HTTP {response.status_code} "
                                  f"(a 500 can mean the token is wrong)")
            try:
                return response.json() if response.content else None
            except ValueError as error:
                raise PsaApiError(f"{method} returned something that is not JSON for cert {cert_number}") from error
        raise PsaApiError(f"{method} failed for cert {cert_number}")  # pragma: no cover

    def cert(self, cert_number: int) -> dict | None:
        """The details of a cert (PSA's ``PSACert`` object), or ``None`` if PSA has no card under that number."""
        response = self._call("GetByCertNumber", cert_number)
        if not isinstance(response, dict):
            return None
        return response.get("PSACert") or None

    def images(self, cert_number: int) -> CertImages:
        return parse_images(self._call("GetImagesByCertNumber", cert_number))


def download_image(session: requests.Session, url: str, path: Path) -> None:
    """Download an image and save it as a JPEG file (other formats are converted)."""
    response = session.get(url, timeout=REQUEST_TIMEOUT_SECONDS)
    response.raise_for_status()
    content = response.content
    path.parent.mkdir(parents=True, exist_ok=True)
    if content[:2] == b"\xff\xd8":
        path.write_bytes(content)
    else:
        with Image.open(io.BytesIO(content)) as image:
            image.convert("RGB").save(path, quality=95)


def read_processed_certs(metadata_path: Path) -> set[int]:
    """The cert numbers an earlier run already recorded."""
    if not metadata_path.is_file():
        return set()
    with open(metadata_path, newline="") as f:
        return {int(row["cert_number"]) for row in csv.DictReader(f)}


def _record(metadata_path: Path, row: dict) -> None:
    is_new = not metadata_path.exists()
    with open(metadata_path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=METADATA_COLUMNS)
        if is_new:
            writer.writeheader()
        writer.writerow({column: row.get(column, "") for column in METADATA_COLUMNS})


def fetch_cert(client: PsaClient, cert_number: int, output_dir: Path, category: str, save_backs: bool) -> dict:
    """Look one cert up and download its scan(s) if it is a card we want. Returns its metadata row."""
    row: dict = {"cert_number": cert_number}
    cert = client.cert(cert_number)
    if cert is None:
        return row | {"status": "no card under this number"}

    grade_text = cert.get("CardGrade") or cert.get("GradeDescription") or ""
    row |= {"category": cert.get("Category", ""), "year": cert.get("Year", ""), "brand": cert.get("Brand", ""),
            "subject": cert.get("Subject", ""), "variety": cert.get("Variety", ""),
            "card_number": cert.get("CardNumber", ""), "grade_text": grade_text}
    if category.lower() not in str(row["category"]).lower():
        return row | {"status": "other category"}
    grade = parse_grade(grade_text)
    if grade is None:
        return row | {"status": "no whole-number grade"}
    row["grade"] = grade

    images = client.images(cert_number)
    row |= {"front_url": images.front or "", "back_url": images.back or ""}
    if images.front is None:
        return row | {"status": "no images"}

    file_name = f"card{cert_number}_PSA{grade}{IMAGE_SUFFIX}"
    try:
        download_image(client.session, images.front, output_dir / f"psa{grade}" / file_name)
        row["front_file"] = f"psa{grade}/{file_name}"
        if save_backs and images.back:
            download_image(client.session, images.back, output_dir / "backs" / f"psa{grade}" / file_name)
            row["back_file"] = f"backs/psa{grade}/{file_name}"
    except (requests.RequestException, OSError) as error:
        return row | {"status": f"image download failed: {error}"[:200]}
    return row | {"status": STATUS_SAVED}


def fetch_certs(client: PsaClient, cert_numbers: list[int], output_dir: Path, category: str = "baseball",
                save_backs: bool = False) -> dict[str, int]:
    """Look up the cert numbers that were not processed yet. Returns how many certs ended in each status."""
    output_dir.mkdir(parents=True, exist_ok=True)
    metadata_path = output_dir / METADATA_FILE
    processed = read_processed_certs(metadata_path)
    counts: dict[str, int] = {}
    for cert_number in cert_numbers:
        if cert_number in processed:
            continue
        try:
            row = fetch_cert(client, cert_number, output_dir, category, save_backs)
        except QuotaExceeded as stop:
            # nothing is recorded for this cert, so the next run looks it up again
            print(f"Stopping: {stop}")
            break
        _record(metadata_path, row)
        processed.add(cert_number)
        counts[row["status"]] = counts.get(row["status"], 0) + 1
    return counts


def read_cert_numbers(path: Path) -> list[int]:
    """Cert numbers from a text file: one per line; blank lines and lines starting with # are ignored."""
    lines = (line.strip() for line in path.read_text().splitlines())
    return [int(line) for line in lines if line and not line.startswith("#")]


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Download PSA's scans of graded cards through the PSA Public API")
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--start_cert", type=int, help="First cert number of a range to look up")
    source.add_argument("--certs_file", type=Path, help="Text file with one cert number per line")
    parser.add_argument("--count", type=int, default=1000,
                        help="How many cert numbers to look up from start_cert on")
    parser.add_argument("-o", "--output_dir", type=Path, required=True,
                        help="Folder for the images (in psa1 ... psa10 folders) and the metadata file")
    parser.add_argument("--category", default="baseball",
                        help="Only download cards whose PSA category contains this text")
    parser.add_argument("--backs", action="store_true",
                        help="Also download the scans of the backs, into a separate backs/ folder")
    parser.add_argument("--max_calls", type=int, default=FREE_TIER_CALLS_PER_DAY,
                        help="Stop after this many API calls (PSA's standard limit is 100 per day)")
    args = parser.parse_args(argv)

    token = os.environ.get(TOKEN_VARIABLE)
    if not token:
        raise SystemExit(f"Set the {TOKEN_VARIABLE} environment variable to your PSA API token "
                         f"(sign in at https://www.psacard.com/publicapi to get one)")
    cert_numbers = (read_cert_numbers(args.certs_file) if args.certs_file
                    else list(range(args.start_cert, args.start_cert + args.count)))

    client = PsaClient(token, max_calls=args.max_calls)
    try:
        counts = fetch_certs(client, cert_numbers, args.output_dir, args.category, args.backs)
    except PsaApiError as error:
        raise SystemExit(f"Stopped: {error}") from error

    print(f"{client.calls_made} API calls made")
    for status, count in sorted(counts.items(), key=lambda item: -item[1]):
        print(f"  {count:6d}  {status}")
    print(f"Details of every cert number: {args.output_dir / METADATA_FILE}")


if __name__ == "__main__":
    main()
