import csv
import io

import pytest
from PIL import Image

requests = pytest.importorskip("requests")

from card_grading import psa_api  # noqa: E402

JPEG = b"\xff\xd8" + b"x" * 100


def png_bytes():
    buffer = io.BytesIO()
    Image.new("RGB", (4, 6), (10, 200, 30)).save(buffer, format="PNG")
    return buffer.getvalue()


class FakeResponse:
    def __init__(self, status_code=200, json_body=None, content=None, text=""):
        self.status_code = status_code
        self._json = json_body
        self.content = content if content is not None else (b"{}" if json_body is not None else b"")
        self.text = text

    def json(self):
        return self._json

    def raise_for_status(self):
        if self.status_code >= 400:
            raise requests.HTTPError(str(self.status_code))


def cert_response(category, grade):
    return {"PSACert": {"CertNumber": "1", "Category": category, "CardGrade": grade, "Year": "1989", "Brand": "TOPPS",
                        "Subject": "NOLAN RYAN", "CardNumber": "530"},
            "IsValidRequest": True, "ServerMessage": "Request successful"}


def images_response(cert):
    return [{"IsFrontImage": False, "ImageURL": f"https://img.test/{cert}/back.jpg"},
            {"IsFrontImage": True, "ImageURL": f"https://img.test/{cert}/front.jpg"}]


class FakeSession:
    """Answers API calls from a dict of certs, and serves image bytes for any other url."""

    def __init__(self, certs, quota=None, image_bytes=JPEG):
        self.certs = certs
        self.quota = quota
        self.image_bytes = image_bytes
        self.api_calls = []
        self.image_requests = []

    def get(self, url, headers=None, timeout=None):
        assert timeout
        if not url.startswith(psa_api.API_URL):
            assert not headers, "the API token must not be sent to the image host"
            self.image_requests.append(url)
            return FakeResponse(content=self.image_bytes)
        assert headers["Authorization"] == "bearer secret-token"
        method, cert = url.split("/")[-2:]
        self.api_calls.append((method, int(cert)))
        if self.quota is not None and len(self.api_calls) > self.quota:
            return FakeResponse(429, text="API calls quota exceeded! maximum admitted 100 per Day.")
        if method == "GetImagesByCertNumber":
            return FakeResponse(json_body=images_response(cert))
        return FakeResponse(json_body=self.certs.get(
            int(cert), {"IsValidRequest": True, "ServerMessage": "No data found"}))


@pytest.fixture(autouse=True)
def no_sleep(monkeypatch):
    monkeypatch.setattr(psa_api.time, "sleep", lambda seconds: None)


CERTS = {
    100: cert_response("BASEBALL CARDS", "NM-MT 8"),
    101: cert_response("TCG CARDS", "GEM MT 10"),
    103: cert_response("BASEBALL CARDS", "NM-MT+ 8.5"),
    104: cert_response("BASEBALL CARDS", "GEM MT 10"),
}


def read_metadata(folder):
    with open(folder / psa_api.METADATA_FILE, newline="") as f:
        return {int(row["cert_number"]): row for row in csv.DictReader(f)}


@pytest.mark.parametrize("text, grade", [
    ("GEM MT 10", 10), ("NM-MT 8", 8), ("PR 1", 1), ("10", 10), ("MINT 9", 9),
    ("NM-MT+ 8.5", None), ("FR 1.5", None), ("MINT 9 (OC)", None), ("NM 7 MK", None),
    ("AUTHENTIC", None), ("", None), (None, None), ("GRADE 11", None),
])
def test_parse_grade(text, grade):
    assert psa_api.parse_grade(text) == grade


def test_parse_images_reads_flagged_wrapped_and_unflagged_responses():
    flagged = psa_api.parse_images(images_response(7))
    assert flagged == psa_api.CertImages(front="https://img.test/7/front.jpg", back="https://img.test/7/back.jpg")
    assert psa_api.parse_images({"Images": images_response(7)}) == flagged
    assert psa_api.parse_images([{"url": "a.jpg"}, {"url": "b.jpg"}]) == psa_api.CertImages(front="a.jpg", back="b.jpg")
    assert psa_api.parse_images(None) == psa_api.CertImages()
    assert psa_api.parse_images({"ServerMessage": "No data found"}) == psa_api.CertImages()


def test_only_wanted_cards_are_downloaded(tmp_path):
    session = FakeSession(CERTS)
    client = psa_api.PsaClient("secret-token", session=session)

    counts = psa_api.fetch_certs(client, [100, 101, 102, 103, 104], tmp_path, save_backs=True)

    assert counts == {"saved": 2, "other category": 1, "no card under this number": 1, "no whole-number grade": 1}
    assert (tmp_path / "psa8" / "card100_PSA8.jpg").read_bytes() == JPEG
    assert (tmp_path / "psa10" / "card104_PSA10.jpg").is_file()
    assert (tmp_path / "backs" / "psa8" / "card100_PSA8.jpg").is_file()
    # the images are only asked for when the card is wanted
    assert [cert for method, cert in session.api_calls if method == "GetImagesByCertNumber"] == [100, 104]
    assert session.image_requests[0] == "https://img.test/100/front.jpg"
    row = read_metadata(tmp_path)[100]
    assert (row["status"], row["grade"], row["subject"], row["front_file"]) == (
        "saved", "8", "NOLAN RYAN", "psa8/card100_PSA8.jpg")


def test_other_image_formats_are_converted_to_jpeg(tmp_path):
    client = psa_api.PsaClient("secret-token", session=FakeSession(CERTS, image_bytes=png_bytes()))
    psa_api.fetch_certs(client, [100], tmp_path)
    with Image.open(tmp_path / "psa8" / "card100_PSA8.jpg") as image:
        assert image.format == "JPEG" and image.size == (4, 6)


def test_stops_at_the_call_limit_and_continues_on_the_next_run(tmp_path):
    first = FakeSession(CERTS, quota=3)   # cert 100, images of 100, cert 101, then the limit
    counts = psa_api.fetch_certs(psa_api.PsaClient("secret-token", session=first), [100, 101, 102, 104], tmp_path)
    assert counts == {"saved": 1, "other category": 1}

    second = FakeSession(CERTS)
    psa_api.fetch_certs(psa_api.PsaClient("secret-token", session=second), [100, 101, 102, 104], tmp_path)
    assert [cert for _, cert in second.api_calls] == [102, 104, 104]   # 100 and 101 are not looked up again
    assert sorted(read_metadata(tmp_path)) == [100, 101, 102, 104]


def test_call_budget_is_respected(tmp_path):
    session = FakeSession(CERTS)
    client = psa_api.PsaClient("secret-token", session=session, max_calls=2)
    psa_api.fetch_certs(client, [101, 102, 103, 104], tmp_path)
    assert len(session.api_calls) == 2


def test_wrong_token_stops_with_an_error(tmp_path):
    class Rejecting(FakeSession):
        def get(self, url, headers=None, timeout=None):
            return FakeResponse(500, text="error")

    with pytest.raises(psa_api.PsaApiError, match="token"):
        psa_api.fetch_certs(psa_api.PsaClient("secret-token", session=Rejecting({})), [100], tmp_path)


def test_account_without_api_access_stops_with_psas_message(tmp_path):
    class NotApproved(FakeSession):
        def get(self, url, headers=None, timeout=None):
            self.api_calls.append(url)
            return FakeResponse(403, text='{"Message":"Access to this API is limited to approved customers."}')

    session = NotApproved({})
    with pytest.raises(psa_api.PsaApiError, match="limited to approved customers"):
        psa_api.fetch_certs(psa_api.PsaClient("secret-token", session=session), [100, 101], tmp_path)
    assert len(session.api_calls) == 1          # no retries and no further certs after a refusal
    assert not (tmp_path / psa_api.METADATA_FILE).exists()


def test_main_needs_a_token(tmp_path, monkeypatch):
    monkeypatch.delenv(psa_api.TOKEN_VARIABLE, raising=False)
    with pytest.raises(SystemExit, match=psa_api.TOKEN_VARIABLE):
        psa_api.main(["--start_cert", "1", "--output_dir", str(tmp_path)])


def test_read_cert_numbers(tmp_path):
    path = tmp_path / "certs.txt"
    path.write_text("# my cards\n80000001\n\n 80000002 \n")
    assert psa_api.read_cert_numbers(path) == [80000001, 80000002]
