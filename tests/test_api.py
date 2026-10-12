from pathlib import Path

from fastapi.testclient import TestClient

from src.api import create_app
from src.invoices import generate_split

_texts, _ents = generate_split(3, seed=2025)
_used = generate_split(1, seed=2025)  # force identical generator state as training-time dev split
SAMPLE = _texts[-1]

cm = TestClient(create_app())


def setup_module(module):
    cm.__enter__()
    module.client = cm


def teardown_module(module):
    cm.__exit__(None, None, None)


def test_health():
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"


def test_invoice_extracts_required_fields():
    r = client.post("/invoice", json={"text": SAMPLE})
    assert r.status_code == 200
    body = r.json()
    labels = {e["label"] for e in body["entities"]}
    for lab in ("COMPANY", "INV_NUM", "DATE", "VAT_NUM", "AMOUNT"):
        assert lab in labels


def test_invoice_offsets_slice_correctly():
    r = client.post("/invoice", json={"text": SAMPLE})
    assert r.status_code == 200
    for e in r.json()["entities"]:
        assert SAMPLE[e["start"]:e["end"]] == e["text"]


def test_invoice_missing_fields_flagged():
    r = client.post("/invoice", json={"text": "Just a note about payment terms."})
    assert r.status_code == 200
    body = r.json()
    assert body["suspicious"] is True
    assert set(body["missing"]) == {"COMPANY", "INV_NUM", "DATE", "VAT_NUM", "AMOUNT"}


def test_upload_endpoint(tmp_path: Path):
    f = tmp_path / "inv.txt"
    f.write_text(SAMPLE, encoding="utf-8")
    with open(f, "rb") as fh:
        r = client.post("/invoice_upload", files={"file": ("inv.txt", fh, "text/plain")})
    assert r.status_code == 200
    assert r.json()["missing"] == []
