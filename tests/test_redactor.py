import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from redactor import Redactor  # noqa: E402

r = Redactor(use_model=False)


def test_ssn_phone_email():
    out = r.redact("SSN 123-45-6789, call (555) 201-7788 or a@b.com")
    assert "123-45-6789" not in out["redacted_text"]
    assert "201-7788" not in out["redacted_text"]
    assert "a@b.com" not in out["redacted_text"]
    assert out["counts"]["SSN"] == 1


def test_dob_and_mrn():
    out = r.redact("DOB: 03/14/1978 MRN: 00482917")
    assert "[DOB]" in out["redacted_text"]
    assert "[MRN]" in out["redacted_text"]


def test_clinical_text_untouched():
    text = "Continue metformin and lisinopril. Chest X-ray ordered."
    assert r.redact(text)["redacted_text"] == text


def test_sample_note():
    with open(os.path.join(os.path.dirname(__file__), "..", "samples", "sample_note.txt")) as f:
        out = r.redact(f.read())
    for leaked in ["123-45-6789", "555-303-4411", "rajesh.reddy@example.com", "00482917", "03/14/1978"]:
        assert leaked not in out["redacted_text"]


def test_ner_grouping_logic():
    from redactor import NerModel
    text = "Anita Sharma visited Hyderabad"
    labels = ["B-PER", "I-PER", "O", "B-LOC"]
    offsets = [(0, 5), (6, 12), (13, 20), (21, 30)]
    ents = NerModel._group(text, labels, offsets)
    assert [(e.type, e.text) for e in ents] == [("NAME", "Anita Sharma"), ("LOCATION", "Hyderabad")]


def test_api():
    os.environ["MEDSHIELD_RULES_ONLY"] = "1"
    import app as appmod
    c = appmod.app.test_client()
    assert c.get("/").status_code == 200
    assert c.get("/api/status").get_json()["mode"] == "rules only"
    assert c.post("/api/redact", json={"text": ""}).status_code == 400
    res = c.post("/api/redact", json={"text": "Call 555-303-4411"}).get_json()
    assert "[PHONE]" in res["redacted_text"]
