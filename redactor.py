"""MedShield Edge - on-device PHI redaction for clinical text.

Two layers, both running locally:
  1. Rule layer (regex): SSN, phone, email, DOB/dates, MRN, ZIP, URLs, IDs.
  2. AI layer (open-source BERT NER model, ONNX): person names, locations, organisations.

The AI layer uses ONNX Runtime. On a Snapdragon-powered Windows PC with the
`onnxruntime-qnn` package installed, it runs through the QNN execution provider
(Hexagon NPU). Anywhere else it falls back to the CPU provider, and if the model
cannot be loaded at all the app keeps working in rules-only mode.
"""
from __future__ import annotations

import json
import os
import re
import time
from dataclasses import dataclass, asdict
from typing import List, Optional

import numpy as np

MODEL_REPO = os.environ.get("MEDSHIELD_MODEL_REPO", "Xenova/bert-base-NER")
MODEL_FILE = os.environ.get("MEDSHIELD_MODEL_FILE", "onnx/model_quantized.onnx")
MODEL_DIR = os.environ.get("MEDSHIELD_MODEL_DIR", os.path.join(os.path.dirname(__file__), "models"))
SEQ_LEN = 128  # fixed sequence length: NPUs prefer static shapes

# ---------------------------------------------------------------- rule layer
RULES = [
    ("SSN", re.compile(r"\b\d{3}-\d{2}-\d{4}\b")),
    ("EMAIL", re.compile(r"\b[\w.+-]+@[\w-]+(?:\.[\w-]+)+\b")),
    ("URL", re.compile(r"\bhttps?://\S+|\bwww\.\S+", re.I)),
    ("MRN", re.compile(r"\b(?:MRN|Medical Record(?: Number| No\.?)?|Patient ID|Acct|Account)\s*[:#\-]?\s*[A-Z0-9-]{5,}\b", re.I)),
    ("DOB", re.compile(r"\b(?:DOB|D\.O\.B\.?|Date of Birth|Born)\s*[:\-]?\s*(?:\d{1,2}[/-]\d{1,2}[/-]\d{2,4}|[A-Z][a-z]+ \d{1,2},? \d{4})", re.I)),
    ("PHONE", re.compile(r"(?<!\d)(?:\+?\d{1,3}[\s.-]?)?(?:\(\d{3}\)|\d{3})[\s.-]?\d{3}[\s.-]?\d{4}(?!\d)")),
    ("PHONE", re.compile(r"(?<!\d)(?:\+91[\s-]?)?[6-9]\d{4}[\s-]?\d{5}(?!\d)")),
    ("DATE", re.compile(r"\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b|\b(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Sept|Oct|Nov|Dec)[a-z]*\.? \d{1,2},? \d{4}\b")),
    ("ZIP", re.compile(r"\b\d{5}(?:-\d{4})?\b|\b\d{6}\b")),
]

NER_KEEP = {"PER": "NAME", "LOC": "LOCATION", "ORG": "ORGANIZATION"}


@dataclass
class Entity:
    type: str
    start: int
    end: int
    text: str
    source: str  # "rule" or "model"


def _rule_entities(text: str) -> List[Entity]:
    found: List[Entity] = []
    for label, pattern in RULES:
        for m in pattern.finditer(text):
            found.append(Entity(label, m.start(), m.end(), m.group(0), "rule"))
    return found


def _resolve_overlaps(entities: List[Entity]) -> List[Entity]:
    """Keep the longest span when spans overlap; earlier rules win ties."""
    ordered = sorted(entities, key=lambda e: (-(e.end - e.start), e.start))
    kept: List[Entity] = []
    for e in ordered:
        if all(e.end <= k.start or e.start >= k.end for k in kept):
            kept.append(e)
    return sorted(kept, key=lambda e: e.start)


# ------------------------------------------------------------------ AI layer
class NerModel:
    def __init__(self) -> None:
        self.ready = False
        self.error: Optional[str] = None
        self.provider = "none"
        self.session = None
        self.tokenizer = None
        self.id2label = {}
        self._load()

    def _fetch_files(self):
        from huggingface_hub import hf_hub_download

        os.makedirs(MODEL_DIR, exist_ok=True)
        model_path = hf_hub_download(MODEL_REPO, MODEL_FILE, local_dir=MODEL_DIR)
        tok_path = hf_hub_download(MODEL_REPO, "tokenizer.json", local_dir=MODEL_DIR)
        cfg_path = hf_hub_download(MODEL_REPO, "config.json", local_dir=MODEL_DIR)
        return model_path, tok_path, cfg_path

    def _make_session(self, model_path: str):
        import onnxruntime as ort

        available = ort.get_available_providers()
        if "QNNExecutionProvider" in available:
            try:
                sess = ort.InferenceSession(
                    model_path,
                    providers=[("QNNExecutionProvider", {"backend_path": "QnnHtp.dll"}), "CPUExecutionProvider"],
                )
                self.provider = "Snapdragon NPU (QNN / Hexagon)" if sess.get_providers()[0] == "QNNExecutionProvider" else "CPU (QNN fallback)"
                return sess
            except Exception as exc:  # QNN present but model/op not accepted
                self.error = f"QNN init failed, using CPU: {exc}"
        self.provider = "CPU"
        return ort.InferenceSession(model_path, providers=["CPUExecutionProvider"])

    def _load(self) -> None:
        try:
            from tokenizers import Tokenizer

            model_path, tok_path, cfg_path = self._fetch_files()
            self.tokenizer = Tokenizer.from_file(tok_path)
            with open(cfg_path, "r", encoding="utf-8") as f:
                self.id2label = {int(k): v for k, v in json.load(f)["id2label"].items()}
            self.session = self._make_session(model_path)
            self.input_names = [i.name for i in self.session.get_inputs()]
            self.cls_id = self.tokenizer.token_to_id("[CLS]")
            self.sep_id = self.tokenizer.token_to_id("[SEP]")
            self.ready = True
        except Exception as exc:
            self.ready = False
            self.error = f"AI model unavailable, running rules-only: {exc}"

    def predict(self, text: str) -> List[Entity]:
        if not self.ready or not text.strip():
            return []
        enc = self.tokenizer.encode(text, add_special_tokens=False)
        ids, offsets = enc.ids, enc.offsets
        step = SEQ_LEN - 2
        out: List[Entity] = []
        for i in range(0, len(ids), step):
            chunk_ids = ids[i : i + step]
            chunk_off = offsets[i : i + step]
            n = len(chunk_ids) + 2
            input_ids = np.zeros((1, SEQ_LEN), dtype=np.int64)
            mask = np.zeros((1, SEQ_LEN), dtype=np.int64)
            input_ids[0, 0] = self.cls_id
            input_ids[0, 1 : n - 1] = chunk_ids
            input_ids[0, n - 1] = self.sep_id
            mask[0, :n] = 1
            feeds = {"input_ids": input_ids, "attention_mask": mask}
            if "token_type_ids" in self.input_names:
                feeds["token_type_ids"] = np.zeros((1, SEQ_LEN), dtype=np.int64)
            logits = self.session.run(None, feeds)[0][0]
            labels = [self.id2label[int(k)] for k in logits[1 : n - 1].argmax(axis=-1)]
            out.extend(self._group(text, labels, chunk_off))
        return out

    @staticmethod
    def _group(text: str, labels: List[str], offsets) -> List[Entity]:
        ents: List[Entity] = []
        cur = None
        for lab, (s, e) in zip(labels, offsets):
            if lab == "O":
                if cur:
                    ents.append(cur)
                    cur = None
                continue
            prefix, _, etype = lab.partition("-")
            if etype not in NER_KEEP:
                if cur:
                    ents.append(cur)
                    cur = None
                continue
            if cur and cur.type == NER_KEEP[etype] and prefix == "I" and s - cur.end <= 1:
                cur.end = e
            else:
                if cur:
                    ents.append(cur)
                cur = Entity(NER_KEEP[etype], s, e, "", "model")
        if cur:
            ents.append(cur)
        for ent in ents:
            ent.text = text[ent.start : ent.end]
        return ents


# ------------------------------------------------------------------ pipeline
class Redactor:
    def __init__(self, use_model: bool = True) -> None:
        self.ner = NerModel() if use_model else None

    @property
    def status(self) -> dict:
        ready = bool(self.ner and self.ner.ready)
        return {
            "mode": "rules + AI model" if ready else "rules only",
            "ai_ready": ready,
            "provider": self.ner.provider if ready else "CPU (regex rules)",
            "model": MODEL_REPO if ready else None,
            "note": (self.ner.error if self.ner else None),
        }

    def redact(self, text: str) -> dict:
        t0 = time.perf_counter()
        ents = _rule_entities(text)
        if self.ner and self.ner.ready:
            # Drop short ALL-CAPS "organisations" such as SSN/MRN/DOB: they are
            # field labels in clinical notes, not organisation names.
            ents += [
                e for e in self.ner.predict(text)
                if not (e.type == "ORGANIZATION" and e.text.isupper() and len(e.text) <= 5)
            ]
        ents = _resolve_overlaps(ents)

        pieces, last = [], 0
        for e in ents:
            pieces.append(text[last : e.start])
            pieces.append(f"[{e.type}]")
            last = e.end
        pieces.append(text[last:])
        elapsed = (time.perf_counter() - t0) * 1000

        counts: dict = {}
        for e in ents:
            counts[e.type] = counts.get(e.type, 0) + 1
        return {
            "redacted_text": "".join(pieces),
            "entities": [asdict(e) for e in ents],
            "counts": counts,
            "total": len(ents),
            "latency_ms": round(elapsed, 1),
            **self.status,
        }
