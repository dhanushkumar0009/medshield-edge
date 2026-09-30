"""MedShield Edge - local web app. Everything runs on this machine; no cloud calls at inference time."""
import os

from flask import Flask, jsonify, request, send_from_directory

from redactor import Redactor

app = Flask(__name__, static_folder="static")
USE_MODEL = os.environ.get("MEDSHIELD_RULES_ONLY", "0") != "1"
redactor = Redactor(use_model=USE_MODEL)


@app.get("/")
def index():
    return send_from_directory("static", "index.html")


@app.get("/api/status")
def status():
    return jsonify(redactor.status)


@app.post("/api/redact")
def redact():
    data = request.get_json(silent=True) or {}
    text = (data.get("text") or "")[:20000]
    if not text.strip():
        return jsonify({"error": "Please paste a clinical note first."}), 400
    return jsonify(redactor.redact(text))


@app.get("/api/sample")
def sample():
    path = os.path.join(os.path.dirname(__file__), "samples", "sample_note.txt")
    with open(path, "r", encoding="utf-8") as f:
        return jsonify({"text": f.read()})


if __name__ == "__main__":
    print("MedShield Edge running at http://127.0.0.1:5000  (local only)")
    app.run(host="127.0.0.1", port=5000, debug=False)
