# MedShield Edge

**On-device PHI redaction for clinical notes, built for Snapdragon-powered HP PCs.**

Clinical notes contain patient names, phone numbers, dates of birth, record numbers and more. Before a note can be shared for research, training, QA or AI tooling, that information has to be removed. Most redaction tools upload the text to a cloud service, which creates the very privacy risk they are meant to remove.

MedShield Edge does the whole job **on the laptop**. The note never leaves the device.

## How it works

1. **Rule layer** (regex): SSNs, phone numbers, emails, dates of birth, dates, MRNs, ZIP/PIN codes, URLs.
2. **AI layer**: an open-source BERT named-entity model (`Xenova/bert-base-NER`, an ONNX export of `dslim/bert-base-NER`, from Hugging Face) finds person names, locations and organisations that no regex can reliably catch.
3. Overlapping detections are merged and every hit is replaced with a label such as `[NAME]` or `[PHONE]`.

The AI layer runs through **ONNX Runtime**. On a Snapdragon-powered PC with the `onnxruntime-qnn` package, it uses the **QNN execution provider (Hexagon NPU)**. Anywhere else it falls back to CPU, and if the model cannot be loaded the app still works in rules-only mode. The screen always shows which mode and processor are in use.

Snapdragon-specific design choices:
- Fixed 128-token input windows (NPUs prefer static shapes), long notes are processed in chunks.
- INT8-quantised model (`model_quantized.onnx`) for NPU-friendly, low-power inference.
- No network calls during inference, so it works offline after the first model download.

## Quick start (Windows, Snapdragon-powered HP PC)

```bat
git clone <this repo>
cd medshield-edge
py -3.11 -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
pip uninstall -y onnxruntime
pip install onnxruntime-qnn
python app.py
```

Open http://127.0.0.1:5000, click **Load sample note**, then **Redact**.

The first run downloads the model (about 100 MB) into `models/`. After that it works offline.

On any other computer, skip the two `onnxruntime` lines and it runs on CPU. To force rules-only mode, set `MEDSHIELD_RULES_ONLY=1`.

## Project layout

```
app.py            Flask server (localhost only)
redactor.py       Rule layer + ONNX NER layer + merge logic
static/index.html Single-page UI
samples/          Fictional sample note (no real patient data)
tests/            pytest tests
```

Run the tests: `pip install pytest && pytest -q tests`

## Troubleshooting

- **Status bar says "rules only"**: the model could not be downloaded or loaded. Check your internet connection for the first run. The `note` field at `/api/status` shows the reason.
- **Status bar says "CPU (QNN fallback)"**: the NPU runtime was found but did not accept the model, so ONNX Runtime used the CPU. Results are identical, only speed differs.
- **`pip install tokenizers` fails on Windows ARM64**: use an x64 Python, or install the Rust toolchain so it can build.

## Limitations

- Automated redaction is a helper, not a guarantee. Review output before sharing it.
- Tuned for English text with US and Indian phone, date and postal formats.
- The NER model is general-purpose, not clinical-specific; fine-tuning on clinical text is the next step.
- All sample data is fictional.

## Models and licences

- Model: `dslim/bert-base-NER` (MIT), ONNX export by Xenova, via Hugging Face.
- Runtime: ONNX Runtime with the Qualcomm QNN execution provider.
