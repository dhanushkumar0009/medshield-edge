@echo off
REM MedShield Edge - one-click start (Windows)
IF NOT EXIST .venv ( py -3.11 -m venv .venv )
call .venv\Scripts\activate
pip install -r requirements.txt
REM On a Snapdragon-powered PC, swap the CPU runtime for the NPU (QNN) runtime:
REM   pip uninstall -y onnxruntime && pip install onnxruntime-qnn
python app.py
