# YOLOv8 Tracking Benchmark (CPU / MPS / CoreML)

This repository contains scripts to benchmark **YOLOv8 tracking** (`model.track` with ByteTrack) across different runtimes on Apple Silicon (M4 Pro).  
The goal is to compare performance between **CPU**, **MPS (GPU)**, and **CoreML**, as well as virtualization (Parallels, UTM).

## Contents
- `export_coreml.py` — converts a PyTorch `.pt` YOLOv8 model to CoreML `.mlpackage`
- `yolo_benchmarking.py` — benchmarking script with FPS reporting (pipeline vs. inference)

## Requirements
```bash
python -m venv .venv && source .venv/bin/activate
pip install ultralytics opencv-python torch coremltools
