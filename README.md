# Whiteboard Content Extraction: An Accessibility Tool for Lecture Video

CS 59300: Application of Deep Learning, Fall 2026

## Team

- Steven LaGoy, lagosm01@pfw.edu
- Dalton Lybarger, lybads01@pfw.edu
- Pranav Rao, raops01@pfw.edu

## Overview

A deep learning application that extracts handwritten content from recorded lecture video and converts it into an accessible, machine-readable format. The system accepts a full lecture recording, locates the whiteboard or blackboard within the frame, and reconstructs the board's content over time as it is written, including equations and prose. Output is exported as a Word document and a PDF and delivered through a Flask web application usable from a computer or a mobile device.

For the full requirements, stakeholders, and system design, see [`docs/PROJECT-PROPOSAL.md`](docs/PROJECT-PROPOSAL.md).

## Pipeline

1. Accept an uploaded lecture video (up to 150 minutes).
2. Sample frames and segment the whiteboard/blackboard from the rest of the scene.
3. Reduce sampled frames to stable, timestamped keyframes.
4. Recognize equations and prose on each keyframe with a swappable DL backend.
5. Reconstruct recognized blocks into one ordered, timestamped document.
6. Export the document to Word and PDF (LaTeX as a nice-to-have).
7. Serve the whole flow through a Flask web app.

## Project Structure

```
whiteboard-content-extraction/
├── data/
│   ├── raw/                     # Untracked raw video and test footage
│   └── processed/               # Untracked processed data
├── docs/
│   └── project-proposal.md
├── scripts/
│   ├── train.py                 # Train the custom CNN recognition backend
│   ├── evaluate.py              # Run accuracy/precision/recall/F1/WER across DL methods
│   └── benchmark.py             # Measure end-to-end processing time vs. video length
├── src/
│   └── whiteboard_extraction/
│       ├── video/               # Frame sampling, board segmentation, keyframe extraction
│       ├── recognition/         # Swappable DL backends: OCR, custom CNN, multimodal LLM
│       ├── reconstruction/      # Assembling recognized blocks into one ordered document
│       ├── export/              # Word, PDF, and LaTeX export
│       ├── evaluation/          # Recognition metrics
│       ├── web/                 # Flask app, templates, static assets
│       ├── data/                # Dataset paths and loading
│       └── utils/               # Shared helpers
├── tests/
├── pyproject.toml
├── requirements.txt
└── .gitignore
```

## Setup

**Linux / macOS**
```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install -e .
```

**Windows (cmd)**
```cmd
python -m venv .venv
.venv\Scripts\activate.bat
pip install -r requirements.txt
pip install -e .
```

**Windows (PowerShell)**
```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
pip install -e .
```

## Run the web app

The simplest cross-platform option, once the virtual environment is active:

```bash
python src/whiteboard_extraction/web/app.py
```

Or use the Flask CLI:

**Linux / macOS**
```bash
export FLASK_APP=src/whiteboard_extraction/web/app.py
flask run
```

**Windows (cmd)**
```cmd
set FLASK_APP=src\whiteboard_extraction\web\app.py
flask run
```

**Windows (PowerShell)**
```powershell
$env:FLASK_APP = "src\whiteboard_extraction\web\app.py"
flask run
```

Visit `http://127.0.0.1:5000` in a browser to upload a video.

## Run tests

Same command on every platform, once the virtual environment is active:

```bash
pytest
```

## Run scripts

Also the same across platforms:

```bash
python scripts/train.py
python scripts/evaluate.py
python scripts/benchmark.py
```

## Status

Core pipeline stages (video sampling, segmentation, recognition backends, reconstruction, export, and the Flask shell) are scaffolded but not yet implemented. See `docs/project-proposal.md` for current requirements and the task list leading up to the midterm review.
