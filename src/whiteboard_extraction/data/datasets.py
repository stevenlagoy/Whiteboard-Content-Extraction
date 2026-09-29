"""
Dataset loading and vocabulary definitions

This module holds paths to labeled test footage and any training-pair datasets needed for the
custom CNN backend. There is no fixed character vocabulary anymore: OCR and multimodal backends
recognize open-vocabulary text, and the custom CNN is evaluated against the same labeled test set
as the other methods.
"""

from pathlib import Path

# Root directory for raw and processed data
DATA_DIR = Path(__file__).resolve().parents[3] / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
TEST_FOOTAGE_DIR = RAW_DIR / "test_footage"
