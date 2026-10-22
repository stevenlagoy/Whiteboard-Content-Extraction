"""Recognition methods behind one swappable interface. See docs/RECOGNITION-INTERFACE.md."""

from .base import KINDS, RecognizedBlock, Recognizer
from .pipeline import Keyframe, ProgressCallback, recognize_keyframes
from .registry import (
    DEFAULT_RECOGNIZER, ENV_SETTINGS, RECOGNIZER_NAMES, SETTING_NAMES, RecognizerOptions, build_recognizer,
    recognizer_from_env,
)

__all__ = [
    "DEFAULT_RECOGNIZER", "ENV_SETTINGS", "KINDS", "Keyframe", "ProgressCallback", "RECOGNIZER_NAMES", "SETTING_NAMES",
    "RecognizedBlock", "Recognizer", "RecognizerOptions", "build_recognizer", "recognize_keyframes", "recognizer_from_env",
]
