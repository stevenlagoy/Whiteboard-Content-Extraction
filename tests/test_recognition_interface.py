"""Tests for the swappable recognition interface (#11): the protocol, the registry, and the call site."""

from __future__ import annotations

import importlib.util

import numpy as np
import pytest

from whiteboard_extraction.export.docx_export import export_docx
from whiteboard_extraction.export.pdf_export import export_pdf
from whiteboard_extraction.recognition import (
    KINDS, RECOGNIZER_NAMES, RecognizedBlock, Recognizer, RecognizerOptions, build_recognizer,
    recognize_keyframes, recognizer_from_env, registry,
)
from whiteboard_extraction.reconstruction.document import build_document


def frame(height: int = 8, width: int = 8) -> np.ndarray:
    return np.zeros((height, width, 3), dtype=np.uint8)


KEYFRAMES = [(0.0, frame()), (30.0, frame()), (65.0, frame())]


class CountingRecognizer:
    """Method one: a numbered prose block on every keyframe."""

    def __init__(self) -> None:
        self.calls: list[float] = []

    def recognize(self, timestamp: float, image: np.ndarray) -> list[RecognizedBlock]:
        self.calls.append(timestamp)
        return [RecognizedBlock(f"frame {len(self.calls)}", "prose", timestamp, 0.9)]


class EquationRecognizer:
    """Method two: an equation after the first minute, nothing before."""

    def recognize(self, timestamp: float, image: np.ndarray) -> list[RecognizedBlock]:
        if timestamp < 60:
            return []
        return [RecognizedBlock(r"E = mc^2", "equation", timestamp)]


class NotARecognizer:
    def detect(self, image):
        return []


# ------------------------------------------------------------------------------ protocol

def test_any_object_with_recognize_satisfies_the_protocol():
    assert isinstance(CountingRecognizer(), Recognizer)
    assert isinstance(EquationRecognizer(), Recognizer)
    assert not isinstance(NotARecognizer(), Recognizer)


def test_kinds_are_the_two_the_pipeline_knows():
    assert KINDS == ("equation", "prose")


# ----------------------------------------------------------------------------- call site

def test_two_methods_run_through_the_same_call_site():
    counting = recognize_keyframes(CountingRecognizer(), KEYFRAMES)
    equations = recognize_keyframes(EquationRecognizer(), KEYFRAMES)
    assert [len(blocks) for blocks in counting] == [1, 1, 1]
    assert [len(blocks) for blocks in equations] == [0, 0, 1]
    assert [block.timestamp for blocks in counting for block in blocks] == [0.0, 30.0, 65.0]


def test_call_site_reports_progress_in_order():
    seen: list[tuple[int, int]] = []
    recognize_keyframes(CountingRecognizer(), KEYFRAMES, on_progress=lambda done, total: seen.append((done, total)))
    assert seen == [(1, 3), (2, 3), (3, 3)]


def test_call_site_rejects_unknown_kinds():
    class Wrong:
        def recognize(self, timestamp, image):
            return [RecognizedBlock("x", "heading", timestamp)]

    with pytest.raises(ValueError, match="heading"):
        recognize_keyframes(Wrong(), KEYFRAMES[:1])


def test_downstream_code_is_untouched_by_the_choice_of_method(tmp_path):
    """Reconstruction and both exports consume whatever a method produced, without knowing which one ran."""
    for name, method in (("counting", CountingRecognizer()), ("equation", EquationRecognizer())):
        document = build_document(recognize_keyframes(method, KEYFRAMES))
        assert export_docx(document, tmp_path / f"{name}.docx").stat().st_size > 0
        assert export_pdf(document, tmp_path / f"{name}.pdf").stat().st_size > 0
    assert build_document(recognize_keyframes(EquationRecognizer(), KEYFRAMES)).as_plain_text() == "[65s] E = mc^2"


# ------------------------------------------------------------------------------- registry

def test_built_in_names():
    assert RECOGNIZER_NAMES == ("ocr", "llm", "cnn")


def test_build_recognizer_instantiates_backends_with_their_settings():
    assert build_recognizer("ocr", ocr_engine="easyocr").engine == "easyocr"
    assert build_recognizer("llm", llm_model="some-model").model == "some-model"


@pytest.mark.skipif(importlib.util.find_spec("torch") is None, reason="the CNN backend needs PyTorch")
def test_build_recognizer_cnn():
    assert isinstance(build_recognizer("cnn", cnn_checkpoint="x.pt"), Recognizer)


def test_build_recognizer_rejects_unknown_names():
    with pytest.raises(ValueError, match="choose one of ocr, llm, cnn"):
        build_recognizer("nope")


def test_build_recognizer_rejects_factories_that_return_the_wrong_thing(monkeypatch):
    monkeypatch.setitem(registry._FACTORIES, "broken", lambda options: NotARecognizer())
    with pytest.raises(TypeError, match="recognize"):
        build_recognizer("broken")


def test_a_new_method_is_one_registry_entry(monkeypatch):
    monkeypatch.setitem(registry._FACTORIES, "counting", lambda options: CountingRecognizer())
    assert isinstance(build_recognizer("counting"), CountingRecognizer)
    assert isinstance(recognizer_from_env({"WCE_RECOGNIZER": "counting"}), CountingRecognizer)


def test_environment_selects_the_method_and_its_settings():
    assert recognizer_from_env({"WCE_RECOGNIZER": "llm", "WCE_LLM_MODEL": "m-1"}).model == "m-1"
    assert recognizer_from_env({}).engine == "tesseract"  # the default method is ocr


def test_options_from_env_keep_defaults_for_unset_variables():
    options = RecognizerOptions.from_env({"WCE_OCR_ENGINE": "e", "WCE_CNN_CHECKPOINT": "c.pt"})
    assert options == RecognizerOptions(ocr_engine="e", llm_model="", cnn_checkpoint="c.pt")


def test_recognizer_from_env_defaults_to_the_process_environment(monkeypatch):
    monkeypatch.setenv("WCE_RECOGNIZER", "llm")
    monkeypatch.setenv("WCE_LLM_MODEL", "m-env")
    assert recognizer_from_env().model == "m-env"


def test_build_recognizer_rejects_unknown_settings():
    with pytest.raises(ValueError, match="Unknown recognizer setting"):
        build_recognizer("ocr", engine="easyocr")


def test_call_site_accepts_an_iterator_and_still_reports_totals():
    seen: list[tuple[int, int]] = []
    results = recognize_keyframes(CountingRecognizer(), iter(KEYFRAMES), on_progress=lambda d, t: seen.append((d, t)))
    assert len(results) == 3 and seen == [(1, 3), (2, 3), (3, 3)]
    assert recognize_keyframes(CountingRecognizer(), []) == []


def test_call_site_explains_a_backend_that_returns_the_wrong_type():
    class Wrong:
        def recognize(self, timestamp, image):
            return None

    with pytest.raises(TypeError, match="returned NoneType"):
        recognize_keyframes(Wrong(), KEYFRAMES[:1])
