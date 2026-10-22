"""Choose a recognition method by name, so nothing else imports a concrete backend.

This is the single configuration point for user story 13. The video pipeline, the evaluation
script and the web app ask for a method by name (``build_recognizer("llm")``) or let the
environment decide (``recognizer_from_env()`` reads ``WCE_RECOGNIZER``). Swapping methods
changes one value here; preprocessing, reconstruction and export code never change.

Adding a method takes two steps: write a class with ``recognize(timestamp, image)`` (see
``base.Recognizer``) and add one entry to ``_FACTORIES`` below. A new setting for it is one
field on ``RecognizerOptions`` plus one entry in ``ENV_SETTINGS``.
"""

from __future__ import annotations

import os
from collections.abc import Callable, Mapping
from dataclasses import dataclass, fields

from .base import Recognizer

DEFAULT_RECOGNIZER = "ocr"
ENV_RECOGNIZER = "WCE_RECOGNIZER"

# RecognizerOptions field -> environment variable that sets it.
ENV_SETTINGS: dict[str, str] = {
    "ocr_engine": "WCE_OCR_ENGINE",
    "llm_model": "WCE_LLM_MODEL",
    "cnn_checkpoint": "WCE_CNN_CHECKPOINT",
}


@dataclass(frozen=True)
class RecognizerOptions:
    """Settings a method may need. Each backend reads only its own field and ignores the rest."""

    ocr_engine: str = "tesseract"
    llm_model: str = ""
    cnn_checkpoint: str = ""

    @classmethod
    def from_env(cls, env: Mapping[str, str]) -> RecognizerOptions:
        """Read every setting that is present in env; the rest keep their defaults."""
        return cls(**{field: env[name] for field, name in ENV_SETTINGS.items() if name in env})


SETTING_NAMES = tuple(field.name for field in fields(RecognizerOptions))


# Backends are imported inside their factory so an unused one (and PyTorch for the CNN) is never loaded.

def _ocr(options: RecognizerOptions) -> Recognizer:
    from .ocr_pretrained import PretrainedOCRRecognizer
    return PretrainedOCRRecognizer(options.ocr_engine)


def _llm(options: RecognizerOptions) -> Recognizer:
    from .llm_multimodal import MultimodalLLMRecognizer
    return MultimodalLLMRecognizer(options.llm_model)


def _cnn(options: RecognizerOptions) -> Recognizer:
    from .cnn import CustomCNNRecognizer
    return CustomCNNRecognizer(options.cnn_checkpoint)


_FACTORIES: dict[str, Callable[[RecognizerOptions], Recognizer]] = {"ocr": _ocr, "llm": _llm, "cnn": _cnn}
RECOGNIZER_NAMES = tuple(_FACTORIES)


def build_recognizer(name: str, **settings: str) -> Recognizer:
    """Instantiate a method by name, e.g. ``build_recognizer("llm", llm_model="...")``.

    Settings are the fields of RecognizerOptions; unset ones keep their defaults.
    """
    unknown = sorted(set(settings) - set(SETTING_NAMES))
    if unknown:
        raise ValueError(f"Unknown recognizer setting(s) {', '.join(unknown)}; choose from {', '.join(SETTING_NAMES)}.")
    return _build(name, RecognizerOptions(**settings))


def recognizer_from_env(env: Mapping[str, str] | None = None) -> Recognizer:
    """Build the method the environment selects: WCE_RECOGNIZER names it, the other WCE_* variables configure it."""
    env = os.environ if env is None else env
    return _build(env.get(ENV_RECOGNIZER, DEFAULT_RECOGNIZER), RecognizerOptions.from_env(env))


def _build(name: str, options: RecognizerOptions) -> Recognizer:
    try:
        factory = _FACTORIES[name]
    except KeyError:
        raise ValueError(f"Unknown recognizer {name!r}; choose one of {', '.join(_FACTORIES)}.") from None
    recognizer = factory(options)
    if not isinstance(recognizer, Recognizer):
        raise TypeError(
            f"Recognizer {name!r} ({type(recognizer).__name__}) does not implement base.Recognizer: "
            "it needs a recognize(timestamp, image) method."
        )
    return recognizer
