"""Choose the processing engine by name. Add an entry to register a new engine."""

from __future__ import annotations

from collections.abc import Callable

from whiteboard_extraction.recognition.base import Recognizer
from whiteboard_extraction.recognition.registry import build_recognizer

from ..config import AppConfig
from .base import ProcessingEngine
from .pipeline import PipelineEngine
from .simulated import SimulatedEngine


def _recognizer(config: AppConfig) -> Recognizer:
    # The recognition registry is the only list of methods. It imports a backend when that
    # backend is asked for, so the app still starts without PyTorch unless the CNN is selected.
    return build_recognizer(config.recognizer, llm_model=config.llm_model, cnn_checkpoint=config.cnn_checkpoint)


ENGINE_FACTORIES: dict[str, Callable[[AppConfig], ProcessingEngine]] = {
    "simulated": lambda config: SimulatedEngine(),
    "pipeline": lambda config: PipelineEngine(_recognizer(config)),
}


def build_engine(config: AppConfig) -> ProcessingEngine:
    try:
        factory = ENGINE_FACTORIES[config.engine]
    except KeyError:
        known = ", ".join(sorted(ENGINE_FACTORIES))
        raise ValueError(f"Unknown engine {config.engine!r}; use one of: {known}.") from None
    return factory(config)
