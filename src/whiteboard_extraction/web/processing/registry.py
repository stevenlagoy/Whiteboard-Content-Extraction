"""Choose the processing engine by name. Add an entry to register a new engine."""

from __future__ import annotations

from collections.abc import Callable

from ..config import AppConfig
from .base import ProcessingEngine
from .pipeline import PipelineEngine
from .simulated import SimulatedEngine


def _recognizer(config: AppConfig):
    # Imported on demand: the CNN backend needs PyTorch.
    if config.recognizer == "ocr":
        from whiteboard_extraction.recognition.ocr_pretrained import PretrainedOCRRecognizer
        return PretrainedOCRRecognizer()
    if config.recognizer == "llm":
        from whiteboard_extraction.recognition.llm_multimodal import MultimodalLLMRecognizer
        return MultimodalLLMRecognizer(config.llm_model)
    if config.recognizer == "cnn":
        from whiteboard_extraction.recognition.cnn import CustomCNNRecognizer
        return CustomCNNRecognizer(config.cnn_checkpoint)
    raise ValueError(f"Unknown recognizer {config.recognizer!r}; use ocr, llm, or cnn.")


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
