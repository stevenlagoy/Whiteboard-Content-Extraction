"""Present evaluation results: one Markdown comparison table, confusion matrices, and JSON files."""

from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import asdict
from pathlib import Path
from typing import Any

from whiteboard_extraction.recognition.base import RecognizedBlock

from .metrics import MethodMetrics
from .runner import MethodResult, SampleResult

HEADERS = (
    "Method", "Samples", "Blocks ref/pred", "Accuracy", "Precision", "Recall", "F1",
    "WER", "CER", "Exact match", "Time (s)",
)


def _cell(value: float) -> str:
    return f"{value:.3f}"


def _text(value: str) -> str:
    """Keep a value on one table row: Markdown pipes and newlines would break the table."""
    return value.replace("|", "\\|").replace("\n", " ")


def markdown_table(results: Sequence[MethodResult]) -> str:
    """One row per method, in the order given; methods that could not run show n/a."""
    lines = ["| " + " | ".join(HEADERS) + " |", "|" + "|".join("---" for _ in HEADERS) + "|"]
    for result in results:
        metrics = result.metrics
        if metrics is None:
            cells = [_text(result.method_name), *(["n/a"] * (len(HEADERS) - 2)), f"{result.seconds:.1f}"]
        else:
            cells = [
                _text(metrics.method_name),
                str(metrics.samples),
                f"{metrics.reference_blocks}/{metrics.predicted_blocks}",
                _cell(metrics.accuracy),
                _cell(metrics.precision),
                _cell(metrics.recall),
                _cell(metrics.f1),
                _cell(metrics.word_error_rate),
                _cell(metrics.char_error_rate),
                _cell(metrics.exact_match),
                f"{result.seconds:.1f}",
            ]
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def confusion_table(metrics: MethodMetrics) -> str:
    """The kind confusion matrix as Markdown: rows are reference kinds, columns predicted kinds."""
    labels = metrics.confusion_labels
    lines = [
        "| reference \\ predicted | " + " | ".join(labels) + " |",
        "|---|" + "|".join("---" for _ in labels) + "|",
    ]
    for label, row in zip(labels, metrics.confusion):
        lines.append(f"| {label} | " + " | ".join(str(count) for count in row) + " |")
    return "\n".join(lines)


def _source(result: MethodResult, fallback: str | None) -> str:
    label = result.test_set or fallback or "unknown test set"
    return f"{label} (casefold)" if result.casefold else label


def render_report(results: Sequence[MethodResult], test_set: str | None = None) -> str:
    """The full text report: where each run came from, the table, failures, confusion matrices."""
    parts: list[str] = []
    sources = {_source(result, test_set) for result in results}
    if len(sources) == 1:
        parts.append(f"Test set: {next(iter(sources))}")
    elif sources:
        parts.append("Sources (runs come from different test sets or settings):\n"
                     + "\n".join(f"- {result.method_name}: {_source(result, test_set)}" for result in results))
    parts.append(markdown_table(results))
    failed = [result for result in results if result.metrics is None]
    if failed:
        parts.append("Not scored:\n" + "\n".join(f"- {result.method_name}: {result.error}" for result in failed))
    partial = [result for result in results if result.metrics is not None and result.failed_samples]
    if partial:
        parts.append("Sample failures (scored as if the method returned nothing):\n" + "\n".join(
            f"- {result.method_name}: {len(result.failed_samples)} of {len(result.samples)} samples failed, "
            f"e.g. {result.failed_samples[0].name}: {result.failed_samples[0].error}"
            for result in partial
        ))
    for result in results:
        if result.metrics is not None:
            parts.append(
                f"Confusion matrix for {result.method_name} (rows: reference kind, columns: predicted kind):\n"
                + confusion_table(result.metrics)
            )
    return "\n\n".join(parts)


# -------------------------------------------------------------------------------------- JSON

def _block_to_json(block: RecognizedBlock) -> dict[str, Any]:
    return {"text": block.text, "kind": block.kind, "timestamp": block.timestamp, "confidence": block.confidence}


def _block_from_json(data: dict[str, Any]) -> RecognizedBlock:
    return RecognizedBlock(
        text=data["text"], kind=data["kind"], timestamp=float(data.get("timestamp", 0.0)),
        confidence=data.get("confidence"),
    )


def results_to_json(results: Sequence[MethodResult], test_set: str | None = None) -> dict[str, Any]:
    return {
        "test_set": test_set,
        "methods": [
            {
                "method_name": result.method_name,
                "test_set": result.test_set or test_set,
                "casefold": result.casefold,
                "metrics": asdict(result.metrics) if result.metrics is not None else None,
                "seconds": result.seconds,
                "error": result.error,
                "samples": [
                    {
                        "name": sample.name,
                        "seconds": sample.seconds,
                        "error": sample.error,
                        "reference": [_block_to_json(block) for block in sample.reference],
                        "predicted": [_block_to_json(block) for block in sample.predicted],
                    }
                    for sample in result.samples
                ],
            }
            for result in results
        ],
    }


def results_from_json(data: dict[str, Any]) -> list[MethodResult]:
    results: list[MethodResult] = []
    for entry in data["methods"]:
        metrics = None
        if entry.get("metrics") is not None:
            raw = dict(entry["metrics"])
            raw["confusion_labels"] = tuple(raw["confusion_labels"])
            metrics = MethodMetrics(**raw)
        samples = [
            SampleResult(
                name=sample["name"],
                reference=tuple(_block_from_json(block) for block in sample["reference"]),
                predicted=tuple(_block_from_json(block) for block in sample["predicted"]),
                seconds=float(sample.get("seconds", 0.0)),
                error=sample.get("error"),
            )
            for sample in entry.get("samples", [])
        ]
        results.append(MethodResult(
            entry["method_name"], metrics, samples, float(entry.get("seconds", 0.0)), entry.get("error"),
            test_set=entry.get("test_set", data.get("test_set")), casefold=bool(entry.get("casefold", False)),
        ))
    return results


def save_results(path: Path, results: Sequence[MethodResult], test_set: str | None = None) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(results_to_json(results, test_set), indent=2, ensure_ascii=False), encoding="utf-8")
    return path


def load_results(path: Path) -> list[MethodResult]:
    """Read a file written by save_results. Raises ValueError when the file is not a results file."""
    try:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        return results_from_json(data)
    except (json.JSONDecodeError, KeyError, TypeError, AttributeError) as error:
        raise ValueError(f"{path} is not an evaluation results file ({type(error).__name__}: {error}).") from None
