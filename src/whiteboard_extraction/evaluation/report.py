"""Present evaluation results: one Markdown comparison table, confusion matrices, and JSON files."""

from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import asdict
from numbers import Real
from pathlib import Path
from typing import Any

from whiteboard_extraction.recognition.base import KINDS, RecognizedBlock

from .metrics import MIN_SIMILARITY, MethodMetrics, kind_scores, score_sample, summarize
from .runner import MethodResult, SampleResult

HEADERS = (
    "Method", "Samples", "Blocks ref/pred", "Accuracy", "Precision", "Recall", "F1",
    "WER", "CER", "Exact match", "s/image",
)
KIND_HEADERS = ("Method", "Kind", "Blocks", "Accuracy", "Precision", "Recall", "F1", "WER", "CER", "Exact match")
NOT_AVAILABLE = "n/a"


def _cell(value: float) -> str:
    return f"{value:.3f}"


def _seconds(result: MethodResult) -> str:
    per_image = result.seconds_per_image
    return NOT_AVAILABLE if per_image is None else f"{per_image:.2f}"


def _rate(errors: int, total: int) -> str:
    """An error rate, or n/a when there is no reference text of this kind to divide by."""
    return _cell(errors / total) if total else NOT_AVAILABLE


def _text(value: str) -> str:
    """Keep a value on one table row: Markdown pipes and newlines would break the table."""
    return value.replace("|", "\\|").replace("\n", " ")


def markdown_table(results: Sequence[MethodResult]) -> str:
    """One row per method, in the order given; methods that could not run show n/a."""
    lines = ["| " + " | ".join(HEADERS) + " |", "|" + "|".join("---" for _ in HEADERS) + "|"]
    for result in results:
        metrics = result.metrics
        if metrics is None:
            cells = [_text(result.method_name), *([NOT_AVAILABLE] * (len(HEADERS) - 2)), _seconds(result)]
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
                _seconds(result),
            ]
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def kind_table(results: Sequence[MethodResult]) -> str:
    """Equations and prose scored separately (user story 7): one row per method and kind.

    Accuracy, precision, recall and F1 say how well the method tells the kind; WER, CER and exact
    match say how well it read the blocks of that kind. A kind that is neither in the labels nor
    in the method's output has no row.
    """
    lines = ["| " + " | ".join(KIND_HEADERS) + " |", "|" + "|".join("---" for _ in KIND_HEADERS) + "|"]
    for result in results:
        metrics = result.metrics
        if metrics is None:
            continue
        scores = kind_scores(metrics)
        for kind in KINDS:
            score, counts = scores[kind], metrics.by_kind.get(kind)
            if not score["actual"] and not score["predicted"]:
                continue
            if counts is None:
                blocks, text = str(score["actual"]), [NOT_AVAILABLE] * 3
            else:
                blocks = str(counts["blocks"])
                text = [
                    _rate(counts["word_errors"], counts["reference_words"]),
                    _rate(counts["char_errors"], counts["reference_chars"]),
                    _cell(counts["exact_matches"] / counts["blocks"]) if counts["blocks"] else NOT_AVAILABLE,
                ]
            lines.append("| " + " | ".join([
                _text(metrics.method_name), kind, blocks, _cell(score["accuracy"]),
                _cell(score["precision"]), _cell(score["recall"]), _cell(score["f1"]), *text,
            ]) + " |")
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
    """Where a run came from and the settings that change its numbers."""
    label = result.test_set or fallback or "unknown test set"
    settings = []
    if result.casefold:
        settings.append("casefold")
    if result.min_similarity != MIN_SIMILARITY:
        settings.append(f"min similarity {result.min_similarity:g}")
    return f"{label} ({', '.join(settings)})" if settings else label


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
    if any(result.metrics is not None for result in results):
        parts.append("By kind (equations and prose scored separately):\n" + kind_table(results))
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
    skipped = [result for result in results if result.skipped]
    if skipped:
        parts.append("Unreadable images (left out of the scores):\n" + "\n".join(
            f"- {result.method_name}: {', '.join(result.skipped)}" for result in skipped
        ))
    for result in results:
        if result.metrics is not None:
            parts.append(
                f"Confusion matrix for {result.method_name} (rows: reference kind, columns: predicted kind):\n"
                + confusion_table(result.metrics)
            )
    return "\n\n".join(parts)


# -------------------------------------------------------------------------------------- JSON

def _plain(value: Any) -> Any:
    """A NumPy number becomes a float so json can write it; anything else is left as it is."""
    return float(value) if isinstance(value, Real) and not isinstance(value, bool) else value


def _block_to_json(block: RecognizedBlock) -> dict[str, Any]:
    return {"text": block.text, "kind": block.kind, "timestamp": _plain(block.timestamp),
            "confidence": _plain(block.confidence)}


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
                "min_similarity": result.min_similarity,
                "skipped": list(result.skipped),
                "timed_per_image": result.timed_per_image,
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


def _stored_metrics(raw: dict[str, Any]) -> MethodMetrics:
    """The metrics as they were saved, for a file that kept no predictions to score again."""
    known = {name: value for name, value in raw.items() if name in MethodMetrics.__dataclass_fields__}
    known["confusion_labels"] = tuple(known["confusion_labels"])
    return MethodMetrics(**known)


def results_from_json(data: dict[str, Any]) -> list[MethodResult]:
    """Rebuild saved runs. A scored run is scored again from its saved predictions, so every run in
    a table follows the rules of the code that prints it, whichever version saved the file."""
    results: list[MethodResult] = []
    for entry in data["methods"]:
        casefold = bool(entry.get("casefold", False))
        min_similarity = float(entry.get("min_similarity", MIN_SIMILARITY))
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
        skipped = list(entry.get("skipped", []))
        if "skipped" not in entry:  # saved when an unreadable image still counted against the method
            unreadable = [sample for sample in samples if (sample.error or "").startswith("TestSetError")]
            skipped = [sample.name for sample in unreadable]
            samples = [sample for sample in samples if sample not in unreadable]
        metrics = None
        if entry.get("metrics") is not None:
            if samples:
                metrics = summarize(entry["method_name"], [
                    score_sample(sample.name, sample.reference, sample.predicted, casefold, min_similarity)
                    for sample in samples
                ])
            else:
                metrics = _stored_metrics(entry["metrics"])
        results.append(MethodResult(
            entry["method_name"], metrics, samples, float(entry.get("seconds", 0.0)), entry.get("error"),
            test_set=entry.get("test_set", data.get("test_set")), casefold=casefold,
            min_similarity=min_similarity, skipped=skipped,
            timed_per_image=bool(entry.get("timed_per_image", False)),
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
