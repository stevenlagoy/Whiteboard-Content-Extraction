"""Run a recognition method over a labeled test set and score it.

A method is anything that turns a Sample into blocks. Real recognizers are wrapped with
recognizer_predictor; two baselines are built in so the pipeline can be checked without a
model: "oracle" returns the labels (every score should be perfect) and "blank" returns
nothing (every score should be at its worst).
"""

from __future__ import annotations

import sys
import time
import traceback
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field

from whiteboard_extraction.recognition.base import RecognizedBlock, Recognizer
from whiteboard_extraction.recognition.registry import build_recognizer

from .metrics import MethodMetrics, score_sample, summarize
from .testset import Sample

Predictor = Callable[[Sample], Sequence[RecognizedBlock]]

BASELINES: dict[str, Predictor] = {
    "oracle": lambda sample: sample.blocks,
    "blank": lambda sample: (),
}


@dataclass(frozen=True)
class SampleResult:
    """What one method predicted for one image, kept so mistakes can be inspected."""

    name: str
    reference: tuple[RecognizedBlock, ...]
    predicted: tuple[RecognizedBlock, ...]
    seconds: float
    error: str | None = None  # set when the method raised on this image; predicted is then empty


@dataclass
class MethodResult:
    """One method's run: its metrics (None when it could not run) and every prediction."""

    method_name: str
    metrics: MethodMetrics | None
    samples: list[SampleResult] = field(default_factory=list)
    seconds: float = 0.0  # wall-clock time spent running the method, scoring excluded
    error: str | None = None  # set when the method could not run at all
    test_set: str | None = None
    casefold: bool = False

    @property
    def failed_samples(self) -> list[SampleResult]:
        return [sample for sample in self.samples if sample.error is not None]


def recognizer_predictor(recognizer: Recognizer) -> Predictor:
    """Adapt a Recognizer to the predictor signature the runner uses."""
    return lambda sample: recognizer.recognize(sample.timestamp, sample.load_image())


def _describe(error: BaseException) -> str:
    return f"{type(error).__name__}: {error}"


def evaluate_predictor(
    method_name: str,
    predict: Predictor,
    samples: Sequence[Sample],
    *,
    casefold: bool = False,
    test_set: str | None = None,
    clock: Callable[[], float] = time.perf_counter,
) -> MethodResult:
    """Predict every sample, then score the method.

    A method that is not implemented stops at once. One that raises on a single image has that
    image scored as if it returned nothing, with the reason kept on the SampleResult, so one bad
    frame does not void a whole run; a method that fails on every image is reported as failed.
    """
    results: list[SampleResult] = []
    started = clock()
    for sample in samples:
        before = clock()
        try:
            predicted = tuple(predict(sample))
        except NotImplementedError as error:  # a stub fails on every image: no point continuing
            return MethodResult(method_name, None, results, clock() - started, error=f"not implemented ({error})",
                                test_set=test_set, casefold=casefold)
        except Exception as error:  # noqa: BLE001 - one broken image must not stop the comparison
            traceback.print_exc(file=sys.stderr)
            results.append(SampleResult(sample.name, sample.blocks, (), clock() - before, error=_describe(error)))
            continue
        results.append(SampleResult(sample.name, sample.blocks, predicted, clock() - before))
    elapsed = clock() - started

    if results and all(result.error is not None for result in results):
        return MethodResult(method_name, None, results, elapsed, error=f"every sample failed; first: {results[0].error}",
                            test_set=test_set, casefold=casefold)
    try:
        scores = [score_sample(result.name, result.reference, result.predicted, casefold) for result in results]
    except ValueError as error:  # the method returned blocks the metrics cannot score, e.g. an unknown kind
        return MethodResult(method_name, None, results, elapsed, error=_describe(error), test_set=test_set, casefold=casefold)
    return MethodResult(method_name, summarize(method_name, scores), results, elapsed, test_set=test_set, casefold=casefold)


def evaluate_named(
    name: str,
    samples: Sequence[Sample],
    *,
    casefold: bool = False,
    test_set: str | None = None,
    **recognizer_options,
) -> MethodResult:
    """Evaluate a baseline or a registered recognizer by name."""
    if name in BASELINES:
        return evaluate_predictor(name, BASELINES[name], samples, casefold=casefold, test_set=test_set)
    try:
        recognizer = build_recognizer(name, **recognizer_options)
    except Exception as error:  # noqa: BLE001 - a backend that cannot even be built is reported, not raised
        traceback.print_exc(file=sys.stderr)
        return MethodResult(name, None, error=_describe(error), test_set=test_set, casefold=casefold)
    return evaluate_predictor(name, recognizer_predictor(recognizer), samples, casefold=casefold, test_set=test_set)
