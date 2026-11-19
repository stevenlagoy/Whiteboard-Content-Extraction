"""Run a recognition method over a labeled test set and score it.

A method is anything that turns a Sample into blocks. Real recognizers go through
evaluate_recognizer; two baselines are built in so the pipeline can be checked without a
model: "oracle" returns the labels (every score should be perfect) and "blank" returns
nothing (every score in the main table should be at its worst).

Timing: the time per image covers the method's own work on one image and nothing else. For a
recognizer the image is decoded before the clock starts and, unless warm-up is turned off, the
method is called once on the first image before anything is timed, so loading a model does not
count. Building the method and scoring are never timed. Total pipeline time is a different
question and belongs in scripts/benchmark.py.
"""

from __future__ import annotations

import sys
import time
import traceback
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Any

from whiteboard_extraction.recognition.base import RecognizedBlock, Recognizer
from whiteboard_extraction.recognition.registry import build_recognizer

from .metrics import MIN_SIMILARITY, MethodMetrics, score_sample, summarize
from .testset import Sample, TestSetError

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
    seconds: float  # time the method spent on this image
    error: str | None = None  # set when the method raised on this image; predicted is then empty


@dataclass
class MethodResult:
    """One method's run: its metrics (None when it could not run) and every prediction."""

    method_name: str
    metrics: MethodMetrics | None
    samples: list[SampleResult] = field(default_factory=list)
    seconds: float = 0.0  # time the method spent on all of its images together
    error: str | None = None  # set when the method could not run at all
    test_set: str | None = None
    casefold: bool = False
    min_similarity: float = MIN_SIMILARITY
    skipped: list[str] = field(default_factory=list)  # images that could not be read; they are not scored
    timed_per_image: bool = True  # False for a run saved before only the method's own work was timed

    @property
    def failed_samples(self) -> list[SampleResult]:
        return [sample for sample in self.samples if sample.error is not None]

    @property
    def seconds_per_image(self) -> float | None:
        """Mean time over the images the method answered, or None when there is nothing comparable to report."""
        answered = [sample.seconds for sample in self.samples if sample.error is None]
        return sum(answered) / len(answered) if answered and self.timed_per_image else None


def recognizer_predictor(recognizer: Recognizer) -> Predictor:
    """Adapt a Recognizer to the predictor signature; the time it reports includes decoding the image."""
    return lambda sample: recognizer.recognize(sample.timestamp, sample.load_image())


def _describe(error: BaseException) -> str:
    return f"{type(error).__name__}: {error}"


def _run(
    method_name: str,
    samples: Sequence[Sample],
    prepare: Callable[[Sample], Any],
    predict: Callable[[Sample, Any], Sequence[RecognizedBlock]],
    *,
    casefold: bool,
    min_similarity: float,
    test_set: str | None,
    clock: Callable[[], float],
) -> MethodResult:
    """Predict every sample, then score the method. Only predict(sample, prepare(sample)) is timed."""
    settings = {"test_set": test_set, "casefold": casefold, "min_similarity": min_similarity}
    results: list[SampleResult] = []
    skipped: list[str] = []
    for sample in samples:
        before = None
        try:
            prepared = prepare(sample)
            before = clock()
            predicted = tuple(predict(sample, prepared))
        except NotImplementedError as error:  # a stub fails on every image: no point continuing
            seconds = sum(result.seconds for result in results)
            return MethodResult(method_name, None, results, seconds, f"not implemented ({error})", skipped=skipped, **settings)
        except TestSetError as error:  # the image is the problem, not the method: leave it out of the scores
            print(f"Skipping {sample.name}: {error}", file=sys.stderr)
            skipped.append(sample.name)
            continue
        except Exception as error:  # noqa: BLE001 - one broken image must not stop the comparison
            traceback.print_exc(file=sys.stderr)
            spent = clock() - before if before is not None else 0.0
            results.append(SampleResult(sample.name, sample.blocks, (), spent, error=_describe(error)))
            continue
        results.append(SampleResult(sample.name, sample.blocks, predicted, clock() - before))
    seconds = sum(result.seconds for result in results)

    if not results and skipped:
        return MethodResult(method_name, None, results, seconds, "no image could be read", skipped=skipped, **settings)
    if results and all(result.error is not None for result in results):
        error = f"every sample failed; first: {results[0].error}"
        return MethodResult(method_name, None, results, seconds, error, skipped=skipped, **settings)
    try:
        scores = [score_sample(result.name, result.reference, result.predicted, casefold, min_similarity)
                  for result in results]
    except ValueError as error:  # the method returned blocks the metrics cannot score, e.g. an unknown kind
        return MethodResult(method_name, None, results, seconds, _describe(error), skipped=skipped, **settings)
    return MethodResult(method_name, summarize(method_name, scores), results, seconds, skipped=skipped, **settings)


def evaluate_predictor(
    method_name: str,
    predict: Predictor,
    samples: Sequence[Sample],
    *,
    casefold: bool = False,
    min_similarity: float = MIN_SIMILARITY,
    test_set: str | None = None,
    clock: Callable[[], float] = time.perf_counter,
) -> MethodResult:
    """Score any function from a Sample to blocks; the whole call is timed.

    A method that is not implemented stops at once. One that raises on a single image has that
    image scored as if it returned nothing, with the reason kept on the SampleResult, so one bad
    frame does not void a whole run; a method that fails on every image is reported as failed.
    An image that cannot be read (TestSetError) is skipped and does not count for or against it.
    """
    return _run(method_name, samples, lambda sample: None, lambda sample, _: predict(sample),
                casefold=casefold, min_similarity=min_similarity, test_set=test_set, clock=clock)


def evaluate_recognizer(
    method_name: str,
    recognizer: Recognizer,
    samples: Sequence[Sample],
    *,
    casefold: bool = False,
    min_similarity: float = MIN_SIMILARITY,
    test_set: str | None = None,
    warmup: bool = True,
    clock: Callable[[], float] = time.perf_counter,
) -> MethodResult:
    """Score a Recognizer, timing only its recognize calls (see the module docstring).

    The warm-up is one extra recognize call on the first readable image; its result is thrown
    away. Turn it off for a method that is billed per call.
    """
    if warmup:
        for sample in samples:
            try:
                image = sample.load_image()
            except Exception:  # noqa: BLE001 - try the next image; the scored run below reports whatever is wrong
                continue
            try:
                recognizer.recognize(sample.timestamp, image)
            except NotImplementedError as error:
                return MethodResult(method_name, None, error=f"not implemented ({error})", test_set=test_set,
                                    casefold=casefold, min_similarity=min_similarity)
            except Exception:  # noqa: BLE001 - the scored run below reports whatever is wrong
                pass
            break
    return _run(method_name, samples, Sample.load_image,
                lambda sample, image: recognizer.recognize(sample.timestamp, image),
                casefold=casefold, min_similarity=min_similarity, test_set=test_set, clock=clock)


def evaluate_named(
    name: str,
    samples: Sequence[Sample],
    *,
    casefold: bool = False,
    min_similarity: float = MIN_SIMILARITY,
    test_set: str | None = None,
    warmup: bool = True,
    **recognizer_options,
) -> MethodResult:
    """Evaluate a baseline or a registered recognizer by name."""
    if name in BASELINES:
        return evaluate_predictor(name, BASELINES[name], samples, casefold=casefold, min_similarity=min_similarity,
                                  test_set=test_set)
    try:
        recognizer = build_recognizer(name, **recognizer_options)
    except Exception as error:  # noqa: BLE001 - a backend that cannot even be built is reported, not raised
        traceback.print_exc(file=sys.stderr)
        return MethodResult(name, None, error=_describe(error), test_set=test_set, casefold=casefold,
                            min_similarity=min_similarity)
    return evaluate_recognizer(name, recognizer, samples, casefold=casefold, min_similarity=min_similarity,
                               test_set=test_set, warmup=warmup)
