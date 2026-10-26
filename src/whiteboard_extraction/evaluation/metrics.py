"""Recognition metrics: text error rates, block alignment, and kind-classification scores.

Everything here is pure: it takes reference and predicted blocks (or plain text) and returns
numbers. Running a recognizer over a test set lives in runner.py; reporting lives in report.py.

How a method is scored
----------------------
* Text: the blocks of each sample are joined in order and compared as text. Word error rate
  (WER) and character error rate (CER) are edit distances divided by the reference length,
  summed over the whole test set (corpus level), so long samples weigh more than short ones.
* Blocks: reference and predicted blocks are aligned in order (see align_blocks). A reference
  block is an exact match when its aligned prediction has the same normalized text.
* Kinds: each aligned pair contributes one label to the equation/prose classification. A
  missed block counts as predicting "none" and an extra block as a "none" reference. Accuracy
  is over all pairs; precision, recall and F1 are macro-averaged over equation and prose; the
  confusion matrix includes "none" so misses and extras stay visible.
"""

from __future__ import annotations

import re
import unicodedata
from collections.abc import Sequence
from dataclasses import dataclass

from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score

from whiteboard_extraction.recognition.base import KINDS, RecognizedBlock
NONE = "none"  # label for a missing or extra block in the kind confusion matrix
KIND_LABELS = (*KINDS, NONE)
MIN_SIMILARITY = 0.5  # a predicted block must share at least this much text to count as the same block

_WHITESPACE = re.compile(r"\s+")


# ------------------------------------------------------------------------------------- text

def normalize_text(text: str, casefold: bool = False) -> str:
    """Normalize Unicode (NFC), collapse whitespace runs and trim; optionally ignore letter case."""
    cleaned = _WHITESPACE.sub(" ", unicodedata.normalize("NFC", text)).strip()
    return cleaned.casefold() if casefold else cleaned


def edit_distance(reference: Sequence, hypothesis: Sequence) -> int:
    """Levenshtein distance between two token sequences (characters, words, or anything comparable)."""
    previous = list(range(len(hypothesis) + 1))
    for i, reference_token in enumerate(reference, start=1):
        current = [i]
        for j, hypothesis_token in enumerate(hypothesis, start=1):
            cost = 0 if reference_token == hypothesis_token else 1
            current.append(min(previous[j] + 1, current[j - 1] + 1, previous[j - 1] + cost))
        previous = current
    return previous[-1]


def word_error_rate(reference: str, hypothesis: str) -> float:
    """Word-level edit distance divided by the number of reference words (can exceed 1.0)."""
    reference_words, hypothesis_words = reference.split(), hypothesis.split()
    return edit_distance(reference_words, hypothesis_words) / max(len(reference_words), 1)


def char_error_rate(reference: str, hypothesis: str) -> float:
    """Character-level edit distance divided by the number of reference characters."""
    reference_chars, hypothesis_chars = normalize_text(reference), normalize_text(hypothesis)
    return edit_distance(reference_chars, hypothesis_chars) / max(len(reference_chars), 1)


def text_similarity(first: str, second: str, casefold: bool = False) -> float:
    """1.0 for identical normalized text, 0.0 for nothing in common."""
    first, second = normalize_text(first, casefold), normalize_text(second, casefold)
    longest = max(len(first), len(second))
    return 1.0 if longest == 0 else 1.0 - edit_distance(first, second) / longest


# --------------------------------------------------------------------------------- alignment

@dataclass(frozen=True)
class AlignedPair:
    """A reference block and the predicted block taken to be the same one; either side may be missing."""

    reference: RecognizedBlock | None
    predicted: RecognizedBlock | None
    similarity: float  # text similarity of the pair; 0.0 when one side is missing


def align_blocks(
    reference: Sequence[RecognizedBlock],
    predicted: Sequence[RecognizedBlock],
    casefold: bool = False,
    min_similarity: float = MIN_SIMILARITY,
) -> list[AlignedPair]:
    """Pair reference blocks with predicted blocks in order, allowing misses and extras.

    This is sequence alignment: pairing two blocks costs 1 - similarity and is only allowed
    when their text similarity reaches min_similarity; leaving a block unpaired costs 1.
    The cheapest alignment wins, so a split, merged, missed, or invented block shows up as
    misses and extras instead of shifting every later pair.
    """
    rows, cols = len(reference), len(predicted)
    similarity = [[text_similarity(r.text, p.text, casefold) for p in predicted] for r in reference]
    cost = [[0.0] * (cols + 1) for _ in range(rows + 1)]
    choice = [[""] * (cols + 1) for _ in range(rows + 1)]
    for i in range(1, rows + 1):
        cost[i][0], choice[i][0] = float(i), "miss"
    for j in range(1, cols + 1):
        cost[0][j], choice[0][j] = float(j), "extra"
    for i in range(1, rows + 1):
        for j in range(1, cols + 1):
            # On a tie "extra" is chosen first while backtracking, so the final order lists a missed
            # reference block before the invented block next to it.
            options = [(cost[i][j - 1] + 1.0, "extra"), (cost[i - 1][j] + 1.0, "miss")]
            if similarity[i - 1][j - 1] >= min_similarity:
                options.insert(0, (cost[i - 1][j - 1] + 1.0 - similarity[i - 1][j - 1], "pair"))
            cost[i][j], choice[i][j] = min(options, key=lambda option: option[0])

    pairs: list[AlignedPair] = []
    i, j = rows, cols
    while i > 0 or j > 0:
        step = choice[i][j]
        if step == "pair":
            pairs.append(AlignedPair(reference[i - 1], predicted[j - 1], similarity[i - 1][j - 1]))
            i, j = i - 1, j - 1
        elif step == "miss":
            pairs.append(AlignedPair(reference[i - 1], None, 0.0))
            i -= 1
        else:
            pairs.append(AlignedPair(None, predicted[j - 1], 0.0))
            j -= 1
    pairs.reverse()
    return pairs


# ----------------------------------------------------------------------------------- scoring

@dataclass(frozen=True)
class SampleScore:
    """Raw counts for one test image, summed later across the test set."""

    name: str
    reference_blocks: int
    predicted_blocks: int
    word_errors: int
    reference_words: int
    char_errors: int
    reference_chars: int
    exact_matches: int
    kinds_true: tuple[str, ...]
    kinds_pred: tuple[str, ...]


@dataclass(frozen=True)
class MethodMetrics:
    """Every number reported for one recognition method on one test set."""

    method_name: str
    samples: int
    reference_blocks: int
    predicted_blocks: int
    accuracy: float
    precision: float
    recall: float
    f1: float
    word_error_rate: float
    char_error_rate: float
    exact_match: float
    confusion: list[list[int]]
    confusion_labels: tuple[str, ...] = KIND_LABELS


def _check_kinds(blocks: Sequence[RecognizedBlock], what: str) -> None:
    for block in blocks:
        if block.kind not in KINDS:
            raise ValueError(f"{what} block has kind {block.kind!r}; expected one of {', '.join(KINDS)}.")


def score_sample(
    name: str,
    reference: Sequence[RecognizedBlock],
    predicted: Sequence[RecognizedBlock],
    casefold: bool = False,
) -> SampleScore:
    """Compare one image's predicted blocks with its labeled blocks."""
    _check_kinds(reference, "Reference")
    _check_kinds(predicted, "Predicted")
    predicted = [block for block in predicted if normalize_text(block.text)]  # blank output carries no content
    pairs = align_blocks(reference, predicted, casefold)
    reference_text = " ".join(normalize_text(block.text, casefold) for block in reference)
    predicted_text = " ".join(normalize_text(block.text, casefold) for block in predicted)
    reference_words, predicted_words = reference_text.split(), predicted_text.split()
    exact_matches = sum(
        1 for pair in pairs
        if pair.reference is not None and pair.predicted is not None
        and normalize_text(pair.reference.text, casefold) == normalize_text(pair.predicted.text, casefold)
    )
    return SampleScore(
        name=name,
        reference_blocks=len(reference),
        predicted_blocks=len(predicted),
        word_errors=edit_distance(reference_words, predicted_words),
        reference_words=len(reference_words),
        char_errors=edit_distance(reference_text, predicted_text),
        reference_chars=len(reference_text),
        exact_matches=exact_matches,
        kinds_true=tuple(pair.reference.kind if pair.reference is not None else NONE for pair in pairs),
        kinds_pred=tuple(pair.predicted.kind if pair.predicted is not None else NONE for pair in pairs),
    )


def summarize(method_name: str, scores: Sequence[SampleScore]) -> MethodMetrics:
    """Combine per-sample counts into the method's metrics."""
    y_true = [kind for score in scores for kind in score.kinds_true]
    y_pred = [kind for score in scores for kind in score.kinds_pred]
    present = set(y_true) | set(y_pred)
    kinds = [kind for kind in KINDS if kind in present]  # average only over kinds that occur in this test set
    if y_true and kinds:
        accuracy = float(accuracy_score(y_true, y_pred))
        precision = float(precision_score(y_true, y_pred, labels=kinds, average="macro", zero_division=0))
        recall = float(recall_score(y_true, y_pred, labels=kinds, average="macro", zero_division=0))
        f1 = float(f1_score(y_true, y_pred, labels=kinds, average="macro", zero_division=0))
        confusion = confusion_matrix(y_true, y_pred, labels=list(KIND_LABELS)).tolist()
    else:  # nothing on any board and nothing predicted: there was nothing to get wrong
        accuracy = precision = recall = f1 = 1.0
        confusion = [[0] * len(KIND_LABELS) for _ in KIND_LABELS]

    reference_blocks = sum(score.reference_blocks for score in scores)
    predicted_blocks = sum(score.predicted_blocks for score in scores)
    reference_words = sum(score.reference_words for score in scores)
    reference_chars = sum(score.reference_chars for score in scores)
    return MethodMetrics(
        method_name=method_name,
        samples=len(scores),
        reference_blocks=reference_blocks,
        predicted_blocks=predicted_blocks,
        accuracy=accuracy,
        precision=precision,
        recall=recall,
        f1=f1,
        word_error_rate=sum(score.word_errors for score in scores) / max(reference_words, 1),
        char_error_rate=sum(score.char_errors for score in scores) / max(reference_chars, 1),
        exact_match=(sum(score.exact_matches for score in scores) / reference_blocks if reference_blocks
                     else float(predicted_blocks == 0)),  # nothing to find: perfect only if nothing was invented
        confusion=confusion,
        confusion_labels=KIND_LABELS,
    )
