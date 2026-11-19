"""Recognition metrics: text error rates, block alignment, and kind-classification scores.

Everything here is pure: it takes reference and predicted blocks (or plain text) and returns
numbers. Running a recognizer over a test set lives in runner.py; reporting lives in report.py.

How a method is scored
----------------------
* Blocks: reference and predicted blocks are aligned in order (see align_blocks). A reference
  block is an exact match when its aligned prediction has the same normalized text.
* Text: the blocks of a board are read in order as one text and compared with what the method
  read. Word error rate (WER) and character error rate (CER) are edit distances divided by the
  reference length, summed over the whole test set (corpus level), so long samples weigh more
  than short ones. Because the board is one text, text that was read correctly costs nothing
  here however its lines of one kind were split or merged, down to one block per word.
* Equations: spacing never counts. An equation is compared without the whitespace that means
  nothing in LaTeX (only the space that ends a command before a letter is kept), and its
  "words" for WER are LaTeX tokens: a command such as \\frac or any other single symbol.
* Reading is kept apart from labeling. The board's label text and the method's text are lined
  up once, character by character, and every predicted block is read as the kind of the label
  most of its characters landed on (see reading_kinds), not as the kind the method gave it and
  not through the pairing threshold. So WER and CER depend on neither: a formula that was read
  correctly but labeled prose costs nothing here, and the wrong label shows up in the kind
  scores.
* By kind: the same counts are also kept for the equations and the prose of a board separately.
  Those need not add up to the whole-board counts: a block that mixes kinds is missing from one
  kind and extra in the other.
* Kinds: each aligned pair contributes one label to the equation/prose classification. A
  missed block counts as predicting "none" and an extra block as a "none" reference. Accuracy
  is over all pairs; precision, recall and F1 are macro-averaged over equation and prose; the
  confusion matrix includes "none" so misses and extras stay visible.
"""

from __future__ import annotations

import re
import unicodedata
from collections.abc import Sequence
from dataclasses import asdict, dataclass, field

from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score

from whiteboard_extraction.recognition.base import KINDS, RecognizedBlock

NONE = "none"  # label for a missing or extra block in the kind confusion matrix
KIND_LABELS = (*KINDS, NONE)
EQUATION = "equation"
PROSE = "prose"
MIN_SIMILARITY = 0.5  # a predicted block must share at least this much text to count as the same block

_WHITESPACE = re.compile(r"\s+")
_LATEX_TOKEN = re.compile(r"\\[A-Za-z]+|\\.|\S")  # a command, an escaped symbol, or any other single symbol
_COMMAND = re.compile(r"\\[A-Za-z]+")
_COMMAND_LETTERS = frozenset("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ")


# ------------------------------------------------------------------------------------- text

def normalize_text(text: str, casefold: bool = False, kind: str | None = None) -> str:
    """Normalize Unicode (NFC) and whitespace; optionally ignore letter case.

    Whitespace runs are collapsed to one space and the ends trimmed. An equation then loses all
    the whitespace that carries no meaning in LaTeX, so a method that writes ``\\frac{5} {2}`` has
    read ``\\frac{5}{2}`` correctly. The one space LaTeX needs is kept: the space that ends a
    command before a letter, so ``\\sin x`` and ``\\sinx`` stay different.
    """
    cleaned = _WHITESPACE.sub(" ", unicodedata.normalize("NFC", text)).strip()
    if casefold:
        cleaned = cleaned.casefold()
    return _join_tokens(_LATEX_TOKEN.findall(cleaned)) if kind == EQUATION else cleaned


def _join_tokens(tokens: Sequence[str]) -> str:
    """LaTeX tokens as text with no spacing except the space that ends a command before a letter."""
    return "".join(
        (" " if index and _COMMAND.fullmatch(tokens[index - 1]) and token[0] in _COMMAND_LETTERS else "") + token
        for index, token in enumerate(tokens)
    )


def words(text: str, casefold: bool = False, kind: str | None = None) -> list[str]:
    """The units WER counts: whitespace-separated words, or LaTeX tokens for an equation.

    An equation token is a command such as ``\\frac``, an escaped symbol, or any other single
    symbol. Spacing changes nothing: ``x ^ 2``, ``x^2`` and ``x^ 2`` are the same three tokens.
    The tokens are the pieces normalize_text joins, so two equations have the same tokens exactly
    when they have the same normalized text.
    """
    cleaned = normalize_text(text, casefold)
    return _LATEX_TOKEN.findall(cleaned) if kind == EQUATION else cleaned.split()


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


Compared = tuple[str, "str | None"]  # a block's text and the kind it is compared as


def _stream_words(blocks: Sequence[Compared], casefold: bool) -> list[str]:
    return [word for text, kind in blocks for word in words(text, casefold, kind)]


def _stream_chars(blocks: Sequence[Compared], casefold: bool) -> str:
    """The blocks as one string, a space apart.

    A run of equations is joined token to token as if it were one equation, since spacing never
    counts there: an equation written on one line or split over two is the same text.
    """
    parts: list[str] = []
    tokens: list[str] = []
    for text, kind in blocks:
        if kind == EQUATION:
            tokens.extend(words(text, casefold, kind))
            continue
        if tokens:
            parts.append(_join_tokens(tokens))
            tokens = []
        parts.append(normalize_text(text, casefold, kind))
    if tokens:
        parts.append(_join_tokens(tokens))
    return " ".join(parts)


def word_counts(reference: Sequence[Compared], hypothesis: Sequence[Compared], casefold: bool = False) -> tuple[int, int]:
    """Word errors and reference word count between two lists of blocks, each read in order as one text."""
    reference_words, hypothesis_words = _stream_words(reference, casefold), _stream_words(hypothesis, casefold)
    return edit_distance(reference_words, hypothesis_words), len(reference_words)


def char_counts(reference: Sequence[Compared], hypothesis: Sequence[Compared], casefold: bool = False) -> tuple[int, int]:
    """Character errors and reference character count between two lists of blocks, each read in order as one text."""
    reference_chars, hypothesis_chars = _stream_chars(reference, casefold), _stream_chars(hypothesis, casefold)
    return edit_distance(reference_chars, hypothesis_chars), len(reference_chars)


def word_error_rate(reference: str, hypothesis: str, casefold: bool = False, kind: str | None = None) -> float:
    """Word-level edit distance divided by the number of reference words (can exceed 1.0)."""
    errors, total = word_counts([(reference, kind)], [(hypothesis, kind)], casefold)
    return errors / max(total, 1)


def char_error_rate(reference: str, hypothesis: str, casefold: bool = False, kind: str | None = None) -> float:
    """Character-level edit distance divided by the number of reference characters."""
    errors, total = char_counts([(reference, kind)], [(hypothesis, kind)], casefold)
    return errors / max(total, 1)


def text_similarity(first: str, second: str, casefold: bool = False, kind: str | None = None) -> float:
    """1.0 for identical normalized text, 0.0 for nothing in common."""
    first, second = normalize_text(first, casefold, kind), normalize_text(second, casefold, kind)
    longest = max(len(first), len(second))
    return 1.0 if longest == 0 else 1.0 - edit_distance(first, second) / longest


# --------------------------------------------------------------------------------- alignment

@dataclass(frozen=True)
class AlignedPair:
    """A reference block and the predicted block taken to be the same one; either side may be missing."""

    reference: RecognizedBlock | None
    predicted: RecognizedBlock | None
    similarity: float  # text similarity of the pair; 0.0 when one side is missing

    @property
    def paired(self) -> bool:
        return self.reference is not None and self.predicted is not None


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
    misses and extras instead of shifting every later pair. Similarity is measured after
    normalizing both texts as the reference block's kind.
    """
    return _align(reference, predicted, _similarities(reference, predicted, casefold), min_similarity)


def _similarities(
    reference: Sequence[RecognizedBlock], predicted: Sequence[RecognizedBlock], casefold: bool,
) -> list[list[float]]:
    """Text similarity of every reference block (rows) with every predicted block (columns)."""
    return [[text_similarity(r.text, p.text, casefold, r.kind) for p in predicted] for r in reference]


def _align(
    reference: Sequence[RecognizedBlock],
    predicted: Sequence[RecognizedBlock],
    similarity: list[list[float]],
    min_similarity: float,
) -> list[AlignedPair]:
    rows, cols = len(reference), len(predicted)
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
class TextCounts:
    """Raw text counts for the blocks of one kind; adding two of them pools their counts."""

    blocks: int = 0  # reference blocks of this kind
    word_errors: int = 0
    reference_words: int = 0
    char_errors: int = 0
    reference_chars: int = 0
    exact_matches: int = 0

    def __add__(self, other: TextCounts) -> TextCounts:
        return TextCounts(*(mine + theirs for mine, theirs in zip(_values(self), _values(other))))


def _values(counts: TextCounts) -> tuple[int, ...]:
    return (counts.blocks, counts.word_errors, counts.reference_words, counts.char_errors,
            counts.reference_chars, counts.exact_matches)


@dataclass(frozen=True)
class SampleScore:
    """Raw counts for one test image, summed later across the test set."""

    name: str
    reference_blocks: int
    predicted_blocks: int
    text: TextCounts  # the board read as one text
    by_kind: dict[str, TextCounts]  # the equations and the prose of the board read separately
    kinds_true: tuple[str, ...]
    kinds_pred: tuple[str, ...]

    @property
    def word_errors(self) -> int:
        return self.text.word_errors

    @property
    def reference_words(self) -> int:
        return self.text.reference_words

    @property
    def char_errors(self) -> int:
        return self.text.char_errors

    @property
    def reference_chars(self) -> int:
        return self.text.reference_chars

    @property
    def exact_matches(self) -> int:
        return self.text.exact_matches


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
    by_kind: dict[str, dict[str, int]] = field(default_factory=dict)  # kind -> the fields of TextCounts


def _check_kinds(blocks: Sequence[RecognizedBlock], what: str) -> None:
    for block in blocks:
        if block.kind not in KINDS:
            raise ValueError(f"{what} block has kind {block.kind!r}; expected one of {', '.join(KINDS)}.")


def _lined_up(reference: Sequence, hypothesis: Sequence) -> list[tuple[int, int]]:
    """Positions (in reference, in hypothesis) that one cheapest edit path sets against each other."""
    rows, cols = len(reference), len(hypothesis)
    moves = [bytearray(cols + 1) for _ in range(rows + 1)]  # 0: set against each other, 1: reference only, 2: hypothesis only
    moves[0][1:] = b"\x02" * cols
    previous = list(range(cols + 1))
    for i in range(1, rows + 1):
        item, row_moves, current = reference[i - 1], moves[i], [i]
        row_moves[0] = 1
        for j in range(1, cols + 1):
            best = previous[j - 1] + (item != hypothesis[j - 1])
            if previous[j] + 1 < best:
                best, row_moves[j] = previous[j] + 1, 1
            if current[j - 1] + 1 < best:
                best, row_moves[j] = current[j - 1] + 1, 2
            current.append(best)
        previous = current
    pairs: list[tuple[int, int]] = []
    i, j = rows, cols
    while i and j:
        move = moves[i][j]
        if move == 0:
            pairs.append((i - 1, j - 1))
        i, j = i - (move != 2), j - (move != 1)
    return pairs


def reading_kinds(
    reference: Sequence[RecognizedBlock], predicted: Sequence[RecognizedBlock], casefold: bool = False,
) -> list[str]:
    """The kind each predicted block's text is read as, one entry per predicted block.

    Reading the text must depend neither on the kind a method gave a block nor on the pairing
    threshold, and it has to cope with a method that splits or merges lines. So block boundaries
    are set aside: everything written on the board and everything the method read are lined up
    once, character by character and ignoring whitespace, the way an edit distance lines two
    texts up. A predicted block is then read as the kind of the label most of its characters
    landed on. A block none of whose characters landed on a label is read like the block before
    it (the first one like the block after it), and as prose when nothing lined up at all.
    """
    def characters(blocks: Sequence[RecognizedBlock]) -> tuple[str, list[int]]:
        texts = [_WHITESPACE.sub("", normalize_text(block.text, casefold)) for block in blocks]
        return "".join(texts), [index for index, text in enumerate(texts) for _ in text]

    (label_text, label_of), (read_text, block_of) = characters(reference), characters(predicted)
    landed = [[0] * len(reference) for _ in predicted]  # landed[block][label]: characters of the block set against the label
    for label_position, read_position in _lined_up(label_text, read_text):
        landed[block_of[read_position]][label_of[label_position]] += 1
    kinds: list[str | None] = [
        reference[counts.index(max(counts))].kind if any(counts) else None for counts in landed
    ]
    known = [kind for kind in kinds if kind is not None]
    last = known[0] if known else PROSE
    for column, kind in enumerate(kinds):
        last = kinds[column] = kind if kind is not None else last
    return kinds


def _text_counts(
    reference: Sequence[Compared], predicted: Sequence[Compared], exact_matches: int, casefold: bool,
) -> TextCounts:
    """Compare the reference blocks with the predicted blocks, each list read in order as one text."""
    word_errors, reference_words = word_counts(reference, predicted, casefold)
    char_errors, reference_chars = char_counts(reference, predicted, casefold)
    return TextCounts(len(reference), word_errors, reference_words, char_errors, reference_chars, exact_matches)


def score_sample(
    name: str,
    reference: Sequence[RecognizedBlock],
    predicted: Sequence[RecognizedBlock],
    casefold: bool = False,
    min_similarity: float = MIN_SIMILARITY,
) -> SampleScore:
    """Compare one image's predicted blocks with its labeled blocks.

    Two things are measured apart. Pairing (align_blocks, with min_similarity) gives the kind
    scores and exact match. Reading (reading_kinds, which uses neither the threshold nor the
    method's kinds) gives WER and CER, which therefore depend on neither.
    """
    _check_kinds(reference, "Reference")
    _check_kinds(predicted, "Predicted")
    predicted = [block for block in predicted if normalize_text(block.text)]  # blank output carries no content
    similarity = _similarities(reference, predicted, casefold)
    pairs = _align(reference, predicted, similarity, min_similarity)
    exact = [
        pair.reference.kind for pair in pairs
        if pair.paired and normalize_text(pair.reference.text, casefold, pair.reference.kind)
        == normalize_text(pair.predicted.text, casefold, pair.reference.kind)
    ]
    read_as = reading_kinds(reference, predicted, casefold)
    reference_text = [(block.text, block.kind) for block in reference]
    predicted_text = [(block.text, kind) for block, kind in zip(predicted, read_as)]
    return SampleScore(
        name=name,
        reference_blocks=len(reference),
        predicted_blocks=len(predicted),
        text=_text_counts(reference_text, predicted_text, len(exact), casefold),
        by_kind={
            kind: _text_counts(
                [entry for entry in reference_text if entry[1] == kind],
                [entry for entry in predicted_text if entry[1] == kind],
                exact.count(kind), casefold,
            )
            for kind in KINDS
        },
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

    by_kind = {kind: sum((score.by_kind[kind] for score in scores), TextCounts()) for kind in KINDS}
    pooled = sum((score.text for score in scores), TextCounts())
    reference_blocks = sum(score.reference_blocks for score in scores)
    predicted_blocks = sum(score.predicted_blocks for score in scores)
    return MethodMetrics(
        method_name=method_name,
        samples=len(scores),
        reference_blocks=reference_blocks,
        predicted_blocks=predicted_blocks,
        accuracy=accuracy,
        precision=precision,
        recall=recall,
        f1=f1,
        word_error_rate=pooled.word_errors / max(pooled.reference_words, 1),
        char_error_rate=pooled.char_errors / max(pooled.reference_chars, 1),
        exact_match=(pooled.exact_matches / reference_blocks if reference_blocks
                     else float(predicted_blocks == 0)),  # nothing to find: perfect only if nothing was invented
        confusion=confusion,
        confusion_labels=KIND_LABELS,
        by_kind={kind: asdict(counts) for kind, counts in by_kind.items()},
    )


def kind_scores(metrics: MethodMetrics) -> dict[str, dict[str, float]]:
    """Accuracy, precision, recall and F1 of each kind, read from the confusion matrix (0.0 where undefined).

    Accuracy is one kind against everything else: the share of all aligned pairs for which the
    method was right about whether the block is of this kind.
    """
    labels, matrix = list(metrics.confusion_labels), metrics.confusion
    pairs = sum(sum(row) for row in matrix)
    scores: dict[str, dict[str, float]] = {}
    for kind in KINDS:
        index = labels.index(kind)
        correct = matrix[index][index]
        predicted = sum(row[index] for row in matrix)
        actual = sum(matrix[index])
        precision = correct / predicted if predicted else 0.0
        recall = correct / actual if actual else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        accuracy = (pairs - (predicted - correct) - (actual - correct)) / pairs if pairs else 0.0
        scores[kind] = {"accuracy": accuracy, "precision": precision, "recall": recall, "f1": f1,
                        "predicted": predicted, "actual": actual}
    return scores
