# Evaluating recognition methods

`scripts/evaluate.py` scores one or more recognition backends against the same labeled test
set and prints a single comparison table, so the custom CNN, the pretrained OCR model, and the
multimodal LLM can be compared on equal terms (issue #14, user story 11). Every number comes
from a repeatable script run, never from a hand calculation.

Today the `oracle` and `blank` baselines and the `llm` method produce numbers; `ocr` and `cnn`
print `n/a` until #12 and #40 land. The `llm` method takes its provider, model and key from the
`WCE_LLM_*` variables described in `docs/RECOGNITION-LLM.md`. Accuracy, precision, recall, F1 and the confusion matrix score the
block kind (prose or equation); how well the text itself was read is WER, CER and exact match.

## Quick start

Check that everything works without any model, using a tiny rendered test set and the two
built-in baselines:

```bash
python scripts/evaluate.py --write-sample data/processed/eval_sample --method oracle blank
```

`oracle` answers with the labels (every score should be perfect) and `blank` answers with nothing
(every score in the main table should be at its worst). Then score the real backends on a labeled
test set:

```bash
python scripts/evaluate.py --test-set data/raw/eval_boards --method ocr llm cnn --json results/eval.json
```

A backend that is not implemented yet, or that cannot start (no API key, missing checkpoint), gets
an `n/a` row and a one-line reason under the table instead of stopping the run. A backend that
raises on one image has that image scored as if it returned nothing, and the reason is listed
under the table. An image that can no longer be read is left out of the scores and named under
the table. Saved JSON runs can
be combined into one table later, which is how results from different machines are compared:

```bash
python scripts/evaluate.py --load results/ocr.json results/llm.json results/cnn.json
```

Run `python scripts/evaluate.py --help` for every option, including `--casefold`,
`--min-similarity`, `--no-warmup`, `--llm-model`, `--cnn-checkpoint`, and `--ocr-engine`.

## Test set format

A test set is a folder containing board images and a `labels.jsonl` manifest with one JSON
object per line:

```json
{"image": "boards/0012.png", "timestamp": 734.0,
 "blocks": [{"text": "Homework 4 is due Friday", "kind": "prose"},
            {"text": "\\det(A - \\lambda I) = 0", "kind": "equation"}]}
```

| Field | Meaning |
|---|---|
| `image` | Path relative to the folder. A cropped board region, as the video pipeline hands it to a recognizer. Anything OpenCV can read; an image it cannot read is an error when the test set loads. |
| `timestamp` | Seconds into the source video (optional, default 0). Passed to the recognizer and attached to its blocks. |
| `blocks` | What is written on the board, top to bottom. Each block needs a `text` and a `kind` of `prose` or `equation` (equations as LaTeX). An empty list labels a blank board. |

A block is one written line or expression, and it has one kind. A sentence that wraps across lines
on the board is one prose block, each line of a derivation is its own equation block, and a
heading or a label is a block of its own. A line that contains mathematics is an equation even
when a few words accompany it. Methods are expected to return blocks of the same size. WER and
CER read the board as one text, so correctly read text costs nothing there however its lines of
one kind were split or merged, down to one block per word; that shows only in the block-level
numbers (exact match and the kind scores). A block that mixes kinds still costs little in CER,
and more in WER because the words of an equation are tokens.

To build one from real footage, run the video pipeline to get cropped keyframes, copy the ones
to label into the folder, and write their text by hand. Keep the manifest in the folder next to
the images so the set can be zipped and shared. `data/raw/` and `data/processed/` are the
intended homes. Git ignores only the sample set and `results/`, so adding a real test set to the
repository is a deliberate choice. `--write-sample` refuses to overwrite a folder
that already holds a test set unless `--force` is given.

## What the numbers mean

| Column | Definition |
|---|---|
| Samples | Images the method was scored on. An image that could not be read is left out; an image the method raised on is counted and scored as if it returned nothing. |
| Blocks ref/pred | Labeled blocks in the test set / blocks the method returned. |
| Accuracy | Share of aligned block pairs whose kind (prose or equation) is right. A missed or invented block counts as wrong. |
| Precision, Recall, F1 | Macro-averaged over the kinds present. A missed block is a false negative for its kind; an invented block is a false positive for the kind it was given. |
| WER | Word error rate. Each board is read as one text, blocks in order, and compared with what the method read; the word-level edit distances are summed over the test set and divided by the total number of reference words. Can exceed 1.0 when the method invents text. |
| CER | Character error rate, computed the same way on characters. More forgiving to small OCR slips. |
| Exact match | Share of labeled blocks whose aligned prediction has exactly the same text after normalization. |
| s/image | Mean time the method spent on one image, over the images it answered. Only the `recognize` call is timed: decoding the image, building the method and scoring are left out, and one untimed warm-up call keeps model loading out as well (`--no-warmup` skips it, for a method that is billed per call). Total pipeline time (user story 12) is a separate measurement for `scripts/benchmark.py`, which is still empty. |

How text is compared:

* Whitespace runs are collapsed and Unicode is normalized (NFC) before anything is compared.
* Spacing never counts inside an equation. An equation is compared without the whitespace that
  means nothing in LaTeX, so `\frac{5} {2}` and `\frac{5}{2}` are the same text. The one space
  that is kept is the one LaTeX needs to end a command before a letter: `\sin x` and `\sinx` are
  different. The "words" of an equation for WER are LaTeX tokens: a command such as `\frac`, or
  any other single symbol.
* Reading is kept apart from labeling. Block boundaries and the kinds a method assigns are set
  aside: the board's label text and the method's text are lined up once, character by character,
  and each predicted block is read as the kind of the label most of its characters landed on.
  So WER and CER depend neither on the kinds a method assigns nor on `--min-similarity`: a
  formula that was read correctly but called prose costs nothing here, and the wrong label
  counts in the kind scores.
* Edit distance is Levenshtein (insert, delete, substitute), the definition WER and CER are
  normally reported with.

A second table, By kind, scores equations and prose separately (user story 7), one row per method
and kind:

| Column | Definition |
|---|---|
| Blocks | Labeled blocks of this kind. |
| Accuracy | Share of all aligned pairs for which the method was right about whether the block is of this kind. A missed block of the other kind counts as right here, so `blank` scores the share of blocks of the other kind, not 0. |
| Precision, Recall, F1 | The same scores as in the main table, for this kind alone; the main table holds their mean over the kinds present. |
| WER, CER | The equations (or the prose) of each board read as one text and compared with what the method read of that kind. |
| Exact match | Share of this kind's labeled blocks read exactly. |

A kind that is neither in the labels nor in the method's output has no row, and a rate shows `n/a`
when there is no label text of that kind to divide by. The two rows need not add up to the main
table: a block that mixes both kinds is missing from one row and extra in the other, while the
main table still reads it in place.

`--casefold` lowercases labels and predictions before alignment and before WER, CER and exact match
are computed, and `--min-similarity` changes which blocks pair (see below) and with that the kind
scores and exact match. Only compare runs made with the same settings; the saved JSON records both
and the report shows them. Blocks a method returns with empty text are ignored, matching the rule
that labels cannot be empty.

Below the tables, one confusion matrix per method shows how reference kinds (rows) were predicted
(columns); the `none` row and column hold invented and missed blocks respectively.

Blocks are aligned in order with standard sequence alignment: a predicted block pairs with a
labeled block only when their text similarity reaches `--min-similarity` (default 0.5, so at least
half of the characters agree), so a split, merged, missed, or invented block shows up as a miss or
an extra instead of shifting every later pair. Lower the threshold for a method that reads
handwriting poorly, where correct blocks would otherwise fail to pair.

The saved JSON holds the same metrics plus every reference and predicted block, so individual
mistakes can be inspected after a run. `--load` scores a saved run again from those blocks, so a
file written by an older version of the script shows numbers under the current rules. Its time
per image shows `n/a` when the file was saved before only the `recognize` call was timed.

## Adding a method

Any object with `recognize(timestamp, image) -> list[RecognizedBlock]` (the `Recognizer`
protocol in `recognition/base.py`) can be scored. Give it a name in
`recognition/registry.py` and it becomes available to `--method`. Inside Python,
`whiteboard_extraction.evaluation.evaluate_recognizer` scores a recognizer object and
`evaluate_predictor` scores any function from a `Sample` to blocks, which is how the baselines
work.

## Output format

Every method has to return its result in the same form, because reconstruction and export read
it without knowing which method ran. For one keyframe, `recognize(timestamp, image)` returns:

* a `list` of `RecognizedBlock`, empty for a blank board;
* `text`: a `str` with something in it (leave empty blocks out);
* `kind`: `"equation"` or `"prose"`, spelled exactly;
* `timestamp`: the keyframe's timestamp, the number that was passed in;
* `confidence`: `None`, or a number from 0 to 1.

`whiteboard_extraction.evaluation.block_problems(blocks, timestamp)` checks these rules and
returns what is wrong in plain words, or an empty list. `tests/test_recognizer_conformance.py`
runs it on every method in the registry, so a new method is checked as soon as it is registered.
Each method is given a board with printed text and has to return at least one block. A method
that is still a stub is skipped until it is implemented. A method that needs a service, an engine
or a model file gets an entry in that file's `OFFLINE_SETUP` so the test runs without it.

The checker cannot see how a method divides a board into blocks. The expected size is the one the
labels use: one block per written line or expression.

## Limits

* Two ways of writing the same formula still differ: `\frac12` and `\frac{1}{2}` are different
  text, so write labels the way the methods are asked to write LaTeX. Spaces inside `\text{...}`
  are dropped like any other spacing in an equation.
* Edit distances and the character line-up are computed in pure Python; a board with thousands
  of characters takes a few seconds.
* The synthetic test set is printed text on a white background. It proves the script works; it
  says nothing about handwriting accuracy.
