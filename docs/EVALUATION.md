# Evaluating recognition methods

`scripts/evaluate.py` scores one or more recognition backends against the same labeled test
set and prints a single comparison table, so the custom CNN, the pretrained OCR model, and the
multimodal LLM can be compared on equal terms (issue #14, user story 11). Every number comes
from a repeatable script run, never from a hand calculation.

Today only the `oracle` and `blank` baselines produce numbers: `ocr`, `llm` and `cnn` print `n/a`
until #12, #13 and #40 land. Accuracy, precision, recall, F1 and the confusion matrix score the
block kind (prose or equation); how well the text itself was read is WER, CER and exact match.

## Quick start

Check that everything works without any model, using a tiny rendered test set and the two
built-in baselines:

```bash
python scripts/evaluate.py --write-sample data/processed/eval_sample --method oracle blank
```

`oracle` answers with the labels (every score should be perfect) and `blank` answers with nothing
(every score should be at its worst). Then score the real backends on a labeled test set:

```bash
python scripts/evaluate.py --test-set data/raw/eval_boards --method ocr llm cnn --json results/eval.json
```

A backend that is not implemented yet, or that cannot start (no API key, missing checkpoint), gets
an `n/a` row and a one-line reason under the table instead of stopping the run. A backend that
raises on one image has that image scored as if it returned nothing, and the reason is listed
under the table. Saved JSON runs can
be combined into one table later, which is how results from different machines are compared:

```bash
python scripts/evaluate.py --load results/ocr.json results/llm.json results/cnn.json
```

Run `python scripts/evaluate.py --help` for every option, including `--casefold`, `--llm-model`,
`--cnn-checkpoint`, and `--ocr-engine`.

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
| `image` | Path relative to the folder. A cropped board region, as the video pipeline hands it to a recognizer. Anything OpenCV can read. |
| `timestamp` | Seconds into the source video (optional, default 0). Passed to the recognizer and attached to its blocks. |
| `blocks` | What is written on the board, top to bottom. Each block has `text` and a `kind` of `prose` or `equation` (equations as LaTeX). An empty list labels a blank board. |

To build one from real footage, run the video pipeline to get cropped keyframes, copy the ones
to label into the folder, and write their text by hand. Keep the manifest in the folder next to
the images so the set can be zipped and shared. `data/raw/` and `data/processed/` are the
intended homes and are not meant to be committed. `--write-sample` refuses to overwrite a folder
that already holds a test set unless `--force` is given.

## What the numbers mean

| Column | Definition |
|---|---|
| Blocks ref/pred | Labeled blocks in the test set / blocks the method returned. |
| Accuracy | Share of aligned block pairs whose kind (prose or equation) is right. A missed or invented block counts as wrong. |
| Precision, Recall, F1 | Macro-averaged over the two kinds. A missed block is a false negative for its kind; an invented block is a false positive for the kind it was given. |
| WER | Word error rate: word-level edit distance between each board's text and the method's text, summed over the test set and divided by the total number of reference words. Can exceed 1.0 when the method invents text. |
| CER | Character error rate, computed the same way on characters. More forgiving to small OCR slips, and the better number for LaTeX, whose tokens are not space-separated. |
| Exact match | Share of labeled blocks whose aligned prediction has exactly the same text (after collapsing whitespace). |
| Time (s) | Wall-clock time spent running the method over the whole set, reading the images included and scoring excluded (user story 12). |

`--casefold` lowercases labels and predictions before alignment and before WER, CER and exact match
are computed, so only compare runs made with the same setting; the saved JSON records it. Blocks a
method returns with empty text are ignored, matching the rule that labels cannot be empty.

Below the table, one confusion matrix per method shows how reference kinds (rows) were predicted
(columns); the `none` row and column hold invented and missed blocks respectively.

Blocks are aligned in order with standard sequence alignment: a predicted block pairs with a
labeled block only when at least half of its characters agree, so a split, merged, missed, or
invented block shows up as a miss or an extra instead of shifting every later pair.

The saved JSON holds the same metrics plus every reference and predicted block, so individual
mistakes can be inspected after a run.

## Adding a method

Any object with `recognize(timestamp, image) -> list[RecognizedBlock]` (the `Recognizer`
protocol in `recognition/base.py`) can be scored. Give it a name in
`recognition/registry.py` and it becomes available to `--method`. Inside Python,
`whiteboard_extraction.evaluation.evaluate_predictor` scores any function from a `Sample` to
blocks, which is how the baselines and the tests work.

## Limits

* Text metrics tokenize on whitespace, so `x^2+1` and `x^2 + 1` count as different words.
  Use CER for equations, or normalize spacing in both labels and outputs.
* Edit distances are computed in pure Python; boards with thousands of characters take a few
  seconds each.
* The synthetic test set is printed text on a white background. It proves the script works; it
  says nothing about handwriting accuracy.
