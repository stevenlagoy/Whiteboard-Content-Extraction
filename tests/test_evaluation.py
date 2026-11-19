"""Tests for the evaluation package and scripts/evaluate.py."""

from __future__ import annotations

import importlib.util
import io
import json
import random
import sys
from pathlib import Path

import numpy as np
import pytest

from whiteboard_extraction.evaluation.metrics import (
    KIND_LABELS, TextCounts, align_blocks, char_error_rate, edit_distance, kind_scores, normalize_text, reading_kinds,
    score_sample, summarize, word_error_rate, words,
)
from whiteboard_extraction.evaluation.report import load_results, render_report, save_results
from whiteboard_extraction.evaluation.runner import (
    MethodResult, SampleResult, evaluate_named, evaluate_predictor, evaluate_recognizer, recognizer_predictor,
)
from whiteboard_extraction.evaluation.synthetic import write_synthetic_test_set
from whiteboard_extraction.evaluation.testset import Sample, TestSetError, load_test_set, write_manifest
from whiteboard_extraction.recognition import registry
from whiteboard_extraction.recognition.base import RecognizedBlock
from whiteboard_extraction.recognition.registry import build_recognizer

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "evaluate.py"


def block(text: str, kind: str = "prose", timestamp: float = 0.0) -> RecognizedBlock:
    return RecognizedBlock(text=text, kind=kind, timestamp=timestamp)


def write_board(path: Path) -> Path:
    """A small image file that OpenCV can read."""
    import cv2

    assert cv2.imwrite(str(path), np.full((8, 8, 3), 255, dtype=np.uint8))
    return path


@pytest.fixture
def samples(tmp_path):
    write_synthetic_test_set(tmp_path / "set")
    return load_test_set(tmp_path / "set")


# ------------------------------------------------------------------------------ metrics

def test_edit_distance():
    assert edit_distance("kitten", "sitting") == 3
    assert edit_distance([], ["a", "b"]) == 2
    assert edit_distance("same", "same") == 0


def test_word_error_rate():
    assert word_error_rate("the cat sat", "the cat sat") == 0
    assert word_error_rate("the cat sat", "the dog sat") == pytest.approx(1 / 3)
    assert word_error_rate("", "x y") == 2.0  # more words than the reference has
    assert word_error_rate("", "") == 0


def test_char_error_rate_ignores_whitespace_runs():
    assert char_error_rate("x = 1", "x  =  1") == 0
    assert char_error_rate("abcd", "abxd") == 0.25


def test_align_blocks_pairs_in_order_and_marks_misses_and_extras():
    reference = [block("alpha beta"), block("x^2", "equation"), block("gamma")]
    predicted = [block("alpha beta"), block("gamma"), block("extra line")]
    pairs = align_blocks(reference, predicted)
    texts = [(p.reference.text if p.reference else None, p.predicted.text if p.predicted else None) for p in pairs]
    assert texts == [("alpha beta", "alpha beta"), ("x^2", None), ("gamma", "gamma"), (None, "extra line")]


def test_align_blocks_pairs_similar_but_not_unrelated_text():
    pairs = align_blocks([block("Homework 4 is due Friday")], [block("Hornework 4 is due Fridav")])
    assert pairs[0].reference is not None and pairs[0].predicted is not None and pairs[0].similarity > 0.8
    unrelated = align_blocks([block("abc")], [block("xyz")])
    assert [(p.reference is None, p.predicted is None) for p in unrelated] == [(False, True), (True, False)]


def test_score_sample_counts_text_errors_and_kinds():
    reference = [block("the cat sat"), block("x^2 + 1", "equation")]
    predicted = [block("the cat sat"), block("x^2 + 1", "prose")]  # right text, wrong kind
    score = score_sample("s", reference, predicted)
    assert (score.exact_matches, score.word_errors, score.char_errors) == (2, 0, 0)
    assert score.kinds_true == ("prose", "equation") and score.kinds_pred == ("prose", "prose")


def test_summarize_oracle_is_perfect_and_blank_is_worst():
    reference = [block("a b c"), block("d e", "equation")]
    perfect = summarize("oracle", [score_sample("s", reference, reference)])
    assert (perfect.accuracy, perfect.precision, perfect.recall, perfect.f1) == (1.0, 1.0, 1.0, 1.0)
    assert (perfect.word_error_rate, perfect.char_error_rate, perfect.exact_match) == (0.0, 0.0, 1.0)
    blank = summarize("blank", [score_sample("s", reference, [])])
    assert (blank.word_error_rate, blank.exact_match, blank.recall, blank.accuracy) == (1.0, 0.0, 0.0, 0.0)
    none = KIND_LABELS.index("none")
    assert blank.confusion[KIND_LABELS.index("prose")][none] == 1
    assert blank.confusion[KIND_LABELS.index("equation")][none] == 1


def test_summarize_with_nothing_on_the_boards_is_perfect():
    metrics = summarize("blank", [score_sample("s", [], [])])
    assert (metrics.accuracy, metrics.exact_match, metrics.word_error_rate) == (1.0, 1.0, 0.0)


def test_casefold_option():
    assert score_sample("s", [block("Hello")], [block("hello")], casefold=True).exact_matches == 1
    assert score_sample("s", [block("Hello")], [block("hello")]).exact_matches == 0


def test_unknown_kind_is_rejected():
    with pytest.raises(ValueError, match="kind"):
        score_sample("s", [block("a")], [block("a", kind="text")])


# ----------------------------------------------------------------------------- test set

def test_load_test_set_reads_images_and_labels(samples):
    assert [len(sample.blocks) for sample in samples] == [2, 1, 3, 0]
    assert samples[0].blocks[1].kind == "equation" and samples[0].timestamp == 12.0
    image = samples[0].load_image()
    assert image.ndim == 3 and image.shape[2] == 3


@pytest.mark.parametrize("line, message", [
    ('{"image": "missing.png", "blocks": []}', "does not exist"),
    ('{"image": "board.png", "blocks": [{"text": "x", "kind": "poem"}]}', "kind"),
    ('{"image": "board.png", "blocks": [{"text": "x"}]}', "needs a kind"),
    ('{"image": "board.png", "blocks": [{"kind": "prose"}]}', "text"),
    ('{"image": "board.png"}', "blocks"),
    ('{"blocks": []}', "image"),
    ("not json", "JSON"),
])
def test_load_test_set_reports_bad_manifest_lines(tmp_path, line, message):
    write_board(tmp_path / "board.png")
    (tmp_path / "labels.jsonl").write_text(line + "\n")
    with pytest.raises(TestSetError, match=message):
        load_test_set(tmp_path)


def test_load_test_set_needs_a_manifest_with_samples(tmp_path):
    with pytest.raises(TestSetError, match="labels.jsonl"):
        load_test_set(tmp_path)
    write_manifest(tmp_path, [])
    with pytest.raises(TestSetError, match="no samples"):
        load_test_set(tmp_path)


# ------------------------------------------------------------------------------- runner

def test_baselines(samples):
    oracle = evaluate_named("oracle", samples)
    assert oracle.error is None and oracle.metrics.exact_match == 1.0 and oracle.metrics.samples == 4
    blank = evaluate_named("blank", samples)
    assert blank.metrics.word_error_rate == 1.0 and blank.metrics.predicted_blocks == 0


def test_a_recognizer_that_is_not_implemented_becomes_an_error_row(samples):
    class Stub:
        def recognize(self, timestamp, image):
            raise NotImplementedError("later")

    result = evaluate_predictor("stub", recognizer_predictor(Stub()), samples)
    assert result.metrics is None and result.error == "not implemented (later)"


def test_a_recognizer_that_crashes_becomes_an_error_row(samples):
    def explode(sample):
        raise RuntimeError("no API key")

    result = evaluate_predictor("boom", explode, samples)
    assert result.metrics is None and result.error == "every sample failed; first: RuntimeError: no API key"


def test_a_recognizer_runs_through_the_runner_with_images(samples):
    class Echo:
        """Answers with the labels of the sample it is given, found by timestamp."""

        def __init__(self, known):
            self.by_timestamp = {sample.timestamp: sample.blocks for sample in known}

        def recognize(self, timestamp, image):
            assert image.ndim == 3
            return list(self.by_timestamp[timestamp])

    result = evaluate_predictor("echo", recognizer_predictor(Echo(samples)), samples)
    assert result.metrics.exact_match == 1.0 and len(result.samples) == 4


def test_registry_builds_backends_lazily_and_rejects_unknown_names():
    assert build_recognizer("ocr").engine == "tesseract"
    assert build_recognizer("llm", llm_model="m").model == "m"
    with pytest.raises(ValueError, match="Unknown recognizer"):
        build_recognizer("nope")


class Stub:
    def recognize(self, timestamp, image):
        raise NotImplementedError("later")


@pytest.fixture
def stub_ocr(monkeypatch):
    """Make `ocr` a stub for the test, so it keeps passing once the real backend is implemented."""
    monkeypatch.setitem(registry._FACTORIES, "ocr", lambda options: Stub())


def test_unimplemented_backends_are_reported_not_raised(samples, stub_ocr):
    result = evaluate_named("ocr", samples)
    assert result.metrics is None and result.error == "not implemented (later)"


# ------------------------------------------------------------------------------- report

def test_report_and_json_round_trip(tmp_path, samples):
    results = [evaluate_named("oracle", samples), evaluate_named("blank", samples)]
    text = render_report(results, test_set="demo")
    assert "Test set: demo" in text and "| oracle | 4 |" in text and "| blank | 4 |" in text
    assert "Confusion matrix for oracle" in text
    path = save_results(tmp_path / "out" / "results.json", results, test_set="demo")
    loaded = load_results(path)
    assert [result.method_name for result in loaded] == ["oracle", "blank"]
    assert loaded[0].metrics == results[0].metrics
    assert loaded[0].samples[0].reference == samples[0].blocks and loaded[1].samples[0].predicted == ()


def test_methods_that_did_not_run_are_listed_under_the_table():
    text = render_report([MethodResult("ocr", None, error="not implemented (later)")])
    assert "| ocr | n/a |" in text and "- ocr: not implemented (later)" in text


# ---------------------------------------------------------------------------------- CLI

def load_script():
    spec = importlib.util.spec_from_file_location("evaluate_script", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_cli_end_to_end(tmp_path, capsys, stub_ocr):
    script = load_script()
    out_json = tmp_path / "results.json"
    code = script.main(["--write-sample", str(tmp_path / "set"), "--method", "oracle", "blank", "--json", str(out_json)])
    assert code == 0
    assert "| oracle | 4 |" in capsys.readouterr().out
    assert json.loads(out_json.read_text())["methods"][0]["metrics"]["exact_match"] == 1.0

    assert script.main(["--load", str(out_json)]) == 0
    assert "| blank | 4 |" in capsys.readouterr().out

    assert script.main(["--test-set", str(tmp_path / "set"), "--method", "ocr"]) == 1  # a stub scores nothing
    assert "not implemented" in capsys.readouterr().out


def test_cli_argument_errors(tmp_path):
    script = load_script()
    with pytest.raises(SystemExit):
        script.main([])
    with pytest.raises(SystemExit):
        script.main(["--method", "oracle"])
    with pytest.raises(SystemExit):
        script.main(["--test-set", str(tmp_path)])  # no manifest


# ----------------------------------------------------------- review follow-ups

def test_single_kind_test_set_can_score_perfectly():
    reference = [block("a b"), block("c d")]  # prose only
    metrics = summarize("oracle", [score_sample("s", reference, reference)])
    assert (metrics.precision, metrics.recall, metrics.f1) == (1.0, 1.0, 1.0)
    invented = summarize("x", [score_sample("s", reference, [*reference, block("y = 1", "equation")])])
    assert invented.precision < 1.0  # an invented equation still counts against the method


def test_blank_predicted_blocks_are_ignored():
    reference = [block("a b"), block("c d")]
    score = score_sample("s", reference, [block("a b"), block("   "), block("c d")])
    assert (score.predicted_blocks, score.exact_matches, score.char_errors) == (2, 2, 0)


def test_exact_match_on_an_empty_board_is_perfect_only_when_nothing_is_invented():
    assert summarize("blank", [score_sample("s", [], [])]).exact_match == 1.0
    assert summarize("x", [score_sample("s", [], [block("ghost")])]).exact_match == 0.0


def test_text_is_unicode_normalized():
    composed, decomposed = "café", "café"
    assert score_sample("s", [block(composed)], [block(decomposed)]).exact_matches == 1


def test_a_failure_on_one_sample_is_scored_as_nothing_and_kept(samples, capsys):
    def flaky(sample):
        if sample.name == "board_02.png":
            raise RuntimeError("decoder crashed")
        return sample.blocks

    result = evaluate_predictor("flaky", flaky, samples)
    assert result.metrics is not None and len(result.failed_samples) == 1
    assert result.failed_samples[0].error == "RuntimeError: decoder crashed" and result.failed_samples[0].predicted == ()
    assert result.metrics.exact_match < 1.0
    assert "RuntimeError: decoder crashed" in capsys.readouterr().err  # the traceback went to stderr
    text = render_report([result])
    assert "1 of 4 samples failed" in text and "board_02.png: RuntimeError: decoder crashed" in text


def test_numpy_numbers_from_a_backend_can_be_saved(tmp_path, samples):
    def predict(sample):
        return [RecognizedBlock(b.text, b.kind, np.float64(sample.timestamp), np.float32(0.5)) for b in sample.blocks]

    path = tmp_path / "results.json"
    save_results(path, [evaluate_predictor("numpy", predict, samples)])
    saved = load_results(path)[0].samples[0].predicted[0]
    assert saved.timestamp == samples[0].timestamp and saved.confidence == 0.5


def test_provenance_survives_the_json_round_trip(tmp_path, samples):
    results = [evaluate_named("oracle", samples, casefold=True, test_set="demo")]
    loaded = load_results(save_results(tmp_path / "r.json", results))
    assert (loaded[0].test_set, loaded[0].casefold) == ("demo", True)
    assert "Test set: demo (casefold)" in render_report(loaded)
    other = evaluate_named("blank", samples, test_set="other")
    assert "Sources (runs come from different test sets" in render_report([*loaded, other])


def test_load_results_rejects_files_that_are_not_results(tmp_path):
    (tmp_path / "bad.json").write_text('{"hello": 1}')
    with pytest.raises(ValueError, match="not an evaluation results file"):
        load_results(tmp_path / "bad.json")
    script = load_script()
    with pytest.raises(SystemExit):
        script.main(["--load", str(tmp_path / "bad.json")])


def test_write_sample_does_not_overwrite_without_force(tmp_path, capsys):
    script = load_script()
    assert script.main(["--write-sample", str(tmp_path / "set"), "--method", "oracle"]) == 0
    with pytest.raises(SystemExit):
        script.main(["--write-sample", str(tmp_path / "set"), "--method", "oracle"])
    assert script.main(["--write-sample", str(tmp_path / "set"), "--method", "oracle", "--force"]) == 0
    capsys.readouterr()


def test_manifest_with_a_bom_loads_and_binary_garbage_is_reported(tmp_path):
    write_board(tmp_path / "board.png")
    (tmp_path / "labels.jsonl").write_bytes("﻿".encode("utf-8") + b'{"image": "board.png", "blocks": []}\n')
    assert len(load_test_set(tmp_path)) == 1
    (tmp_path / "labels.jsonl").write_bytes(b"\xff\xfe\x00bad")
    with pytest.raises(TestSetError, match="UTF-8"):
        load_test_set(tmp_path)


# ------------------------------------------------------------------ equations and kinds

def test_whitespace_never_counts_inside_an_equation():
    assert normalize_text(r"\frac{5} {2}", kind="equation") == r"\frac{5}{2}"
    assert normalize_text("  two   words ", kind="prose") == "two words"
    score = score_sample("s", [block(r"\frac{5}{2}", "equation")], [block(r"\frac{5} {2}", "equation")])
    assert (score.exact_matches, score.word_errors, score.char_errors) == (1, 0, 0)
    assert char_error_rate(r"\frac{5}{2}", r"\frac{5} {2}", kind="equation") == 0
    assert char_error_rate(r"\frac{5}{2}", r"\frac{5} {2}") > 0  # as prose the space is a character


def test_the_words_of_an_equation_are_latex_tokens():
    tokens = ["\\frac", "{", "d", "y", "}", "{", "d", "x", "}", "=", "2", "x"]
    assert words(r"\frac{dy}{dx} = 2x", kind="equation") == tokens
    assert words(r"50\% of x", kind="equation") == ["5", "0", "\\%", "o", "f", "x"]
    assert words(r"\det(A - \lambda I)", kind="equation") == ["\\det", "(", "A", "-", "\\lambda", "I", ")"]
    assert word_error_rate("x^2 + 1", "x^3+1", kind="equation") == pytest.approx(1 / 5)
    assert words("the cat  sat") == ["the", "cat", "sat"]


def test_spacing_never_changes_the_tokens_of_an_equation():
    assert words(r"\frac { 5 } { 2 } = 2 . 5", kind="equation") == words(r"\frac{5}{2}=2.5", kind="equation")
    assert words(r"\sin x", kind="equation") == ["\\sin", "x"]  # the space ends the command
    assert words(r"\sin x", kind="equation") != words(r"\sinx", kind="equation")  # a different command


def test_the_space_that_ends_a_command_is_the_only_one_an_equation_keeps():
    assert normalize_text(r"\sin  x + \alpha _ 1", kind="equation") == r"\sin x+\alpha_1"
    assert normalize_text(r"\frac { 5 } { 2 }", kind="equation") == r"\frac{5}{2}"
    wrong = score_sample("s", [block(r"\sin x", "equation")], [block(r"\sinx", "equation")])
    assert (wrong.exact_matches, wrong.word_errors, wrong.char_errors) == (0, 2, 1)  # the three numbers agree
    right = score_sample("s", [block(r"\sin x", "equation")], [block(r"\sin   x", "equation")])
    assert (right.exact_matches, right.word_errors, right.char_errors) == (1, 0, 0)


def test_a_formula_labeled_as_prose_is_still_compared_as_a_formula():
    score = score_sample("s", [block(r"\frac{5}{2}", "equation")], [block(r"\frac{5} {2}", "prose")])
    assert (score.exact_matches, score.char_errors) == (1, 0)  # read correctly
    assert score.by_kind["equation"].blocks == 1 and score.by_kind["prose"].reference_words == 0
    assert score.kinds_pred == ("prose",)  # the wrong label is counted in the kind scores instead


def test_text_counts_are_kept_for_the_board_and_for_each_kind():
    reference = [block("the cat sat"), block("x^2 + 1", "equation"), block("on the mat")]
    predicted = [block("the cat sat"), block("x^3 + 1", "equation"), block("on a mat")]
    score = score_sample("s", reference, predicted)
    prose, equation = score.by_kind["prose"], score.by_kind["equation"]
    assert (prose.blocks, prose.word_errors, prose.reference_words, prose.exact_matches) == (2, 1, 6, 1)
    assert (equation.blocks, equation.word_errors, equation.reference_words, equation.exact_matches) == (1, 1, 5, 0)
    assert (score.word_errors, score.reference_words, score.exact_matches) == (2, 11, 1)
    metrics = summarize("m", [score, score])
    assert metrics.word_error_rate == pytest.approx(2 / 11)
    assert metrics.by_kind["equation"]["word_errors"] == 2 and metrics.by_kind["prose"]["blocks"] == 4


def test_an_extra_block_is_read_like_the_line_next_to_it():
    score = score_sample("s", [block("a b")], [block("a b"), block("y = 1", "equation")])
    assert score.by_kind["equation"] == TextCounts()  # there is no equation on this board to compare it with
    assert (score.by_kind["prose"].word_errors, score.word_errors) == (3, 3)  # y, =, 1 with nothing to match
    assert score.kinds_pred == ("prose", "equation")  # the invented equation still counts in the kind scores


def test_kind_scores_match_the_macro_average():
    reference = [block("a b"), block("x = 1", "equation"), block("c d"), block("y = 2", "equation")]
    predicted = [block("a b"), block("x = 1", "prose"), block("c d"), block("y = 2", "equation")]
    metrics = summarize("m", [score_sample("s", reference, predicted)])
    scores = kind_scores(metrics)
    assert (scores["equation"]["precision"], scores["equation"]["recall"]) == (1.0, 0.5)
    assert (scores["prose"]["precision"], scores["prose"]["recall"]) == (pytest.approx(2 / 3), 1.0)
    assert (scores["equation"]["accuracy"], scores["prose"]["accuracy"]) == (0.75, 0.75)  # one mistake in four pairs
    assert metrics.precision == pytest.approx((scores["equation"]["precision"] + scores["prose"]["precision"]) / 2)
    assert metrics.f1 == pytest.approx((scores["equation"]["f1"] + scores["prose"]["f1"]) / 2)


def test_report_has_a_table_by_kind(samples):
    text = render_report([evaluate_named("oracle", samples), evaluate_named("blank", samples)])
    assert "| Method | Kind | Blocks | Accuracy | Precision | Recall | F1 | WER | CER | Exact match |" in text
    assert "| oracle | equation | 3 | 1.000 | 1.000 | 1.000 | 1.000 | 0.000 | 0.000 | 1.000 |" in text
    assert "| blank | prose | 3 | 0.500 | 0.000 | 0.000 | 0.000 | 1.000 | 1.000 | 0.000 |" in text


def test_a_run_saved_by_an_older_version_is_scored_again_when_loaded(tmp_path, samples):
    path = save_results(tmp_path / "r.json", [evaluate_named("oracle", samples)])
    data = json.loads(path.read_text(encoding="utf-8"))
    for entry in data["methods"]:  # what the earlier format lacked, with numbers the earlier rules gave
        del entry["min_similarity"], entry["skipped"], entry["timed_per_image"], entry["metrics"]["by_kind"]
        entry["metrics"]["word_error_rate"] = 0.9
    path.write_text(json.dumps(data), encoding="utf-8")
    loaded = load_results(path)[0]
    assert (loaded.min_similarity, loaded.skipped, loaded.metrics.word_error_rate) == (0.5, [], 0.0)
    text = render_report([loaded])
    assert "| oracle | equation | 3 | 1.000 | 1.000 | 1.000 | 1.000 | 0.000 | 0.000 | 1.000 |" in text
    # the time it recorded meant something else, so it is not shown
    assert "| oracle | 4 | 6/6 | 1.000 | 1.000 | 1.000 | 1.000 | 0.000 | 0.000 | 1.000 | n/a |" in text


def test_an_older_run_that_charged_an_unreadable_image_is_scored_without_it(tmp_path, samples):
    path = save_results(tmp_path / "r.json", [evaluate_named("oracle", samples)])
    data = json.loads(path.read_text(encoding="utf-8"))
    entry = data["methods"][0]
    del entry["skipped"]
    entry["samples"][1].update(predicted=[], error="TestSetError: Could not read image board.png.")
    path.write_text(json.dumps(data), encoding="utf-8")
    loaded = load_results(path)[0]
    assert loaded.skipped == [samples[1].name] and len(loaded.samples) == 3
    assert loaded.metrics.samples == 3 and loaded.metrics.exact_match == 1.0


def test_a_saved_run_without_predictions_keeps_its_stored_metrics(tmp_path, samples):
    path = save_results(tmp_path / "r.json", [evaluate_named("oracle", samples)])
    data = json.loads(path.read_text(encoding="utf-8"))
    data["methods"][0]["samples"] = []
    data["methods"][0]["metrics"]["a_field_from_the_future"] = 1
    path.write_text(json.dumps(data), encoding="utf-8")
    assert load_results(path)[0].metrics.exact_match == 1.0


# ------------------------------------------------------------------ block size and labels

def test_a_board_returned_as_one_block_is_still_read_as_one_text():
    reference = [block("the cat sat"), block("x^2 + 1", "equation"), block("on the mat")]
    score = score_sample("s", reference, [block("the cat sat x^2 + 1 on the mat")])
    assert (score.char_errors, score.reference_chars) == (2, 28)  # only the two spaces inside the formula
    assert (score.word_errors, score.reference_words) == (3, 11)  # x^2 is one word here and three tokens in the label
    assert score.by_kind["equation"].word_errors == 5  # by kind, the formula is missing from the equations


def test_splitting_or_merging_lines_of_one_kind_costs_no_text_errors():
    prose = score_sample("s", [block("the cat sat"), block("on the mat")], [block("the cat sat on the mat")])
    assert (prose.word_errors, prose.char_errors) == (0, 0)
    merged = score_sample("s", [block("a=b", "equation"), block("c=d", "equation")], [block("a=b c=d", "equation")])
    assert (merged.word_errors, merged.char_errors) == (0, 0)
    split = score_sample("s", [block("a=b+c=d", "equation")], [block("a=b+", "equation"), block("c=d", "equation")])
    assert (split.word_errors, split.char_errors) == (0, 0)


def test_a_weak_reading_costs_the_same_text_errors_whether_or_not_it_pairs():
    reference, weak = [block("x^2+y^2=z^2", "equation")], [block("x^Z + g^Z = 2^7 q k", "prose")]  # similarity 0.46
    unpaired = score_sample("s", reference, weak)
    paired = score_sample("s", reference, weak, min_similarity=0.2)
    assert unpaired.kinds_pred == ("none", "prose") and paired.kinds_pred == ("prose",)  # the kind scores do differ
    assert unpaired.text == paired.text and paired.word_errors == 7  # read as the formula it tried to be
    assert unpaired.by_kind == paired.by_kind and unpaired.by_kind["prose"] == TextCounts()


def test_two_weak_lines_in_a_row_are_each_read_as_their_own_label():
    reference = [block("the proof is left as an exercise"), block("x^2+y^2=z^2", "equation"), block("gamma delta")]
    predicted = [block("th prf lft as exrcse for u"), block("x^Z + g^Z = 2^7 q k", "equation"), block("gamma delta")]
    assert reading_kinds(reference, predicted) == ["prose", "equation", "prose"]
    strict, loose = score_sample("s", reference, predicted), score_sample("s", reference, predicted, min_similarity=0.2)
    assert strict.kinds_pred.count("none") == 2 and "none" not in loose.kinds_pred  # neither weak line pairs at 0.5
    assert strict.text == loose.text and strict.by_kind == loose.by_kind


def test_the_kind_a_method_gives_a_block_never_changes_its_text_errors():
    reference = [block("the cat sat"), block("x^2 + 1", "equation"), block("on the mat")]
    kinds = ["prose", "equation", "prose", "prose"]
    for texts in (["the cat sat", "x^2+1", "on the mat"], ["the bat sat", "x^3 + 1", "on a mat", "and more"], ["x^2 + 1"]):
        honest = score_sample("s", reference, [block(text, kind) for text, kind in zip(texts, kinds)])
        all_prose = score_sample("s", reference, [block(text, "prose") for text in texts])
        all_equations = score_sample("s", reference, [block(text, "equation") for text in texts])
        assert honest.text == all_prose.text == all_equations.text
        assert honest.by_kind == all_prose.by_kind == all_equations.by_kind


def text_errors(score):
    by_kind = {kind: (counts.word_errors, counts.char_errors) for kind, counts in score.by_kind.items()}
    return score.word_errors, score.char_errors, by_kind


def test_text_errors_never_depend_on_the_threshold_or_on_the_kinds_a_method_assigns():
    rng = random.Random(42)
    vocabulary = "the cat sat on a mat x y = + ^ 2 1 \\frac \\alpha { }".split()

    def line():
        return " ".join(rng.choice(vocabulary) for _ in range(rng.randint(1, 6)))

    for _ in range(150):
        reference = [block(line(), rng.choice(["prose", "equation"])) for _ in range(rng.randint(1, 5))]
        predicted = [
            block("".join(ch for ch in label.text if rng.random() > 0.2) or "q", rng.choice(["prose", "equation"]))
            for label in reference
        ]
        if rng.random() < 0.3:
            predicted.insert(rng.randrange(len(predicted) + 1), block(line()))
        expected = text_errors(score_sample("s", reference, predicted))
        for threshold in (0.0, 0.3, 0.9):
            assert text_errors(score_sample("s", reference, predicted, min_similarity=threshold)) == expected
        for kind in ("prose", "equation"):
            assert text_errors(score_sample("s", reference, [block(item.text, kind) for item in predicted])) == expected


def test_reading_kinds_for_left_over_blocks():
    labels = [block("a=b+c=d", "equation")]
    assert reading_kinds(labels, [block("a=b+", "prose"), block("c=d", "prose")]) == ["equation", "equation"]  # a split line
    assert reading_kinds([block("alpha beta")], [block("zzz", "equation"), block("alpha beta")]) == ["prose", "prose"]
    assert reading_kinds([], [block("y = 1", "equation")]) == ["prose"]  # nothing to match: read as plain text
    # a piece of a split line belongs with its own line, whichever side that is on
    labels = [block("the cat sat"), block("x^2 + y^2 = 1", "equation")]
    pieces = [block("the cat sat"), block("x^2 +"), block("y^2 = 1")]
    assert reading_kinds(labels, pieces) == ["prose", "equation", "equation"]
    labels = [block("x^2 + y^2 = 1", "equation"), block("the cat sat on the mat")]
    pieces = [block("x^2 + y^2 = 1"), block("the cat sat"), block("on the mat")]
    assert reading_kinds(labels, pieces) == ["equation", "prose", "prose"]


def test_a_split_and_a_merge_on_one_board_still_cost_nothing():
    reference = [block("Homework 4 is due Friday"), block("x^2 + y^2 = r^2", "equation"), block("x = 3", "equation")]
    predicted = [block("Homework 4 is"), block("due Friday"), block("x^2 + y^2 = r^2 x = 3", "equation")]
    assert reading_kinds(reference, predicted) == ["prose", "prose", "equation"]
    score = score_sample("s", reference, predicted)
    assert (score.word_errors, score.char_errors) == (0, 0)
    assert all((counts.word_errors, counts.char_errors) == (0, 0) for counts in score.by_kind.values())


def test_a_split_next_to_a_missed_line_costs_only_the_missed_line():
    reference = [block("Homework 4 is due Friday"), block("x^2 + y^2 = r^2", "equation"), block("x = 3", "equation")]
    missed = score_sample("s", reference, [block("Homework 4 is due Friday"), block("x^2 + y^2 = r^2", "equation")])
    split = score_sample("s", reference, [block("Homework 4 is"), block("due Friday"),
                                          block("x^2 + y^2 = r^2", "equation")])
    assert (split.word_errors, split.char_errors) == (missed.word_errors, missed.char_errors) and missed.word_errors == 3


def test_text_errors_on_a_blank_board_do_not_depend_on_the_kind_of_what_was_invented():
    as_equation = score_sample("s", [], [block("x^2 + 1", "equation")])
    as_prose = score_sample("s", [], [block("x^2 + 1", "prose")])
    assert (as_equation.word_errors, as_equation.char_errors) == (as_prose.word_errors, as_prose.char_errors) == (3, 7)


def test_a_board_returned_one_word_per_block_costs_nothing():
    reference = [block("Homework 4 is due Friday"), block(r"\frac{dy}{dx} = 2 x", "equation"), block("see page 12")]
    pieces = [piece for label in reference for piece in label.text.split()]
    score = score_sample("s", reference, [block(piece, "prose") for piece in pieces])
    assert (score.word_errors, score.char_errors, score.exact_matches) == (0, 0, 0)
    assert reading_kinds(reference, [block(piece) for piece in pieces]) == ["prose"] * 5 + ["equation"] * 4 + ["prose"] * 3


def test_an_equation_split_at_a_command_is_the_same_text():
    whole = [block(r"y = \alpha x", "equation")]
    split = score_sample("s", whole, [block(r"y = \alpha", "equation"), block("x", "equation")])
    assert (split.word_errors, split.char_errors) == (0, 0)
    merged = score_sample("s", [block(r"y = \alpha", "equation"), block("x", "equation")], whole)
    assert (merged.word_errors, merged.char_errors) == (0, 0)


def test_a_board_returned_as_one_block_is_read_the_same_wherever_the_formula_stands():
    lines = [block("the cat sat"), block("on the mat"), block("x^2 + 1", "equation")]
    for reference in (lines, [lines[2], lines[0], lines[1]], [lines[0], lines[2], lines[1]]):
        one_block = [block(" ".join(line.text for line in reference))]
        score = score_sample("s", reference, one_block)
        assert (score.word_errors, score.reference_words, score.char_errors) == (3, 11, 2)


def test_splitting_a_line_next_to_a_weak_reading_changes_no_text_errors():
    reference = [block("Homework 4 is due Friday"), block("x^2+y^2=z^2", "equation")]
    whole = score_sample("s", reference, [block("Homework 4 is due Friday"), block("x^Z + g^Z = 2^7 q k", "equation")])
    split = score_sample("s", reference, [block("Homework 4 is"), block("due Friday"),
                                          block("x^Z + g^Z = 2^7 q k", "equation")])
    assert (whole.word_errors, whole.char_errors) == (split.word_errors, split.char_errors) and whole.word_errors == 7
    assert (whole.exact_matches, split.exact_matches) == (1, 0)  # the split shows in the block-level number


# ------------------------------------------------------------------ the pairing threshold

def test_min_similarity_decides_which_blocks_pair():
    reference, weak = [block("abcdefghij")], [block("abcdxxxxxx")]  # similarity 0.4
    strict = score_sample("s", reference, weak)
    assert (strict.kinds_true, strict.kinds_pred) == (("prose", "none"), ("none", "prose"))
    loose = score_sample("s", reference, weak, min_similarity=0.3)
    assert (loose.kinds_true, loose.kinds_pred) == (("prose",), ("prose",))


def test_min_similarity_reaches_the_scores_and_the_report(tmp_path):
    sample = Sample("a.png", tmp_path / "a.png", 0.0, (block("abcdefghij"),))
    strict = evaluate_predictor("weak", lambda _: [block("abcdxxxxxx")], [sample])
    loose = evaluate_predictor("weak", lambda _: [block("abcdxxxxxx")], [sample], min_similarity=0.3, test_set="demo")
    assert strict.metrics.accuracy == 0.0 and loose.metrics.accuracy == 1.0
    assert "Test set: demo (min similarity 0.3)" in render_report([loose])
    saved = load_results(save_results(tmp_path / "r.json", [loose]))
    assert saved[0].min_similarity == 0.3


def test_cli_min_similarity(tmp_path, capsys):
    script = load_script()
    out_json = tmp_path / "r.json"
    arguments = ["--write-sample", str(tmp_path / "set"), "--method", "oracle", "--json", str(out_json)]
    assert script.main([*arguments, "--min-similarity", "0.3"]) == 0
    assert json.loads(out_json.read_text(encoding="utf-8"))["methods"][0]["min_similarity"] == 0.3
    with pytest.raises(SystemExit):
        script.main([*arguments, "--force", "--min-similarity", "1.5"])
    capsys.readouterr()


# ------------------------------------------------------------------ images that cannot be read

class LabelEcho:
    """A recognizer that answers with the labels of the sample it is given, found by timestamp."""

    def __init__(self, known):
        self.by_timestamp = {sample.timestamp: sample.blocks for sample in known}
        self.calls = 0

    def recognize(self, timestamp, image):
        self.calls += 1
        return list(self.by_timestamp[timestamp])


def test_an_unreadable_image_is_rejected_when_the_test_set_loads(tmp_path):
    (tmp_path / "board.png").write_bytes(b"not an image")
    (tmp_path / "labels.jsonl").write_text('{"image": "board.png", "blocks": []}\n')
    with pytest.raises(TestSetError, match="OpenCV cannot read"):
        load_test_set(tmp_path)


def test_an_image_that_cannot_be_read_later_is_skipped_not_scored(tmp_path, samples, capsys):
    samples[1].image_path.write_bytes(b"damaged after the test set was loaded")
    result = evaluate_recognizer("echo", LabelEcho(samples), samples, warmup=False)
    assert result.skipped == [samples[1].name] and len(result.samples) == 3 and not result.failed_samples
    assert result.metrics.samples == 3 and result.metrics.exact_match == 1.0
    assert f"Skipping {samples[1].name}" in capsys.readouterr().err
    assert f"- echo: {samples[1].name}" in render_report([result])
    assert load_results(save_results(tmp_path / "r.json", [result]))[0].skipped == [samples[1].name]


def test_a_method_is_not_scored_when_no_image_can_be_read(samples):
    for sample in samples:
        sample.image_path.write_bytes(b"damaged")
    result = evaluate_recognizer("echo", LabelEcho(samples), samples)
    assert result.metrics is None and result.error == "no image could be read" and len(result.skipped) == 4


# ------------------------------------------------------------------ time per image

def slow_setup(monkeypatch):
    """A fake clock, images that take 5 s to decode, and a method whose first call takes 100 s."""
    now = [0.0]

    def decode(sample):
        now[0] += 5.0
        return np.zeros((4, 4, 3), dtype=np.uint8)

    class Slow:
        calls = 0

        def recognize(self, timestamp, image):
            self.calls += 1
            now[0] += 100.0 if self.calls == 1 else 2.0  # the first call loads the model
            return []

    monkeypatch.setattr(Sample, "load_image", decode)
    return Slow(), (lambda: now[0])


def test_time_per_image_leaves_out_decoding_and_the_warm_up_call(samples, monkeypatch):
    recognizer, clock = slow_setup(monkeypatch)
    result = evaluate_recognizer("slow", recognizer, samples, clock=clock)
    assert recognizer.calls == len(samples) + 1  # one untimed warm-up call
    assert [sample.seconds for sample in result.samples] == [2.0] * 4
    assert (result.seconds, result.seconds_per_image) == (8.0, 2.0)
    assert "| 2.00 |" in render_report([result])


def test_without_warm_up_the_first_image_carries_the_model_load(samples, monkeypatch):
    recognizer, clock = slow_setup(monkeypatch)
    result = evaluate_recognizer("slow", recognizer, samples, warmup=False, clock=clock)
    assert recognizer.calls == len(samples)
    assert [sample.seconds for sample in result.samples] == [100.0, 2.0, 2.0, 2.0]


def test_time_per_image_is_taken_over_the_images_the_method_answered():
    answered, failed = SampleResult("a.png", (), (), 10.0), SampleResult("b.png", (), (), 0.01, error="RuntimeError: x")
    assert MethodResult("m", None, [answered, failed, failed], 10.02).seconds_per_image == 10.0
    assert MethodResult("m", None, [failed], 0.01).seconds_per_image is None


def test_a_failure_while_decoding_for_the_warm_up_is_left_to_the_scored_run(samples, monkeypatch):
    def broken(sample):
        raise OSError("disk gone")

    monkeypatch.setattr(Sample, "load_image", broken)
    result = evaluate_recognizer("echo", LabelEcho(samples), samples)
    assert result.metrics is None and result.error == "every sample failed; first: OSError: disk gone"


def test_a_stub_is_reported_from_the_warm_up_call(samples):
    result = evaluate_recognizer("stub", Stub(), samples)
    assert result.metrics is None and result.error == "not implemented (later)" and result.seconds_per_image is None
    assert "| stub | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a |" in render_report([result])


def test_cli_no_warmup_reaches_the_method(tmp_path, capsys, monkeypatch):
    seen = []

    def factory(options):
        recognizer = LabelEcho(load_test_set(tmp_path / "set"))
        seen.append(recognizer)
        return recognizer

    monkeypatch.setitem(registry._FACTORIES, "ocr", factory)
    script = load_script()
    assert script.main(["--write-sample", str(tmp_path / "set"), "--method", "ocr"]) == 0
    assert script.main(["--test-set", str(tmp_path / "set"), "--method", "ocr", "--no-warmup"]) == 0
    assert [recognizer.calls for recognizer in seen] == [5, 4]
    capsys.readouterr()


# ------------------------------------------------------------------ console encoding

def test_cli_output_survives_a_console_that_is_not_utf8(tmp_path, capsys, monkeypatch):
    class Garbled:
        def recognize(self, timestamp, image):
            raise RuntimeError("模型输出无法解析")

    monkeypatch.setitem(registry._FACTORIES, "ocr", lambda options: Garbled())
    script = load_script()
    assert script.main(["--write-sample", str(tmp_path / "set"), "--method", "oracle"]) == 0
    console = io.TextIOWrapper(io.BytesIO(), encoding="cp1252")
    monkeypatch.setattr(sys, "stdout", console)
    assert script.main(["--test-set", str(tmp_path / "set"), "--method", "ocr"]) == 1
    console.flush()
    assert "模型输出无法解析" in console.buffer.getvalue().decode("utf-8")
    capsys.readouterr()
