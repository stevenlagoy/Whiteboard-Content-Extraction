"""Tests for the evaluation package and scripts/evaluate.py."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

from whiteboard_extraction.evaluation.metrics import (
    KIND_LABELS, align_blocks, char_error_rate, edit_distance, score_sample, summarize, word_error_rate,
)
from whiteboard_extraction.evaluation.report import load_results, render_report, save_results
from whiteboard_extraction.evaluation.runner import MethodResult, evaluate_named, evaluate_predictor, recognizer_predictor
from whiteboard_extraction.evaluation.synthetic import write_synthetic_test_set
from whiteboard_extraction.evaluation.testset import TestSetError, load_test_set, write_manifest
from whiteboard_extraction.recognition.base import RecognizedBlock
from whiteboard_extraction.recognition.registry import build_recognizer

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "evaluate.py"


def block(text: str, kind: str = "prose", timestamp: float = 0.0) -> RecognizedBlock:
    return RecognizedBlock(text=text, kind=kind, timestamp=timestamp)


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
    ('{"image": "board.png", "blocks": [{"kind": "prose"}]}', "text"),
    ('{"image": "board.png"}', "blocks"),
    ('{"blocks": []}', "image"),
    ("not json", "JSON"),
])
def test_load_test_set_reports_bad_manifest_lines(tmp_path, line, message):
    (tmp_path / "board.png").write_bytes(b"")
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


def test_unimplemented_backends_are_reported_not_raised(samples):
    result = evaluate_named("ocr", samples)
    assert result.metrics is None and result.error.startswith("not implemented")


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


def test_cli_end_to_end(tmp_path, capsys):
    script = load_script()
    out_json = tmp_path / "results.json"
    code = script.main(["--write-sample", str(tmp_path / "set"), "--method", "oracle", "blank", "--json", str(out_json)])
    assert code == 0
    assert "| oracle | 4 |" in capsys.readouterr().out
    assert json.loads(out_json.read_text())["methods"][0]["metrics"]["exact_match"] == 1.0

    assert script.main(["--load", str(out_json)]) == 0
    assert "| blank | 4 |" in capsys.readouterr().out

    assert script.main(["--test-set", str(tmp_path / "set"), "--method", "ocr"]) == 1  # nothing scored yet
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
    (tmp_path / "board.png").write_bytes(b"")
    (tmp_path / "labels.jsonl").write_bytes("﻿".encode("utf-8") + b'{"image": "board.png", "blocks": []}\n')
    assert len(load_test_set(tmp_path)) == 1
    (tmp_path / "labels.jsonl").write_bytes(b"\xff\xfe\x00bad")
    with pytest.raises(TestSetError, match="UTF-8"):
        load_test_set(tmp_path)
