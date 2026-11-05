"""Scoring recognition methods against a labeled test set. See docs/EVALUATION.md."""

from .metrics import MethodMetrics, align_blocks, char_error_rate, score_sample, summarize, word_error_rate
from .report import load_results, render_report, save_results
from .runner import BASELINES, MethodResult, evaluate_named, evaluate_predictor, recognizer_predictor
from .testset import Sample, TestSetError, load_test_set

__all__ = [
    "BASELINES", "MethodMetrics", "MethodResult", "Sample", "TestSetError", "align_blocks", "char_error_rate",
    "evaluate_named", "evaluate_predictor", "load_results", "load_test_set", "recognizer_predictor", "render_report",
    "save_results", "score_sample", "summarize", "word_error_rate",
]
