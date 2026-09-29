"""Compute recognition metrics for a single DL method against a labeled test set."""

from dataclasses import dataclass

from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score


@dataclass
class MethodMetrics:
    method_name: str
    accuracy: float
    precision: float
    recall: float
    f1: float
    word_error_rate: float
    confusion: list[list[int]]


def word_error_rate(reference: str, hypothesis: str) -> float:
    """Compute WER between a reference and hypothesis string via edit distance."""
    ref_words, hyp_words = reference.split(), hypothesis.split()
    dp = [[0] * (len(hyp_words) + 1) for _ in range(len(ref_words) + 1)]
    for i in range(len(ref_words) + 1):
        dp[i][0] = i
    for j in range(len(hyp_words) + 1):
        dp[0][j] = j
    for i in range(1, len(ref_words) + 1):
        for j in range(1, len(hyp_words) + 1):
            cost = 0 if ref_words[i - 1] == hyp_words[j - 1] else 1
            dp[i][j] = min(dp[i - 1][j] + 1, dp[i][j - 1] + 1, dp[i - 1][j - 1] + cost)
    return dp[-1][-1] / max(len(ref_words), 1)


def evaluate_method(
    method_name: str, y_true: list[str], y_pred: list[str],
    reference_text: str, hypothesis_text: str,
) -> MethodMetrics:
    """Compute accuracy/precision/recall/F1/confusion matrix plus WER."""
    return MethodMetrics(
        method_name=method_name,
        accuracy=float(accuracy_score(y_true, y_pred)),
        precision=float(precision_score(y_true, y_pred, average="macro", zero_division=0)),
        recall=float(recall_score(y_true, y_pred, average="macro", zero_division=0)),
        f1=float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        word_error_rate=word_error_rate(reference_text, hypothesis_text),
        confusion=confusion_matrix(y_true, y_pred).tolist(),
    )