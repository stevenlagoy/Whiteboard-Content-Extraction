"""Every recognition method must return its output in the same form.

``test_method_output_conforms`` runs each method in the registry on a board with printed text
and checks the result with ``block_problems``. A method added to ``recognition/registry.py`` is
picked up automatically, with nothing to add here.

The methods run offline. One that needs a service gets an entry in ``OFFLINE_SETUP`` that answers
its request locally, so the method's own code still runs. A stub that raises
``NotImplementedError`` is skipped and starts being checked the day it is implemented, so a
finished method must not raise it. A new method that needs an engine or a model file should add an
``OFFLINE_SETUP`` entry the same way; every method that runs has to find at least one block.
"""

import dataclasses
import json

import pytest

from whiteboard_extraction.evaluation import block_problems, load_test_set
from whiteboard_extraction.evaluation.synthetic import write_synthetic_test_set
from whiteboard_extraction.recognition.base import RecognizedBlock
from whiteboard_extraction.recognition.registry import RECOGNIZER_NAMES, build_recognizer

TIMESTAMP = 12.5
OPTIONAL_PACKAGES = {"anthropic", "easyocr", "pytesseract", "torch"}  # a method is skipped where its package is absent

# What a careless model might send: the backend has to turn this into conforming blocks.
CANNED_LLM_REPLY = json.dumps([
    {"text": "Induction", "kind": "prose"},
    {"text": r"S(n) = \frac{n(n+1)}{2}", "kind": "equation", "confidence": 0.9},
    {"text": "S(1) = 1", "kind": "Equation"},  # wrong case
    {"text": "Base case", "kind": "prose", "confidence": 95},  # a 0-100 score
    {"text": "   ", "kind": "prose"},  # blank
])


@pytest.fixture
def board(tmp_path):
    """A rendered BGR board with printed text, so a method that reads has something to find."""
    write_synthetic_test_set(tmp_path)
    return load_test_set(tmp_path)[0].load_image()


def _llm_offline(monkeypatch) -> None:
    """Send the LLM method down its OpenAI-style transport and answer its one HTTP call here."""
    from whiteboard_extraction.recognition import llm_multimodal

    def post(self, url, headers, payload, timeout):
        message = {"role": "assistant", "content": CANNED_LLM_REPLY}
        return {"choices": [{"index": 0, "message": message, "finish_reason": "stop"}]}

    for name in (llm_multimodal.MODEL_ENV, llm_multimodal.BASE_URL_ENV, llm_multimodal.KEY_NAME_ENV,
                 llm_multimodal.TIMEOUT_ENV, llm_multimodal.REASONING_EFFORT_ENV, llm_multimodal.JSON_SCHEMA_ENV,
                 llm_multimodal.EXTRA_BODY_ENV):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv(llm_multimodal.PROVIDER_ENV, "openai")
    monkeypatch.setattr(llm_multimodal.UrllibHTTPClient, "post", post)


OFFLINE_SETUP = {"llm": _llm_offline}  # method name -> function(monkeypatch) that lets it run without a service


@pytest.mark.parametrize("name", RECOGNIZER_NAMES)
def test_method_output_conforms(name, board, monkeypatch):
    setup = OFFLINE_SETUP.get(name)
    if setup is not None:
        setup(monkeypatch)
    try:
        blocks = build_recognizer(name).recognize(TIMESTAMP, board)
    except NotImplementedError as error:
        pytest.skip(f"{name} is not implemented yet ({error})")
    except ModuleNotFoundError as error:  # a missing package; a wrong import from an installed one still fails
        if error.name not in OPTIONAL_PACKAGES:
            raise
        pytest.skip(f"{name} needs the {error.name} package")
    assert block_problems(blocks, TIMESTAMP) == []
    assert blocks, f"{name} was given a board with text on it and returned nothing"


def test_llm_backend_cleans_up_a_careless_reply(board, monkeypatch):
    _llm_offline(monkeypatch)
    blocks = build_recognizer("llm").recognize(TIMESTAMP, board)
    assert [(block.kind, block.confidence) for block in blocks] == [
        ("prose", None), ("equation", 0.9), ("equation", None), ("prose", None),
    ]


def test_labels_have_the_same_form_as_predictions(tmp_path):
    write_synthetic_test_set(tmp_path)
    for sample in load_test_set(tmp_path):
        assert block_problems(list(sample.blocks), sample.timestamp) == []


# --- the checker itself ---------------------------------------------------------------------------

def good(**changes) -> RecognizedBlock:
    return dataclasses.replace(RecognizedBlock("x = 1", "equation", TIMESTAMP, 0.8), **changes)


def test_conforming_output_has_no_problems():
    assert block_problems([good(), good(kind="prose", confidence=None)], TIMESTAMP) == []
    assert block_problems([], TIMESTAMP) == []  # a blank board


def test_numpy_numbers_are_accepted():
    np = pytest.importorskip("numpy")
    blocks = [good(timestamp=np.float64(TIMESTAMP), confidence=np.float32(0.5))]
    assert block_problems(blocks, TIMESTAMP) == []


@pytest.mark.parametrize("blocks, fragment", [
    (None, "returned NoneType"),
    ((good(),), "returned tuple"),
    ([{"text": "x", "kind": "prose"}], "block 0 is dict"),
    ([good(text=None)], "text is NoneType"),
    ([good(text="  ")], "text is blank"),
    ([good(kind="formula")], "kind is 'formula'"),
    ([good(kind="Equation")], "kind is 'Equation'"),
    ([good(kind=None)], "kind is None"),
    ([good(timestamp=0.0)], "expected the keyframe's 12.5"),
    ([good(timestamp="12.5")], "expected a number of seconds"),
    ([good(timestamp=float("nan"))], "expected a number of seconds"),
    ([good(confidence=95)], "confidence is 95"),
    ([good(confidence=-0.1)], "confidence is -0.1"),
    ([good(confidence="high")], "confidence is 'high'"),
    ([good(confidence=True)], "confidence is True"),
])
def test_each_broken_rule_is_named(blocks, fragment):
    problems = block_problems(blocks, TIMESTAMP)
    assert len(problems) == 1
    assert fragment in problems[0]


def test_every_problem_is_reported_not_only_the_first():
    problems = block_problems([good(text=""), good(), good(kind="math", confidence=2)], TIMESTAMP)
    assert [problem.split(":")[0] for problem in problems] == ["block 0", "block 2", "block 2"]


def test_a_method_that_returns_the_wrong_form_is_caught(board):
    class Sloppy:
        def recognize(self, timestamp, image):
            return [RecognizedBlock("x", "math", 0.0, confidence=87)]

    assert len(block_problems(Sloppy().recognize(TIMESTAMP, board), TIMESTAMP)) == 3
