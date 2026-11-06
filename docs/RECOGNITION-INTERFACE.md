# The swappable recognition interface

Issue #11, user story 13. Three deep-learning methods are planned for reading the board, a
pretrained OCR model (#12), a multimodal LLM (#13), and a custom CNN (#40), and the project
needs to compare them without rewriting anything else. This page describes the seam that makes
that possible. Everything lives in `src/whiteboard_extraction/recognition/`.

## The contract

A recognition method is any object with one method:

```python
def recognize(self, timestamp: float, image: np.ndarray) -> list[RecognizedBlock]: ...
```

* `image` is one cropped board keyframe as OpenCV reads it (BGR, `uint8`, height × width × 3).
* `timestamp` is the keyframe's position in the video, in seconds.
* The result lists what is written on the board, top to bottom. Each `RecognizedBlock` carries
  `text`, a `kind` of `"prose"` or `"equation"` (equations as LaTeX), the `timestamp`, and an
  optional `confidence` between 0 and 1. A blank board returns an empty list.

`Recognizer` in `base.py` is a runtime-checkable protocol, so `isinstance(obj, Recognizer)` tells
you whether an object fits. No base class is needed; a method does not import anything from the
pipeline.

## Choosing a method

Nothing outside the recognition package names a backend class. Callers ask the registry:

```python
from whiteboard_extraction.recognition import build_recognizer, recognizer_from_env

method = build_recognizer("llm", llm_model="claude-opus-5-5")   # by name, in code
method = recognizer_from_env()                                   # by configuration
```

`recognizer_from_env` reads these variables. `WCE_RECOGNIZER`, `WCE_LLM_MODEL` and `WCE_CNN_CHECKPOINT` are
the names the web app (PR #39) already reads; `WCE_OCR_ENGINE` is new here.

| Variable | Meaning | Default |
|---|---|---|
| `WCE_RECOGNIZER` | `ocr`, `llm`, or `cnn` | `ocr` |
| `WCE_OCR_ENGINE` | engine name for the OCR method | `tesseract` |
| `WCE_LLM_MODEL` | model name for the LLM method | empty (the backend's default) |
| `WCE_CNN_CHECKPOINT` | checkpoint path for the CNN method | empty |

Backends are imported inside the registry only when requested, so running the OCR method never
imports PyTorch.

## The call site

The pipeline calls the chosen method in exactly one place, `recognize_keyframes` in
`pipeline.py`. It takes the keyframes the video stage produced, `(timestamp, image)` pairs, runs
the method over them in order, checks that every block has a valid kind, and returns one block
list per keyframe. That list is what `reconstruction.document.build_document` takes, and the
document is what the Word and PDF exporters take:

```python
from pathlib import Path

from whiteboard_extraction.recognition import recognize_keyframes, recognizer_from_env
from whiteboard_extraction.reconstruction.document import build_document
from whiteboard_extraction.export.docx_export import export_docx

blocks = recognize_keyframes(recognizer_from_env(), keyframes, on_progress=lambda done, total: print(done, total))
export_docx(build_document(blocks), Path("notes.docx"))
```

Switching from the OCR method to the LLM method is `WCE_RECOGNIZER=llm`; the video stage, the
document builder and the exporters are not edited. `tests/test_recognition_interface.py` proves
this with two test doubles (`CountingRecognizer`, `EquationRecognizer`) run through the same call
site into both exporters; the real backends are still stubs on `main` (see the status table).

## Adding a method

1. Write a class with `recognize(timestamp, image)` returning `RecognizedBlock`s. Keep heavy
   imports inside the class or its module.
2. Add one line to `_FACTORIES` in `registry.py` that builds it from `RecognizerOptions`. If it
   needs a new setting, add a field to `RecognizerOptions` (with its default) and one entry to
   `ENV_SETTINGS` mapping the field to its `WCE_*` variable; `build_recognizer(name, **settings)`
   and `recognizer_from_env` pick it up from there.
3. It is now available to `build_recognizer` and `WCE_RECOGNIZER`, and through them to the
   evaluation script's `--method` flag (#14). The web app (PR #39) still carries its own selector
   in `web/processing/registry.py`; once both merge, that selector should call `build_recognizer`
   so the registry is the only list of methods.

## Status of the methods

| Method | Name | Issue | State |
|---|---|---|---|
| Pretrained OCR | `ocr` | #12 | stub, raises `NotImplementedError` |
| Multimodal LLM | `llm` | #13 | implemented on its feature branch |
| Custom CNN | `cnn` | #40 | stub, needs a trained checkpoint |

The acceptance criterion "two methods plugged in and running" is met by the LLM method plus
whichever of the other two lands next; the evaluation script (#14) compares them on the same
labeled test set.
