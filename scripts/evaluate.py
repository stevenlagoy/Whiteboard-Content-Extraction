#!/usr/bin/env python
"""Score recognition methods against a labeled test set and print one comparison table.

Examples
--------
Make a small synthetic test set and check the pipeline with the built-in baselines::

    python scripts/evaluate.py --write-sample data/processed/eval_sample --method oracle blank

Score the real backends on a labeled test set (see docs/EVALUATION.md for the format)::

    python scripts/evaluate.py --test-set data/raw/eval_boards --method ocr llm cnn --json results/eval.json

Build the table from runs saved earlier, for example on different machines::

    python scripts/evaluate.py --load results/ocr.json results/llm.json

The exit status is 0 when at least one method was scored and 1 when none could run.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

try:
    import whiteboard_extraction  # noqa: F401
except ImportError:  # running from a checkout without `pip install -e .`
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from whiteboard_extraction.evaluation.report import load_results, render_report, save_results  # noqa: E402
from whiteboard_extraction.evaluation.runner import BASELINES, MethodResult, evaluate_named  # noqa: E402
from whiteboard_extraction.evaluation.synthetic import write_synthetic_test_set  # noqa: E402
from whiteboard_extraction.evaluation.testset import MANIFEST_NAME, TestSetError, load_test_set  # noqa: E402
from whiteboard_extraction.recognition.registry import RECOGNIZER_NAMES  # noqa: E402

METHOD_CHOICES = (*RECOGNIZER_NAMES, *BASELINES)
DEFAULT_METHODS = list(RECOGNIZER_NAMES)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--test-set", type=Path, metavar="DIR", help="folder with labels.jsonl and the board images")
    parser.add_argument(
        "--method", nargs="+", choices=METHOD_CHOICES, metavar="NAME",
        help=f"methods to score, from: {', '.join(METHOD_CHOICES)} (default: {' '.join(DEFAULT_METHODS)})",
    )
    parser.add_argument("--json", type=Path, metavar="FILE", help="save every method's metrics and predictions as JSON")
    parser.add_argument(
        "--load", nargs="+", type=Path, default=[], metavar="FILE", help="add methods saved earlier with --json to the table",
    )
    parser.add_argument("--casefold", action="store_true", help="ignore letter case when comparing text")
    parser.add_argument(
        "--write-sample", type=Path, metavar="DIR",
        help="write a small synthetic test set to DIR; it is scored when --test-set is omitted",
    )
    parser.add_argument("--force", action="store_true", help="let --write-sample overwrite an existing test set")
    backend = parser.add_argument_group("backend options")
    backend.add_argument("--ocr-engine", default="tesseract", help="engine for the ocr method (default: tesseract)")
    backend.add_argument("--llm-model", default="", help="model name for the llm method")
    backend.add_argument("--cnn-checkpoint", default="", help="checkpoint path for the cnn method")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    test_set = _resolve_test_set(args, parser)
    results = _load_saved(args.load, parser)
    if test_set is not None:
        results.extend(_score_methods(test_set, args, parser))
    label = str(test_set) if test_set is not None else None
    print(render_report(results, test_set=label))
    if args.json is not None:
        save_results(args.json, results, test_set=label)
        print(f"Saved results to {args.json}", file=sys.stderr)
    return 0 if any(result.metrics is not None for result in results) else 1


def _resolve_test_set(args: argparse.Namespace, parser: argparse.ArgumentParser) -> Path | None:
    """Write the synthetic sample when asked, then decide which folder (if any) to score."""
    test_set = args.test_set
    if args.write_sample is not None:
        if (args.write_sample / MANIFEST_NAME).exists() and not args.force:
            parser.error(f"{args.write_sample} already holds a test set; pass --force to overwrite it")
        manifest = write_synthetic_test_set(args.write_sample)
        print(f"Wrote a synthetic test set to {manifest.parent}", file=sys.stderr)
        test_set = test_set if test_set is not None else args.write_sample
    if args.method and test_set is None:
        parser.error("--method needs --test-set (or --write-sample)")
    if test_set is None and not args.load:
        parser.error("give --test-set, --write-sample, or --load")
    return test_set


def _load_saved(paths: list[Path], parser: argparse.ArgumentParser) -> list[MethodResult]:
    results: list[MethodResult] = []
    for path in paths:
        try:
            results.extend(load_results(path))
        except (OSError, ValueError) as error:
            parser.error(f"could not read results from {path}: {error}")
    return results


def _score_methods(test_set: Path, args: argparse.Namespace, parser: argparse.ArgumentParser) -> list[MethodResult]:
    try:
        samples = load_test_set(test_set)
    except TestSetError as error:
        parser.error(str(error))
    results: list[MethodResult] = []
    for name in args.method or DEFAULT_METHODS:
        print(f"Scoring {name} on {len(samples)} samples...", file=sys.stderr)
        results.append(evaluate_named(
            name, samples, casefold=args.casefold, test_set=str(test_set),
            ocr_engine=args.ocr_engine, llm_model=args.llm_model, cnn_checkpoint=args.cnn_checkpoint,
        ))
    return results


if __name__ == "__main__":
    sys.exit(main())
