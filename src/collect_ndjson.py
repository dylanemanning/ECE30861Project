#!/usr/bin/env python3
"""Aggregate outputs from src/analyze_repo.py and
src/HF_API_Integration.py into NDJSON.

Usage examples:

Collect from two repos and two models and write to out.ndjson:
    python src/collect_ndjson.py \
        --repo https://github.com/user/repo1 \
        --repo https://github.com/user/repo2 \
        --model google/gemma-3-270m \
        --model bert-base-uncased \
        --output out.ndjson

Read repo list and model list from files:
    python src/collect_ndjson.py \
        --repos-file repos.txt \
        --models-file models.txt \
        --output out.ndjson

The script writes one JSON object per line (NDJSON). Each object has a
top-level `type` field of either "repo" or "hf_model" and a `data` field
with the original payload returned by the source functions.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from typing import Iterable, List, Optional
import importlib

# Ensure importing the sibling modules in src works when this file is executed
# directly.  Insert `src` dir into sys.path first so module imports work.
THIS_DIR = os.path.dirname(os.path.abspath(__file__))
if THIS_DIR not in sys.path:
    sys.path.insert(0, THIS_DIR)

# Import the sibling modules using importlib after ensuring THIS_DIR is
# on sys.path.
_analyze_mod = None
_hf_mod = None


def _ensure_modules_loaded() -> None:
    """Import analyze_repo and HF_API_Integration modules lazily.

    This avoids requiring heavy dependencies (GitPython, requests)
    at import time (for example when running --help).
    """
    global _analyze_mod, _hf_mod
    if _analyze_mod is None:
        _analyze_mod = importlib.import_module("analyze_repo")
    if _hf_mod is None:
        _hf_mod = importlib.import_module("HF_API_Integration")


def _analyze_repo_fn(repo_url: str) -> dict:
    _ensure_modules_loaded()
    return _analyze_mod.analyze_repo(repo_url)


def _hf_meta_fn(model_id: str) -> dict:
    _ensure_modules_loaded()
    return _hf_mod.get_huggingface_model_metadata(model_id)


def read_lines_file(path: str) -> List[str]:
    with open(path, "r", encoding="utf-8") as f:
        lines = []
        for line in f:
            s = line.strip()
            if not s or s.startswith("#"):
                continue
            lines.append(s)
        return lines


def write_ndjson_line(fp, obj: dict) -> None:
    fp.write(json.dumps(obj, ensure_ascii=False))
    fp.write("\n")


def collect_and_write(
    repos: Iterable[str],
    models: Iterable[str],
    output: Optional[str],
    append: bool = False,
) -> int:
    """Collect data and write NDJSON.

    If output is None or '-' the NDJSON is written to stdout. Otherwise
    it's written to the named file (honoring append mode).
    """
    mode = "a" if append else "w"
    written = 0
    start_ts = time.time()
    use_stdout = output is None or output == "-"
    if use_stdout:
        out = sys.stdout
    else:
        # mypy/static check: ensure output is a string path here
        assert isinstance(output, str)
        out = open(output, mode, encoding="utf-8")
    try:
        for repo in repos:
            try:
                data = _analyze_repo_fn(repo)
            except Exception as e:
                data = {"error": str(e)}
            obj = {
                "type": "repo",
                "source": repo,
                "collected_at": time.strftime(
                    "%Y-%m-%dT%H:%M:%SZ", time.gmtime()
                ),
                "data": data,
            }
            write_ndjson_line(out, obj)
            written += 1

        for model in models:
            try:
                data = _hf_meta_fn(model)
            except Exception as e:
                data = {"error": str(e)}
            obj = {
                "type": "hf_model",
                "source": model,
                "collected_at": time.strftime(
                    "%Y-%m-%dT%H:%M:%SZ", time.gmtime()
                ),
                "data": data,
            }
            write_ndjson_line(out, obj)
            written += 1

    finally:
        if not use_stdout:
            out.close()

    elapsed = time.time() - start_ts
    # Print summary to stderr so stdout remains pure NDJSON when used in pipes
    print(
        f"Wrote {written} records to {output or 'stdout'} in {elapsed:.2f}s",
        file=sys.stderr,
    )
    return 0


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(prog="collect_ndjson")
    p.add_argument(
        "--repo",
        action="append",
        default=[],
        help=(
            "Repository URL to analyze; can be provided multiple times"
        ),
    )
    p.add_argument(
        "--repos-file",
        help="Path to a file with one repo URL per line",
    )
    p.add_argument(
        "--model",
        action="append",
        default=[],
        help=(
            "Hugging Face model id to query; can be provided multiple times"
        ),
    )
    p.add_argument(
        "--models-file",
        help="Path to a file with one model id per line",
    )
    p.add_argument(
        "--output",
        "-o",
        default=None,
        help=(
            "Output NDJSON file path. If omitted or '-' the NDJSON is "
            "printed to stdout"
        ),
    )
    p.add_argument(
        "--append",
        action="store_true",
        help="Append to output instead of overwriting",
    )
    return p.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> int:
    args = parse_args(argv)

    repos: List[str] = list(args.repo or [])
    models: List[str] = list(args.model or [])

    if args.repos_file:
        repos.extend(read_lines_file(args.repos_file))
    if args.models_file:
        models.extend(read_lines_file(args.models_file))

    if not repos and not models:
        print(
            "No repos or models provided. Use --repo/--model or "
            "--repos-file/--models-file."
        )
        return 2

    return collect_and_write(repos, models, args.output, append=args.append)


if __name__ == "__main__":
    raise SystemExit(main())
