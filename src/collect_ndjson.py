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
from typing import Iterable, List, Optional, Tuple, Dict, Any
import importlib

# Ensure importing the sibling modules in src works when this file is executed
# directly.  Insert `src` dir into sys.path first so module imports work.
THIS_DIR = os.path.dirname(os.path.abspath(__file__))
PARENT_DIR = os.path.dirname(THIS_DIR)
if THIS_DIR not in sys.path:
    sys.path.insert(0, THIS_DIR)
if PARENT_DIR not in sys.path:
    sys.path.insert(0, PARENT_DIR)

# Import the sibling modules using importlib after ensuring THIS_DIR is
# on sys.path.
_analyze_mod = None
_hf_mod = None


def log(message: str, level: int = 1) -> None:
    """Log to $LOG_FILE only, honoring $LOG_LEVEL (0 silent, 1 info, 2 debug).
    If $LOG_FILE is not set or verbosity is lower than level, do nothing.
    """
    try:
        log_level = int(os.environ.get("LOG_LEVEL", "0"))
    except Exception:
        log_level = 0
    if log_level < level:
        return
    log_file = os.environ.get("LOG_FILE")
    if not log_file:
        return
    try:
        with open(log_file, "a", encoding="utf-8") as f:
            f.write(message + "\n")
    except Exception:
        # Swallow logging errors to avoid interfering with stdout NDJSON
        pass


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
    return _hf_mod.fetch_model_metrics(model_id)


def _hf_dataset_quality_fn(dataset_url: str) -> dict:
    _ensure_modules_loaded()
    if hasattr(_hf_mod, "fetch_dataset_quality"):
        return _hf_mod.fetch_dataset_quality(dataset_url)
    return {}


def _hf_discover_dataset_for_model(model_id_or_url: str) -> str:
    _ensure_modules_loaded()
    if hasattr(_hf_mod, "discover_hf_dataset_url_for_model"):
        return _hf_mod.discover_hf_dataset_url_for_model(model_id_or_url)
    return ""

def _hf_readme_text_fn(model_id_or_url: str) -> str:
    _ensure_modules_loaded()
    if hasattr(_hf_mod, "fetch_model_card_text"):
        return _hf_mod.fetch_model_card_text(model_id_or_url)
    return ""


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


## URL parsing moved to src/url_handler.py


def collect_and_write(
    repos: Iterable[str],
    models: Iterable[object],
    output: Optional[str] = None,
    append: bool = False,
) -> int:
    """Collect repo/model data and write each record as NDJSON."""

    start_ts = time.time()
    written = 0

    use_stdout = output is None or output == "-"
    mode = "a" if append else "w"
    if use_stdout:
        out_fp = sys.stdout
    else:
        out_fp = open(output, mode, encoding="utf-8")

    def _timestamp() -> str:
        return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())

    try:
        for repo in repos:
            try:
                payload = _analyze_repo_fn(repo)
            except Exception as exc:
                payload = {"error": str(exc)}
            record = {
                "type": "repo",
                "source": repo,
                "collected_at": _timestamp(),
                "data": payload,
            }
            write_ndjson_line(out_fp, record)
            written += 1

        for model in models:
            code_url: Optional[str] = None
            dataset_url: Optional[str] = None
            model_id = model
            if isinstance(model, tuple) and len(model) == 3:
                code_url, dataset_url, model_id = model  # type: ignore[misc]
            model_str = str(model_id)
            try:
                payload = _hf_meta_fn(model_str)
            except Exception as exc:
                payload = {"error": str(exc)}

            record: Dict[str, Any] = {
                "type": "hf_model",
                "source": model_str,
                "collected_at": _timestamp(),
                "data": payload,
            }
            if code_url:
                record["code_url"] = code_url
            if dataset_url:
                record["dataset_url"] = dataset_url

            write_ndjson_line(out_fp, record)
            written += 1
    finally:
        if not use_stdout:
            out_fp.close()

    elapsed = time.time() - start_ts
    target = output or "stdout"
    log(f"Wrote {written} records to {target} in {elapsed:.2f}s", level=1)
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
        help="Path to write NDJSON output (default: stdout)",
    )
    p.add_argument(
        "--append",
        action="store_true",
        help="Append to the output file instead of overwriting",
    )
    return p.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> int:
    args = parse_args(argv)

    repos: List[str] = list(args.repo or [])
    models: List[str] = list(args.model or [])

    if args.repos_file:
        repos.extend(read_lines_file(args.repos_file))
    if args.models_file:
        # Delegate to url_handler for full processing and NDJSON emission
        try:
            from url_handler import handle_input_file  # type: ignore
        except Exception:
            handle_input_file = None  # type: ignore
        if handle_input_file:
            try:
                records = handle_input_file(args.models_file)
            except Exception as e:
                log(f"url_handler failed: {e}", level=1)
                records = []
            use_stdout = not args.output or args.output == "-"
            mode = "a" if args.append else "w"
            if use_stdout:
                out = sys.stdout
            else:
                out = open(args.output, mode, encoding="utf-8")
            written = 0
            try:
                for rec in records:
                    write_ndjson_line(out, rec)
                    written += 1
            finally:
                if not use_stdout:
                    out.close()
            log(
                f"Wrote {written} records via url_handler to {args.output or 'stdout'}",
                level=1,
            )
            return 0
        else:
            log("url_handler not available; no models processed", level=1)
            return 1

    if not repos and not models:
        print(
            "No repos or models provided. Use --repo/--model or "
            "--repos-file/--models-file."
        )
        return 2

    return collect_and_write(
        repos,
        models,
        output=args.output,
        append=bool(args.append),
    )


if __name__ == "__main__":
    raise SystemExit(main())
