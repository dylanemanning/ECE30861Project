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
) -> int:
    """Collect data and write NDJSON to stdout only."""
    written = 0
    start_ts = time.time()
    out = sys.stdout
    try:
        # Optional: still support repos, but don't emit to NDJSON to keep schema consistent.
        for repo in repos:
            try:
                _ = _analyze_repo_fn(repo)
                log(f"Analyzed repo {repo}", level=2)
            except Exception as e:
                log(f"Repo analysis failed for {repo}: {e}", level=1)

        # Lazy import to avoid overhead when not needed
        try:
            from genai_readme_analysis import analyze_metrics, discover_dataset_url_with_genai  # type: ignore
        except Exception:
            analyze_metrics = None  # type: ignore
            discover_dataset_url_with_genai = None  # type: ignore

        def _round2(x: Any) -> Any:
            try:
                if isinstance(x, float):
                    return round(x, 2)
                if isinstance(x, (int,)):
                    return x
                return x
            except Exception:
                return x

        for model in models:
            code_url = ""
            dataset_url = ""
            model_url = model
            if isinstance(model, tuple) and len(model) == 3:
                code_url, dataset_url, model_url = model  # type: ignore
            # Fetch README markdown from model card to help GenAI find dataset description sections
            readme_text = ""
            try:
                readme_text = _hf_readme_text_fn(str(model_url)) or ""
                if readme_text:
                    log(f"Fetched README for {model_url} ({len(readme_text)} chars)", level=2)
            except Exception as e:
                log(f"Failed to fetch README for {model_url}: {e}", level=1)
            # Normalize dataset_url: treat placeholders and non-HF links as empty to trigger discovery
            raw_dataset = str(dataset_url or "").strip()
            if raw_dataset and raw_dataset.lower() in {"none", "null", "na", "n/a", "-"}:
                log(f"Input dataset placeholder detected; will attempt discovery (value={raw_dataset})", level=2)
                dataset_url = ""
            elif raw_dataset and not raw_dataset.startswith("https://huggingface.co/datasets/"):
                # Provided but not a HF datasets URL; ignore to allow discovery
                log(f"Non-HF dataset link provided; ignoring for HF quality and attempting discovery: {raw_dataset}", level=2)
                dataset_url = ""
            else:
                dataset_url = raw_dataset
            try:
                hf = _hf_meta_fn(str(model_url))
            except Exception as e:
                log(f"HF fetch failed for {model_url}: {e}", level=1)
                hf = {"model_id": str(model_url), "error": str(e)}

            genai: Dict[str, Any] = {}
            if analyze_metrics:
                try:
                    genai = analyze_metrics(
                        readme=readme_text,
                        code=code_url,
                        metadata="",
                        dataset_link=dataset_url,
                        model=str(model_url),
                    ) or {}
                except Exception as ee:
                    log(f"GenAI analysis failed for code={code_url} dataset={dataset_url}: {ee}", level=1)

            # If dataset_url is a HF dataset, compute dataset_quality using HF API (preferred)
            dsq: Dict[str, Any] = {}
            # If dataset not provided, try GenAI discovery using README context
            if not dataset_url:
                log(f"No dataset URL provided for model={model_url}; attempting GenAI discovery", level=2)
                try:
                    if 'discover_dataset_url_with_genai' in locals() and discover_dataset_url_with_genai:
                        ds_disc = discover_dataset_url_with_genai(readme=readme_text, model=str(model_url)) or {}
                    else:
                        ds_disc = {}
                except Exception as e:
                    log(f"GenAI dataset discovery failed for {model_url}: {e}", level=1)
                    ds_disc = {}
                discovered = str(ds_disc.get("dataset_url") or "").strip() if isinstance(ds_disc, dict) else ""
                if discovered:
                    # Normalize if only id was returned
                    if not discovered.startswith("http") and "/" in discovered:
                        discovered = f"https://huggingface.co/datasets/{discovered}"
                    if discovered.startswith("https://huggingface.co/datasets/"):
                        dataset_url = discovered
                        log(f"Using GenAI-discovered dataset URL: {dataset_url}", level=2)
                    else:
                        log(f"GenAI discovery returned non-HF dataset URL; ignoring: {discovered}", level=1)
            # HF fallback discovery if still missing
            if (not dataset_url):
                fallback = _hf_discover_dataset_for_model(str(model_url))
                if fallback:
                    dataset_url = fallback
                    log(f"Using HF-discovered dataset URL: {dataset_url}", level=2)
                else:
                    log(f"HF fallback could not find a dataset for model={model_url}", level=2)
            if dataset_url and dataset_url.startswith("http"):
                try:
                    log(f"Computing dataset_quality for dataset={dataset_url}", level=2)
                    dsq = _hf_dataset_quality_fn(dataset_url) or {}
                except Exception as e:
                    log(f"HF dataset quality failed for {dataset_url}: {e}", level=1)

            # Build flat record per expected schema
            name = str(hf.get("model_id", str(model_url))).split("/")[-1]
            record: Dict[str, Any] = {
                "name": name,
                "category": "MODEL",
            }

            # Intentionally omit size and license metrics. HF module now only provides model_id.

            # GenAI-derived fields (pass through if present)
            for k in [
                "ramp_up_time",
                "performance_claims",
                "dataset_and_code_score",
                # Only include the metrics we now support from GenAI
            ]:
                if k in genai:
                    record[k] = _round2(genai[k])
                lat_k = f"{k}_latency"
                if lat_k in genai:
                    try:
                        record[lat_k] = int(genai[lat_k])
                    except Exception:
                        pass

            # Prefer HF dataset_quality if available; otherwise, if GenAI provided dataset_quality (because dataset was missing), include it
            if dsq.get("dataset_quality") is not None:
                record["dataset_quality"] = _round2(dsq.get("dataset_quality"))
                if dsq.get("dataset_quality_latency") is not None:
                    record["dataset_quality_latency"] = int(dsq.get("dataset_quality_latency") or 0)
            elif genai.get("dataset_quality") is not None:
                record["dataset_quality"] = _round2(genai.get("dataset_quality"))
                lat = genai.get("dataset_quality_latency")
                if lat is not None:
                    try:
                        record["dataset_quality_latency"] = int(lat)
                    except Exception:
                        pass
            # If HF dataset quality not computed, we do not include it (no GenAI dataset_quality now)

            write_ndjson_line(out, record)
            written += 1

    finally:
        pass

    elapsed = time.time() - start_ts
    # Print summary to stderr so stdout remains pure NDJSON when used in pipes
    log(f"Wrote {written} records to stdout in {elapsed:.2f}s", level=1)
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
    # Output is always stdout; no --output flag.
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
            out = sys.stdout
            written = 0
            for rec in records:
                write_ndjson_line(out, rec)
                written += 1
            log(f"Wrote {written} records via url_handler", level=1)
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

    return collect_and_write(repos, models)


if __name__ == "__main__":
    raise SystemExit(main())
