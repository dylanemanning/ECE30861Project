from typing import Any
from typing import List, Tuple, Dict
import os
import json
import re
import requests
<<<<<<< HEAD
import sys
import os
# Ensure src is in sys.path for imports
SRC_DIR = os.path.dirname(os.path.abspath(__file__))
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)
import HF_API_Integration as hf
import genai_readme_analysis
from genai_readme_analysis import analyze_metrics, discover_dataset_url_with_genai
=======
import importlib

# Robust import for HF_API_Integration: try package import, then top-level, then relative.
hf = None
for modname in ("src.HF_API_Integration", "HF_API_Integration"):
    try:
        hf = importlib.import_module(modname)
        break
    except Exception:
        hf = None
        continue
if hf is None:
    try:
        # Try relative import when running as package
        hf = importlib.import_module(f".{'HF_API_Integration'}", package=__package__)
    except Exception:
        hf = None

# Robust import for genai_readme_analysis
gra = None
for modname in ("src.genai_readme_analysis", "genai_readme_analysis"):
    try:
        gra = importlib.import_module(modname)
        break
    except Exception:
        gra = None
        continue
if gra is None:
    try:
        gra = importlib.import_module(f".{ 'genai_readme_analysis' }", package=__package__)
    except Exception:
        gra = None

if gra is None:
    # fallback stubs to avoid runtime crashes; functions will raise if used
    def analyze_metrics(*args, **kwargs):
        raise ImportError("genai_readme_analysis not available")

    def discover_dataset_url_with_genai(*args, **kwargs):
        raise ImportError("genai_readme_analysis not available")
else:
    analyze_metrics = getattr(gra, "analyze_metrics")
    discover_dataset_url_with_genai = getattr(gra, "discover_dataset_url_with_genai")
>>>>>>> ce17111e561f80f8ca8f87ed48f22c395d41f875


def parse_triple(line: str) -> Tuple[str, str, str]:
    """Parse a line in the format: code_url,dataset_url,model_url.
    - Single token lines are treated as model-only (code,dataset empty)
    - Extra commas beyond three are merged into the model field
    Returns (code, dataset, model) trimmed.
    """
    parts = [p.strip() for p in line.split(",")]
    if len(parts) == 1:
        return "", "", parts[0]
    if len(parts) < 3:
        parts += [""] * (3 - len(parts))
    if len(parts) > 3:
        parts = [parts[0], parts[1], ",".join(parts[2:]).strip()]
    return parts[0], parts[1], parts[2]


def read_url_file(path: str) -> List[Tuple[str, str, str]]:
    triples: List[Tuple[str, str, str]] = []
    with open(path, "r", encoding="utf-8") as f:
        for raw in f:
            s = raw.strip()
            if not s or s.startswith("#"):
                continue
            triples.append(parse_triple(s))
    return triples


def is_placeholder_or_non_hf_dataset(url: str) -> bool:
    if not url:
        return True
    s = url.strip()
    if s.lower() in {"none", "null", "na", "n/a", "-"}:
        return True
    if not s.startswith("https://huggingface.co/datasets/"):
        return True
    return False


def handle_input_file(path: str) -> List[Dict[str, Any]]:
    """High-level orchestrator:
    - For each line (code,dataset,model), call GenAI for metrics.
    - If dataset URL present and valid HF, compute dataset_quality via HF.
    - If dataset URL missing/invalid, set dataset_url_flag=False and invoke GenAI dataset discovery; if found, compute HF dataset_quality.
    Returns list of flat records ready for NDJSON emission.
    """
    results: List[Dict[str, Any]] = []
    triples = read_url_file(path)

    def extract_model_id(url_or_id: str) -> str:
        s = url_or_id.strip()
        if s.startswith("https://huggingface.co/"):
            path = s[len("https://huggingface.co/"):]
            parts = [p for p in path.split("/") if p]
            if len(parts) >= 2:
                return f"{parts[0]}/{parts[1]}"
            elif len(parts) == 1:
                return parts[0]
            else:
                return s
        return s

    for code_url, dataset_url, model_url in triples:
        code_url = code_url.strip() if code_url else ""
        dataset_url = dataset_url.strip() if dataset_url else ""
        model_url = model_url.strip() if model_url else ""
    # print(f"[DEBUG] Model: {model_url}")
    # print(f"[DEBUG] code_url: {code_url}")
    # print(f"[DEBUG] dataset_url: {dataset_url}")

        # Extract model ID for HF API
        model_id = extract_model_id(model_url)
        license_info = hf.get_license_info(model_id)
        compat_score = license_info.get("lgplv21_compat_score", 0)
        # Ensure license_latency is integer milliseconds, rounded
        raw_license_latency = license_info.get("license_latency", 0)
        try:
            license_latency = int(round(float(raw_license_latency) * 1000))
        except Exception:
            license_latency = 0

        # Model name extraction: always use the second and third segment for Hugging Face URLs
        def extract_model_name(url):
            s = url.strip()
            if s.startswith("https://huggingface.co/"):
                parts = [p for p in s[len("https://huggingface.co/"):].split("/") if p]
                if len(parts) >= 2:
                    return parts[1]
                elif len(parts) == 1:
                    return parts[0]
                else:
                    return s
            parts = s.split("/")
            if parts[-1].lower() == "main" and len(parts) > 1:
                return parts[-2]
            return parts[-1]

        name = extract_model_name(model_url)

        # All other metrics from GenAI, passing code and dataset URLs for relevant metrics
        metrics = analyze_metrics(
            readme="",  # Optionally fetch README if needed
            code=code_url,
            metadata="",
            dataset_link=dataset_url,
            model=model_url,
        ) or {}

        # Calculate dataset_and_code_score and latency based on code/dataset link presence
        if code_url or dataset_url:
            dataset_and_code_score = 1.0
            dataset_and_code_score_latency = 1
        else:
            dataset_and_code_score = 0.0
            dataset_and_code_score_latency = 1
        dataset_quality = metrics.get("dataset_quality", None)
        if dataset_quality is None:
            dataset_quality = 0.0
        code_quality = metrics.get("code_quality", None)
        if code_quality is None:
            code_quality = 0.0
        dataset_quality_latency = metrics.get("dataset_quality_latency", None)
        if dataset_quality_latency is None:
            dataset_quality_latency = 0
        code_quality_latency = metrics.get("code_quality_latency", None)
        if code_quality_latency is None:
            code_quality_latency = 0
        # if not code_url and not dataset_url:
        #     try:
        #         dataset_quality = float(dataset_quality) * 0.1 if dataset_quality is not None else 0.0
        #     except Exception:
        #         dataset_quality = 0.0
        # Round dataset_quality to 3 decimal places
        try:
            dataset_quality = round(float(dataset_quality), 3) if dataset_quality is not None else None
        except Exception:
            pass
        rec: Dict[str, Any] = {
            "name": name,
            "category": "MODEL",
            "license": 1 if compat_score else 0,
            "license_latency": license_latency,
            "bus_factor": metrics.get("bus_factor", None),
            "bus_factor_latency": int(round(float(metrics.get("bus_factor_latency", 0)))) if "bus_factor_latency" in metrics else None,
            "dataset_quality": dataset_quality,
            "dataset_quality_latency": int(round(float(dataset_quality_latency))),
            "code_quality": code_quality,
            "code_quality_latency": int(round(float(code_quality_latency))),
            "dataset_and_code_score": dataset_and_code_score,
            "dataset_and_code_score_latency": dataset_and_code_score_latency,
        }
        # Add remaining GenAI metrics, rounding latency fields to int ms
        for k, v in metrics.items():
            if k in rec:
                continue
            if k.endswith("_latency"):
                try:
                    rec[k] = int(round(float(v)))
                except Exception:
                    rec[k] = v
            else:
                rec[k] = v
        results.append(rec)
    return results


if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python -m src.url_handler <input_file>")
        sys.exit(2)
    out = handle_input_file(sys.argv[1])
    for rec in out:
        print(json.dumps(rec, ensure_ascii=False))
