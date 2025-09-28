from typing import List, Tuple, Dict, Any
import os
import json
import re
import requests
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

    for code_url, dataset_url, model_url in triples:
        raw_model = str(model_url or "").strip()

        def is_hf_model_url(u: str) -> bool:
            return u.startswith("https://huggingface.co/") or ("/" in u and not u.startswith("http"))

        def is_github_url(u: str) -> bool:
            return u.startswith("https://github.com/")

        def normalize_model_id_and_name(u: str) -> Tuple[str, str]:
            # Returns (model_id_for_calls, display_name)
            if not u:
                return "", ""
            if is_hf_model_url(u):
                try:
                    mid = hf._extract_model_id(u)  # type: ignore[attr-defined]
                except Exception:
                    mid = u
                name = mid.split("/")[-1] if "/" in mid else mid
                return mid, name
            if is_github_url(u):
                # https://github.com/{owner}/{repo}[/...]
                path = u[len("https://github.com/"):]
                parts = [p for p in path.split("/") if p]
                repo = parts[1] if len(parts) >= 2 else parts[0] if parts else u
                return u, repo
            # Fallback: plain id or URL
            return u, (u.split("/")[-1] if "/" in u else u)

        def fetch_readme_text(u: str) -> str:
            # Try HF README when it's an HF id/url
            mid, _ = normalize_model_id_and_name(u)
            if is_hf_model_url(u) or ("/" in mid and not mid.startswith("http")):
                if hasattr(hf, "fetch_model_card_text"):
                    txt = hf.fetch_model_card_text(mid)
                    if txt:
                        return txt
            # GitHub README fallback
            if is_github_url(u):
                try:
                    path = u[len("https://github.com/"):]
                    parts = [p for p in path.split("/") if p]
                    if len(parts) >= 2:
                        owner, repo = parts[0], parts[1]
                        for branch in ("main", "master"):
                            raw_url = f"https://raw.githubusercontent.com/{owner}/{repo}/{branch}/README.md"
                            resp = requests.get(raw_url, timeout=10)
                            if resp.status_code == 200 and resp.text:
                                return resp.text
                except Exception:
                    pass
            return ""

        model_id, display_name = normalize_model_id_and_name(raw_model)
        readme_text = fetch_readme_text(raw_model)

        # Normalize dataset URL and set flag
        ds_input = dataset_url.strip() if dataset_url else ""
        dataset_url_flag = not is_placeholder_or_non_hf_dataset(ds_input)
        ds_url = ds_input if dataset_url_flag else ""

        # GenAI metrics (always)
        metrics = analyze_metrics(
            readme=readme_text,
            code=code_url or "",
            metadata="",
            dataset_link=ds_url,
            model=model_id,
        ) or {}

        # Dataset quality path
        dsq: Dict[str, Any] = {}
        if dataset_url_flag and ds_url:
            dsq = hf.fetch_dataset_quality(ds_url) if hasattr(hf, "fetch_dataset_quality") else {}
        else:
            # Discover dataset via GenAI if missing
            disc = discover_dataset_url_with_genai(readme=readme_text, model=model_id) or {}
            discovered = str(disc.get("dataset_url") or "").strip()
            if discovered:
                if not discovered.startswith("http") and "/" in discovered:
                    discovered = f"https://huggingface.co/datasets/{discovered}"
                if discovered.startswith("https://huggingface.co/datasets/"):
                    dsq = hf.fetch_dataset_quality(discovered) if hasattr(hf, "fetch_dataset_quality") else {}

        # Build output record
        rec: Dict[str, Any] = {"name": display_name, "category": "MODEL"}
        for k in [
            "ramp_up_time",
            "performance_claims",
            "dataset_and_code_score",
        ]:
            if k in metrics:
                rec[k] = metrics[k]
            lat_k = f"{k}_latency"
            if lat_k in metrics:
                try:
                    rec[lat_k] = int(metrics[lat_k])
                except Exception:
                    pass
        if dsq.get("dataset_quality") is not None:
            rec["dataset_quality"] = dsq.get("dataset_quality")
            if dsq.get("dataset_quality_latency") is not None:
                try:
                    rec["dataset_quality_latency"] = int(dsq.get("dataset_quality_latency") or 0)
                except Exception:
                    pass
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
