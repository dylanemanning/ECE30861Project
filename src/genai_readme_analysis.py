import os
import requests
import re
import json
from typing import Dict, Any


def analyze_with_genai(readme: str = "", code: str = "", metadata: str = "", dataset_link: str = "", model: str = ""):
    """
    Call Purdue GenAI Studio to compute ONLY the metrics:
    - ramp_up_time, ramp_up_time_latency
    - performance_claims, performance_claims_latency
    - dataset_and_code_score, dataset_and_code_score_latency

    This call must NOT include dataset_url or dataset_quality.
    """
    api_key = os.environ.get("GEN_AI_STUDIO_API_KEY") or "sk-91a78299639b4da595d22241ca2161d0"
    prompt = (
        "You are an expert evaluator. Return ONLY a JSON object with exactly these keys:\n"
        "- ramp_up_time (float in [0,1])\n"
        "- ramp_up_time_latency (int milliseconds)\n"
        "- performance_claims (float in [0,1])\n"
        "- performance_claims_latency (int milliseconds)\n"
        "- dataset_and_code_score (float in [0,1])\n"
        "- dataset_and_code_score_latency (int milliseconds)\n\n"
        "Metric Operationalization:\n"
        "- Ramp Up Time: Assess ease of getting started from README/tutorials/examples.\n"
        "- Performance Claims: Check README/paper claims and whether they cite/align with recognized benchmarks; score verified/credible claims higher.\n"
        "- Dataset and Code Score: Presence and usability of dataset links and example code; score 1 if clearly documented and accessible, else lower.\n\n"
        "Strict output rules:\n"
        "- Output ONLY JSON. No code fences, no commentary.\n"
        "- Latencies must be integers in milliseconds.\n"
        "- Do NOT include dataset URLs or dataset quality in this response.\n\n"
        f"Model (may be a full HF URL or <owner>/<name>):\n{model}\n\n"
        "Context\n"
        f"README (may be empty):\n{readme}\n"
        f"Code:\n{code}\n"
        f"Metadata:\n{metadata}\n"
        f"Dataset Link provided by user (may be empty; ignore for this response):\n{dataset_link}\n"
    )
    url = "https://genai.rcac.purdue.edu/api/chat/completions"
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    body = {"model": "llama3.1:latest", "messages": [{"role": "user", "content": prompt}], "stream": False}
    timeout = float(os.environ.get("GENAI_TIMEOUT", "15"))
    response = requests.post(url, headers=headers, json=body, timeout=timeout)
    response.raise_for_status()
    return response.json()


def _parse_llm_content_to_json(content: str) -> Dict[str, Any]:
    """
    Extract a JSON object from the LLM content. Handles fenced ```json blocks
    and raw JSON in the message. Returns a dict; returns {} on failure.
    """
    match = re.search(r"```json\n(.*?)```", content, re.DOTALL)
    raw = None
    if match:
        raw = match.group(1)
    else:
        brace_match = re.search(r"\{[\s\S]*\}$", content.strip())
        if brace_match:
            raw = brace_match.group(0)
    if not raw:
        return {}
    try:
        return json.loads(raw)
    except Exception:
        return {}


def _flatten_metrics(m: Dict[str, Any]) -> Dict[str, Any]:
    """Flatten supported metrics into a flat dict: {key: float, key_latency: int}."""
    allowed_scores = {"ramp_up_time", "performance_claims", "dataset_and_code_score"}
    flat: Dict[str, Any] = {}
    for key, val in m.items():
        if key in allowed_scores:
            if isinstance(val, dict):
                score = val.get("score")
                latency = val.get("latency")
                if score is not None:
                    try:
                        flat[key] = float(score)
                    except Exception:
                        pass
                if latency is not None:
                    try:
                        flat[f"{key}_latency"] = int(round(float(latency)))
                    except Exception:
                        pass
            else:
                try:
                    flat[key] = float(val)
                except Exception:
                    pass
        elif key.endswith("_latency") and key[:-8] in allowed_scores:
            try:
                flat[key] = int(round(float(val)))
            except Exception:
                pass
    return flat


def analyze_metrics(readme: str = "", code: str = "", metadata: str = "", dataset_link: str = "", model: str = "") -> Dict[str, Any]:
    resp = analyze_with_genai(readme=readme, code=code, metadata=metadata, dataset_link=dataset_link, model=model)
    try:
        content = resp["choices"][0]["message"]["content"]
    except Exception:
        return {}
    parsed = _parse_llm_content_to_json(content)
    if not parsed:
        return {}
    return _flatten_metrics(parsed)


def discover_dataset_url_with_genai(readme: str = "", model: str = "") -> Dict[str, Any]:
    """Ask GenAI to find the most relevant HF dataset URL from the README/model context.
    Returns { dataset_url: str, dataset_discovery_latency: int } or {} on failure.
    """
    api_key = os.environ.get("GEN_AI_STUDIO_API_KEY") or "sk-91a78299639b4da595d22241ca2161d0"
    prompt = (
        "You are a precise information extractor. Return ONLY a JSON object with exactly these keys:\n"
        "- dataset_url (string): The most relevant Hugging Face dataset URL in the form https://huggingface.co/datasets/<owner>/<name> extracted from the README/model context. If none is clearly indicated, return an empty string \"\".\n"
        "- dataset_discovery_latency (int milliseconds): Time you hypothetically spent extracting this info.\n\n"
        "Rules:\n"
        "- Output ONLY JSON. No code fences, no commentary.\n"
        "- If the README mentions multiple datasets, pick the primary one used for training/pretraining.\n"
        "- If no dataset is mentioned, set dataset_url to \"\".\n"
        "- Prefer an exact Hugging Face datasets URL. If only a dataset name is present, return owner/name when known, else just \"\".\n"
        "Hints by model family (only if applicable):\n"
        "- Speech/ASR (whisper, wav2vec, hubert): Common Voice, LibriSpeech/LibriLight, VoxPopuli, TED-LIUM.\n"
        "- QA/NLP: SQuAD/SQuAD2.0; NLI: MNLI/SNLI; Sentiment: SST-2/IMDB; GLUE tasks for general NLU.\n"
        "- Pretraining (BERT-like): BookCorpus + English Wikipedia; OpenWebText or C4 for T5-like.\n"
        "- Vision: ImageNet, COCO; Detection/segmentation variants as noted.\n"
        "- Diffusion: LAION-5B or derivatives.\n\n"
        f"Model (may be a full HF URL or <owner>/<name>):\n{model}\n\n"
        f"README (full text):\n{readme}\n"
    )
    url = "https://genai.rcac.purdue.edu/api/chat/completions"
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    body = {"model": "llama3.1:latest", "messages": [{"role": "user", "content": prompt}], "stream": False}
    timeout = float(os.environ.get("GENAI_TIMEOUT", "15"))
    response = requests.post(url, headers=headers, json=body, timeout=timeout)
    response.raise_for_status()
    try:
        content = response.json()["choices"][0]["message"]["content"]
    except Exception:
        return {}
    parsed = _parse_llm_content_to_json(content)
    if not isinstance(parsed, dict):
        return {}
    out: Dict[str, Any] = {}
    ds = str(parsed.get("dataset_url") or "").strip()
    if ds:
        out["dataset_url"] = ds
    try:
        out["dataset_discovery_latency"] = int(round(float(parsed.get("dataset_discovery_latency", 0))))
    except Exception:
        pass
    return out
