
import requests
import time
import os
import json
from typing import Dict, Any, Optional

HF_API_BASE: str = "https://huggingface.co/api/models"

def log(message: str, level: int = 1) -> None:
    log_level = int(os.environ.get("LOG_LEVEL", "0"))
    log_file = os.environ.get("LOG_FILE")
    if log_level >= level:
        if log_file:
            with open(log_file, "a") as f:
                f.write(message + "\n")
        else:
            print(message)

def fetch_model_metrics(model_id: str) -> Optional[Dict[str, Any]]:
    """
    Fetches and normalizes quality metrics and latency for a Hugging Face model.
    Returns a dict with metrics in [0,1] and latency in seconds.
    """
    url = f"{HF_API_BASE}/{model_id}"
    start = time.time()
    try:
        response = requests.get(url, timeout=10)
        latency = time.time() - start
        response.raise_for_status()
        data = response.json()
        # Example metrics: downloads, likes (normalize to [0,1] for demo)
        downloads = data.get("downloads", 0)
        likes = data.get("likes", 0)
        # Simple normalization (for demo): assume max 100k downloads, 10k likes
        norm_downloads = min(downloads / 100_000, 1.0)
        norm_likes = min(likes / 10_000, 1.0)
        metrics = {
            "model_id": model_id,
            "downloads": downloads,
            "likes": likes,
            "downloads_norm": round(norm_downloads, 3),
            "likes_norm": round(norm_likes, 3),
            "lastModified": data.get("lastModified", "unknown"),
            "latency": round(latency, 3)
        }
        log(f"Fetched metrics for {model_id}: {metrics}", level=2)
        return metrics
    except Exception as e:
        log(f"Error fetching {model_id}: {e}", level=1)
        return None

def output_ndjson(record: Dict[str, Any]) -> None:
    print(json.dumps(record, separators=(",", ":")))


# Backwards-compatible name used by the runner/tests
def get_huggingface_model_metadata(model_id: str) -> Optional[Dict[str, Any]]:
    return fetch_model_metrics(model_id)

if __name__ == "__main__":
    # Example: process a list of model IDs (for CLI integration)
    sample_models = ["google/gemma-3-270m", "bert-base-uncased"]
    for model_id in sample_models:
        metrics = fetch_model_metrics(model_id)
        if metrics:
            output_ndjson(metrics)
        else:
            log(f"Failed to fetch metrics for {model_id}", level=1)
