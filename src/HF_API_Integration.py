"""HF API integration helpers used by tests and other modules.

This module provides a small, deterministic set of helper functions for
working with Hugging Face model and dataset metadata. Tests mock
HTTP calls so functions can be simple wrappers around requests.
"""

import math
import time
import logging
from typing import Any, Dict
import requests

logger = logging.getLogger("HF_API_Integration")


def get_huggingface_model_metadata(model_id: str) -> dict | None:
    url = f"https://huggingface.co/api/models/{model_id}"
    try:
        resp = requests.get(url, timeout=10)
        resp.raise_for_status()
        data = resp.json()
        return {
            "model_id": data.get("modelId") or data.get("model_id") or model_id,
            "downloads": int(data.get("downloads", 0) or 0),
            "likes": int(data.get("likes", 0) or 0),
            "lastModified": data.get("lastModified", "unknown"),
        }
    except Exception as e:
        logger.debug("get_huggingface_model_metadata error: %s", e)
        return None


def _normalize_size(size_bytes: int) -> float:
    try:
        b = int(size_bytes or 0)
        if b <= 1_000_000:
            return 0.0
        if b >= 10_000_000_000:
            return 1.0
        lo = math.log(1_000_000)
        hi = math.log(10_000_000_000)
        val = (math.log(b) - lo) / (hi - lo)
        return max(0.0, min(1.0, round(val, 6)))
    except Exception:
        return 0.0


def _compat_from_capacity(size_bytes: int, capacity_bytes: int) -> float:
    try:
        if not capacity_bytes or capacity_bytes <= 0:
            return 0.0
        ratio = float(size_bytes or 0) / float(capacity_bytes)
        if ratio <= 0.25:
            return 1.0
        if ratio <= 0.5:
            return 0.85
        if ratio <= 1.0:
            return 0.70
        if ratio <= 2.0:
            return 0.40
        if ratio <= 4.0:
            return 0.20
        return 0.0
    except Exception:
        return 0.0


def _compute_size_score(total_bytes: int) -> Dict[str, float]:
    capacities = {
        "raspberry_pi": 200_000_000,
        "jetson_nano": 1_000_000_000,
        "desktop_pc": 8_000_000_000,
        "aws_server": 20_000_000_000,
    }
    out: Dict[str, float] = {}
    for dev, cap in capacities.items():
        out[dev] = _compat_from_capacity(int(total_bytes or 0), cap)
    return out


def _normalize_label(tags: list | None) -> float:
    if not tags:
        return 0.0
    ignored = {"model", "test", "example", "pytorch", "tf"}
    meaningful = [t for t in tags if isinstance(t, str) and t and t.lower() not in ignored]
    n = len(meaningful)
    if n == 0:
        return 0.0
    if n <= 2:
        return 0.5
    return 1.0


def _normalize_completeness(data: dict) -> float:
    keys = ["description", "tags", "license", "downloads", "likes"]
    if not data:
        return 0.0
    present = 0
    for k in keys:
        v = data.get(k)
        if v is None:
            continue
        if isinstance(v, (str, list, dict)) and not v:
            continue
        present += 1
    return round(present / len(keys), 6)


def _score_tags(tags: list | None) -> float:
    if not isinstance(tags, (list, tuple)):
        return 0.0
    meaningful = [t for t in tags if isinstance(t, str) and t.strip()]
    n = len(meaningful)
    if n <= 0:
        return 0.0
    if n <= 2:
        return 0.5
    if n == 3:
        return 0.8
    if n == 4:
        return 0.9
    return 1.0


def _score_description(text: str | None) -> float:
    if not text:
        return 0.0
    L = len(str(text))
    return min(1.0, round(L / 2000.0, 6))


def _score_license(lic: Any) -> float:
    if not lic:
        return 0.0
    if isinstance(lic, str):
        s = lic.strip().lower()
        if not s:
            return 0.0
        if "mit" in s or "apache" in s or "bsd" in s:
            return 1.0
        return 0.5
    if isinstance(lic, dict):
        if lic.get("spdx"):
            return 1.0
        return 0.0
    return 0.5


def _score_schema(data: dict) -> float:
    if not isinstance(data, dict):
        return 0.0
    cd = data.get("cardData")
    if isinstance(cd, dict) and (cd.get("features") or cd.get("dataset_info")):
        return 1.0
    if data.get("configs") is not None:
        return 0.5
    return 0.0


def _score_completeness(data: dict) -> float:
    return _normalize_completeness(data)


def _score_downloads(n: int | None) -> float:
    try:
        x = int(n or 0)
        if x <= 0:
            return 0.0
        if x < 100:
            return 0.2
        if x < 1000:
            return 0.5
        if x < 10000:
            return 0.8
        return 1.0
    except Exception:
        return 0.0


def _extract_dataset_id(s: str) -> str:
    if not s:
        return ""
    s = str(s).strip()
    if s.startswith(','):
        s = s[1:]
    prefix = "https://huggingface.co/datasets/"
    if s.startswith(prefix):
        rest = s[len(prefix):]
        parts = [p for p in rest.split("/") if p]
        if not parts:
            return ""
        if len(parts) >= 2:
            return f"{parts[0]}/{parts[1]}"
        return parts[0]
    return s


def _extract_model_id(s: str) -> str:
    if not s:
        return ""
    s = str(s).strip()
    if s.startswith(','):
        s = s[1:]
    prefix = "https://huggingface.co/"
    if s.startswith(prefix):
        rest = s[len(prefix):]
        parts = [p for p in rest.split("/") if p]
        if not parts:
            return ""
        if len(parts) >= 2:
            return f"{parts[0]}/{parts[1]}"
        return parts[0]
    return s


def fetch_model_card_text(model_id: str) -> str:
    urls = [
        f"https://huggingface.co/{model_id}/raw/main/README.md",
        f"https://huggingface.co/{model_id}/raw/main/README",
    ]
    for u in urls:
        try:
            r = requests.get(u, timeout=5)
            if getattr(r, "status_code", 0) == 200:
                return r.text or ""
        except Exception:
            continue
    return ""


def discover_hf_dataset_url_for_model(model_id: str) -> str:
    url = f"https://huggingface.co/api/models/{model_id}"
    try:
        r = requests.get(url, timeout=10)
        r.raise_for_status()
        data = r.json()
        cd = data.get("cardData", {}) or {}
        datasets = cd.get("datasets") or []
        if datasets:
            ds = datasets[0]
            return f"https://huggingface.co/datasets/{ds}"
        tags = data.get("tags") or []
        for t in tags:
            if isinstance(t, str) and t.startswith("dataset:"):
                return "https://huggingface.co/datasets/" + t.split(":", 1)[1]
    except Exception:
        pass
    return ""


def fetch_dataset_quality(dataset_id: str) -> dict:
    t0 = time.perf_counter()
    try:
        url = f"https://huggingface.co/api/datasets/{dataset_id}"
        r = requests.get(url, timeout=10)
        r.raise_for_status()
        data = r.json()
        card = data.get("cardData", {}) or {}
        tags = data.get("tags", []) or []
        downloads = data.get("downloads", 0) or 0
        score = (
            0.5 * _normalize_completeness(card)
            + 0.3 * _score_tags(tags)
            + 0.2 * _score_downloads(downloads)
        )
        return {
            "dataset_id": dataset_id,
            "dataset_quality": round(float(score), 6),
            "dataset_quality_latency": int(round((time.perf_counter() - t0) * 1000)),
        }
    except Exception as e:
        return {"dataset_id": dataset_id, "error": str(e)}


def fetch_model_metrics(model_id: str) -> dict:
    t0 = time.perf_counter()
    meta = get_huggingface_model_metadata(model_id)
    if not meta:
        return {"model_id": model_id, "error": "no metadata"}
    downloads = meta.get("downloads", 0) or 0
    likes = meta.get("likes", 0) or 0
    return {
        "model_id": meta.get("model_id", model_id),
        "downloads": int(downloads),
        "likes": int(likes),
        "downloads_norm": _score_downloads(downloads),
        "likes_norm": _score_downloads(likes),
        "latency": int(round((time.perf_counter() - t0) * 1000)),
    }


def check_license_lgplv21(license_str: str) -> bool:
    if not license_str:
        return False
    s = str(license_str).lower()
    return any(k in s for k in ("lgpl", "gpl", "mit", "bsd"))


def log(msg: str) -> None:
    logger.info(msg)


from license_compat import license_compat

def get_license_info(model_id: str) -> dict:
    """
    Fetch license info and LGPLv2.1 compatibility for a Hugging Face model.
    Returns a dict with model_id, license, lgplv21_compat_score, and error (if any).
    """
    return license_compat(model_id)
