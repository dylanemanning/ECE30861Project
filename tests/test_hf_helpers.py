"""Additional unit tests for HF_API_Integration helpers to improve coverage."""

from __future__ import annotations

import json
import sys
import types
from collections import deque

import pytest
import requests

import src.HF_API_Integration as hf


class DummyResponse:
    """Simple response stub mimicking the subset of requests.Response we rely on."""

    def __init__(self, *, status_code: int = 200, json_data: dict | None = None, text: str = "") -> None:
        self.status_code = status_code
        self._json_data = json_data or {}
        self.text = text

    def json(self) -> dict:
        return self._json_data

    def raise_for_status(self) -> None:
        if self.status_code >= 400:
            raise requests.HTTPError(f"status {self.status_code}")


def test_check_license_lgplv21_uses_license_module(monkeypatch):
    fake_mod = types.SimpleNamespace(license_compat=lambda model_id: {"model_id": model_id, "score": 1})
    monkeypatch.setitem(sys.modules, "license_compat", fake_mod)

    result = hf.check_license_lgplv21("demo-model")

    assert result == {"model_id": "demo-model", "score": 1}


def test_log_writes_to_file(tmp_path, monkeypatch):
    log_path = tmp_path / "log.txt"
    monkeypatch.setenv("LOG_LEVEL", "2")
    monkeypatch.setenv("LOG_FILE", str(log_path))

    hf.log("hello world", level=1)

    assert log_path.read_text().strip() == "hello world"


def test_log_respects_level(monkeypatch, capsys):
    monkeypatch.delenv("LOG_FILE", raising=False)
    monkeypatch.setenv("LOG_LEVEL", "0")

    hf.log("should not print", level=2)

    captured = capsys.readouterr()
    assert captured.out == ""


def test_normalize_size_and_compute_scores():
    assert hf._normalize_size(0) == 0.0
    assert hf._normalize_size(10_000_000_000) == 1.0

    scores = hf._compute_size_score(500_000_000)
    assert scores["raspberry_pi"] == pytest.approx(0.2)
    assert scores["jetson_nano"] == pytest.approx(0.85)
    assert scores["aws_server"] == pytest.approx(1.0)


def test_compat_from_capacity_bands():
    capacity = 100
    assert hf._compat_from_capacity(10, capacity) == 1.0
    assert hf._compat_from_capacity(40, capacity) == 0.85
    assert hf._compat_from_capacity(100, capacity) == 0.70
    assert hf._compat_from_capacity(150, capacity) == 0.40
    assert hf._compat_from_capacity(350, capacity) == 0.20
    assert hf._compat_from_capacity(500, capacity) == 0.0


def test_normalize_label_and_completeness():
    assert hf._normalize_label(None) == 0.0
    assert hf._normalize_label(["model"]) == 0.0
    assert hf._normalize_label(["useful"]) == 0.5
    assert hf._normalize_label(["tag1", "tag2", "tag3"]) == 1.0

    data = {"description": "info", "tags": [1], "license": "MIT", "downloads": 5, "likes": 2}
    assert hf._normalize_completeness(data) == pytest.approx(1.0)


def test_extract_model_and_dataset_ids():
    assert hf._extract_model_id("https://huggingface.co/google/gemma-2b") == "google/gemma-2b"
    assert hf._extract_model_id("bert-base") == "bert-base"

    assert hf._extract_dataset_id("https://huggingface.co/datasets/user/set") == "user/set"
    assert hf._extract_dataset_id("dataset-only") == "dataset-only"


@pytest.mark.parametrize(
    "downloads, expected",
    [
        (0, 0.0),
        (50, 0.2),
        (150, 0.5),
        (1500, 0.8),
        (20_000, 1.0),
    ],
)
def test_score_downloads(downloads, expected):
    assert hf._score_downloads(downloads) == expected


def test_score_helpers_cover_paths():
    assert hf._score_tags(["a", "b", "c", "d", "e"]) == 1.0
    assert hf._score_description("x" * 1000) == 0.5
    assert hf._score_license("MIT") == 1.0
    assert hf._score_license({"type": "custom"}) == 1.0
    assert hf._score_license(123) == 0.5
    assert hf._score_schema({"cardData": {"features": {}}}) == 1.0
    assert hf._score_schema({"configs": ["cpu"]}) == 0.5
    assert hf._score_completeness({"cardData": 1, "tags": 1, "license": 1, "downloads": 1, "siblings": None}) == pytest.approx(0.8)


@pytest.mark.parametrize(
    "exc",
    [requests.exceptions.ConnectionError("boom"), requests.exceptions.Timeout("slow"), ValueError("bad json")],
)
def test_fetch_model_metrics_fallback(exc, monkeypatch):
    def failing_get(*_args, **_kwargs):
        raise exc

    monkeypatch.setattr(hf.requests, "get", failing_get)

    result = hf.fetch_model_metrics("demo-model")

    assert result["downloads"] == 0
    assert result["likes"] == 0


def test_fetch_model_metrics_nonexistent(monkeypatch):
    def failing_get(*_args, **_kwargs):
        raise requests.exceptions.HTTPError("404")

    monkeypatch.setattr(hf.requests, "get", failing_get)

    assert hf.fetch_model_metrics("nonexistent-model") is None


def test_fetch_model_metrics_success(monkeypatch):
    meta = {"downloads": 500000, "likes": 250, "lastModified": "2024-01-01"}

    def fake_get(url, timeout):
        _ = url
        return DummyResponse(json_data={
            "downloads": meta["downloads"],
            "likes": meta["likes"],
            "lastModified": meta["lastModified"],
        })

    monkeypatch.setattr(hf.requests, "get", fake_get)

    result = hf.fetch_model_metrics("abc")

    assert result["downloads_norm"] == pytest.approx(0.5)
    assert result["likes_norm"] == pytest.approx(0.03)
    assert result["model_id"] == "abc"


def test_output_ndjson_writes_compact_json(capsys):
    hf.output_ndjson({"a": 1, "b": 2})
    captured = capsys.readouterr()
    assert captured.out.strip() == json.dumps({"a": 1, "b": 2}, separators=(",", ":"))


def test_fetch_model_card_text_attempts_multiple_urls(monkeypatch):
    responses = deque([
        DummyResponse(status_code=500),
        DummyResponse(status_code=404),
        DummyResponse(status_code=200, text="# README\ncontent"),
    ])

    def fake_get(url, timeout):
        assert timeout == 10
        resp = responses.popleft()
        if resp.status_code >= 400:
            raise requests.HTTPError("fail")
        return resp

    monkeypatch.setattr(hf.requests, "get", fake_get)

    text = hf.fetch_model_card_text("owner/model")
    assert "content" in text


def test_fetch_model_card_text_returns_empty_on_fail(monkeypatch):
    def fake_get(*_args, **_kwargs):
        raise requests.RequestException("unavailable")

    monkeypatch.setattr(hf.requests, "get", fake_get)

    assert hf.fetch_model_card_text("owner/model") == ""


def test_discover_hf_dataset_from_card_data(monkeypatch):
    payload = {
        "cardData": {"datasets": ["user/dataset"]},
        "tags": [],
    }

    monkeypatch.setattr(hf.requests, "get", lambda *_args, **_kwargs: DummyResponse(json_data=payload))

    url = hf.discover_hf_dataset_url_for_model("demo")
    assert url == "https://huggingface.co/datasets/user/dataset"


def test_discover_hf_dataset_from_tags(monkeypatch):
    payload = {
        "cardData": {},
        "tags": ["dataset:news/corpus"],
    }

    monkeypatch.setattr(hf.requests, "get", lambda *_args, **_kwargs: DummyResponse(json_data=payload))

    url = hf.discover_hf_dataset_url_for_model("demo")
    assert url == "https://huggingface.co/datasets/news/corpus"


def test_discover_hf_dataset_handles_errors(monkeypatch):
    def failing_get(*_args, **_kwargs):
        raise requests.RequestException("error")

    monkeypatch.setattr(hf.requests, "get", failing_get)

    assert hf.discover_hf_dataset_url_for_model("demo") == ""


def test_fetch_dataset_quality_success(monkeypatch):
    payload = {
        "cardData": {
            "description": "d" * 1000,
            "license": "mit",
            "datasets": ["owner/name"],
            "features": {"text": "string"},
        },
        "tags": ["dataset:owner/name", "extra", "another"],
        "license": "apache-2.0",
        "downloads": 15000,
        "configs": ["cpu"],
    }

    monkeypatch.setattr(
        hf.requests,
        "get",
        lambda *_args, **_kwargs: DummyResponse(json_data=payload),
    )

    result = hf.fetch_dataset_quality("https://huggingface.co/datasets/owner/name")

    assert result["dataset_id"] == "owner/name"
    assert result["dataset_quality"] == pytest.approx(0.86)
    assert result["dataset_quality_latency"] >= 0


def test_fetch_dataset_quality_error(monkeypatch):
    def failing_get(*_args, **_kwargs):
        raise requests.Timeout("slow")

    monkeypatch.setattr(hf.requests, "get", failing_get)

    result = hf.fetch_dataset_quality("owner/name")
    assert result["dataset_id"] == "owner/name"
    assert "error" in result
