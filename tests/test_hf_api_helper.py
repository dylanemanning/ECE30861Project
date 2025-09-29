"""Comprehensive tests for helper utilities in src.HF_API_Integration."""

import math
import pytest
from unittest.mock import patch, Mock

from src.HF_API_Integration import (
    _normalize_size,
    _compat_from_capacity,
    _compute_size_score,
    _normalize_label,
    _normalize_completeness,
    _score_tags,
    _score_description,
    _score_license,
    _score_schema,
    _score_completeness,
    _score_downloads,
    _extract_dataset_id,
    _extract_model_id,
    fetch_model_card_text,
    discover_hf_dataset_url_for_model,
    fetch_dataset_quality,
    fetch_model_metrics,
    check_license_lgplv21,
    log,
)


class TestHFAPIHelpers:
    def test_normalize_size_range(self):
        assert _normalize_size(0) == 0.0
        assert _normalize_size(1_000_000) == 0.0
        assert _normalize_size(10_000_000_000) == 1.0
        mid = _normalize_size(50_000_000)
        assert 0.0 < mid < 1.0

    def test_compat_from_capacity_ratio_buckets(self):
        assert _compat_from_capacity(0, 1_000_000_000) == 1.0
        assert _compat_from_capacity(300_000_000, 1_000_000_000) == 0.85
        assert _compat_from_capacity(1_500_000_000, 1_000_000_000) == 0.40
        assert _compat_from_capacity(5_000_000_000, 1_000_000_000) == 0.0
        assert _compat_from_capacity(100, 0) == 0.0

    def test_compute_size_score_devices(self):
        scores = _compute_size_score(500_000_000)
        assert set(scores.keys()) == {
            "raspberry_pi",
            "jetson_nano",
            "desktop_pc",
            "aws_server",
        }
        assert scores["raspberry_pi"] == _compat_from_capacity(500_000_000, 200_000_000)

    def test_normalize_label_filters(self):
        assert _normalize_label([]) == 0.0
        assert _normalize_label(["model", "pytorch"]) == 0.0
        assert _normalize_label(["bert"]) == 0.5
        assert _normalize_label(["bert", "nlp", "qa"]) == 1.0

    def test_normalize_completeness_counts(self):
        assert _normalize_completeness({}) == 0.0
        full = {
            "description": "desc",
            "tags": ["a"],
            "license": "MIT",
            "downloads": 10,
            "likes": 1,
        }
        assert math.isclose(_normalize_completeness(full), 1.0)
        partial = {"description": "desc"}
        assert math.isclose(_normalize_completeness(partial), 0.2)

    def test_score_tags_gradations(self):
        assert _score_tags(None) == 0.0
        assert _score_tags(["tag"]) == 0.5
        assert _score_tags(["tag1", "tag2", "tag3"]) == 0.8
        assert _score_tags(["tag1", "tag2", "tag3", "tag4", "tag5"]) == 1.0

    def test_score_description_bounds(self):
        assert _score_description("") == 0.0
        assert math.isclose(_score_description("A" * 100), 0.05)
        assert _score_description("A" * 5000) == 1.0

    def test_score_license_variants(self):
        assert _score_license("MIT") == 1.0
        assert _score_license("Apache-2.0") == 1.0
        assert _score_license({"spdx": "MIT"}) == 1.0
        assert _score_license({}) == 0.0
        assert _score_license(123) == 0.5

    def test_score_schema_variants(self):
        assert _score_schema({}) == 0.0
        assert _score_schema({"cardData": {"features": {"a": 1}}}) == 1.0
        assert _score_schema({"cardData": {"dataset_info": {"b": 2}}}) == 1.0
        assert _score_schema({"configs": []}) == 0.5

    def test_score_completeness_delegates(self):
        data = {"description": "d"}
        assert _score_completeness(data) == _normalize_completeness(data)

    def test_score_downloads_thresholds(self):
        assert _score_downloads(0) == 0.0
        assert _score_downloads(50) == 0.2
        assert _score_downloads(500) == 0.5
        assert _score_downloads(5000) == 0.8
        assert _score_downloads(50000) == 1.0

    def test_extract_helpers(self):
        assert _extract_dataset_id("https://huggingface.co/datasets/org/name/tree/main") == "org/name"
        assert _extract_model_id("https://huggingface.co/org/model") == "org/model"
        assert _extract_model_id(",bert-base") == "bert-base"

    @patch('src.HF_API_Integration.requests.get')
    def test_fetch_model_card_text_tries_fallback(self, mock_get):
        mock_get.side_effect = [Mock(status_code=404, text=""), Mock(status_code=200, text="# doc")]

        result = fetch_model_card_text("owner/model")

        assert result == "# doc"
        assert mock_get.call_count == 2

    @patch('src.HF_API_Integration.requests.get')
    def test_fetch_model_card_text_all_fail(self, mock_get):
        mock_get.return_value = Mock(status_code=404, text="")
        assert fetch_model_card_text("owner/model") == ""

    @patch('src.HF_API_Integration.requests.get')
    def test_discover_dataset_from_carddata(self, mock_get):
        mock_resp = Mock()
        mock_resp.raise_for_status = Mock()
        mock_resp.json.return_value = {"cardData": {"datasets": ["org/name"]}}
        mock_get.return_value = mock_resp

        assert discover_hf_dataset_url_for_model("model") == "https://huggingface.co/datasets/org/name"

    @patch('src.HF_API_Integration.requests.get')
    def test_discover_dataset_from_tags(self, mock_get):
        mock_resp = Mock()
        mock_resp.raise_for_status = Mock()
        mock_resp.json.return_value = {
            "cardData": {"datasets": []},
            "tags": ["dataset:common_voice", "pytorch"],
        }
        mock_get.return_value = mock_resp

        assert discover_hf_dataset_url_for_model("model") == "https://huggingface.co/datasets/common_voice"

    @patch('src.HF_API_Integration.requests.get', side_effect=Exception("boom"))
    def test_discover_dataset_error(self, _mock_get):
        assert discover_hf_dataset_url_for_model("model") == ""

    @patch('src.HF_API_Integration.time.perf_counter', side_effect=[1.0, 1.123])
    @patch('src.HF_API_Integration.requests.get')
    def test_fetch_dataset_quality_success(self, mock_get, _mock_perf):
        mock_resp = Mock()
        mock_resp.raise_for_status = Mock()
        mock_resp.json.return_value = {
            "cardData": {
                "description": "desc",
                "tags": ["meta"],
                "license": "MIT",
                "downloads": 100,
                "likes": 5,
            },
            "tags": ["tag1", "tag2", "tag3"],
            "downloads": 5000,
        }
        mock_get.return_value = mock_resp

        result = fetch_dataset_quality("org/name")

        assert result["dataset_id"] == "org/name"
        assert math.isclose(result["dataset_quality"], 0.9)
        assert result["dataset_quality_latency"] == 123

    @patch('src.HF_API_Integration.requests.get', side_effect=Exception("boom"))
    def test_fetch_dataset_quality_error(self, _mock_get):
        result = fetch_dataset_quality("bad")
        assert result == {"dataset_id": "bad", "error": "boom"}

    @patch('src.HF_API_Integration.get_huggingface_model_metadata')
    def test_fetch_model_metrics_success(self, mock_meta):
        mock_meta.return_value = {"model_id": "x", "downloads": 50, "likes": 5}

        result = fetch_model_metrics("x")

        assert result["model_id"] == "x"
        assert result["downloads"] == 50
        assert result["downloads_norm"] > 0
        assert "latency" in result

    @patch('src.HF_API_Integration.get_huggingface_model_metadata', return_value=None)
    def test_fetch_model_metrics_error(self, _mock_meta):
        result = fetch_model_metrics("missing")
        assert result == {"model_id": "missing", "error": "no metadata"}

    def test_check_license_lgplv21(self):
        assert check_license_lgplv21("MIT") is True
        assert check_license_lgplv21("Proprietary") is False
        assert check_license_lgplv21("") is False

    def test_log_emits_info(self, caplog):
        caplog.set_level("INFO")
        log("hello")
        assert "hello" in caplog.text
