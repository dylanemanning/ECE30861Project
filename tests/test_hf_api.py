"""Tests for get_huggingface_model_metadata."""

import pytest
from unittest.mock import patch, Mock

from src.HF_API_Integration import get_huggingface_model_metadata


class TestHFAPIIntegration:
    @patch('src.HF_API_Integration.requests.get')
    def test_successful_api_call(self, mock_get):
        mock_resp = Mock()
        mock_resp.raise_for_status = Mock()
        mock_resp.json.return_value = {
            "modelId": "bert-base-uncased",
            "downloads": 123,
            "likes": 5,
            "lastModified": "2024-01-01T00:00:00"
        }
        mock_get.return_value = mock_resp

        result = get_huggingface_model_metadata("bert-base-uncased")

        assert result == {
            "model_id": "bert-base-uncased",
            "downloads": 123,
            "likes": 5,
            "lastModified": "2024-01-01T00:00:00"
        }
        mock_get.assert_called_once_with("https://huggingface.co/api/models/bert-base-uncased", timeout=10)

    @patch('src.HF_API_Integration.requests.get')
    def test_missing_fields_default(self, mock_get):
        mock_resp = Mock()
        mock_resp.raise_for_status = Mock()
        mock_resp.json.return_value = {"modelId": "tiny"}
        mock_get.return_value = mock_resp

        result = get_huggingface_model_metadata("tiny")

        assert result["model_id"] == "tiny"
        assert result["downloads"] == 0
        assert result["likes"] == 0
        assert result["lastModified"] == "unknown"

    @patch('src.HF_API_Integration.requests.get')
    def test_http_error_returns_none(self, mock_get):
        mock_resp = Mock()
        mock_resp.raise_for_status.side_effect = Exception("boom")
        mock_get.return_value = mock_resp

        assert get_huggingface_model_metadata("bad") is None

    @patch('src.HF_API_Integration.requests.get')
    def test_json_error_returns_none(self, mock_get):
        mock_resp = Mock()
        mock_resp.raise_for_status = Mock()
        mock_resp.json.side_effect = ValueError("bad json")
        mock_get.return_value = mock_resp

        assert get_huggingface_model_metadata("bad-json") is None
