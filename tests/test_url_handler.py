"""Tests for url_handler that match the current implementation."""

import pytest
from unittest.mock import patch

from src.url_handler import (
    parse_triple,
    read_url_file,
    is_placeholder_or_non_hf_dataset,
    handle_input_file,
)


class TestURLHandler:
    def test_parse_triple_variants(self):
        assert parse_triple("code,dataset,model") == ("code", "dataset", "model")
        assert parse_triple("model-only") == ("", "", "model-only")
        assert parse_triple("code,model") == ("code", "model", "")

    def test_read_url_file_ignores_comments(self, tmp_path):
        path = tmp_path / "urls.txt"
        path.write_text("# comment\ncode,dataset,model\n\n,,model2\n")
        assert read_url_file(str(path)) == [
            ("code", "dataset", "model"),
            ("", "", "model2"),
        ]

    def test_is_placeholder(self):
        assert is_placeholder_or_non_hf_dataset("") is True
        assert is_placeholder_or_non_hf_dataset("https://huggingface.co/datasets/name") is False
        assert is_placeholder_or_non_hf_dataset("https://example.com") is True

    @patch('src.url_handler.read_url_file')
    @patch('src.url_handler.analyze_metrics')
    @patch('src.url_handler.hf')
    def test_handle_input_file_with_metrics(self, mock_hf, mock_analyze, mock_read):
        mock_read.return_value = [("", "", "https://huggingface.co/owner/model"), ("code", "dataset", "model2")]
        mock_analyze.side_effect = [
            {"ramp_up_time": 0.8, "ramp_up_time_latency": 100.7},
            {}
        ]
        mock_hf.get_license_info.side_effect = [
            {"lgplv21_compat_score": 1, "license_latency": 0.05},
            {"lgplv21_compat_score": 0, "license_latency": "bad"}
        ]

        result = handle_input_file("dummy.txt")

        assert result[0]["name"] == "model"
        assert result[0]["ramp_up_time_latency"] == 101
        assert result[0]["license"] == 1
        assert result[1]["license_latency"] == 0

    @patch('src.url_handler.read_url_file')
    @patch('src.url_handler.analyze_metrics')
    def test_handle_input_file_error_propagates(self, mock_analyze, mock_read):
        mock_read.return_value = [("", "", "model1")]
        mock_analyze.side_effect = Exception("boom")

        with pytest.raises(Exception):
            handle_input_file("dummy")

    @patch('src.url_handler.read_url_file')
    def test_handle_input_file_empty(self, mock_read):
        mock_read.return_value = []
        assert handle_input_file("empty") == []
