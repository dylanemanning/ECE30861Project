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

    def test_handle_input_file_uses_model_size_scalar(self):
        triples = [("code", "https://huggingface.co/datasets/org/ds", "https://huggingface.co/owner/model")]
        metrics = {
            "dataset_quality": 0.4567,
            "dataset_quality_latency": 10.7,
            "code_quality": 0.91,
            "code_quality_latency": 3.2,
            "bus_factor": 0.25,
            "bus_factor_latency": 4.4,
            "extra_metric": 0.33,
        }
        with patch('src.url_handler.read_url_file', return_value=triples), \
             patch('src.url_handler.analyze_metrics', return_value=metrics) as mock_analyze, \
             patch('src.url_handler.hf') as mock_hf, \
             patch('src.url_handler.time.perf_counter', side_effect=[100.0, 100.042]), \
             patch('src.url_handler.hf_model_size') as mock_model_size:
            mock_hf.get_license_info.return_value = {
                "lgplv21_compat_score": 1,
                "license_latency": 0.004,
            }
            mock_model_size.get_model_file_sizes.return_value = {"total_size_bytes": 400_000_000}
            mock_model_size.calculate_size_metric.return_value = {"size_metric": 0.75}
            mock_model_size.HARDWARE_CONSTRAINTS = {
                "raspberry_pi": 1,
                "jetson_nano": 2,
                "desktop_pc": 4,
                "aws_server": 8,
            }

            records = handle_input_file("dummy.txt")

        assert mock_analyze.called
        assert len(records) == 1
        rec = records[0]
        assert rec["name"] == "model"
        assert rec["license"] == 1
        assert rec["license_latency"] == 4
        assert rec["dataset_and_code_score"] == 1.0
        assert rec["dataset_quality"] == 0.457
        assert rec["dataset_quality_latency"] == 11
        assert rec["code_quality_latency"] == 3
        assert rec["size_score_latency"] == 42
        assert rec["size_score"]["raspberry_pi"] == 0.75
        assert rec["extra_metric"] == 0.33

    def test_handle_input_file_preserves_existing_size_score(self):
        metrics = {
            "size_score": {
                "raspberry_pi": 0.1,
                "jetson_nano": 0.2,
                "desktop_pc": 0.3,
                "aws_server": 0.4,
            },
            "size_score_latency": "5.6",
            "dataset_quality": "bad",
            "code_quality_latency": "2.9",
            "bus_factor_latency": "1.9",
            "extra_latency": "7.49",
            "raspberry_pi": 0.9,
        }
        with patch('src.url_handler.read_url_file', return_value=[("", "", ",bert-base")]), \
             patch('src.url_handler.analyze_metrics', return_value=metrics), \
             patch('src.url_handler.hf') as mock_hf, \
             patch('src.url_handler.hf_model_size', None):
            mock_hf.get_license_info.return_value = {
                "lgplv21_compat_score": 0,
                "license_latency": "NaN",
            }

            records = handle_input_file("dummy.txt")

        rec = records[0]
        assert rec["name"] == ",bert-base"
        assert rec["license"] == 0
        assert rec["license_latency"] == 0
        assert rec["dataset_and_code_score"] == 0.0
        assert rec["size_score"]["aws_server"] == 0.4
        assert rec["size_score_latency"] == 6
        assert rec["dataset_quality"] == "bad"
        assert rec["extra_latency"] == 7
        # metrics without specific keys should be carried over unchanged
        assert rec["raspberry_pi"] == 0.9

    def test_handle_input_file_derives_size_score_from_scalar(self):
        metrics = {
            "size_score": 0.5,
            "raspberry_pi": "0.8",
            "jetson_nano": None,
            "desktop_pc": "",
            "aws_server": 0,
            "size_score_latency": "bad",
            "dataset_quality": None,
            "code_quality": None,
            "dataset_quality_latency": None,
            "code_quality_latency": None,
        }
        with patch('src.url_handler.read_url_file', return_value=[("", "", "https://huggingface.co/owner/model")]), \
             patch('src.url_handler.analyze_metrics', return_value=metrics), \
             patch('src.url_handler.hf') as mock_hf, \
             patch('src.url_handler.hf_model_size', None):
            mock_hf.get_license_info.return_value = {
                "lgplv21_compat_score": 1,
                "license_latency": 0,
            }

            records = handle_input_file("dummy.txt")

        rec = records[0]
        assert rec["size_score"] == {
            "raspberry_pi": 0.8,
            "jetson_nano": 0.0,
            "desktop_pc": 0.0,
            "aws_server": 0.0,
        }
        assert rec["size_score_latency"] == 0
        assert rec["dataset_quality"] == 0.0
        assert rec["code_quality"] == 0.0

