"""
Additional tests for HF_API_Integration.py helper functions
"""
import pytest
from unittest.mock import patch, Mock, MagicMock
import time
import json
import os

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
    log
)


class TestHFAPIHelpers:
    
    def test_normalize_size(self):
        """Test size normalization"""
        # 0 or negative should return 0
        assert _normalize_size(0) == 0.0
        assert _normalize_size(-100) == 0.0
        
        # 1MB (min) should be 0
        assert _normalize_size(1_000_000) == 0.0
        
        # 10GB (max) should be 1
        assert _normalize_size(10_000_000_000) == 1.0
        
        # Values between should be normalized logarithmically
        # 100MB should be somewhere in middle
        size_100mb = _normalize_size(100_000_000)
        assert 0.0 < size_100mb < 1.0
        
        # 1GB should be higher than 100MB but less than 10GB
        size_1gb = _normalize_size(1_000_000_000)
        assert size_100mb < size_1gb < 1.0
    
    def test_compat_from_capacity(self):
        """Test device compatibility scoring"""
        capacity = 1_000_000_000  # 1GB
        
        # 250MB (0.25x) should score 1.0
        assert _compat_from_capacity(250_000_000, capacity) == 1.0
        
        # 500MB (0.5x) should score 0.85
        assert _compat_from_capacity(500_000_000, capacity) == 0.85
        
        # 1GB (1x) should score 0.70
        assert _compat_from_capacity(1_000_000_000, capacity) == 0.70
        
        # 2GB (2x) should score 0.40
        assert _compat_from_capacity(2_000_000_000, capacity) == 0.40
        
        # 4GB (4x) should score 0.20
        assert _compat_from_capacity(4_000_000_000, capacity) == 0.20
        
        # >4x should score 0.0
        assert _compat_from_capacity(5_000_000_000, capacity) == 0.0
        
        # Edge cases
        assert _compat_from_capacity(0, capacity) == 0.0
        assert _compat_from_capacity(100, 0) == 0.0
    
    def test_compute_size_score(self):
        """Test computing size scores for all devices"""
        # Small model (100MB)
        scores = _compute_size_score(100_000_000)
        
        assert "raspberry_pi" in scores
        assert "jetson_nano" in scores
        assert "desktop_pc" in scores
        assert "aws_server" in scores
        
        # Should fit well on all devices
        assert scores["raspberry_pi"] == 0.85  # <0.5x of 200MB
        assert scores["jetson_nano"] == 1.0   # <0.25x of 1GB
        assert scores["desktop_pc"] == 1.0    # <0.25x of 8GB
        assert scores["aws_server"] == 1.0    # <0.25x of 20GB
        
        # Large model (10GB)
        scores_large = _compute_size_score(10_000_000_000)
        
        # Too big for smaller devices
        assert scores_large["raspberry_pi"] == 0.0
        assert scores_large["jetson_nano"] == 0.0
        assert scores_large["desktop_pc"] == 0.40  # Between 1x and 2x
        assert scores_large["aws_server"] == 0.85   # 0.5x of 20GB
    
    def test_normalize_label(self):
        """Test label/tag normalization"""
        # No tags
        assert _normalize_label(None) == 0.0
        assert _normalize_label([]) == 0.0
        
        # Ignored tags
        assert _normalize_label(["model", "test", "example"]) == 0.0
        
        # 1-2 meaningful tags
        assert _normalize_label(["transformer"]) == 0.5
        assert _normalize_label(["bert", "nlp"]) == 0.5
        
        # 3+ meaningful tags
        assert _normalize_label(["bert", "nlp", "qa", "pytorch"]) == 1.0
        
        # Mixed meaningful and ignored
        assert _normalize_label(["model", "bert", "test", "nlp", "qa"]) == 1.0
    
    def test_normalize_completeness(self):
        """Test completeness scoring"""
        # Empty data
        assert _normalize_completeness({}) == 0.0
        
        # All fields present
        full_data = {
            "description": "Test",
            "tags": ["tag1"],
            "license": "MIT",
            "downloads": 100,
            "likes": 10
        }
        assert _normalize_completeness(full_data) == 1.0
        
        # Partial fields
        partial_data = {
            "description": "Test",
            "downloads": 100
        }
        assert _normalize_completeness(partial_data) == 0.4  # 2/5
        
        # Empty values count as missing
        empty_data = {
            "description": "",
            "tags": [],
            "license": None
        }
        assert _normalize_completeness(empty_data) == 0.0
    
    def test_score_tags(self):
        """Test tag scoring"""
        assert _score_tags(None) == 0.0
        assert _score_tags([]) == 0.0
        assert _score_tags("not a list") == 0.0
        
        assert _score_tags(["tag"]) == 0.5
        assert _score_tags(["tag1", "tag2"]) == 0.5
        assert _score_tags(["tag1", "tag2", "tag3"]) == 0.8
        assert _score_tags(["tag1", "tag2", "tag3", "tag4", "tag5"]) == 1.0
        
        # Empty strings don't count
        assert _score_tags(["", "", ""]) == 0.0
    
    def test_score_description(self):
        """Test description scoring"""
        assert _score_description("") == 0.0
        assert _score_description(None) == 0.0
        
        # Short description
        assert _score_description("A" * 100) == 0.05  # 100/2000
        
        # Medium description
        assert _score_description("A" * 1000) == 0.5  # 1000/2000
        
        # Long description (capped at 1.0)
        assert _score_description("A" * 2000) == 1.0
        assert _score_description("A" * 3000) == 1.0
    
    def test_score_license(self):
        """Test license scoring"""
        assert _score_license(None) == 0.0
        assert _score_license("") == 0.0
        assert _score_license("   ") == 0.0
        
        # String licenses
        assert _score_license("MIT") == 1.0
        assert _score_license("Apache-2.0") == 1.0
        
        # Dict licenses
        assert _score_license({"spdx": "MIT"}) == 1.0
        assert _score_license({}) == 0.0
        
        # Other types
        assert _score_license(123) == 0.5
        assert _score_license(["MIT"]) == 0.5
    
    def test_score_schema(self):
        """Test schema/structure scoring"""
        # No cardData
        assert _score_schema({}) == 0.0
        
        # Has features in cardData
        assert _score_schema({"cardData": {"features": {}}}) == 1.0
        
        # Has dataset_info in cardData
        assert _score_schema({"cardData": {"dataset_info": {}}}) == 1.0
        
        # Has configs
        assert _score_schema({"configs": []}) == 0.5
        
        # cardData not a dict
        assert _score_schema({"cardData": "string"}) == 0.0
    
    def test_score_downloads(self):
        """Test download count scoring"""
        assert _score_downloads(None) == 0.0
        assert _score_downloads("not a number") == 0.0
        assert _score_downloads(0) == 0.0
        
        assert _score_downloads(50) == 0.2
        assert _score_downloads(500) == 0.5
        assert _score_downloads(5000) == 0.8
        assert _score_downloads(50000) == 1.0
    
    def test_extract_dataset_id(self):
        """Test extracting dataset ID from URLs"""
        # Full HF dataset URL
        assert _extract_dataset_id("https://huggingface.co/datasets/squad/squad_v2") == "squad/squad_v2"
        assert _extract_dataset_id("https://huggingface.co/datasets/owner/name/tree/main") == "owner/name"
        
        # Just the ID
        assert _extract_dataset_id("squad/squad_v2") == "squad/squad_v2"
        assert _extract_dataset_id("single_name") == "single_name"
        
        # With leading comma (edge case)
        assert _extract_dataset_id(",squad/squad_v2") == "squad/squad_v2"
        
        # Empty or whitespace
        assert _extract_dataset_id("  ") == ""
    
    def test_extract_model_id(self):
        """Test extracting model ID from URLs"""
        # Full HF model URL
        assert _extract_model_id("https://huggingface.co/bert-base-uncased") == "bert-base-uncased"
        assert _extract_model_id("https://huggingface.co/google/gemma-2b") == "google/gemma-2b"
        assert _extract_model_id("https://huggingface.co/owner/model/tree/main") == "owner/model"
        
        # Just the ID
        assert _extract_model_id("bert-base-uncased") == "bert-base-uncased"
        assert _extract_model_id("google/gemma-2b") == "google/gemma-2b"
        
        # With leading comma
        assert _extract_model_id(",bert-base") == "bert-base"
    
    @patch('src.HF_API_Integration.requests.get')
    def test_fetch_model_card_text_success(self, mock_get):
        """Test fetching model card text"""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.text = "# Model Card\nThis is a test model."
        mock_get.return_value = mock_response
        
        result = fetch_model_card_text("bert-base-uncased")
        
        assert result == "# Model Card\nThis is a test model."
        
        # Should try multiple URLs
        assert mock_get.call_count >= 1
    
    @patch('src.HF_API_Integration.requests.get')
    def test_fetch_model_card_text_fallback(self, mock_get):
        """Test model card fetch with fallback URLs"""
        # First two URLs fail, third succeeds
        responses = [
            Mock(status_code=404, text=""),
            Mock(status_code=404, text=""),
            Mock(status_code=200, text="# Found")
        ]
        mock_get.side_effect = responses
        
        result = fetch_model_card_text("test-model")
        
        assert result == "# Found"
        assert mock_get.call_count == 3
    
    @patch('src.HF_API_Integration.requests.get')
    def test_fetch_model_card_text_all_fail(self, mock_get):
        """Test model card fetch when all URLs fail"""
        mock_get.return_value = Mock(status_code=404, text="")
        
        result = fetch_model_card_text("nonexistent-model")
        
        assert result == ""
    
    @patch('src.HF_API_Integration.requests.get')
    def test_discover_hf_dataset_url_carddata(self, mock_get):
        """Test discovering dataset from cardData"""
        mock_response = Mock()
        mock_response.json.return_value = {
            "cardData": {
                "datasets": ["squad"]
            }
        }
        mock_response.raise_for_status = Mock()
        mock_get.return_value = mock_response
        
        result = discover_hf_dataset_url_for_model("bert-qa")
        
        assert result == "https://huggingface.co/datasets/squad"
    
    @patch('src.HF_API_Integration.requests.get')
    def test_discover_hf_dataset_url_tags(self, mock_get):
        """Test discovering dataset from tags"""
        mock_response = Mock()
        mock_response.json.return_value = {
            "cardData": {},
            "tags": ["dataset:common_voice", "pytorch"]
        }
        mock_response.raise_for_status = Mock()
        mock_get.return_value = mock_response
        
        result = discover_hf_dataset_url_for_model("whisper")
        
        assert result == "https://huggingface.co/datasets/common_voice"
    
    @patch('src.HF_API_Integration.requests.get')
    def test_discover_hf_dataset_url_none(self, mock_get):
        """Test when no dataset is found"""
        mock_response = Mock()
        mock_response.json.return_value = {
            "cardData": {},
            "tags": ["pytorch", "transformer"]
        }
        mock_response.raise_for_status = Mock()
        mock_get.return_value = mock_response
        
        result = discover_hf_dataset_url_for_model("model")
        
        assert result == ""
    
    @patch('src.HF_API_Integration.requests.get')
    def test_fetch_dataset_quality_success(self, mock_get):
        """Test dataset quality computation"""
        mock_response = Mock()
        mock_response.json.return_value = {
            "cardData": {
                "description": "A" * 1000,
                "features": {"text": "string"},
                "license": "MIT"
            },
            "tags": ["nlp", "text", "english", "qa", "benchmark"],
            "downloads": 10000,
            "license": "MIT"
        }
        mock_response.raise_for_status = Mock()
        mock_get.return_value = mock_response
        
        result = fetch_dataset_quality("squad/squad_v2")
        
        assert "dataset_id" in result
        assert "dataset_quality" in result
        assert "dataset_quality_latency" in result
        assert 0 <= result["dataset_quality"] <= 1
        assert result["dataset_quality_latency"] >= 0
    
    @patch('src.HF_API_Integration.requests.get')
    def test_fetch_dataset_quality_error(self, mock_get):
        """Test dataset quality with API error"""
        mock_get.side_effect = Exception("API Error")
        
        result = fetch_dataset_quality("bad-dataset")
        
        assert result["dataset_id"] == "bad-dataset"
        assert "error" in result
        assert "API Error" in result["error"]
    
    @patch('src.HF_API_Integration.get_huggingface_model_metadata')
    def test_fetch_model_metrics_success(self, mock_get_meta):
        """Test fetching model metrics"""
        mock_get_meta.return_value = {
            "model_id": "bert-base",
            "downloads": 1500000,
            "likes": 5000,
            "lastModified": "2024-01-01"
        }
        
        result = fetch_model_metrics("bert-base")
        
        assert result["model_id"] == "bert-base"
        assert result["downloads"] == 1500000
        assert result["likes"] == 5000
        assert "downloads_norm" in result
        assert "likes_norm" in result
        assert "latency" in result
        assert 0 <= result["downloads_norm"] <= 1
        assert 0 <= result["likes_norm"] <= 1
    
    @patch('src.HF_API_Integration.get_huggingface_model_metadata')
    def test_fetch_model_metrics_nonexistent(self, mock_get_meta):
        """Test fetching metrics for nonexistent model"""
        mock_get_meta.return_value = None
        
        result = fetch_model_metrics("nonexistent-model")
        
        # Should return None for clearly invalid models
        assert result is None
    
    @patch('src.HF_API_Integration.get_huggingface_model_metadata')
    def test_fetch_model_metrics_fallback(self, mock_get_meta):
        """Test fetching metrics with fallback for other failures"""
        mock_get_meta.return_value = None
        
        result = fetch_model_metrics("some-model")
        
        # Should return minimal metrics
        assert result["model_id"] == "some-model"
        assert result["downloads"] == 0
        assert result["likes"] == 0
        assert result["downloads_norm"] == 0.0
        assert result["likes_norm"] == 0.0
    
    @patch('src.license_compat.license_compat')
    def test_check_license_lgplv21(self, mock_license_compat):
        """Test license compatibility check wrapper"""
        mock_license_compat.return_value = {
            "model_id": "test-model",
            "license": "MIT",
            "lgplv21_compat_score": 1
        }
        
        result = check_license_lgplv21("test-model")
        
        assert result == mock_license_compat.return_value
        mock_license_compat.assert_called_once_with("test-model")
    
    def test_log_with_file(self):
        """Test logging to file"""
        import tempfile
        
        with tempfile.NamedTemporaryFile(mode='w', delete=False) as f:
            log_file = f.name
        
        try:
            os.environ["LOG_LEVEL"] = "1"
            os.environ["LOG_FILE"] = log_file
            
            log("Test message", level=1)
            
            with open(log_file, 'r') as f:
                content = f.read()
            
            assert "Test message" in content
        finally:
            os.unlink(log_file)
            del os.environ["LOG_LEVEL"]
            del os.environ["LOG_FILE"]
    
    def test_log_with_stdout(self, capsys):
        """Test logging to stdout"""
        os.environ["LOG_LEVEL"] = "2"
        if "LOG_FILE" in os.environ:
            del os.environ["LOG_FILE"]
        
        log("Debug message", level=2)
        
        captured = capsys.readouterr()
        assert "Debug message" in captured.out
        
        del os.environ["LOG_LEVEL"]
    
    def test_log_level_filtering(self, capsys):
        """Test log level filtering"""
        os.environ["LOG_LEVEL"] = "1"
        if "LOG_FILE" in os.environ:
            del os.environ["LOG_FILE"]
        
        log("Should not appear", level=2)  # Level 2 > 1
        log("Should appear", level=1)  # Level 1 = 1
        
        captured = capsys.readouterr()
        assert "Should not appear" not in captured.out
        assert "Should appear" in captured.out
        
        del os.environ["LOG_LEVEL"]