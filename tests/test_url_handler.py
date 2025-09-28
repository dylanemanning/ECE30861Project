"""
Test suite for url_handler.py module
"""
import pytest
from unittest.mock import patch, Mock, MagicMock, mock_open, call
import json
import tempfile
import os

from src.url_handler import (
    parse_triple,
    read_url_file,
    is_placeholder_or_non_hf_dataset,
    handle_input_file
)


class TestURLHandler:
    
    def test_parse_triple_full(self):
        """Test parsing a full triple"""
        result = parse_triple("https://github.com/repo,https://huggingface.co/datasets/squad,bert-base")
        assert result == ("https://github.com/repo", "https://huggingface.co/datasets/squad", "bert-base")
    
    def test_parse_triple_single_token(self):
        """Test parsing a single token (model only)"""
        result = parse_triple("bert-base-uncased")
        assert result == ("", "", "bert-base-uncased")
    
    def test_parse_triple_two_parts(self):
        """Test parsing two parts"""
        result = parse_triple("https://github.com/repo,bert-base")
        assert result == ("https://github.com/repo", "bert-base", "")
    
    def test_parse_triple_extra_commas(self):
        """Test parsing with extra commas (merged into model field)"""
        result = parse_triple("code,dataset,model,with,extra,commas")
        assert result == ("code", "dataset", "model,with,extra,commas")
    
    def test_parse_triple_with_spaces(self):
        """Test parsing with spaces"""
        result = parse_triple("  code_url  ,  dataset_url  ,  model_url  ")
        assert result == ("code_url", "dataset_url", "model_url")
    
    def test_parse_triple_empty_fields(self):
        """Test parsing with empty fields"""
        result = parse_triple(",,model")
        assert result == ("", "", "model")
    
    def test_parse_triple_only_commas(self):
        """Test parsing only commas"""
        result = parse_triple(",,")
        assert result == ("", "", "")
    
    def test_read_url_file_basic(self):
        """Test reading a basic URL file"""
        content = """# Comment line
https://github.com/code,https://huggingface.co/datasets/data,model1
,,model2

code2,dataset2,model2"""
        
        with patch("builtins.open", mock_open(read_data=content)):
            result = read_url_file("test.txt")
        
        assert len(result) == 3
        assert result[0] == ("https://github.com/code", "https://huggingface.co/datasets/data", "model1")
        assert result[1] == ("", "", "model2")
        assert result[2] == ("code2", "dataset2", "model2")
    
    def test_read_url_file_empty(self):
        """Test reading an empty file"""
        with patch("builtins.open", mock_open(read_data="")):
            result = read_url_file("empty.txt")
        assert result == []
    
    def test_read_url_file_only_comments(self):
        """Test reading a file with only comments"""
        content = """# Comment 1
# Comment 2
# Comment 3"""
        
        with patch("builtins.open", mock_open(read_data=content)):
            result = read_url_file("comments.txt")
        assert result == []
    
    def test_read_url_file_encoding(self):
        """Test reading a file with special characters"""
        content = "código,données,模型"
        
        with patch("builtins.open", mock_open(read_data=content)) as mock_file:
            result = read_url_file("unicode.txt")
            mock_file.assert_called_once_with("unicode.txt", "r", encoding="utf-8")
        
        assert result == [("código", "données", "模型")]
    
    def test_is_placeholder_or_non_hf_dataset_placeholders(self):
        """Test detecting placeholder values"""
        assert is_placeholder_or_non_hf_dataset("")
        assert is_placeholder_or_non_hf_dataset("none")
        assert is_placeholder_or_non_hf_dataset("None")
        assert is_placeholder_or_non_hf_dataset("NONE")
        assert is_placeholder_or_non_hf_dataset("null")
        assert is_placeholder_or_non_hf_dataset("NA")
        assert is_placeholder_or_non_hf_dataset("n/a")
        assert is_placeholder_or_non_hf_dataset("N/A")
        assert is_placeholder_or_non_hf_dataset("-")
        assert is_placeholder_or_non_hf_dataset("  ")
    
    def test_is_placeholder_or_non_hf_dataset_non_hf(self):
        """Test detecting non-HF dataset URLs"""
        assert is_placeholder_or_non_hf_dataset("https://example.com/dataset")
        assert is_placeholder_or_non_hf_dataset("http://github.com/dataset")
        assert is_placeholder_or_non_hf_dataset("dataset_name")
        assert is_placeholder_or_non_hf_dataset("/path/to/dataset")
    
    def test_is_placeholder_or_non_hf_dataset_valid_hf(self):
        """Test valid HF dataset URLs return False"""
        assert not is_placeholder_or_non_hf_dataset("https://huggingface.co/datasets/squad")
        assert not is_placeholder_or_non_hf_dataset("https://huggingface.co/datasets/owner/name")
    
    @patch('src.url_handler.read_url_file')
    @patch('src.url_handler.analyze_metrics')
    @patch('src.url_handler.discover_dataset_url_with_genai')
    @patch('src.url_handler.hf')
    def test_handle_input_file_basic(self, mock_hf, mock_discover, mock_analyze, mock_read):
        """Test basic input file handling"""
        # Setup mocks
        mock_read.return_value = [
            ("", "", "bert-base-uncased"),
            ("https://github.com/code", "https://huggingface.co/datasets/squad", "model2")
        ]
        
        mock_analyze.return_value = {
            "ramp_up_time": 0.8,
            "ramp_up_time_latency": 100,
            "performance_claims": 0.7,
            "dataset_and_code_score": 0.9
        }
        
        mock_discover.return_value = {}
        
        mock_hf._extract_model_id = Mock(side_effect=lambda x: x)
        mock_hf.fetch_model_card_text = Mock(return_value="# Model Card")
        mock_hf.fetch_dataset_quality = Mock(return_value={
            "dataset_quality": 0.85,
            "dataset_quality_latency": 200
        })
        
        result = handle_input_file("test.txt")
        
        assert len(result) == 2
        assert result[0]["name"] == "bert-base-uncased"
        assert result[0]["category"] == "MODEL"
        assert result[0]["ramp_up_time"] == 0.8
        assert result[1]["dataset_quality"] == 0.85
    
    @patch('src.url_handler.read_url_file')
    @patch('src.url_handler.analyze_metrics')
    @patch('src.url_handler.discover_dataset_url_with_genai')
    @patch('src.url_handler.hf')
    def test_handle_input_file_dataset_discovery(self, mock_hf, mock_discover, mock_analyze, mock_read):
        """Test dataset discovery when URL is missing"""
        mock_read.return_value = [
            ("", "", "bert-base-uncased")  # No dataset URL
        ]
        
        mock_analyze.return_value = {}
        
        # Mock dataset discovery
        mock_discover.return_value = {
            "dataset_url": "squad/squad_v2",
            "dataset_discovery_latency": 150
        }
        
        mock_hf._extract_model_id = Mock(return_value="bert-base-uncased")
        mock_hf.fetch_model_card_text = Mock(return_value="")
        mock_hf.fetch_dataset_quality = Mock(return_value={
            "dataset_quality": 0.9,
            "dataset_quality_latency": 100
        })
        
        result = handle_input_file("test.txt")
        
        assert len(result) == 1
        # Dataset should be discovered and quality fetched
        assert result[0]["dataset_quality"] == 0.9
        
        # Verify dataset discovery was called
        mock_discover.assert_called_once()
        # Verify quality was fetched for discovered dataset
        mock_hf.fetch_dataset_quality.assert_called_with("https://huggingface.co/datasets/squad/squad_v2")
    
    @patch('src.url_handler.read_url_file')
    @patch('src.url_handler.analyze_metrics')
    @patch('src.url_handler.hf')
    @patch('src.url_handler.requests.get')
    def test_handle_input_file_github_readme(self, mock_requests, mock_hf, mock_analyze, mock_read):
        """Test fetching README from GitHub"""
        mock_read.return_value = [
            ("https://github.com/owner/repo", "", "")
        ]
        
        mock_analyze.return_value = {}
        
        # Mock GitHub README fetch
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.text = "# GitHub README Content"
        mock_requests.return_value = mock_response
        
        mock_hf._extract_model_id = Mock(return_value="owner/repo")
        
        result = handle_input_file("test.txt")
        
        # Verify README was fetched from GitHub
        assert mock_requests.called
        call_args = mock_requests.call_args[0][0]
        assert "raw.githubusercontent.com" in call_args
        assert "README.md" in call_args
    
    @patch('src.url_handler.read_url_file')
    @patch('src.url_handler.analyze_metrics')
    @patch('src.url_handler.hf')
    def test_handle_input_file_hf_model(self, mock_hf, mock_analyze, mock_read):
        """Test handling HuggingFace model URLs"""
        mock_read.return_value = [
            ("", "", "https://huggingface.co/bert-base-uncased")
        ]
        
        mock_analyze.return_value = {
            "ramp_up_time": 0.75,
            "performance_claims": 0.85
        }
        
        mock_hf._extract_model_id = Mock(return_value="bert-base-uncased")
        mock_hf.fetch_model_card_text = Mock(return_value="# BERT Model Card")
        
        result = handle_input_file("test.txt")
        
        assert len(result) == 1
        assert result[0]["name"] == "bert-base-uncased"
        assert result[0]["ramp_up_time"] == 0.75
        
        # Verify model card was fetched
        mock_hf.fetch_model_card_text.assert_called_with("bert-base-uncased")
    
    @patch('src.url_handler.read_url_file')
    @patch('src.url_handler.analyze_metrics')
    def test_handle_input_file_error_handling(self, mock_analyze, mock_read):
        """Test error handling in file processing"""
        mock_read.return_value = [
            ("", "", "model1"),
            ("", "", "model2")
        ]
        
        # First call succeeds, second fails
        mock_analyze.side_effect = [
            {"ramp_up_time": 0.8},
            Exception("API Error")
        ]
        
        # Should not raise, but handle the error gracefully
        result = handle_input_file("test.txt")
        
        assert len(result) == 2
        assert result[0]["ramp_up_time"] == 0.8
        # Second result should have empty metrics due to error
        assert "ramp_up_time" not in result[1] or result[1]["ramp_up_time"] == 0
    
    @patch('src.url_handler.read_url_file')
    def test_handle_input_file_empty(self, mock_read):
        """Test handling empty input file"""
        mock_read.return_value = []
        
        result = handle_input_file("empty.txt")
        
        assert result == []
    
    @patch('src.url_handler.read_url_file')
    @patch('src.url_handler.analyze_metrics')
    @patch('src.url_handler.hf')
    def test_handle_input_file_placeholder_dataset(self, mock_hf, mock_analyze, mock_read):
        """Test handling placeholder dataset URLs"""
        mock_read.return_value = [
            ("", "none", "model1"),  # Placeholder dataset
            ("", "N/A", "model2"),   # Another placeholder
            ("", "https://example.com/data", "model3")  # Non-HF URL
        ]
        
        mock_analyze.return_value = {}
        mock_hf._extract_model_id = Mock(side_effect=lambda x: x)
        
        result = handle_input_file("test.txt")
        
        assert len(result) == 3
        # All should have dataset_url_flag as False (handled internally)
        # No dataset quality should be fetched for placeholders
    
    @patch('src.url_handler.read_url_file')
    @patch('src.url_handler.analyze_metrics')
    def test_handle_input_file_latency_conversion(self, mock_analyze, mock_read):
        """Test that latencies are converted to integers"""
        mock_read.return_value = [
            ("", "", "model1")
        ]
        
        mock_analyze.return_value = {
            "ramp_up_time": 0.8,
            "ramp_up_time_latency": 100.7,  # Float value
            "performance_claims_latency": "200"  # String value
        }
        
        result = handle_input_file("test.txt")
        
        assert isinstance(result[0].get("ramp_up_time_latency", 0), int)
        assert result[0].get("ramp_up_time_latency", 0) == 100  # Should be converted to int
        if "performance_claims_latency" in result[0]:
            assert isinstance(result[0]["performance_claims_latency"], int)