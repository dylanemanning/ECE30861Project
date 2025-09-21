"""
Test suite for HF_API_Integration.py module
"""
import pytest
from unittest.mock import patch, Mock, MagicMock
import requests

# Import the function we're testing
from src.HF_API_Integration import get_huggingface_model_metadata


class TestHFAPIIntegration:
    
    @patch('HF_API_Integration.requests.get')
    def test_successful_api_call(self, mock_get):
        """Test successful API response parsing with all fields"""
        # Create a mock response with complete data
        mock_response = Mock()
        mock_response.json.return_value = {
            "modelId": "bert-base-uncased",
            "downloads": 1500000,
            "likes": 234,
            "lastModified": "2024-03-15T10:30:00.000Z",
            "private": False,
            "tags": ["pytorch", "transformers"],
            "pipeline_tag": "fill-mask"
        }
        mock_response.raise_for_status = Mock()  # Should not raise
        mock_get.return_value = mock_response
        
        # Call the function
        result = get_huggingface_model_metadata("bert-base-uncased")
        
        # Verify the result
        assert result is not None, "Result should not be None for successful API call"
        assert result["model_id"] == "bert-base-uncased"
        assert result["downloads"] == 1500000
        assert result["likes"] == 234
        assert result["lastModified"] == "2024-03-15T10:30:00.000Z"
        
        # Verify the API was called with correct URL
        mock_get.assert_called_once_with(
            "https://huggingface.co/api/models/bert-base-uncased",
            timeout=10
        )
    
    @patch('HF_API_Integration.requests.get')
    def test_api_network_error(self, mock_get):
        """Test that network errors return None and don't crash"""
        # Simulate a network error
        mock_get.side_effect = requests.exceptions.ConnectionError("Network is unreachable")
        
        # Call the function
        result = get_huggingface_model_metadata("some-model")
        
        # Should return None on network error
        assert result is None, "Should return None when network error occurs"
        
        # Verify the API was attempted
        mock_get.assert_called_once()
    
    @patch('HF_API_Integration.requests.get')
    def test_api_timeout_error(self, mock_get):
        """Test that timeout errors are handled gracefully"""
        # Simulate a timeout error
        mock_get.side_effect = requests.exceptions.Timeout("Request timed out")
        
        result = get_huggingface_model_metadata("slow-model")
        
        assert result is None, "Should return None when timeout occurs"
    
    @patch('HF_API_Integration.requests.get')
    def test_api_http_error(self, mock_get):
        """Test handling of HTTP error responses (404, 500, etc.)"""
        # Create a mock response that raises HTTP error
        mock_response = Mock()
        mock_response.raise_for_status.side_effect = requests.exceptions.HTTPError("404 Not Found")
        mock_get.return_value = mock_response
        
        result = get_huggingface_model_metadata("non-existent-model")
        
        assert result is None, "Should return None for HTTP errors"
    
    @patch('HF_API_Integration.requests.get')
    def test_api_missing_fields(self, mock_get):
        """Test handling of API response with missing fields"""
        # Create a mock response with partial data
        mock_response = Mock()
        mock_response.json.return_value = {
            "modelId": "incomplete-model",
            # Missing downloads, likes, lastModified
        }
        mock_response.raise_for_status = Mock()
        mock_get.return_value = mock_response
        
        result = get_huggingface_model_metadata("incomplete-model")
        
        # Should still work but with default values
        assert result is not None
        assert result["model_id"] == "incomplete-model"
        assert result["downloads"] == 0, "Missing downloads should default to 0"
        assert result["likes"] == 0, "Missing likes should default to 0"
        assert result["lastModified"] == "unknown", "Missing lastModified should default to 'unknown'"
    
    @patch('HF_API_Integration.requests.get')
    def test_api_invalid_json(self, mock_get):
        """Test handling of invalid JSON response"""
        # Create a mock response that raises JSON decode error
        mock_response = Mock()
        mock_response.json.side_effect = ValueError("Invalid JSON")
        mock_response.raise_for_status = Mock()
        mock_get.return_value = mock_response
        
        result = get_huggingface_model_metadata("bad-json-model")
        
        assert result is None, "Should return None for invalid JSON"
    
    @patch('HF_API_Integration.requests.get')
    def test_large_download_count(self, mock_get):
        """Test handling of very large download numbers"""
        mock_response = Mock()
        mock_response.json.return_value = {
            "downloads": 999999999999,  # Very large number
            "likes": 0,
            "lastModified": "2024-01-01T00:00:00Z"
        }
        mock_response.raise_for_status = Mock()
        mock_get.return_value = mock_response
        
        result = get_huggingface_model_metadata("popular-model")
        
        assert result is not None
        assert result["downloads"] == 999999999999
        assert isinstance(result["downloads"], int)
    
    @patch('HF_API_Integration.requests.get')  
    def test_special_characters_in_model_id(self, mock_get):
        """Test that special characters in model IDs are handled correctly"""
        mock_response = Mock()
        mock_response.json.return_value = {
            "downloads": 100,
            "likes": 10,
            "lastModified": "2024-01-01T00:00:00Z"
        }
        mock_response.raise_for_status = Mock()
        mock_get.return_value = mock_response
        
        # Model ID with organization/name format
        result = get_huggingface_model_metadata("google/gemma-2b-it")
        
        assert result is not None
        assert result["model_id"] == "google/gemma-2b-it"
        
        # Verify URL encoding is correct
        mock_get.assert_called_with(
            "https://huggingface.co/api/models/google/gemma-2b-it",
            timeout=10
        )