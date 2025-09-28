"""
Test suite for genai_readme_analysis.py module
"""
import pytest
from unittest.mock import patch, Mock, MagicMock
import json
import os

from src.genai_readme_analysis import (
    analyze_with_genai,
    _parse_llm_content_to_json,
    _flatten_metrics,
    analyze_metrics,
    discover_dataset_url_with_genai
)


class TestGenAIReadmeAnalysis:
    
    def test_parse_llm_content_to_json_with_code_fence(self):
        """Test parsing JSON from LLM response with code fence"""
        content = '''Here is the analysis:
        ```json
        {
            "ramp_up_time": 0.8,
            "performance_claims": 0.7
        }
        ```
        '''
        result = _parse_llm_content_to_json(content)
        assert result == {"ramp_up_time": 0.8, "performance_claims": 0.7}
    
    def test_parse_llm_content_to_json_raw(self):
        """Test parsing raw JSON from LLM response"""
        content = '{"dataset_and_code_score": 0.9, "dataset_and_code_score_latency": 150}'
        result = _parse_llm_content_to_json(content)
        assert result == {"dataset_and_code_score": 0.9, "dataset_and_code_score_latency": 150}
    
    def test_parse_llm_content_to_json_with_text_before(self):
        """Test parsing JSON when there's text before it"""
        content = '''Let me analyze this:
        The model looks good.
        {"ramp_up_time": 0.6}'''
        result = _parse_llm_content_to_json(content)
        assert result == {"ramp_up_time": 0.6}
    
    def test_parse_llm_content_to_json_invalid(self):
        """Test parsing invalid JSON returns empty dict"""
        content = "This is not JSON at all"
        result = _parse_llm_content_to_json(content)
        assert result == {}
    
    def test_parse_llm_content_malformed_json(self):
        """Test parsing malformed JSON returns empty dict"""
        content = '{"incomplete": '
        result = _parse_llm_content_to_json(content)
        assert result == {}
    
    def test_flatten_metrics_simple(self):
        """Test flattening simple metrics"""
        metrics = {
            "ramp_up_time": 0.8,
            "ramp_up_time_latency": 100,
            "performance_claims": 0.7,
            "performance_claims_latency": 200
        }
        result = _flatten_metrics(metrics)
        assert result["ramp_up_time"] == 0.8
        assert result["ramp_up_time_latency"] == 100
        assert result["performance_claims"] == 0.7
        assert result["performance_claims_latency"] == 200
    
    def test_flatten_metrics_nested(self):
        """Test flattening nested metrics with score/latency structure"""
        metrics = {
            "ramp_up_time": {
                "score": 0.9,
                "latency": 150.5
            },
            "dataset_and_code_score": {
                "score": 0.85,
                "latency": 250
            }
        }
        result = _flatten_metrics(metrics)
        assert result["ramp_up_time"] == 0.9
        assert result["ramp_up_time_latency"] == 151  # Rounded
        assert result["dataset_and_code_score"] == 0.85
        assert result["dataset_and_code_score_latency"] == 250
    
    def test_flatten_metrics_mixed(self):
        """Test flattening mixed format metrics"""
        metrics = {
            "ramp_up_time": 0.7,
            "performance_claims": {
                "score": 0.6,
                "latency": 100
            },
            "dataset_and_code_score_latency": 300,
            "ignored_metric": 0.5  # Should be ignored
        }
        result = _flatten_metrics(metrics)
        assert result["ramp_up_time"] == 0.7
        assert result["performance_claims"] == 0.6
        assert result["performance_claims_latency"] == 100
        assert result["dataset_and_code_score_latency"] == 300
        assert "ignored_metric" not in result
    
    def test_flatten_metrics_invalid_values(self):
        """Test flattening with invalid values"""
        metrics = {
            "ramp_up_time": "invalid",
            "performance_claims": None,
            "dataset_and_code_score": 0.5
        }
        result = _flatten_metrics(metrics)
        assert "ramp_up_time" not in result
        assert "performance_claims" not in result
        assert result["dataset_and_code_score"] == 0.5
    
    @patch('src.genai_readme_analysis.requests.post')
    def test_analyze_with_genai_success(self, mock_post):
        """Test successful GenAI API call"""
        mock_response = Mock()
        mock_response.json.return_value = {
            "choices": [{
                "message": {
                    "content": '{"ramp_up_time": 0.8, "ramp_up_time_latency": 100}'
                }
            }]
        }
        mock_post.return_value = mock_response
        
        # Set a short timeout for testing
        os.environ['GENAI_TIMEOUT'] = '5'
        
        result = analyze_with_genai(
            readme="# Test README",
            code="print('hello')",
            model="test/model"
        )
        
        assert result is not None
        assert "choices" in result
        
        # Verify API call
        mock_post.assert_called_once()
        call_args = mock_post.call_args
        assert call_args[0][0] == "https://genai.rcac.purdue.edu/api/chat/completions"
        assert "Authorization" in call_args[1]["headers"]
        assert call_args[1]["timeout"] == 5.0
    
    @patch('src.genai_readme_analysis.requests.post')
    def test_analyze_with_genai_timeout(self, mock_post):
        """Test GenAI API call with timeout"""
        import requests
        mock_post.side_effect = requests.exceptions.Timeout("Request timed out")
        
        with pytest.raises(requests.exceptions.Timeout):
            analyze_with_genai(readme="Test")
    
    @patch('src.genai_readme_analysis.requests.post')
    def test_analyze_with_genai_http_error(self, mock_post):
        """Test GenAI API call with HTTP error"""
        import requests
        mock_response = Mock()
        mock_response.raise_for_status.side_effect = requests.exceptions.HTTPError("500 Server Error")
        mock_post.return_value = mock_response
        
        with pytest.raises(requests.exceptions.HTTPError):
            analyze_with_genai(readme="Test")
    
    @patch('src.genai_readme_analysis.analyze_with_genai')
    def test_analyze_metrics_success(self, mock_analyze):
        """Test analyze_metrics with successful response"""
        mock_analyze.return_value = {
            "choices": [{
                "message": {
                    "content": '''{
                        "ramp_up_time": 0.75,
                        "ramp_up_time_latency": 120,
                        "performance_claims": 0.85,
                        "performance_claims_latency": 150,
                        "dataset_and_code_score": 0.9,
                        "dataset_and_code_score_latency": 200
                    }'''
                }
            }]
        }
        
        result = analyze_metrics(
            readme="# Model Card",
            code="train.py",
            model="bert-base"
        )
        
        assert result["ramp_up_time"] == 0.75
        assert result["ramp_up_time_latency"] == 120
        assert result["performance_claims"] == 0.85
        assert result["dataset_and_code_score"] == 0.9
    
    @patch('src.genai_readme_analysis.analyze_with_genai')
    def test_analyze_metrics_api_failure(self, mock_analyze):
        """Test analyze_metrics when API fails"""
        mock_analyze.side_effect = Exception("API Error")
        
        result = analyze_metrics(readme="Test")
        assert result == {}
    
    @patch('src.genai_readme_analysis.analyze_with_genai')
    def test_analyze_metrics_no_content(self, mock_analyze):
        """Test analyze_metrics with missing content in response"""
        mock_analyze.return_value = {"choices": [{"message": {}}]}
        
        result = analyze_metrics(readme="Test")
        assert result == {}
    
    @patch('src.genai_readme_analysis.analyze_with_genai')
    def test_analyze_metrics_invalid_json(self, mock_analyze):
        """Test analyze_metrics with invalid JSON in response"""
        mock_analyze.return_value = {
            "choices": [{
                "message": {"content": "Not valid JSON"}
            }]
        }
        
        result = analyze_metrics(readme="Test")
        assert result == {}
    
    @patch('src.genai_readme_analysis.requests.post')
    def test_discover_dataset_url_with_genai_success(self, mock_post):
        """Test successful dataset URL discovery"""
        mock_response = Mock()
        mock_response.json.return_value = {
            "choices": [{
                "message": {
                    "content": '''{
                        "dataset_url": "https://huggingface.co/datasets/squad/squad_v2",
                        "dataset_discovery_latency": 250
                    }'''
                }
            }]
        }
        mock_post.return_value = mock_response
        
        result = discover_dataset_url_with_genai(
            readme="This model was trained on SQuAD v2",
            model="bert-qa"
        )
        
        assert result["dataset_url"] == "https://huggingface.co/datasets/squad/squad_v2"
        assert result["dataset_discovery_latency"] == 250
    
    @patch('src.genai_readme_analysis.requests.post')
    def test_discover_dataset_url_empty(self, mock_post):
        """Test dataset discovery with no dataset found"""
        mock_response = Mock()
        mock_response.json.return_value = {
            "choices": [{
                "message": {
                    "content": '{"dataset_url": "", "dataset_discovery_latency": 100}'
                }
            }]
        }
        mock_post.return_value = mock_response
        
        result = discover_dataset_url_with_genai(
            readme="No dataset mentioned",
            model="test"
        )
        
        assert "dataset_url" not in result  # Empty strings are not included
        assert result["dataset_discovery_latency"] == 100
    
    @patch('src.genai_readme_analysis.requests.post')
    def test_discover_dataset_url_api_error(self, mock_post):
        """Test dataset discovery with API error"""
        import requests
        mock_post.side_effect = requests.exceptions.RequestException("Network error")
        
        with pytest.raises(requests.exceptions.RequestException):
            discover_dataset_url_with_genai(readme="Test")
    
    @patch('src.genai_readme_analysis.requests.post')
    def test_discover_dataset_url_invalid_response(self, mock_post):
        """Test dataset discovery with invalid response structure"""
        mock_response = Mock()
        mock_response.json.return_value = {"error": "Invalid request"}
        mock_post.return_value = mock_response
        
        result = discover_dataset_url_with_genai(readme="Test")
        assert result == {}
    
    def test_api_key_from_environment(self):
        """Test that API key can be set from environment"""
        test_key = "test-key-12345"
        os.environ["GEN_AI_STUDIO_API_KEY"] = test_key
        
        with patch('src.genai_readme_analysis.requests.post') as mock_post:
            mock_response = Mock()
            mock_response.json.return_value = {"choices": [{"message": {"content": "{}"}}]}
            mock_post.return_value = mock_response
            
            analyze_with_genai(readme="Test")
            
            # Check that the test key was used
            call_args = mock_post.call_args
            auth_header = call_args[1]["headers"]["Authorization"]
            assert auth_header == f"Bearer {test_key}"
        
        # Clean up
        del os.environ["GEN_AI_STUDIO_API_KEY"]