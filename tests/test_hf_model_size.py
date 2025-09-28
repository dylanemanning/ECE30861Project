"""
Test suite for hf_model_size.py module
"""
import pytest
from unittest.mock import patch, Mock, MagicMock, call
import subprocess
import tempfile
import shutil
import os

from src.hf_model_size import get_model_file_sizes


class TestHFModelSize:
    
    @patch('src.hf_model_size.requests.get')
    def test_get_model_file_sizes_success(self, mock_get):
        """Test successful model file size retrieval"""
        mock_response = Mock()
        mock_response.json.return_value = {
            "siblings": [
                {"rfilename": "model.safetensors", "size": 1024000},
                {"rfilename": "config.json", "size": 500},
                {"rfilename": "tokenizer.json", "size": 2000}
            ]
        }
        mock_response.raise_for_status = Mock()
        mock_get.return_value = mock_response
        
        result = get_model_file_sizes("bert-base-uncased")
        
        assert result["model_id"] == "bert-base-uncased"
        assert result["total_size_bytes"] == 1026500
        assert len(result["files"]) == 3
        assert result["files"][0]["filename"] == "model.safetensors"
        assert result["files"][0]["size"] == 1024000
    
    @patch('src.hf_model_size.requests.get')
    def test_get_model_file_sizes_empty_siblings(self, mock_get):
        """Test model with no files"""
        mock_response = Mock()
        mock_response.json.return_value = {"siblings": []}
        mock_response.raise_for_status = Mock()
        mock_get.return_value = mock_response
        
        result = get_model_file_sizes("empty-model")
        
        assert result["model_id"] == "empty-model"
        assert result["total_size_bytes"] == 0
        assert result["files"] == []
    
    @patch('src.hf_model_size.requests.get')
    def test_get_model_file_sizes_missing_size_field(self, mock_get):
        """Test handling files with missing size field"""
        mock_api_response = Mock()
        mock_api_response.json.return_value = {
            "siblings": [
                {"rfilename": "model.bin"},  # No size field
                {"rfilename": "config.json", "size": 1000}
            ]
        }
        mock_api_response.raise_for_status = Mock()
        
        # Mock HEAD request for missing size
        mock_head_response = Mock()
        mock_head_response.headers = {"Content-Length": "5000000"}
        
        mock_get.side_effect = [mock_api_response]
        
        with patch('src.hf_model_size.requests.head', return_value=mock_head_response):
            result = get_model_file_sizes("test-model")
        
        assert result["total_size_bytes"] == 5001000
        assert result["files"][0]["size"] == 5000000
        assert result["files"][1]["size"] == 1000
    
    @patch('src.hf_model_size.requests.get')
    @patch('src.hf_model_size.requests.head')
    def test_get_model_file_sizes_head_request_failure(self, mock_head, mock_get):
        """Test fallback when HEAD request fails"""
        mock_response = Mock()
        mock_response.json.return_value = {
            "siblings": [{"rfilename": "model.bin", "size": 0}]
        }
        mock_response.raise_for_status = Mock()
        mock_get.return_value = mock_response
        
        # HEAD request fails
        mock_head.side_effect = Exception("Network error")
        
        result = get_model_file_sizes("test-model")
        
        assert result["total_size_bytes"] == 0
        assert result["files"][0]["size"] == 0
    
    @patch('src.hf_model_size.requests.get')
    @patch('src.hf_model_size.subprocess.run')
    @patch('src.hf_model_size.tempfile.mkdtemp')
    @patch('src.hf_model_size.shutil.rmtree')
    def test_get_model_file_sizes_git_lfs_fallback(self, mock_rmtree, mock_mkdtemp, mock_subprocess, mock_get):
        """Test git-lfs fallback for missing sizes"""
        mock_response = Mock()
        mock_response.json.return_value = {
            "siblings": [
                {"rfilename": "model.bin", "size": 0},
                {"rfilename": "config.json", "size": 500}
            ]
        }
        mock_response.raise_for_status = Mock()
        mock_get.return_value = mock_response
        
        # Mock tempdir
        mock_mkdtemp.return_value = "/tmp/test"
        
        # Mock git clone success
        clone_result = Mock(returncode=0, stdout=b"", stderr=b"")
        
        # Mock git lfs output
        lfs_result = Mock(
            returncode=0,
            stdout=b"abc123 - model.bin (250 MB)\ndef456 - other.txt (1 KB)",
            stderr=b""
        )
        
        mock_subprocess.side_effect = [clone_result, lfs_result]
        
        with patch('src.hf_model_size.requests.head', side_effect=Exception("No HEAD")):
            result = get_model_file_sizes("test-model")
        
        assert result["total_size_bytes"] == 262144500  # 250MB + 500 bytes
        assert result["files"][0]["size"] == 262144000  # 250MB in bytes
        assert result["files"][1]["size"] == 500
        
        # Verify cleanup
        mock_rmtree.assert_called_once_with("/tmp/test")
    
    @patch('src.hf_model_size.requests.get')
    @patch('src.hf_model_size.subprocess.run')
    @patch('src.hf_model_size.tempfile.mkdtemp')
    @patch('src.hf_model_size.shutil.rmtree')
    def test_get_model_file_sizes_git_lfs_gb_unit(self, mock_rmtree, mock_mkdtemp, mock_subprocess, mock_get):
        """Test git-lfs output parsing with GB units"""
        mock_response = Mock()
        mock_response.json.return_value = {
            "siblings": [{"rfilename": "large_model.bin", "size": 0}]
        }
        mock_response.raise_for_status = Mock()
        mock_get.return_value = mock_response
        
        mock_mkdtemp.return_value = "/tmp/test"
        
        # Mock git lfs with GB unit
        lfs_result = Mock(
            returncode=0,
            stdout=b"abc123 - large_model.bin (2.5 GB)",
            stderr=b""
        )
        
        mock_subprocess.side_effect = [Mock(returncode=0), lfs_result]
        
        with patch('src.hf_model_size.requests.head', side_effect=Exception("No HEAD")):
            result = get_model_file_sizes("test-model")
        
        # 2.5 GB = 2.5 * 1024^3 bytes
        expected_size = int(2.5 * (1024 ** 3))
        assert result["total_size_bytes"] == expected_size
        assert result["files"][0]["size"] == expected_size
    
    @patch('src.hf_model_size.requests.get')
    @patch('src.hf_model_size.subprocess.run')
    @patch('src.hf_model_size.tempfile.mkdtemp')
    @patch('src.hf_model_size.shutil.rmtree')
    def test_get_model_file_sizes_git_clone_failure(self, mock_rmtree, mock_mkdtemp, mock_subprocess, mock_get):
        """Test handling of git clone failure"""
        mock_response = Mock()
        mock_response.json.return_value = {
            "siblings": [{"rfilename": "model.bin", "size": 0}]
        }
        mock_response.raise_for_status = Mock()
        mock_get.return_value = mock_response
        
        mock_mkdtemp.return_value = "/tmp/test"
        
        # Git clone fails
        mock_subprocess.side_effect = Exception("Git not installed")
        
        with patch('src.hf_model_size.requests.head', side_effect=Exception("No HEAD")):
            result = get_model_file_sizes("test-model")
        
        # Should still return result with 0 size
        assert result["total_size_bytes"] == 0
        assert result["files"][0]["size"] == 0
        
        # Cleanup should still be called
        mock_rmtree.assert_called_once()
    
    @patch('src.hf_model_size.requests.get')
    def test_get_model_file_sizes_api_error(self, mock_get):
        """Test handling of API errors"""
        mock_get.side_effect = Exception("API Error")
        
        result = get_model_file_sizes("invalid-model")
        
        assert result["model_id"] == "invalid-model"
        assert "error" in result
        assert "API Error" in result["error"]
    
    @patch('src.hf_model_size.requests.get')
    def test_get_model_file_sizes_http_error(self, mock_get):
        """Test handling of HTTP errors"""
        import requests
        mock_response = Mock()
        mock_response.raise_for_status.side_effect = requests.exceptions.HTTPError("404 Not Found")
        mock_get.return_value = mock_response
        
        result = get_model_file_sizes("nonexistent-model")
        
        assert result["model_id"] == "nonexistent-model"
        assert "error" in result
    
    @patch('src.hf_model_size.requests.get')
    @patch('src.hf_model_size.subprocess.run')
    @patch('src.hf_model_size.tempfile.mkdtemp')
    @patch('src.hf_model_size.shutil.rmtree')
    def test_get_model_file_sizes_lfs_parsing_edge_cases(self, mock_rmtree, mock_mkdtemp, mock_subprocess, mock_get):
        """Test edge cases in git-lfs output parsing"""
        mock_response = Mock()
        mock_response.json.return_value = {
            "siblings": [
                {"rfilename": "model1.bin", "size": 0},
                {"rfilename": "model2.bin", "size": 0}
            ]
        }
        mock_response.raise_for_status = Mock()
        mock_get.return_value = mock_response
        
        mock_mkdtemp.return_value = "/tmp/test"
        
        # Mock git lfs with various formats
        lfs_result = Mock(
            returncode=0,
            stdout=b"""
            abc123 - model1.bin (100 B)
            def456 - model2.bin (0.5 KB)
            invalid line without proper format
            """,
            stderr=b""
        )
        
        mock_subprocess.side_effect = [Mock(returncode=0), lfs_result]
        
        with patch('src.hf_model_size.requests.head', side_effect=Exception("No HEAD")):
            result = get_model_file_sizes("test-model")
        
        assert result["files"][0]["size"] == 100  # 100 bytes
        assert result["files"][1]["size"] == 512  # 0.5 KB = 512 bytes
        assert result["total_size_bytes"] == 612