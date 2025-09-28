"""
Additional tests for analyze_repo.py helper functions
"""
import pytest
from unittest.mock import patch, Mock, MagicMock
import tempfile
import os
import subprocess
import sys
from pathlib import Path

from src.analyze_repo import (
    normalize_score,
    is_lgpl_compatible,
    compute_code_quality,
    compute_size_metric,
    compute_local_metrics
)


class TestAnalyzeRepoHelpers:
    
    def test_normalize_score_basic(self):
        """Test basic score normalization"""
        # Value at min should be 0
        assert normalize_score(0, 0, 10) == 0.0
        # Value at max should be 1
        assert normalize_score(10, 0, 10) == 1.0
        # Value at midpoint should be 0.5
        assert normalize_score(5, 0, 10) == 0.5
    
    def test_normalize_score_edge_cases(self):
        """Test edge cases for normalize_score"""
        # When min equals max
        assert normalize_score(5, 5, 5) == 1.0
        assert normalize_score(3, 5, 5) == 0.0
        
        # Value below min should be clamped to 0
        assert normalize_score(-5, 0, 10) == 0.0
        
        # Value above max should be clamped to 1
        assert normalize_score(15, 0, 10) == 1.0
    
    def test_normalize_score_different_ranges(self):
        """Test normalization with different ranges"""
        # Negative range
        assert normalize_score(-5, -10, 0) == 0.5
        
        # Large range
        assert normalize_score(500, 0, 1000) == 0.5
        
        # Fractional values
        assert normalize_score(0.5, 0, 1) == 0.5
    
    def test_is_lgpl_compatible_compatible_licenses(self):
        """Test LGPL compatibility for compatible licenses"""
        compatible = [
            "MIT", "MIT/X11", "BSD", "BSD-2-Clause", "BSD-3-Clause",
            "LGPL-2.1", "LGPL-2.1-only", "LGPL-2.1-or-later",
            "GPL-2.0", "GPL-2.0-or-later"
        ]
        
        for license_str in compatible:
            assert is_lgpl_compatible(license_str) == 1
            # Test case insensitivity
            assert is_lgpl_compatible(license_str.lower()) == 1
            assert is_lgpl_compatible(license_str.upper()) == 1
    
    def test_is_lgpl_compatible_incompatible_licenses(self):
        """Test LGPL compatibility for incompatible licenses"""
        incompatible = [
            "Apache-2.0", "AGPL-3.0", "Proprietary", 
            "Custom", "Unknown", ""
        ]
        
        for license_str in incompatible:
            assert is_lgpl_compatible(license_str) == 0
    
    def test_is_lgpl_compatible_none(self):
        """Test LGPL compatibility with None"""
        assert is_lgpl_compatible(None) == 0
    
    def test_is_lgpl_compatible_whitespace(self):
        """Test LGPL compatibility with whitespace"""
        assert is_lgpl_compatible("  MIT  ") == 1
        assert is_lgpl_compatible("  ") == 0
    
    def test_compute_size_metric_empty_repo(self):
        """Test size metric for empty repository"""
        with tempfile.TemporaryDirectory() as tmpdir:
            score = compute_size_metric(tmpdir)
            assert score == 0.0
    
    def test_compute_size_metric_with_weights(self):
        """Test size metric with weight files"""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create weight files
            weight_files = [
                ("model.bin", 5 * 1024 * 1024),  # 5MB
                ("weights.h5", 10 * 1024 * 1024),  # 10MB
                ("checkpoint.ckpt", 20 * 1024 * 1024),  # 20MB
                ("model.safetensors", 15 * 1024 * 1024),  # 15MB
                ("other.txt", 1024),  # Should be ignored
            ]
            
            for filename, size in weight_files:
                filepath = os.path.join(tmpdir, filename)
                with open(filepath, 'wb') as f:
                    f.write(b'\0' * size)
            
            score = compute_size_metric(tmpdir)
            
            # Total weight files: 50MB out of 16GB max
            expected = 50 / (16 * 1024)
            assert abs(score - expected) < 0.01
    
    def test_compute_size_metric_nested_directories(self):
        """Test size metric with nested directories"""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create nested structure
            subdir = os.path.join(tmpdir, "models", "bert")
            os.makedirs(subdir)
            
            model_path = os.path.join(subdir, "model.bin")
            with open(model_path, 'wb') as f:
                f.write(b'\0' * (100 * 1024 * 1024))  # 100MB
            
            score = compute_size_metric(tmpdir)
            
            # 100MB out of 16GB
            expected = 100 / (16 * 1024)
            assert abs(score - expected) < 0.01
    
    def test_compute_size_metric_custom_capacity(self):
        """Test size metric with custom max capacity"""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create a 50MB file
            filepath = os.path.join(tmpdir, "model.bin")
            with open(filepath, 'wb') as f:
                f.write(b'\0' * (50 * 1024 * 1024))
            
            # Test with 100MB max capacity
            score = compute_size_metric(tmpdir, max_capacity_mb=100)
            assert score == 0.5  # 50MB/100MB = 0.5
            
            # Test with 25MB max capacity (should cap at 1.0)
            score = compute_size_metric(tmpdir, max_capacity_mb=25)
            assert score == 1.0
    
    def test_compute_size_metric_permission_error(self):
        """Test size metric handles permission errors gracefully"""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create a file that might fail to read
            filepath = os.path.join(tmpdir, "model.bin")
            with open(filepath, 'wb') as f:
                f.write(b'\0' * 1024)
            
            # Mock getsize to raise permission error
            with patch('os.path.getsize', side_effect=PermissionError("Access denied")):
                score = compute_size_metric(tmpdir)
                assert score == 0.0  # Should handle error gracefully
    
    @patch('subprocess.run')
    @patch('os.walk')
    def test_compute_code_quality_success(self, mock_walk, mock_run):
        """Test successful code quality computation"""
        # Mock file system walk
        mock_walk.return_value = [
            ("/repo", [], ["file1.py", "file2.py", "README.md"]),
            ("/repo/src", [], ["module.py"])
        ]
        
        # Mock flake8 output
        mock_result = Mock()
        mock_result.stdout = "file1.py:10:1: E302 expected 2 blank lines\nfile2.py:5:80: E501 line too long"
        mock_result.stderr = ""
        mock_result.returncode = 0
        mock_run.return_value = mock_result
        
        score = compute_code_quality("/repo")
        
        # 3 Python files, 2 issues
        # avg_issues = 2/3 = 0.667
        # score = 1.0 - 0.667/150 = ~0.996
        assert score > 0.99
        assert score <= 1.0
    
    @patch('subprocess.run')
    @patch('os.walk')
    def test_compute_code_quality_many_issues(self, mock_walk, mock_run):
        """Test code quality with many issues"""
        # Mock 1 Python file
        mock_walk.return_value = [("/repo", [], ["file.py"])]
        
        # Mock many flake8 issues
        issues = "\n".join([f"file.py:{i}:1: E302 issue" for i in range(200)])
        mock_result = Mock()
        mock_result.stdout = issues
        mock_result.returncode = 1
        mock_run.return_value = mock_result
        
        score = compute_code_quality("/repo")
        
        # 200 issues / 1 file = 200 avg
        # score = 1.0 - 200/150 = negative, clamped to 0
        assert score == 0.0
    
    @patch('subprocess.run')
    @patch('os.walk')
    def test_compute_code_quality_no_python_files(self, mock_walk, mock_run):
        """Test code quality with no Python files"""
        mock_walk.return_value = [("/repo", [], ["README.md", "package.json"])]
        
        score = compute_code_quality("/repo")
        
        # No Python files, should still return a score
        assert score == 1.0  # No issues when no files
    
    @patch('subprocess.run')
    @patch('os.walk')
    def test_compute_code_quality_flake8_error(self, mock_walk, mock_run):
        """Test code quality when flake8 fails"""
        mock_walk.return_value = [("/repo", [], ["file.py"])]
        
        # Mock flake8 error
        mock_result = Mock()
        mock_result.stdout = ""
        mock_result.stderr = "flake8: command not found"
        mock_result.returncode = 1
        mock_run.return_value = mock_result
        
        score = compute_code_quality("/repo")
        
        # Should return 0 on flake8 error with no output
        assert score == 0.0
    
    @patch('subprocess.run')
    @patch('os.walk')
    def test_compute_code_quality_exception(self, mock_walk, mock_run):
        """Test code quality handles exceptions"""
        mock_walk.return_value = [("/repo", [], ["file.py"])]
        mock_run.side_effect = Exception("Subprocess failed")
        
        score = compute_code_quality("/repo")
        
        # Should return default 0.5 on exception
        assert score == 0.5
    
    @patch('subprocess.run')
    @patch('shutil.which')
    @patch('os.walk')
    def test_compute_code_quality_flake8_path(self, mock_walk, mock_which, mock_run):
        """Test code quality uses correct flake8 path"""
        mock_walk.return_value = [("/repo", [], ["file.py"])]
        
        # Test when flake8 is in PATH
        mock_which.return_value = "/usr/bin/flake8"
        mock_result = Mock(stdout="", returncode=0)
        mock_run.return_value = mock_result
        
        compute_code_quality("/repo")
        
        mock_run.assert_called_once()
        assert mock_run.call_args[0][0] == ["/usr/bin/flake8", "."]
        
        # Test when flake8 is not in PATH
        mock_run.reset_mock()
        mock_which.return_value = None
        
        compute_code_quality("/repo")
        
        mock_run.assert_called_once()
        assert mock_run.call_args[0][0] == [sys.executable, "-m", "flake8", "."]
    
    @patch('subprocess.run')
    def test_compute_local_metrics_full(self, mock_run):
        """Test compute_local_metrics with all components"""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create repo structure
            os.makedirs(os.path.join(tmpdir, "src"))
            
            # Create Python files
            for i in range(5):
                Path(os.path.join(tmpdir, f"file{i}.py")).touch()
            
            # Create weight file
            with open(os.path.join(tmpdir, "model.bin"), 'wb') as f:
                f.write(b'\0' * (100 * 1024 * 1024))  # 100MB
            
            # Mock git output
            git_result = Mock()
            git_result.stdout = "  10\tAlice <alice@example.com>\n  5\tBob <bob@example.com>"
            git_result.returncode = 0
            
            # Mock flake8 output
            flake8_result = Mock()
            flake8_result.stdout = "file1.py:1:1: E302 issue"
            flake8_result.returncode = 0
            
            mock_run.side_effect = [git_result, flake8_result]
            
            metrics = compute_local_metrics(tmpdir, license_str="MIT")
            
            assert "bus_factor" in metrics
            assert "code_quality" in metrics
            assert "license_score" in metrics
            assert "size" in metrics
            
            assert metrics["license_score"] == 1  # MIT is compatible
            assert metrics["bus_factor"] > 0  # Has contributors
            assert metrics["code_quality"] > 0  # Has some quality
            assert metrics["size"] > 0  # Has weight files
    
    @patch('subprocess.run')
    def test_compute_local_metrics_git_failure(self, mock_run):
        """Test compute_local_metrics when git fails"""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Mock git failure
            mock_run.side_effect = subprocess.CalledProcessError(1, "git")
            
            metrics = compute_local_metrics(tmpdir)
            
            assert metrics["bus_factor"] == 0.0  # Fallback value
            assert "code_quality" in metrics
            assert "size" in metrics
    
    @patch('subprocess.run')
    def test_compute_local_metrics_single_contributor(self, mock_run):
        """Test bus factor with single contributor"""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Mock single contributor
            git_result = Mock()
            git_result.stdout = "  100\tAlice <alice@example.com>"
            git_result.returncode = 0
            
            mock_run.side_effect = [git_result, Mock(stdout="", returncode=0)]
            
            metrics = compute_local_metrics(tmpdir)
            
            assert metrics["bus_factor"] == 0.0  # Single contributor = 0 bus factor