import pytest
import tempfile
import os
import shutil
from unittest.mock import patch, MagicMock
from pathlib import Path

# Import the functions we're testing
from src.analyze_repo import (
    clone_repo,
    extract_license,
    extract_repo_stats,
    analyze_repo
)


class TestAnalyzeRepo:
    
    def test_mit_license_detection(self):
        """Test that MIT license is correctly detected from LICENSE file"""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create a LICENSE file with MIT content
            license_path = os.path.join(tmpdir, "LICENSE")
            with open(license_path, "w") as f:
                f.write("MIT License\n\n"
                       "Copyright (c) 2024 Test Author\n\n"
                       "Permission is hereby granted, free of charge...")
            
            result = extract_license(tmpdir)
            assert result == "MIT", f"Expected 'MIT' but got '{result}'"
    
    def test_repo_stats_extraction(self):
        """Test that file counts are calculated correctly"""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create test files in different directories
            # Root level files
            open(os.path.join(tmpdir, "main.py"), "w").close()
            open(os.path.join(tmpdir, "utils.py"), "w").close()
            open(os.path.join(tmpdir, "README.md"), "w").close()
            open(os.path.join(tmpdir, "data.json"), "w").close()
            
            # Create a subdirectory with more files
            subdir = os.path.join(tmpdir, "src")
            os.makedirs(subdir)
            open(os.path.join(subdir, "helper.py"), "w").close()
            open(os.path.join(subdir, "config.yaml"), "w").close()
            
            stats = extract_repo_stats(tmpdir)
            
            assert stats["total_files"] == 6, f"Expected 6 total files but got {stats['total_files']}"
            assert stats["python_files"] == 3, f"Expected 3 Python files but got {stats['python_files']}"
    
    @patch('analyze_repo.clone_repo')
    @patch('analyze_repo.extract_license')
    @patch('analyze_repo.extract_repo_stats')
    def test_lgpl_compatibility_mit(self, mock_stats, mock_license, mock_clone):
        """Test that MIT license is marked as LGPL compatible"""
        # Mock the functions to avoid actual git operations
        mock_clone.return_value = None
        mock_license.return_value = "MIT"
        mock_stats.return_value = {"total_files": 10, "python_files": 5}
        
        with patch('tempfile.mkdtemp', return_value='/fake/temp'):
            with patch('shutil.rmtree'):
                result = analyze_repo("https://github.com/test/repo")
        
        # MIT should be LGPL compatible
        assert result["lgpl_compatible"] == True, "MIT should be LGPL compatible"
        assert result["license"] == "MIT"
        assert result["total_files"] == 10
        assert result["python_files"] == 5
        assert result["repo"] == ["test", "repo"]
    
    def test_apache_license_detection(self):
        """Test that Apache license is correctly detected"""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create a LICENSE file with Apache content
            license_path = os.path.join(tmpdir, "LICENSE.txt")
            with open(license_path, "w") as f:
                f.write("Apache License\n"
                       "Version 2.0, January 2004\n"
                       "http://www.apache.org/licenses/")
            
            result = extract_license(tmpdir)
            assert result == "Apache", f"Expected 'Apache' but got '{result}'"
    
    def test_unknown_license_detection(self):
        """Test that unknown licenses are properly handled"""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create a LICENSE file with unknown content
            license_path = os.path.join(tmpdir, "LICENSE")
            with open(license_path, "w") as f:
                f.write("Custom Proprietary License\n"
                       "All rights reserved.\n"
                       "No permissions granted.")
            
            result = extract_license(tmpdir)
            assert result == "Unknown", f"Expected 'Unknown' but got '{result}'"
    
    def test_no_license_file(self):
        """Test behavior when no license file exists"""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Don't create any LICENSE file
            result = extract_license(tmpdir)
            assert result == "Unknown", f"Expected 'Unknown' for missing license but got '{result}'"
    
    @patch('analyze_repo.clone_repo')
    @patch('analyze_repo.extract_license')
    @patch('analyze_repo.extract_repo_stats')
    def test_lgpl_compatibility_unknown(self, mock_stats, mock_license, mock_clone):
        """Test that unknown licenses are marked as NOT LGPL compatible"""
        mock_clone.return_value = None
        mock_license.return_value = "Unknown"
        mock_stats.return_value = {"total_files": 5, "python_files": 2}
        
        with patch('tempfile.mkdtemp', return_value='/fake/temp'):
            with patch('shutil.rmtree'):
                result = analyze_repo("https://github.com/test/repo")
        
        # Unknown should NOT be LGPL compatible
        assert result["lgpl_compatible"] == False, "Unknown license should not be LGPL compatible"
        assert result["license"] == "Unknown"
    
    def test_gpl_license_detection(self):
        """Test that GPL/LGPL licenses are correctly detected"""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Test GPL detection
            license_path = os.path.join(tmpdir, "COPYING")
            with open(license_path, "w") as f:
                f.write("GNU GENERAL PUBLIC LICENSE\n"
                       "Version 3, 29 June 2007\n"
                       "Copyright (C) 2007 Free Software Foundation")
            
            result = extract_license(tmpdir)
            assert result == "GPL/LGPL", f"Expected 'GPL/LGPL' but got '{result}'"
    
    def test_bsd_license_detection(self):
        """Test that BSD license is correctly detected"""
        with tempfile.TemporaryDirectory() as tmpdir:
            license_path = os.path.join(tmpdir, "LICENSE")
            with open(license_path, "w") as f:
                f.write("BSD 3-Clause License\n\n"
                       "Copyright (c) 2024, Test\n"
                       "Redistribution and use in source and binary forms...")
            
            result = extract_license(tmpdir)
            assert result == "BSD", f"Expected 'BSD' but got '{result}'"