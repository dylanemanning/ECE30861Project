"""
Test suite for collect_ndjson.py module
"""
import pytest
import json
import tempfile
import os
from unittest.mock import patch, Mock, MagicMock, mock_open
from io import StringIO
import sys

# Import the functions we're testing
from src.collect_ndjson import (
    read_lines_file,
    write_ndjson_line,
    collect_and_write,
    parse_args,
    main
)


class TestCollectNDJSON:
    
    def test_read_lines_file_basic(self):
        """Test reading a file with URLs, skipping comments and empty lines"""
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.txt') as f:
            f.write("https://github.com/user/repo1\n")
            f.write("# This is a comment\n")
            f.write("\n")  # empty line
            f.write("https://github.com/user/repo2\n")
            f.write("   \n")  # whitespace only
            f.write("https://github.com/user/repo3\n")
            temp_file = f.name
        
        try:
            lines = read_lines_file(temp_file)
            assert len(lines) == 3, f"Expected 3 lines, got {len(lines)}"
            assert lines[0] == "https://github.com/user/repo1"
            assert lines[1] == "https://github.com/user/repo2"
            assert lines[2] == "https://github.com/user/repo3"
        finally:
            os.unlink(temp_file)
    
    def test_write_ndjson_line(self):
        """Test writing NDJSON line format correctly"""
        output = StringIO()
        test_obj = {
            "type": "repo",
            "source": "https://github.com/test/repo",
            "data": {"license": "MIT", "files": 10}
        }
        
        write_ndjson_line(output, test_obj)
        
        output.seek(0)
        written = output.read()
        
        # Should end with newline
        assert written.endswith('\n')
        
        # Should be valid JSON (minus the newline)
        json_part = written.strip()
        parsed = json.loads(json_part)
        assert parsed == test_obj
    
    @patch('src.collect_ndjson._analyze_repo_fn')
    @patch('src.collect_ndjson._hf_meta_fn')
    def test_collect_and_write_mixed_sources(self, mock_hf, mock_repo):
        """Test collecting data from both repos and models"""
        # Mock the analysis functions
        mock_repo.return_value = {"license": "MIT", "total_files": 50}
        mock_hf.return_value = {"model_id": "test-model", "downloads": 1000}
        
        # Create temporary output file
        with tempfile.NamedTemporaryFile(mode='w+', delete=False, suffix='.ndjson') as f:
            temp_output = f.name
        
        try:
            # Collect data for 1 repo and 1 model
            repos = ["https://github.com/test/repo"]
            models = ["test-model"]
            
            result = collect_and_write(repos, models, temp_output, append=False)
            
            assert result == 0, "Should return 0 on success"
            
            # Read and verify the output
            with open(temp_output, 'r') as f:
                lines = f.readlines()
            
            assert len(lines) == 2, f"Expected 2 NDJSON lines, got {len(lines)}"
            
            # Check first line (repo)
            obj1 = json.loads(lines[0])
            assert obj1["type"] == "repo"
            assert obj1["source"] == "https://github.com/test/repo"
            assert obj1["data"]["license"] == "MIT"
            
            # Check second line (model)
            obj2 = json.loads(lines[1])
            assert obj2["type"] == "hf_model"
            assert obj2["source"] == "test-model"
            assert obj2["data"]["downloads"] == 1000
            
        finally:
            if os.path.exists(temp_output):
                os.unlink(temp_output)
    
    @patch('src.collect_ndjson._analyze_repo_fn')
    def test_collect_handles_analysis_errors(self, mock_repo):
        """Test that collection handles errors gracefully"""
        # Make the analysis function raise an exception
        mock_repo.side_effect = Exception("Network error: unable to clone")
        
        output = StringIO()
        repos = ["https://github.com/bad/repo"]
        models = []
        
        # Mock sys.stdout to capture output
        with patch('sys.stdout', output):
            result = collect_and_write(repos, models, None, append=False)
        
        assert result == 0, "Should still return 0 even with errors"
        
        output.seek(0)
        written = output.read()
        
        # Should have written an error record
        obj = json.loads(written.strip())
        assert obj["type"] == "repo"
        assert "error" in obj["data"]
        assert "Network error" in obj["data"]["error"]
    
    def test_parse_args_multiple_repos_and_models(self):
        """Test parsing command line arguments with multiple repos and models"""
        argv = [
            "--repo", "https://github.com/user/repo1",
            "--repo", "https://github.com/user/repo2",
            "--model", "bert-base",
            "--model", "gpt2",
            "--output", "results.ndjson"
        ]
        
        args = parse_args(argv)
        
        assert len(args.repo) == 2
        assert "https://github.com/user/repo1" in args.repo
        assert "https://github.com/user/repo2" in args.repo
        assert len(args.model) == 2
        assert "bert-base" in args.model
        assert "gpt2" in args.model
        assert args.output == "results.ndjson"
        assert args.append == False  # default value
    
    def test_main_no_inputs_provided(self):
        """Test main function when no repos or models are provided"""
        argv = ["--output", "out.ndjson"]
        
        # Capture stdout to check error message
        with patch('sys.stdout', new=StringIO()) as fake_out:
            result = main(argv)
        
        assert result == 2, "Should return error code 2 when no inputs provided"
        output = fake_out.getvalue()
        assert "No repos or models provided" in output