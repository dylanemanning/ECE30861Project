"""Tests for src.hf_model_size matching the current implementation."""

import importlib.util
import sys
from unittest.mock import patch, Mock


def _load_module_with_constraints():
    spec = importlib.util.find_spec("src.hf_model_size")
    module = importlib.util.module_from_spec(spec)
    module.HARDWARE_CONSTRAINTS = {
        "raspberry_pi": 256 * 1024 * 1024,
        "desktop": 16 * 1024 * 1024 * 1024,
    }
    sys.modules["src.hf_model_size"] = module
    sys.modules["hf_model_size"] = module
    spec.loader.exec_module(module)
    return module


hf_model_size = _load_module_with_constraints()
get_model_file_sizes = hf_model_size.get_model_file_sizes
calculate_size_metric = hf_model_size.calculate_size_metric


class TestHFModelSize:
    def test_basic_size_collection(self):
        with patch('hf_model_size.requests.get') as mock_get:
            mock_resp = Mock()
            mock_resp.raise_for_status = Mock()
            mock_resp.json.return_value = {
                "siblings": [
                    {"rfilename": "model.safetensors", "size": 1024},
                    {"rfilename": "config.json", "size": 256},
                ]
            }
            mock_get.return_value = mock_resp

            result = get_model_file_sizes("bert-base")

        assert result["total_size_bytes"] == 1280
        assert result["files"][0]["filename"] == "model.safetensors"

    def test_head_fallback(self):
        with patch('hf_model_size.requests.get') as mock_get, patch('hf_model_size.requests.head') as mock_head:
            api_resp = Mock()
            api_resp.raise_for_status = Mock()
            api_resp.json.return_value = {"siblings": [{"rfilename": "weights.bin", "size": 0}]}
            mock_get.return_value = api_resp

            head_resp = Mock()
            head_resp.headers = {"Content-Length": "4096"}
            mock_head.return_value = head_resp

            result = get_model_file_sizes("fallback")

        assert result["total_size_bytes"] == 4096

    def test_get_model_file_sizes_error(self):
        with patch('hf_model_size.requests.get') as mock_get:
            mock_get.side_effect = RuntimeError("down")
            result = get_model_file_sizes("broken")

        assert result["model_id"] == "broken"
        assert "error" in result

    def test_calculate_size_metric(self):
        info = {"model_id": "tiny", "total_size_bytes": 128 * 1024 * 1024, "files": []}
        result = calculate_size_metric(info)
        assert 0 < result["size_metric"] <= 1
        assert set(result["size_score"].keys()) == {
            "raspberry_pi",
            "jetson_nano",
            "desktop_pc",
            "aws_server",
        }

    def test_calculate_size_metric_error_passthrough(self):
        info = {"model_id": "bad", "error": "fail"}
        result = calculate_size_metric(info)
        assert result["size_metric"] == 0.0

    def test_calculate_size_metric_zero(self):
        info = {"model_id": "zero", "total_size_bytes": 0}
        result = calculate_size_metric(info)
        assert result["size_metric"] == 0.0

    @patch('hf_model_size.requests.get')
    @patch('hf_model_size.requests.head')
    @patch('subprocess.run')
    @patch('tempfile.mkdtemp', return_value='/tmp/repo')
    @patch('shutil.rmtree')
    def test_git_lfs_fallback(self, mock_rmtree, mock_mkdtemp, mock_run, mock_head, mock_get):
        mock_resp = Mock()
        mock_resp.raise_for_status = Mock()
        mock_resp.json.return_value = {
            "siblings": [{"rfilename": "weights.bin", "size": 0}]
        }
        mock_get.return_value = mock_resp
        mock_head.side_effect = RuntimeError("no head")

        clone_result = Mock(stdout=b"", returncode=0)
        lfs_output = Mock(stdout=b"abc123 - weights.bin 2.5 GB\n", returncode=0)
        mock_run.side_effect = [clone_result, lfs_output]

        result = get_model_file_sizes("owner/model")

        expected_size = int(2.5 * (1024 ** 3))
        assert result["files"][0]["size"] == expected_size
        assert result["total_size_bytes"] == expected_size
        mock_rmtree.assert_called_once_with('/tmp/repo')
