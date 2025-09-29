"""Tests for the limited HF API integration surface."""

from unittest.mock import patch

import pytest

import src.HF_API_Integration as hf


class TestHFAPIIntegration:
    def test_get_license_info_passthrough(self):
        expected = {"model_id": "owner/model", "license": "mit"}
        with patch("src.HF_API_Integration.license_compat", return_value=expected) as mock_compat:
            assert hf.get_license_info("owner/model") == expected
            mock_compat.assert_called_once_with("owner/model")

    def test_get_license_info_requires_model(self):
        with patch("src.HF_API_Integration.license_compat", return_value={}) as mock_compat:
            result = hf.get_license_info(" ")
            mock_compat.assert_called_once_with(" ")
            assert isinstance(result, dict)

    def test_missing_helpers_raise_attribute_error(self):
        with pytest.raises(AttributeError):
            getattr(hf, "get_huggingface_model_metadata")
