"""Tests reflecting the pared-down HF API integration module."""

import importlib
import inspect

import pytest


hf = importlib.import_module("src.HF_API_Integration")


class TestHFAPIHelperSurface:
    def test_only_expected_exports_present(self):
        public = {name for name in dir(hf) if not name.startswith("_")}
        assert "get_license_info" in public
        # Ensure previously removed helpers are not silently reintroduced.
        unexpected = {
            "get_huggingface_model_metadata",
            "fetch_model_metrics",
            "fetch_dataset_quality",
        }
        assert not (public & unexpected)

    @pytest.mark.parametrize(
        "symbol",
        [
            "get_huggingface_model_metadata",
            "fetch_dataset_quality",
            "fetch_model_metrics",
            "_normalize_size",
            "check_license_lgplv21",
        ],
    )
    def test_removed_symbols_raise_attribute_error(self, symbol):
        with pytest.raises(AttributeError):
            getattr(hf, symbol)

    def test_get_license_info_signature(self):
        sig = inspect.signature(hf.get_license_info)
        assert list(sig.parameters) == ["model_id"]
