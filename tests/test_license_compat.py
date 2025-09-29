"""
Test suite for license_compat.py module
"""
import pytest
from unittest.mock import patch, Mock
import requests

from src.license_compat import (
    license_compat,
    extract_license,
    is_lgpl_compatible,
    COMPATIBLE_LICENSES
)


class TestLicenseCompat:
    
    def test_compatible_licenses_set(self):
        """Test that COMPATIBLE_LICENSES contains expected licenses"""
        assert "mit" in COMPATIBLE_LICENSES
        assert "apache-2.0" in COMPATIBLE_LICENSES
        assert "bsd-3-clause" in COMPATIBLE_LICENSES
        assert "lgpl-2.1" in COMPATIBLE_LICENSES
    
    def test_is_lgpl_compatible_mit(self):
        """Test MIT license is compatible"""
        assert is_lgpl_compatible("MIT") == 1
        assert is_lgpl_compatible("mit") == 1
        assert is_lgpl_compatible("MiT") == 1
    
    def test_is_lgpl_compatible_apache(self):
        """Test Apache license is compatible"""
        assert is_lgpl_compatible("apache-2.0") == 1
        assert is_lgpl_compatible("Apache-2.0") == 1
        assert is_lgpl_compatible("APACHE-2.0") == 1
    
    def test_is_lgpl_compatible_bsd(self):
        """Test BSD licenses are compatible"""
        assert is_lgpl_compatible("bsd-2-clause") == 1
        assert is_lgpl_compatible("bsd-3-clause") == 1
        assert is_lgpl_compatible("BSD-3-Clause") == 1
    
    def test_is_lgpl_compatible_lgpl_itself(self):
        """Test LGPL is compatible with itself"""
        assert is_lgpl_compatible("lgpl-2.1") == 1
        assert is_lgpl_compatible("lgpl-2.1-or-later") == 1
        assert is_lgpl_compatible("LGPL-2.1") == 1
    
    def test_is_lgpl_compatible_gpl(self):
        """Test GPL is compatible"""
        assert is_lgpl_compatible("gpl-2.0") == 1
        assert is_lgpl_compatible("GPL-2.0") == 1
    
    def test_is_lgpl_compatible_incompatible(self):
        """Test incompatible licenses"""
        assert is_lgpl_compatible("proprietary") == 0
        assert is_lgpl_compatible("unknown") == 0
        assert is_lgpl_compatible("custom") == 0
        assert is_lgpl_compatible("") == 0
        with pytest.raises(AttributeError):
            is_lgpl_compatible(None)
    
    def test_is_lgpl_compatible_other_open_source(self):
        """Test other compatible open source licenses"""
        assert is_lgpl_compatible("isc") == 1
        assert is_lgpl_compatible("zlib") == 1
        assert is_lgpl_compatible("mpl-2.0") == 1
        assert is_lgpl_compatible("epl-2.0") == 1
        assert is_lgpl_compatible("cddl-1.0") == 1
    
    def test_extract_license_top_level_string(self):
        """Test extracting license from top-level field"""
        data = {"license": "MIT"}
        assert extract_license(data) == "MIT"
    
    def test_extract_license_carddata_string(self):
        """Test extracting license from cardData field"""
        data = {
            "license": "unknown",
            "cardData": {"license": "Apache-2.0"}
        }
        assert extract_license(data) == "Apache-2.0"
    
    def test_extract_license_carddata_list(self):
        """Test extracting license from cardData list"""
        data = {
            "cardData": {"license": ["MIT", "Apache-2.0"]}
        }
        assert extract_license(data) == "MIT"  # Takes first from list
    
    def test_extract_license_empty_list(self):
        """Test extracting license from empty list"""
        data = {
            "cardData": {"license": []}
        }
        assert extract_license(data) == "unknown"
    
    def test_extract_license_no_license(self):
        """Test when no license field exists"""
        data = {"other_field": "value"}
        assert extract_license(data) == "unknown"
    
    def test_extract_license_unknown_value(self):
        """Test when license field is 'unknown'"""
        data = {"license": "unknown"}
        assert extract_license(data) == "unknown"
    
    def test_extract_license_none_carddata(self):
        """Test when cardData is None"""
        data = {
            "license": "unknown",
            "cardData": None
        }
        assert extract_license(data) == "unknown"
    
    def test_extract_license_prefer_top_level(self):
        """Test that valid top-level license is preferred"""
        data = {
            "license": "MIT",
            "cardData": {"license": "Apache-2.0"}
        }
        assert extract_license(data) == "MIT"
    
    @patch('src.license_compat.requests.get')
    def test_license_compat_success(self, mock_get):
        """Test successful license compatibility check"""
        mock_response = Mock()
        mock_response.json.return_value = {
            "license": "MIT",
            "cardData": None
        }
        mock_response.raise_for_status = Mock()
        mock_get.return_value = mock_response
        
        result = license_compat("bert-base-uncased")
        
        assert result["model_id"] == "bert-base-uncased"
        assert result["license"] == "MIT"
        assert result["lgplv21_compat_score"] == 1
        assert "error" not in result
    
    @patch('src.license_compat.requests.get')
    def test_license_compat_incompatible(self, mock_get):
        """Test incompatible license detection"""
        mock_response = Mock()
        mock_response.json.return_value = {
            "license": "proprietary"
        }
        mock_response.raise_for_status = Mock()
        mock_get.return_value = mock_response
        
        result = license_compat("closed-model")
        
        assert result["model_id"] == "closed-model"
        assert result["license"] == "proprietary"
        assert result["lgplv21_compat_score"] == 0
    
    @patch('src.license_compat.requests.get')
    def test_license_compat_network_error(self, mock_get):
        """Test handling network errors"""
        mock_get.side_effect = requests.exceptions.ConnectionError("Network error")
        
        result = license_compat("test-model")
        
        assert result["model_id"] == "test-model"
        assert result["license"] == "error"
        assert result["lgplv21_compat_score"] == 0
        assert "error" in result
        assert "Network error" in result["error"]
    
    @patch('src.license_compat.requests.get')
    def test_license_compat_http_error(self, mock_get):
        """Test handling HTTP errors"""
        mock_response = Mock()
        mock_response.raise_for_status.side_effect = requests.exceptions.HTTPError("404 Not Found")
        mock_get.return_value = mock_response
        
        result = license_compat("nonexistent-model")
        
        assert result["model_id"] == "nonexistent-model"
        assert result["license"] == "error"
        assert result["lgplv21_compat_score"] == 0
        assert "error" in result
        assert "404" in result["error"]
    
    @patch('src.license_compat.requests.get')
    def test_license_compat_timeout(self, mock_get):
        """Test handling timeout errors"""
        mock_get.side_effect = requests.exceptions.Timeout("Request timed out")
        
        result = license_compat("slow-model")
        
        assert result["model_id"] == "slow-model"
        assert result["license"] == "error"
        assert result["lgplv21_compat_score"] == 0
        assert "error" in result
    
    @patch('src.license_compat.requests.get')
    def test_license_compat_invalid_json(self, mock_get):
        """Test handling invalid JSON response"""
        mock_response = Mock()
        mock_response.json.side_effect = ValueError("Invalid JSON")
        mock_response.raise_for_status = Mock()
        mock_get.return_value = mock_response
        
        result = license_compat("bad-response-model")
        
        assert result["model_id"] == "bad-response-model"
        assert result["license"] == "error"
        assert result["lgplv21_compat_score"] == 0
        assert "error" in result
    
    @patch('src.license_compat.requests.get')
    def test_license_compat_complex_carddata(self, mock_get):
        """Test extracting license from complex cardData structure"""
        mock_response = Mock()
        mock_response.json.return_value = {
            "license": "unknown",
            "cardData": {
                "license": ["Apache-2.0", "MIT"],
                "other_field": "value"
            }
        }
        mock_response.raise_for_status = Mock()
        mock_get.return_value = mock_response
        
        result = license_compat("multi-license-model")
        
        assert result["model_id"] == "multi-license-model"
        assert result["license"] == "Apache-2.0"  # First from list
        assert result["lgplv21_compat_score"] == 1  # Apache is compatible
