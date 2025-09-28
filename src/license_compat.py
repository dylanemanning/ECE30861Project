import requests

def license_compat(model_id: str) -> dict:
    """
    Fetch model info from Hugging Face and check LGPLv2.1 compatibility.
    Returns a dict with model_id, license, lgplv21_compat_score, and error (if any).
    """
    url = f"https://huggingface.co/api/models/{model_id}"
    try:
        resp = requests.get(url, timeout=10)
        resp.raise_for_status()
        data = resp.json()
        license_str = extract_license(data)
        compat = is_lgpl_compatible(license_str)
        return {
            "model_id": model_id,
            "license": license_str,
            "lgplv21_compat_score": compat
        }
    except Exception as e:
        return {
            "model_id": model_id,
            "license": "error",
            "lgplv21_compat_score": 0,
            "error": str(e)
        }
"""
License compatibility checker for Hugging Face to verify compatibility with LGPLv2.1 and related licenses.
"""

def extract_license(data: dict) -> str:
    """
    Try to extract license from Hugging Face model API response.
    Checks top-level 'license', then cardData['license'] if present.
    """
    license_str = data.get("license")
    if license_str and license_str != "unknown":
        return license_str
    card_data = data.get("cardData")
    if card_data:
        # cardData['license'] can be a string or a list
        lic = card_data.get("license")
        if isinstance(lic, str):
            return lic
        elif isinstance(lic, list) and lic:
            return lic[0]
    return "unknown"

COMPATIBLE_LICENSES = {
    "mit", "bsd-2-clause", "bsd-3-clause",
    "apache-2.0", "isc", "zlib", "mpl-2.0",
    "epl-2.0", "cddl-1.0", "lgpl-2.1", "lgpl-2.1-or-later", "gpl-2.0"
}

def is_lgpl_compatible(license_str: str) -> int:
    """
    Return 1 if license is compatible with LGPLv2.1, else 0.
    Uses COMPATIBLE_LICENSES set.
    """
    return 1 if license_str.lower() in COMPATIBLE_LICENSES else 0
