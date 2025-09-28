from license_compat import license_compat

def get_license_info(model_id: str) -> dict:
    """
    Fetch license info and LGPLv2.1 compatibility for a Hugging Face model.
    Returns a dict with model_id, license, lgplv21_compat_score, and error (if any).
    """
    return license_compat(model_id)
