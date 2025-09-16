import requests
from typing import Dict, Any, Optional

HF_API_BASE = "https://huggingface.co/api/models"

def get_huggingface_model_metadata(model_id: str) -> Optional[Dict[str, Any]]:
    """
    Fetch basic metadata for a Hugging Face model using the Hugging Face API.

    Args:
        model_id (str): The Hugging Face model ID, e.g. "google/gemma-3-270m"

    Returns:
        Optional[Dict[str, Any]]: Dictionary containing selected metadata (downloads, likes, lastModified)
                                  or None if the request fails.
    """
    url = f"{HF_API_BASE}/{model_id}"
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()  # Raise error for bad responses (4xx/5xx)
        data = response.json()

        # Extract useful fields (can expand later)
        parsed_metadata = {
            "model_id": model_id,
            "downloads": data.get("downloads", 0),
            "likes": data.get("likes", 0),
            "lastModified": data.get("lastModified", "unknown")
        }
        return parsed_metadata

    except requests.exceptions.RequestException as e:
        print(f"Error fetching metadata for {model_id}: {e}")
        return None


if __name__ == "__main__":
    sample_models = ["google/gemma-3-270m", "bert-base-uncased"]
    for model in sample_models:
        metadata = get_huggingface_model_metadata(model)
        print(metadata)
