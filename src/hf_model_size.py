import requests
import subprocess
import shutil
import tempfile
import os

HF_TOKEN = os.getenv("HF_TOKEN")

# Example hardware constraints (in bytes)
# You can tune these based on actual target devices
HARDWARE_CONSTRAINTS = {
    "aws_t2_micro": 1 * 1024**3,    # 1 GB
    "aws_t2_large": 8 * 1024**3,    # 8 GB
    "aws_p3_2xlarge": 16 * 1024**3, # 16 GB
    "local_gpu": 12 * 1024**3       # 12 GB VRAM
}

def get_model_file_sizes(model_id: str) -> dict:
    """
    Fetches the list of files for a Hugging Face model and sums their sizes in bytes.
    Returns a dict with model_id, total_size_bytes, and a list of file details.
    """
    import subprocess
    url = f"https://huggingface.co/api/models/{model_id}"
    try:
        resp = requests.get(url, timeout=10)
        resp.raise_for_status()
        data = resp.json()
        siblings = data.get("siblings", [])
        total_size = 0
        file_details = []
        missing_files = []
        for file in siblings:
            fname = file.get("rfilename")
            fsize = file.get("size", 0)
            # Fallback: if size is missing or 0, try HEAD request
            if (not fsize or fsize == 0) and fname:
                file_url = f"https://huggingface.co/{model_id}/resolve/main/{fname}"
                try:
                    head = requests.head(file_url, timeout=10, allow_redirects=True)
                    cl = head.headers.get("Content-Length")
                    if cl:
                        fsize = int(cl)
                except Exception:
                    pass
            if not fsize and fname:
                missing_files.append(fname)
            if fsize:
                total_size += fsize
            file_details.append({"filename": fname, "size": fsize})
        # If any files are still missing size, try git-lfs fallback
        if missing_files:
            import os, tempfile, shutil
            repo_dir = None
            try:
                repo_dir = tempfile.mkdtemp(prefix="hfmodel_")
                subprocess.run(["git", "clone", "--no-checkout", f"https://huggingface.co/{model_id}", repo_dir], check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
                lfs_out = subprocess.run(["git", "lfs", "ls-files", "-s"], cwd=repo_dir, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
                lfs_lines = lfs_out.stdout.decode().splitlines()
                lfs_sizes = {}
                for line in lfs_lines:
                    parts = line.strip().split()
                    if len(parts) >= 3:
                        # Format: <hash> - <filename> (<size> <unit>)
                        fname_lfs = parts[2]
                        size_str = parts[-2] if len(parts) > 3 else None
                        unit = parts[-1].strip('()') if len(parts) > 3 else None
                        if size_str and unit:
                            # Convert to bytes
                            size_map = {"B": 1, "KB": 1024, "MB": 1024**2, "GB": 1024**3}
                            try:
                                size_val = float(size_str)
                                size_bytes = int(size_val * size_map.get(unit, 1))
                                lfs_sizes[fname_lfs] = size_bytes
                            except Exception:
                                pass
                # Update missing file sizes
                for f in file_details:
                    if f["filename"] in lfs_sizes and (not f["size"] or f["size"] == 0):
                        f["size"] = lfs_sizes[f["filename"]]
                        total_size += lfs_sizes[f["filename"]]
            except Exception:
                pass
            finally:
                if repo_dir:
                    shutil.rmtree(repo_dir)
        return {
            "model_id": model_id,
            "total_size_bytes": total_size,
            "files": file_details
        }
    except Exception as e:
        return {"model_id": model_id, "error": str(e)}

def calculate_size_metric(model_info: dict, constraints: dict = HARDWARE_CONSTRAINTS) -> dict:
    """
    Given model_info (from get_model_file_sizes), compute a normalized size metric.
    Metric = min(normalized ratios across devices), where 1 = fits perfectly, 0 = does not fit anywhere.
    """
    if "error" in model_info:
        return {**model_info, "size_metric": 0.0}

    total_size = model_info.get("total_size_bytes", 0)
    if total_size == 0:
        return {**model_info, "size_metric": 0.0}

    # Normalize by checking fit against each device
    normalized_scores = []
    for device, capacity in constraints.items():
        ratio = total_size / capacity
        # If it fits, score decreases the closer you are to the limit
        if ratio <= 1:
            score = 1 - ratio  # 1 if tiny, ~0 if barely fits
        else:
            score = 0  # Doesn't fit at all
        normalized_scores.append(score)

    # Final score = max score across devices (best-case deployability)
    size_metric = max(normalized_scores) if normalized_scores else 0.0

    return {**model_info, "size_metric": round(size_metric, 3)}

if __name__ == "__main__":
    import sys, json
    if len(sys.argv) < 2:
        print("Usage: python hf_model_size.py <model_id>")
        sys.exit(1)
    result = get_model_file_sizes(sys.argv[1])
    print(json.dumps(result, indent=2))
