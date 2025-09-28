import requests


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


if __name__ == "__main__":
    import sys, json
    if len(sys.argv) < 2:
        print("Usage: python hf_model_size.py <model_id>")
        sys.exit(1)
    result = get_model_file_sizes(sys.argv[1])
    print(json.dumps(result, indent=2))
