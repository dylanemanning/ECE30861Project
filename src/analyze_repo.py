# --- IMPORTS ---
import os
import subprocess
from git import Repo  # pip install GitPython
import math

# --- METRIC HELPERS ---
def normalize_score(value, min_val, max_val):
    """Normalize a value to [0,1] given min and max."""
    if max_val == min_val:
        return 1.0 if value >= max_val else 0.0
    return max(0.0, min(1.0, (value - min_val) / (max_val - min_val)))

def is_lgpl_compatible(license_str):
    """Return 1 if license is compatible with LGPLv2.1, else 0."""
    compatible_licenses = [
        "LGPL-2.1",
        "LGPL-2.1-only",
        "LGPL-2.1-or-later",
        "GPL-2.0",
        "GPL-2.0-or-later",
        "MIT",
        "MIT/X11",
        "BSD",
        "BSD-2-Clause",
        "BSD-3-Clause"
    ]
    return 1 if license_str in compatible_licenses else 0

def compute_code_quality(repo_path: str) -> float:
    try:
        python_files = sum(
            1 for root, dirs, files in os.walk(repo_path) 
            for file in files if file.endswith(".py")
        )
        print(f"Found {python_files} Python files.")

        flake8_path = shutil.which("flake8")
        if flake8_path:
            cmd = [flake8_path, "."]
        else:
            cmd = [sys.executable, "-m", "flake8", "."]

        result = subprocess.run(

            capture_output=True,
            text=True,
            check=False
        )

        issues = len(result.stdout.strip().splitlines())
        if result.returncode == 1 and issues == 0:
            # flake8 errored before reporting results
            print("flake8 reported an error without lint output:", result.stderr.strip())
            return 0.0
        print(f"Found {issues} flake8 issues.")

        avg_issues = issues / max(python_files, 1)
        score = 1.0 - avg_issues / 150
        return max(0.0, min(1.0, score))
    except Exception as e:
        print(f"Error running flake8: {e}")
        return 0.5


def compute_size_metric(repo_path: str, max_capacity_mb: float = 16*1024) -> float:
    """
    Compute normalized size score for ML model weight files.
    Args:
        repo_path: Local path to repository.
        max_capacity_mb: Maximum hardware capacity in MB for normalization.
    Returns:
        Normalized score in [0,1], higher means repo fits better within hardware limits.
    """
    weight_exts = (".bin", ".h5", ".ckpt", ".safetensors")
    total_bytes = 0
    for root, _, files in os.walk(repo_path):
        for f in files:
            if f.endswith(weight_exts):
                try:
                    total_bytes += os.path.getsize(os.path.join(root, f))
                except Exception:
                    pass
    size_mb = total_bytes / (1024 * 1024)
    normalized = min(1.0, size_mb / max_capacity_mb) # Assumes 16GB max capacity
    return normalized

def compute_local_metrics(repo_path, license_str=None):
    """Compute normalized metrics for a local repo."""

    # Bus Factor: based on commit history
    try:
        # Count commits per contributor
        result = subprocess.run(
            ["git", "shortlog", "-sne", "HEAD"],
            cwd=repo_path,
            capture_output=True,
            text=True,
            check=True
        )
        contrib_lines = result.stdout.strip().splitlines()
        contrib_counts = []
        for line in contrib_lines:
            parts = line.strip().split("\t")
            if len(parts) >= 2:
                count = int(parts[0].strip().split()[0])
                contrib_counts.append(count)

        total_commits = sum(contrib_counts)
        if total_commits > 0 and len(contrib_counts) > 1:
            # Normalize using entropy (better than just max share)
            p = [c / total_commits for c in contrib_counts]
            entropy = -sum(pi * math.log2(pi) for pi in p)
            max_entropy = math.log2(len(p))
            bus_factor = entropy / max_entropy if max_entropy > 0 else 0
        else:
            bus_factor = 0.0
    except Exception:
        bus_factor = 0.0  # fallback if git fails

    # Size Metric (model weights)
    size_score = compute_size_metric(repo_path)

    # Code Quality: placeholder value
    code_quality = compute_code_quality(repo_path)

    # License Compatibility: 1 if compatible, 0 otherwise
    license_score = is_lgpl_compatible(license_str) if license_str else 0

    return {
        "bus_factor": bus_factor,
        #"ramp_up_time": ramp_up_time,
        "code_quality": code_quality,
        "license": license_score,
        "size": size_score
    }

# --- REPO ANALYSIS ---
def clone_repo(repo_url: str, local_dir: str) -> None:
    """
    Clone a GitHub repository into the specified local directory.
    Args:
        repo_url (str): URL of the GitHub repository.
        local_dir (str): Local directory where the repo will be cloned.
    """
    # Clone the remote repository from repo_url into the local_dir using GitPython
    Repo.clone_from(repo_url, local_dir)


def extract_license(local_dir: str) -> str:
    """
    Look for a LICENSE file in the repo and return its contents (or type).
    Args:
        local_dir (str): Path to the cloned repo.
    Returns:
        str: Detected license type or 'Unknown'.
    """
    # List of common license file names to look for in the repo
    license_files = ["LICENSE", "LICENSE.txt", "LICENSE.rst", "COPYING", "COPYING.txt"]
    # Walk through all directories and files in the cloned repo
    for root, dirs, files in os.walk(local_dir):
        for lf in license_files:
            if lf in files:
                path = os.path.join(root, lf)
                try:
                    # Try to open and read the license file
                    with open(path, "r", encoding="utf-8", errors="ignore") as f:
                        content = f.read().lower()
                        print(content)
                        # Check for common license keywords in the file content
                        if "mit" in content:
                            return "MIT"
                        elif "apache" in content:
                            return "Apache"
                        elif "gpl" in content or "lgpl" in content:
                            return "GPL/LGPL"
                        elif "bsd" in content:
                            return "BSD"
                        else:
                            # If no known license is found, return Unknown
                            return "Unknown"
                except Exception:
                    # If the file can't be read, treat as Unknown
                    return "Unknown"
    # If no license file is found, return Unknown
    return "Unknown"


def extract_repo_stats(local_dir: str) -> dict:
    """
    Walk the repo directory and gather basic statistics.
    Args:
        local_dir (str): Path to the cloned repo.
    Returns:
        dict: Contains total_files and python_files counts.
    """
    # Initialize counters for total files and Python files
    total_files = 0
    python_files = 0

    # Walk through every directory and file in the repo
    for root, dirs, files in os.walk(local_dir):
        for file in files:
            total_files += 1  # Increment for every file found
            if file.endswith(".py"):
                python_files += 1  # Increment if the file is a Python file
    # Return the statistics as a dictionary
    return {
        "total_files": total_files, 
        "python_files": python_files
    }

def analyze_repo(repo_url: str) -> dict:
    """
    Clone a repo, extract license and stats, then clean up.
    Returns a dictionary with bus_factor, code_quality, license, lgpl_compatible, and size.
    Handles invalid URLs and incomplete repos gracefully.
    """
    import tempfile, shutil, os, requests

    # Default fallback metrics
    default_metrics = {
        "bus_factor": 0.0,
        "size": 0.0,
        "code_quality": 0.0,
        "license": "Unknown",
        "lgpl_compatible": False
    }

    temp_dir = tempfile.mkdtemp()

    try:
        # Determine if repo_url is local path
        if os.path.exists(repo_url):
            local_dir = repo_url
        else:
            # Try to clone GitHub repo
            local_dir = temp_dir
            try:
                clone_repo(repo_url, local_dir)
            except Exception:
                return {"repo": [repo_url], **default_metrics, "error": "Invalid or not supported URL"}

        # Attempt to detect license
        license_type = "Unknown"
        compatible_licenses = [
            "LGPL-2.1", "LGPL-2.1-only", "LGPL-2.1-or-later",
            "GPL-2.0", "GPL-2.0-or-later",
            "MIT", "MIT/X11",
            "BSD", "BSD-2-Clause", "BSD-3-Clause"
        ]
        owner = repo_name = None

        if repo_url.startswith("https://github.com/"):
            try:
                parts = repo_url.rstrip("/").split("/")
                owner, repo_name = parts[-2], parts[-1]
                api_url = f"https://api.github.com/repos/{owner}/{repo_name}"
                resp = requests.get(api_url, timeout=10)
                if resp.status_code == 200:
                    data = resp.json()
                    license_type = data.get("license", {}).get("spdx_id", "Unknown")
            except Exception:
                pass

        # Fallback: extract license from local files
        if license_type == "Unknown":
            try:
                license_type = extract_license(local_dir)
            except Exception:
                license_type = "Unknown"

        # Compute local metrics (bus_factor, code_quality, size)
        try:
            metrics = compute_local_metrics(local_dir, license_type)
        except Exception:
            metrics = default_metrics

        # Build result dictionary
        repo_info = {
            "repo": [repo_url] if not owner else [owner, repo_name],
            "license": license_type,
            "lgpl_compatible": license_type in compatible_licenses
        }
        repo_info.update(metrics)
        return repo_info

    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)

# --- MAIN FOR MANUAL TESTING ---
if __name__ == "__main__":
    print("Testing compute_local_metrics on two repos:")

    # Small repo: just the current working directory
    small_repo = os.path.dirname(os.path.abspath(__file__))
    print("Small repo:", compute_local_metrics(small_repo, license_str="MIT"))

    # Large repo: clone Linux repo into a temp dir (full history for contributors)
    '''temp_dir = tempfile.mkdtemp()
    try:
        print("Cloning Linux repo (this may take a while)...")
        # Full clone (slow but needed for accurate bus factor)
        Repo.clone_from("https://github.com/torvalds/linux",
                        temp_dir,
                        depth=100, # shallow clone: last 100 commits
                        branch="master"
                        )
        print("Large repo:", compute_local_metrics(temp_dir, license_str="GPL-2.0"))
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)  # cleanup'''

    # Analyze the large repo using analyze_repo
    test_repo = "https://huggingface.co/bigscience/bloom"
    info = analyze_repo(test_repo)
    print("analyze_repo result:", info)
