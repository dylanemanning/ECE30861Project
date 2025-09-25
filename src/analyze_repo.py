# --- IMPORTS ---
import os
import subprocess
import tempfile
import shutil
import sys
import requests  # For GitHub API license lookup
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
    """
    Compute a code quality score [0,1] for a repo using flake8.
    Higher = better.
    """
    try:
        # Count Python files first
        python_files = 0
        for root, dirs, files in os.walk(repo_path):
            for file in files:
                if file.endswith(".py"):
                    python_files += 1

        print(f"Found {python_files} Python files.")

        flake8_path = shutil.which("flake8")
        if flake8_path:
            cmd = [flake8_path, "."]
        else:
            cmd = [sys.executable, "-m", "flake8", "."]

        result = subprocess.run(
            cmd,
            cwd=repo_path,
            capture_output=True,
            text=True,
            check=False  # don't raise exception on lint errors
        )
        if result.returncode not in (0, 1):
            # Non-linting failure, treat as poor quality
            print("flake8 execution failed:", result.stderr.strip())
            return 0.0

        # Count issues: each line = one problem
        issues = len(result.stdout.strip().splitlines())
        if result.returncode == 1 and issues == 0:
            # flake8 errored before reporting results
            print("flake8 reported an error without lint output:", result.stderr.strip())
            return 0.0
        print(f"Found {issues} flake8 issues.")

        # Normalize issues per Python file
        avg_issues = issues / max(python_files, 1)  # avoid divide by zero
        # Log-based soft normalization
        score = 1.0 - avg_issues / 150 # average of 150 issues/file = 0.0
        score = max(0.0, min(1.0, score))
        return score
    except FileNotFoundError:
        # flake8 not installed: neutral score
        return 0.5


def compute_local_metrics(repo_path, license_str=None):
    """Compute normalized metrics for a local repo."""

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

    # Ramp Up Time
    total_files = 0
    loc = 0
    for root, dirs, files in os.walk(repo_path):
        total_files += len(files)
        for file in files:
            if file.endswith(('.py', '.c', '.cpp', '.h', '.js', '.java', '.ts', '.go', '.rb', '.rs', '.cs')):
                try:
                    with open(os.path.join(root, file), 'r', encoding='utf-8', errors='ignore') as f:
                        loc += sum(1 for _ in f)
                except Exception:
                    pass
    ramp_up_time = min(1.0, math.log1p(total_files) / math.log1p(1000000))

    # Code Quality: placeholder value
    code_quality = compute_code_quality(repo_path)

    # License Compatibility: 1 if compatible, 0 otherwise
    license_score = is_lgpl_compatible(license_str) if license_str else 0

    return {
        "bus_factor": bus_factor,
        "ramp_up_time": ramp_up_time,
        "code_quality": code_quality,
        "license": license_score
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
    Args:
        repo_url (str): GitHub repository URL.
    Returns:
        dict: Metadata including repo name, license, and stats.
    """
    # Create a temporary directory to clone the repo into
    temp_dir = tempfile.mkdtemp()
    # List of licenses considered compatible with LGPL
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
    # Parse owner/repo from the URL for GitHub API
    try:
        parts = repo_url.rstrip('/').split('/')
        owner, repo_name = parts[-2], parts[-1]
        api_url = f"https://api.github.com/repos/{owner}/{repo_name}"
        license_type = "Unknown"
        # Try to get license info from GitHub API
        try:
            resp = requests.get(api_url, timeout=10)
            if resp.status_code == 200:
                data = resp.json()
                if data.get("license") and data["license"].get("spdx_id"):
                    license_type = data["license"]["spdx_id"]
        except Exception:
            pass

        # Clone the repo into the temp directory
        clone_repo(repo_url, temp_dir)
        # If license_type is still unknown, try to extract from files
        if license_type == "Unknown":
            license_type = extract_license(temp_dir)
        # Gather statistics about the repo's files
        stats = extract_repo_stats(temp_dir)

        # Build and return the result dictionary
        return {
            # Get the last two parts of the repo URL for identification
            "repo": [owner, repo_name],
            "license": license_type,
            **stats,
            # Check if the detected license is in the compatible list
            "lgpl_compatible": license_type in compatible_licenses
        }
    finally:
        # Clean up the temporary directory after analysis
        shutil.rmtree(temp_dir, ignore_errors=True)



# --- TESTING ---
if __name__ == "__main__":
    print("Testing compute_local_metrics on two repos:")

    # Small repo: just the current working directory
    small_repo = os.path.dirname(os.path.abspath(__file__))
    print("Small repo:", compute_local_metrics(small_repo, license_str="MIT"))

    # Large repo: clone Linux repo into a temp dir (full history for contributors)
    temp_dir = tempfile.mkdtemp()
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
        shutil.rmtree(temp_dir, ignore_errors=True)  # cleanup

    # Analyze the large repo using analyze_repo
    test_repo = "https://github.com/pallets/flask"
    info = analyze_repo(test_repo)
    print("analyze_repo result:", info)
