import os
import tempfile
import shutil
# Import the Repo class from GitPython to interact with git repositories
from git import Repo  # pip install GitPython

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
    license_files = ["LICENSE", "LICENSE.txt", "COPYING", "COPYING.txt"]
    # Walk through all directories and files in the cloned repo
    for root, dirs, files in os.walk(local_dir):
        for lf in license_files:
            if lf in files:
                path = os.path.join(root, lf)
                try:
                    # Try to open and read the license file
                    with open(path, "r", encoding="utf-8", errors="ignore") as f:
                        content = f.read()
                        # Check for common license keywords in the file content
                        if "MIT" in content:
                            return "MIT"
                        elif "Apache" in content:
                            return "Apache"
                        elif "GPL" in content or "LGPL" in content:
                            return "GPL/LGPL"
                        elif "BSD" in content:
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
        "GPL-3.0",
        "MIT",
        "BSD-2-Clause",
        "BSD-3-Clause",
        "Apache-2.0"
    ]
    try:
        # Clone the repo into the temp directory
        clone_repo(repo_url, temp_dir)
        # Extract the license type from the repo
        license_type = extract_license(temp_dir)
        # Gather statistics about the repo's files
        stats = extract_repo_stats(temp_dir)

        # Build and return the result dictionary
        return {
            # Get the last two parts of the repo URL for identification
            "repo": repo_url.split("/")[-2:] if repo_url else "Unknown",
            "license": license_type,
            **stats,
            # Check if the detected license is in the compatible list
            "lgpl_compatible": license_type in compatible_licenses
        }
    finally:
        # Clean up the temporary directory after analysis
        shutil.rmtree(temp_dir, ignore_errors=True)


if __name__ == "__main__":
    # Example usage: analyze a public GitHub repository
    test_repo = "https://github.com/copperspice/copperspice"
    # Call analyze_repo to get metadata and stats for the repo
    info = analyze_repo(test_repo)
    # Print the results to the console
    print(info)
