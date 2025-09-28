import subprocess
import sys
import pytest
from pathlib import Path

# Assume your CLI entrypoint is "analyze_repo.py"
CLI = [sys.executable, "analyze_repo.py"]

def run_cli(args):
    """Helper to run CLI and return (exit_code, stdout, stderr)."""
    proc = subprocess.run(
        CLI + args,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True
    )
    return proc.returncode, proc.stdout, proc.stderr


from analyze_repo import analyze_repo

def test_invalid_url_returns_error():
    result = analyze_repo("not_a_repo_url")
    assert isinstance(result, dict)
    assert "error" in result
    assert "invalid" in result["error"].lower()

def test_fake_github_repo_returns_error():
    result = analyze_repo("https://github.com/fakeowner/fakerepo")
    assert isinstance(result, dict)
    assert "error" in result

def test_non_github_url_returns_error():
    result = analyze_repo("https://google.com")
    assert isinstance(result, dict)
    assert "error" in result


@pytest.mark.parametrize("repo_path", [
    "tests/data/empty_repo",         # contains no files
    "tests/data/repo_no_license",    # has files but no LICENSE
])
def test_incomplete_repo_defaults(repo_path):
    from analyze_repo import analyze_repo

    results = analyze_repo(repo_path)
    assert isinstance(results, dict)

    # license should default to unknown if missing
    if "no_license" in repo_path:
        assert results.get("license", "").lower() in ["unknown", ""]

    # should still provide bus_factor, code_quality, and size
    assert "bus_factor" in results
    assert "code_quality" in results
    assert "size" in results