from pathlib import Path
from types import SimpleNamespace

import pytest

from src import analyze_repo


def test_normalize_score_bounds():
    assert analyze_repo.normalize_score(5, 0, 10) == 0.5
    assert analyze_repo.normalize_score(-5, 0, 10) == 0.0
    assert analyze_repo.normalize_score(10, 10, 10) == 1.0


def test_is_lgpl_compatible_variants():
    assert analyze_repo.is_lgpl_compatible("MIT") == 1
    assert analyze_repo.is_lgpl_compatible("Proprietary") == 0


def test_compute_size_metric(tmp_path):
    weights = tmp_path / "model.safetensors"
    weights.write_bytes(b"0" * 1024 * 1024)  # 1 MiB
    score = analyze_repo.compute_size_metric(str(tmp_path), max_capacity_mb=2)
    assert pytest.approx(0.5, rel=1e-3) == score


def test_compute_code_quality_success(monkeypatch, tmp_path):
    (tmp_path / "file.py").write_text("print('hi')\n")

    monkeypatch.setattr(analyze_repo.shutil, "which", lambda _: None)

    def fake_run(cmd, cwd, capture_output, text, check):
        assert cwd == str(tmp_path)
        return SimpleNamespace(stdout="file.py:1:1: F401 Foo\n", returncode=0)

    monkeypatch.setattr(analyze_repo.subprocess, "run", fake_run)
    score = analyze_repo.compute_code_quality(str(tmp_path))
    assert 0.0 < score <= 1.0


def test_compute_code_quality_flake8_error(monkeypatch, tmp_path):
    (tmp_path / "file.py").write_text("print('hi')\n")

    monkeypatch.setattr(analyze_repo.shutil, "which", lambda _: "flake8")

    def fake_run(cmd, cwd, capture_output, text, check):
        return SimpleNamespace(stdout="", stderr="boom", returncode=2)

    monkeypatch.setattr(analyze_repo.subprocess, "run", fake_run)
    score = analyze_repo.compute_code_quality(str(tmp_path))
    assert score == 0.0


def test_compute_code_quality_exception(monkeypatch, tmp_path):
    (tmp_path / "file.py").write_text("print('hi')\n")

    def blow_up(*args, **kwargs):
        raise OSError("no flake8")

    monkeypatch.setattr(analyze_repo.subprocess, "run", blow_up)
    score = analyze_repo.compute_code_quality(str(tmp_path))
    assert score == 0.5


def test_extract_license_detects_mit(tmp_path):
    lic = tmp_path / "LICENSE"
    lic.write_text("This project is released under the MIT license\n")
    found = analyze_repo.extract_license(str(tmp_path))
    assert found == "MIT"


def test_extract_repo_stats(tmp_path):
    pkg = tmp_path / "pkg"
    pkg.mkdir()
    (pkg / "a.py").write_text("print('hello')\n")
    (tmp_path / "data.bin").write_bytes(b"123")
    stats = analyze_repo.extract_repo_stats(str(tmp_path))
    assert stats["total_files"] == 2
    assert stats["python_files"] == 1


def test_compute_local_metrics(monkeypatch, tmp_path):
    (tmp_path / "weights.bin").write_bytes(b"0" * 4096)

    def fake_run(cmd, cwd, capture_output, text, check):
        assert cmd[:3] == ["git", "shortlog", "-sne"]
        return SimpleNamespace(stdout="   5\tAlice\n   3\tBob\n", returncode=0)

    monkeypatch.setattr(analyze_repo.subprocess, "run", fake_run)
    monkeypatch.setattr(analyze_repo, "compute_code_quality", lambda _path: 0.8)

    metrics = analyze_repo.compute_local_metrics(str(tmp_path), "MIT")
    assert pytest.approx(0.8, rel=1e-6) == metrics["code_quality"]
    assert metrics["license"] == 1
    assert metrics["size"] > 0
    assert metrics["bus_factor"] > 0


def test_analyze_repo_with_local_path(monkeypatch, tmp_path):
    (tmp_path / "LICENSE").write_text("MIT License text\n")

    monkeypatch.setattr(
        analyze_repo,
        "compute_local_metrics",
        lambda _path, _license: {
            "bus_factor": 0.2,
            "code_quality": 0.9,
            "license": 1,
            "size": 0.3,
        },
    )

    result = analyze_repo.analyze_repo(str(tmp_path))
    assert result["license"] == 1
    assert result["lgpl_compatible"] is True
    assert result["bus_factor"] == 0.2
    assert str(tmp_path) in result["repo"]


def test_analyze_repo_invalid_url(monkeypatch):
    def raise_clone(*_args, **_kwargs):
        raise RuntimeError("bad")

    monkeypatch.setattr(analyze_repo, "clone_repo", raise_clone)

    result = analyze_repo.analyze_repo("https://github.com/example/invalid")
    assert result["error"] == "Invalid or not supported URL"
    assert result["license"] == "Unknown"


def test_compute_size_metric_handles_getsize_error(monkeypatch, tmp_path):
    (tmp_path / "weights.bin").write_bytes(b"123")

    def fail_getsize(_path):
        raise OSError("boom")

    monkeypatch.setattr(analyze_repo.os.path, "getsize", fail_getsize)
    score = analyze_repo.compute_size_metric(str(tmp_path))
    assert score == 0.0


def test_compute_local_metrics_git_failure(monkeypatch, tmp_path):
    monkeypatch.setattr(
        analyze_repo.subprocess,
        "run",
        lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("git")),
    )
    monkeypatch.setattr(analyze_repo, "compute_code_quality", lambda _path: 0.3)

    metrics = analyze_repo.compute_local_metrics(str(tmp_path), "MIT")
    assert metrics["bus_factor"] == 0.0
    assert metrics["code_quality"] == 0.3


def test_analyze_repo_github_api(monkeypatch, tmp_path):
    temp_dir = tmp_path / "clone"
    temp_dir.mkdir()

    monkeypatch.setattr("tempfile.mkdtemp", lambda: str(temp_dir))

    def fake_clone(_url, local_dir):
        Path(local_dir).mkdir(exist_ok=True)

    def fake_get(url, timeout):
        assert url == "https://api.github.com/repos/owner/repo"
        return SimpleNamespace(
            status_code=200,
            json=lambda: {"license": {"spdx_id": "BSD-3-Clause"}},
        )

    def raise_extract(_path):
        raise RuntimeError("license read error")

    def raise_metrics(_path, _license):
        raise ValueError("metric failure")

    monkeypatch.setattr(analyze_repo, "clone_repo", fake_clone)
    monkeypatch.setattr("requests.get", fake_get)
    monkeypatch.setattr(analyze_repo, "extract_license", raise_extract)
    monkeypatch.setattr(analyze_repo, "compute_local_metrics", raise_metrics)

    result = analyze_repo.analyze_repo("https://github.com/owner/repo")
    assert result["repo"] == ["owner", "repo"]
    assert result["license"] == "Unknown"
    assert result["lgpl_compatible"] is False
