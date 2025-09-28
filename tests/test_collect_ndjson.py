import json
from io import StringIO

from src import collect_ndjson


def test_read_lines_file(tmp_path):
    path = tmp_path / "items.txt"
    path.write_text("repo-one\n# ignore\n\nsecond\n")
    assert collect_ndjson.read_lines_file(str(path)) == ["repo-one", "second"]


def test_write_ndjson_line():
    buffer = StringIO()
    collect_ndjson.write_ndjson_line(buffer, {"type": "repo", "data": {}})
    assert buffer.getvalue().strip() == '{"type": "repo", "data": {}}'


def test_collect_and_write(tmp_path, monkeypatch):
    monkeypatch.setattr(collect_ndjson, "_analyze_repo_fn", lambda repo: {"repo": repo})
    monkeypatch.setattr(collect_ndjson, "_hf_meta_fn", lambda model: {"model": model})
    monkeypatch.setattr(collect_ndjson.time, "strftime", lambda *_args, **_kwargs: "2024-01-01T00:00:00Z")

    output_path = tmp_path / "out.ndjson"
    rc = collect_ndjson.collect_and_write(["repo"], ["model"], str(output_path))
    assert rc == 0

    lines = [json.loads(line) for line in output_path.read_text().splitlines()]
    assert len(lines) == 2
    assert {entry["type"] for entry in lines} == {"repo", "hf_model"}


def test_collect_and_write_error_paths(tmp_path, monkeypatch):
    def fail_repo(_repo):
        raise RuntimeError("boom")

    def fail_model(_model):
        raise ValueError("kaboom")

    monkeypatch.setattr(collect_ndjson, "_analyze_repo_fn", fail_repo)
    monkeypatch.setattr(collect_ndjson, "_hf_meta_fn", fail_model)
    monkeypatch.setattr(collect_ndjson.time, "strftime", lambda *_args, **_kwargs: "2024-01-01T00:00:00Z")

    output_path = tmp_path / "err.ndjson"
    collect_ndjson.collect_and_write(["repo"], ["model"], str(output_path))
    payloads = [json.loads(line) for line in output_path.read_text().splitlines()]
    assert payloads[0]["data"] == {"error": "boom"}
    assert payloads[1]["data"] == {"error": "kaboom"}


def test_main_no_inputs(capsys):
    rc = collect_ndjson.main([])
    assert rc == 2
    captured = capsys.readouterr()
    assert "No repos or models" in captured.out


def test_main_with_inputs(monkeypatch):
    captured = {}

    def fake_collect(repos, models, output, append):
        captured["repos"] = repos
        captured["models"] = models
        captured["output"] = output
        captured["append"] = append
        return 0

    monkeypatch.setattr(collect_ndjson, "collect_and_write", fake_collect)
    rc = collect_ndjson.main(["--repo", "https://example.com", "--model", "abc", "--output", "file", "--append"])
    assert rc == 0
    assert captured == {
        "repos": ["https://example.com"],
        "models": ["abc"],
        "output": "file",
        "append": True,
    }
