import json

import load_latest_run
import run_pipeline
from run_pipeline import (
    build_run_info,
    save_final_artifacts,
    save_round_findings,
    save_run_info,
    termination_reason,
)


def test_run_artifacts_save_round_findings_and_unified_diff(tmp_path):
    update = {
        "score": 72,
        "raw_penalty": 28,
        "passed": False,
        "merged": [{"line": 1, "severity": "medium", "title": "Example"}],
    }
    findings_path = save_round_findings(tmp_path, 1, update)
    findings = json.loads(findings_path.read_text(encoding="utf-8"))

    assert findings["round"] == 1
    assert findings["score"] == 72
    assert findings["raw_penalty"] == 28
    assert findings["findings"] == update["merged"]

    original = "value = 1\n"
    fixed = "value = 2\n"
    fixed_path, diff_path = save_final_artifacts(
        tmp_path, "sample.py", original, fixed)

    assert fixed_path.read_text(encoding="utf-8") == fixed
    diff = diff_path.read_text(encoding="utf-8")
    assert "--- original/sample.py" in diff
    assert "+++ fixed/sample.py" in diff
    assert "-value = 1" in diff
    assert "+value = 2" in diff


def test_run_info_records_run_metadata_without_environment_secrets(tmp_path, monkeypatch):
    secret = "test-google-secret-value-do-not-save"
    monkeypatch.setenv("GOOGLE_API_KEY", secret)
    monkeypatch.setenv("GEMINI_MODEL", "test-gemini-model")
    monkeypatch.setenv("OLLAMA_MODEL", "test-ollama-model")
    settings = {
        "force_local": False,
        "max_iterations": 3,
    }
    final_state = {
        "providers": ["gemini", "ollama", "gemini"],
        "penalty_history": [124, 80, 10],
        "rejected_fix_count": 2,
    }

    run_info = build_run_info(
        settings, final_state, "2026-10-15T00:00:00+00:00", 12.34567, "passed")
    info_path = save_run_info(tmp_path, run_info)
    contents = info_path.read_text(encoding="utf-8")
    saved = json.loads(contents)

    assert saved["providers"] == ["gemini", "ollama", "gemini"]
    assert saved["models"] == {
        "gemini": "test-gemini-model",
        "ollama": "test-ollama-model",
    }
    assert saved["cloud"] is True
    assert saved["force_local"] is False
    assert saved["max_iterations"] == 3
    assert saved["start_time"] == "2026-10-15T00:00:00+00:00"
    assert saved["total_elapsed_seconds"] == 12.346
    assert saved["termination_reason"] == "passed"
    assert saved["raw_penalty_history"] == [124, 80, 10]
    assert saved["rejected_fix_count"] == 2
    assert secret not in contents
    assert "GOOGLE_API_KEY" not in contents


def test_termination_reason_distinguishes_pass_early_stop_and_round_cap():
    assert termination_reason({"passed": True, "iteration": 3}, 3) == "passed"
    assert termination_reason(
        {"passed": False, "stop_no_improvement": True, "iteration": 1}, 3
    ) == "early stop"
    assert termination_reason(
        {"passed": False, "stop_no_improvement": False, "iteration": 3}, 3
    ) == "round cap"
    assert termination_reason({"passed": False, "iteration": 0}, 3) == "error"


def test_latest_run_loader_prints_info_only_run(tmp_path, monkeypatch, capsys):
    run_dir = tmp_path / "runs" / "20261015_080000"
    run_dir.mkdir(parents=True)
    (run_dir / "run_info.json").write_text(
        json.dumps({"termination_reason": "error", "providers": []}),
        encoding="utf-8",
    )
    monkeypatch.setattr(load_latest_run, "__file__", str(tmp_path / "load_latest_run.py"))

    load_latest_run.main()

    output = capsys.readouterr().out
    assert "Latest run: 20261015_080000" in output
    assert "Run information:" in output
    assert '"termination_reason": "error"' in output
    assert "Final fixed file has not been saved yet." in output


def test_runner_saves_safe_run_info_when_pipeline_raises(tmp_path, monkeypatch):
    secret = "test-google-secret-value-do-not-save"
    monkeypatch.setenv("GOOGLE_API_KEY", secret)
    project_dir = tmp_path / "project"
    sample_dir = project_dir / "sample"
    sample_dir.mkdir(parents=True)
    (sample_dir / "buggy_sample.py").write_text("print('sample')\n", encoding="utf-8")
    run_dir = project_dir / "runs" / "test-run"
    run_dir.parent.mkdir()
    monkeypatch.setattr(run_pipeline, "__file__", str(project_dir / "run_pipeline.py"))
    monkeypatch.setattr(run_pipeline, "create_run_directory", lambda _runs_dir: _make_run_dir(run_dir))
    monkeypatch.setattr(
        run_pipeline,
        "build_graph",
        lambda: (_ for _ in ()).throw(RuntimeError("simulated failure")),
    )

    try:
        run_pipeline.main([])
    except RuntimeError as exc:
        assert str(exc) == "simulated failure"
    else:
        raise AssertionError("Expected pipeline error to propagate")

    info_path = run_dir / "run_info.json"
    contents = info_path.read_text(encoding="utf-8")
    saved = json.loads(contents)
    assert saved["termination_reason"] == "error"
    assert secret not in contents
    assert "GOOGLE_API_KEY" not in contents


def _make_run_dir(run_dir):
    run_dir.mkdir(parents=True)
    return run_dir