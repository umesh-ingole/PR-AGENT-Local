import run_pipeline
from run_pipeline import cloud_startup_warning, parse_args, pipeline_settings
from static_analysis import static_analysis_mode


def test_pipeline_defaults_to_local_only_with_two_rounds():
    settings = pipeline_settings(parse_args([]).cloud)

    assert settings["force_local"] is True
    assert settings["max_iterations"] == 2
    assert settings["provider_order"] == "ollama"


def test_cloud_flag_selects_gemini_fallback_and_three_rounds():
    args = parse_args(["--cloud"])
    settings = pipeline_settings(args.cloud, args.no_fallback)

    assert settings["force_local"] is False
    assert settings["max_iterations"] == 3
    assert settings["provider_order"] == "gemini,ollama"


def test_no_fallback_selects_only_gemini_for_cloud_runs():
    args = parse_args(["--cloud", "--no-fallback"])
    settings = pipeline_settings(args.cloud, args.no_fallback)

    assert settings["force_local"] is False
    assert settings["provider_order"] == "gemini"
    assert settings["no_fallback"] is True


def test_cloud_warns_when_google_api_key_is_unset(monkeypatch):
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)

    warning = cloud_startup_warning(parse_args(["--cloud"]).cloud)

    assert warning == (
        "GOOGLE_API_KEY not set: --cloud will fall back to the local model and be slow."
    )


def test_cloud_no_fallback_warns_that_missing_key_will_not_use_ollama(monkeypatch):
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)

    warning = cloud_startup_warning(cloud=True, no_fallback=True)

    assert warning == (
        "GOOGLE_API_KEY not set: --cloud --no-fallback cannot call Gemini."
    )


def test_cloud_does_not_warn_or_expose_google_api_key_when_set(monkeypatch):
    secret_value = "do-not-print-this-key"
    monkeypatch.setenv("GOOGLE_API_KEY", secret_value)

    warning = cloud_startup_warning(parse_args(["--cloud"]).cloud)

    assert warning is None
    assert secret_value not in str(warning)


def test_static_analysis_mode_respects_explicit_setting_over_windows_default(
    monkeypatch,
):
    monkeypatch.setattr("static_analysis.sys.platform", "win32")
    monkeypatch.setenv("USE_MCP", "1")
    assert static_analysis_mode() == "MCP (USE_MCP=1; explicit setting)"

    monkeypatch.setenv("USE_MCP", "0")
    assert static_analysis_mode() == "direct calls (USE_MCP=0; explicit setting)"

    monkeypatch.delenv("USE_MCP")
    assert static_analysis_mode() == "direct calls (USE_MCP=0; Windows default)"


def test_runner_prints_static_analysis_mode_at_startup(
    tmp_path, monkeypatch, capsys,
):
    project_dir = tmp_path / "project"
    sample_dir = project_dir / "sample"
    sample_dir.mkdir(parents=True)
    (sample_dir / "buggy_sample.py").write_text("print('sample')\n", encoding="utf-8")
    run_dir = project_dir / "runs" / "test-run"
    run_dir.parent.mkdir()

    def make_run_dir(_runs_dir):
        run_dir.mkdir()
        return run_dir

    monkeypatch.setattr(run_pipeline, "__file__", str(project_dir / "run_pipeline.py"))
    monkeypatch.setattr(run_pipeline, "create_run_directory", make_run_dir)
    monkeypatch.setattr(
        run_pipeline,
        "build_graph",
        lambda: (_ for _ in ()).throw(RuntimeError("offline test stop")),
    )
    monkeypatch.setenv("USE_MCP", "0")

    try:
        run_pipeline.main([])
    except RuntimeError as exc:
        assert str(exc) == "offline test stop"
    else:
        raise AssertionError("Expected mocked pipeline failure")

    output = capsys.readouterr().out
    assert "Static analysis: direct calls (USE_MCP=0; explicit setting)" in output
    assert "Static analysis:" in output.splitlines()[0]