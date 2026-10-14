from run_pipeline import cloud_startup_warning, parse_args, pipeline_settings


def test_pipeline_defaults_to_local_only_with_two_rounds():
    settings = pipeline_settings(parse_args([]).cloud)

    assert settings["force_local"] is True
    assert settings["max_iterations"] == 2
    assert settings["provider_order"] == "ollama"


def test_cloud_flag_selects_gemini_fallback_and_three_rounds():
    settings = pipeline_settings(parse_args(["--cloud"]).cloud)

    assert settings["force_local"] is False
    assert settings["max_iterations"] == 3
    assert settings["provider_order"] == "gemini,ollama"


def test_cloud_warns_when_google_api_key_is_unset(monkeypatch):
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)

    warning = cloud_startup_warning(parse_args(["--cloud"]).cloud)

    assert warning == (
        "GOOGLE_API_KEY not set: --cloud will fall back to the local model and be slow."
    )


def test_cloud_does_not_warn_or_expose_google_api_key_when_set(monkeypatch):
    secret_value = "do-not-print-this-key"
    monkeypatch.setenv("GOOGLE_API_KEY", secret_value)

    warning = cloud_startup_warning(parse_args(["--cloud"]).cloud)

    assert warning is None
    assert secret_value not in str(warning)