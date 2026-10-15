import agents
import graph
from tools import scan_secrets


def test_detects_sensitive_name_and_common_key_formats():
    code = (
        'API_KEY = "sk-live-123"\n'
        'aws_access_key = "AKIA1234567890ABCDEF"\n'
        'PRIVATE_KEY = "-----BEGIN PRIVATE KEY-----"\n'
    )

    findings = scan_secrets(code)

    assert [finding["line"] for finding in findings] == [1, 2, 3]
    assert all(finding["severity"] == "high" for finding in findings)
    assert all(finding["evidence"] == "tool" for finding in findings)
    assert all(finding["agent"] == "secrets-scan" for finding in findings)
    assert "sk-live-123" not in str(findings)


def test_ignores_empty_placeholders_and_environment_lookups():
    code = (
        'API_KEY = ""\n'
        'API_SECRET = "changeme"\n'
        'TOKEN = "placeholder"\n'
        'API_KEY = os.getenv("API_KEY")\n'
        'PASSWORD = os.environ.get("PASSWORD")\n'
    )

    assert scan_secrets(code) == []


def test_flags_non_placeholder_environment_fallbacks_with_specific_title():
    code = (
        'API_KEY = os.getenv("API_KEY", "default_api_key")\n'
        'API_KEY = os.environ.get("API_KEY", "default_api_key")\n'
    )

    findings = scan_secrets(code)

    assert [finding["line"] for finding in findings] == [1, 2]
    assert all(finding["severity"] == "high" for finding in findings)
    assert all(finding["title"] == "Hard-coded fallback for a secret" for finding in findings)


def test_flags_short_and_plain_word_password_values():
    code = (
        'PASSWORD = "hunter2"\n'
        'DB_PASSWORD = "mysecretpassword"\n'
    )

    findings = scan_secrets(code)

    assert [finding["line"] for finding in findings] == [1, 2]
    assert all(finding["severity"] == "high" for finding in findings)


def test_ignores_environment_lookups_without_nonempty_default():
    code = (
        'API_KEY = os.getenv("API_KEY")\n'
        'API_KEY = os.getenv("API_KEY", "")\n'
        'API_KEY = os.environ.get("API_KEY", "changeme")\n'
    )

    assert scan_secrets(code) == []


def test_ignores_short_words_and_non_secret_name_suffixes():
    code = (
        'token_type = "bearer"\n'
        'sort_key = "name"\n'
        'KEY_FIELD = "id"\n'
        'API_KEY = "short"\n'
        'API_KEY = "ordinaryword"\n'
    )

    assert scan_secrets(code) == []


def test_static_and_coder_nodes_refresh_secret_findings(monkeypatch):
    initial_code = 'API_KEY = "sk-live-123"\n'
    fixed_code = 'import os\nAPI_KEY = os.getenv("API_KEY")\n'
    monkeypatch.setattr(graph, "run_static_analysis", lambda *_args: ("pylint", "lint output"))

    initial = graph.static_node({"language": "python", "current_code": initial_code})
    assert len(initial["secret_findings"]) == 1
    assert "Deterministic secrets scan" in initial["static_report"]

    monkeypatch.setattr(
        graph.agents,
        "coder_agent",
        lambda _state: {"current_code": fixed_code, "log": []},
    )
    updated = graph.coder_node({"language": "python", "current_code": initial_code})
    assert updated["secret_findings"] == []


def test_summary_includes_scanner_finding_without_llm_call(monkeypatch):
    finding = scan_secrets('API_KEY = "sk-live-123"\n')[0]
    monkeypatch.setattr(
        agents,
        "call_llm",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("LLM called")),
    )

    result = agents.summary_agent({
        "security_findings": [],
        "bug_findings": [],
        "secret_findings": [finding],
    })

    assert result["merged"] == [finding]
    assert result["score"] == 80
    assert result["passed"] is False