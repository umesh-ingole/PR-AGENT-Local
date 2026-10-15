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
    monkeypatch.setattr(graph, "run_pylint_errors_only", lambda _code: [])

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


def test_summary_includes_agent_response_error_alongside_scanner_findings():
    from tools import scan_python_patterns

    secret = scan_secrets('API_KEY = "sk-live-123"\n')
    ast_findings = scan_python_patterns("eval(value)\n")
    response_error = {
        "line": None,
        "severity": "high",
        "title": "Security Agent response error",
        "detail": "The agent returned invalid JSON twice.",
        "code": "",
        "evidence": "reasoning",
        "agent": "agent-response-error",
    }
    unavailable = agents._llm_unavailable_finding(
        "Bug Agent", "provider=ollama error=TimeoutError"
    )
    result = agents.summary_agent({
        "security_findings": [],
        "bug_findings": [],
        "secret_findings": secret,
        "ast_findings": ast_findings,
        "response_errors": [response_error, unavailable],
        "interface_findings": [],
    })

    assert response_error in result["merged"]
    assert unavailable in result["merged"]
    assert secret[0] in result["merged"]
    assert ast_findings[0] in result["merged"]
    baseline = agents.summary_agent({
        "security_findings": [],
        "bug_findings": [],
        "secret_findings": secret,
        "ast_findings": ast_findings,
        "interface_findings": [],
    })
    assert result["score"] == baseline["score"]
    assert result["passed"] == baseline["passed"]


def test_graph_merges_deterministic_secret_when_review_agents_return_no_findings(
    monkeypatch,
):
    source = 'API_KEY = "sk-live-123"\n'

    monkeypatch.setattr(graph, "run_static_analysis", lambda *_args: ("pylint", ""))
    monkeypatch.setattr(graph, "retrieve_context", lambda *_args: "")
    monkeypatch.setattr(graph, "run_pylint_errors_only", lambda _code: [])

    def mocked_call_llm(system, _user, _force_local=False):
        if system == agents.CODER_SYSTEM:
            return f"```python\n{source}```", "mock"
        return "[]", "mock"

    monkeypatch.setattr(agents, "call_llm", mocked_call_llm)
    monkeypatch.setattr(
        agents, "write_summary_with_provider", lambda _state: ("summary", "mock")
    )

    final_state = graph.build_graph().invoke({
        "filename": "sample.py",
        "language": "python",
        "original_code": source,
        "current_code": source,
        "force_local": True,
        "max_iterations": 2,
        "providers": [],
        "log": [],
        "score_history": [],
        "penalty_history": [],
        "rejected_fix_count": 0,
        "dropped_finding_counts": [],
        "dropped_unverified_history": [],
        "dropped_finding_count_index": 0,
    })

    assert all(
        finding["agent"] == "secrets-scan"
        for finding in final_state["merged"]
    )
    assert len(final_state["merged"]) == 1