import json

import pytest

import agents


def _state():
    return {
        "language": "python",
        "static_tool": "pylint",
        "static_report": "ok",
        "context": "style guidance",
        "current_code": "print('hello')\n",
        "force_local": True,
    }


def test_ollama_invalid_json_gets_one_strict_retry(monkeypatch):
    calls = []
    responses = [
        ("Here are the issues: none", "ollama"),
        ("[]", "ollama"),
    ]

    def fake_call_llm(system, user, force_local, **kwargs):
        assert kwargs == {"format": "json"}
        calls.append((system, user, force_local))
        return responses.pop(0)

    monkeypatch.setattr(agents, "call_llm", fake_call_llm)

    result = agents.bug_agent(_state())

    assert result["bug_findings"] == []
    assert len(calls) == 2
    assert "STRICT FORMAT REMINDER" in calls[1][0]
    assert calls[1][2] is True
    assert "retried invalid JSON once" in result["log"][0]
    assert "via ollama, ollama" in result["log"][0]


def test_second_invalid_ollama_response_becomes_unscored_warning(monkeypatch):
    calls = []

    def fake_call_llm(system, user, force_local, **kwargs):
        assert kwargs == {"format": "json"}
        calls.append((system, user, force_local))
        return "not json", "ollama"

    monkeypatch.setattr(agents, "call_llm", fake_call_llm)

    result = agents.bug_agent(_state())

    assert len(calls) == 2
    assert "STRICT FORMAT REMINDER" in calls[1][0]
    assert "PREVIOUS RESPONSE JSON PARSE ERROR (exact)" in calls[1][1]
    assert "Expecting value (line 1, column 1)" in calls[1][1]
    assert result["bug_findings"] == []
    assert len(result["response_errors"]) == 1
    warning = result["response_errors"][0]
    assert warning["title"] == "LLM unavailable"
    assert warning["evidence"] == "tool"
    assert warning["agent"] == "llm-unavailable"
    assert "after one retry" in warning["detail"]
    assert "InvalidJSON" in warning["detail"]
    assert any("LLM unavailable" in line for line in result["log"])


def test_valid_empty_retry_is_not_reported_as_response_error(monkeypatch):
    responses = iter([("invalid", "ollama"), ("[]", "ollama")])
    monkeypatch.setattr(
        agents, "call_llm", lambda *_args, **_kwargs: next(responses)
    )

    result = agents.security_agent(_state())

    assert result["security_findings"] == []
    assert result["response_errors"] == []


def _finding(code, line: int | None = 1):
    return {
        "line": line,
        "severity": "high",
        "title": "Example issue",
        "detail": "Example detail",
        "evidence": "reasoning",
        "code": code,
    }


def test_drops_llm_finding_when_quoted_code_is_not_in_file(monkeypatch):
    monkeypatch.setattr(
        agents,
        "call_llm",
        lambda *_args, **_kwargs: (json.dumps([_finding("dangerous_call()")]), "ollama"),
    )

    result = agents.bug_agent(_state())

    assert result["bug_findings"] == []
    assert any("dropped 1 finding(s) as unverified" in line for line in result["log"])


def test_keeps_llm_finding_when_quoted_code_is_present(monkeypatch):
    finding = _finding("print('hello')")
    monkeypatch.setattr(
        agents,
        "call_llm",
        lambda *_args, **_kwargs: (json.dumps([finding]), "ollama"),
    )

    result = agents.bug_agent(_state())

    assert len(result["bug_findings"]) == 1
    assert result["bug_findings"][0]["code"] == "print('hello')"
    assert any("dropped 0 finding(s) as unverified" in line for line in result["log"])


def test_code_verification_ignores_whitespace_differences(monkeypatch):
    state = {**_state(), "current_code": "password = 'secret'\n"}
    finding = _finding("password='secret'")
    monkeypatch.setattr(
        agents,
        "call_llm",
        lambda *_args, **_kwargs: (json.dumps([finding]), "ollama"),
    )

    result = agents.bug_agent(state)

    assert len(result["bug_findings"]) == 1
    assert result["bug_findings"][0]["code"] == "password='secret'"


def test_code_verification_allows_quote_two_lines_from_reported_line(monkeypatch):
    state = {**_state(), "current_code": "first()\nsecond()\nprint('hello')\n"}
    finding = _finding("print('hello')", line=1)
    monkeypatch.setattr(
        agents,
        "call_llm",
        lambda *_args, **_kwargs: (json.dumps([finding]), "ollama"),
    )

    result = agents.bug_agent(state)

    assert len(result["bug_findings"]) == 1


def test_line_null_requires_quoted_code_somewhere_in_file(monkeypatch):
    state = {**_state(), "current_code": "first()\nprint('hello')\nlast()\n"}
    valid = _finding("print('hello')", line=None)
    invalid = _finding("absent()", line=None)
    monkeypatch.setattr(
        agents,
        "call_llm",
        lambda *_args, **_kwargs: (json.dumps([valid, invalid]), "ollama"),
    )

    result = agents.bug_agent(state)

    assert [finding["code"] for finding in result["bug_findings"]] == ["print('hello')"]
    assert any("dropped 1 finding(s) as unverified" in line for line in result["log"])


def test_agent_prompts_require_current_code_and_quoted_line():
    reminder = "This code may already have been corrected. Report only problems present in the code shown, and quote the line."

    assert '"code"' in agents.FINDING_SCHEMA
    assert '"findings"' in agents.FINDING_SCHEMA
    assert reminder in agents.SECURITY_SYSTEM
    assert reminder in agents.BUG_SYSTEM


def test_review_accepts_findings_object_and_plain_array(monkeypatch):
    responses = iter([
        (json.dumps({"findings": []}), "ollama"),
        ("[]", "ollama"),
    ])
    monkeypatch.setattr(
        agents, "call_llm", lambda *_args, **_kwargs: next(responses)
    )

    assert agents.bug_agent(_state())["bug_findings"] == []
    assert agents.bug_agent(_state())["bug_findings"] == []


def test_tool_finding_severity_is_capped_by_pylint_category():
    raw = [
        _finding("line one", line=1),
        _finding("line two", line=2),
        _finding("line three", line=3),
    ]
    for item, severity in zip(raw, ("high", "critical", "high")):
        item["severity"] = severity

    normalized, capped = agents._normalize(
        raw,
        "bug-agent",
        {1, 2, 3},
        {1: "convention", 2: "warning", 3: "error"},
    )

    assert capped == 2
    assert [item["severity"] for item in normalized] == ["low", "low", "high"]
    assert all(item["evidence"] == "tool" for item in normalized)


def test_ast_scanner_finding_severity_is_not_capped_as_pylint_tool():
    from tools import scan_python_patterns

    finding = scan_python_patterns("eval(value)\n")[0]

    assert finding["severity"] == "critical"
    assert finding["agent"] == "ast-scan"


def test_invalid_replies_are_saved_per_agent_and_round_with_secrets_redacted(
    tmp_path, monkeypatch,
):
    secret = "mock-private-api-key-value"
    monkeypatch.setenv("GOOGLE_API_KEY", secret)
    replies = iter([
        (f'{{"API_KEY": "{secret}", broken}}', "ollama"),
        (f'{{"API_KEY": "{secret}", still broken}}', "ollama"),
    ])
    monkeypatch.setattr(
        agents, "call_llm", lambda *_args, **_kwargs: next(replies)
    )
    state = {**_state(), "run_dir": str(tmp_path), "iteration": 0}

    result = agents.bug_agent(state)

    debug_file = tmp_path / "bug-agent_round_1_invalid_json.txt"
    contents = debug_file.read_text(encoding="utf-8")
    assert len(result["response_errors"]) == 1
    assert contents.count("[REDACTED]") == 2
    assert secret not in contents
    assert "broken" in contents


def test_verifies_line_12_quote_with_line_number_prefix(monkeypatch):
    source_lines = [f"# filler {line}" for line in range(1, 12)]
    source_lines.append('query = f"SELECT ..."')
    state = {**_state(), "current_code": "\n".join(source_lines) + "\n"}
    finding = _finding('12: query = f"SELECT ..."', line=12)
    monkeypatch.setattr(
        agents,
        "call_llm",
        lambda *_args, **_kwargs: (json.dumps([finding]), "ollama"),
    )

    result = agents.bug_agent(state)

    assert len(result["bug_findings"]) == 1


@pytest.mark.parametrize("prefix", ["12:", "12 |", "12.", "L12:", "  12  "])
def test_strips_supported_line_number_prefix_styles(prefix):
    source_lines = [f"# filler {line}" for line in range(1, 12)]
    source_lines.append('query = f"SELECT ..."')
    finding = _finding(f'{prefix}query = f"SELECT ..."', line=12)

    assert agents._verify_finding_code(finding, source_lines)


def test_keeps_quote_that_starts_with_decimal_number():
    finding = _finding("2.5 * rate", line=1)

    assert agents._verify_finding_code(finding, ["2.5 * rate"])


def test_verifies_multiline_quote_with_numbered_lines(monkeypatch):
    state = {
        **_state(),
        "current_code": (
            'conn = sqlite3.connect("app.db")\n'
            "cur = conn.cursor()\n"
            "cur.execute(query)\n"
        ),
    }
    finding = _finding(
        '1: conn = sqlite3.connect("app.db")\n'
        "2 | cur = conn.cursor()\n"
        "3. cur.execute(query)",
        line=1,
    )
    monkeypatch.setattr(
        agents,
        "call_llm",
        lambda *_args, **_kwargs: (json.dumps([finding]), "ollama"),
    )

    result = agents.bug_agent(state)

    assert len(result["bug_findings"]) == 1


def test_drops_quote_that_matches_but_is_more_than_two_lines_away(monkeypatch):
    state = {
        **_state(),
        "current_code": "reported_line()\nnearby_one()\nnearby_two()\nsecret_call()\n",
    }
    finding = _finding("secret_call()", line=1)
    monkeypatch.setattr(
        agents,
        "call_llm",
        lambda *_args, **_kwargs: (json.dumps([finding]), "ollama"),
    )

    result = agents.bug_agent(state)

    assert result["bug_findings"] == []
    assert any("dropped 1 finding(s) as unverified" in line for line in result["log"])


def test_finding_evidence_is_derived_from_report_and_secret_scan_lines(monkeypatch):
    state = {
        **_state(),
        "current_code": "first()\nreported()\nsecret()\n",
        "static_report": "2:warning:example:reported diagnostic",
        "secret_findings": [{"line": 3}],
    }
    findings = [
        {**_finding("first()", line=1), "evidence": "tool"},
        {**_finding("reported()", line=2), "evidence": "reasoning"},
        {**_finding("secret()", line=3), "evidence": "reasoning"},
    ]
    monkeypatch.setattr(
        agents,
        "call_llm",
        lambda *_args, **_kwargs: (json.dumps(findings), "ollama"),
    )

    result = agents.bug_agent(state)

    assert [finding["evidence"] for finding in result["bug_findings"]] == [
        "reasoning", "tool", "tool",
    ]


def test_merge_deduplicates_nearby_cross_agent_quotes_and_keeps_highest_severity():
    security = {
        "line": 10,
        "severity": "medium",
        "title": "SQL injection",
        "detail": "The query is injectable.",
        "code": "cursor.execute(query)",
        "evidence": "reasoning",
        "agent": "security-agent",
    }
    bug = {
        **security,
        "line": 12,
        "severity": "critical",
        "title": "SQL injection vulnerability",
        "agent": "bug-agent",
    }

    merged, _score, _passed = agents.merge_and_score([security], [bug])

    assert len(merged) == 1
    assert merged[0]["severity"] == "critical"
    assert merged[0]["agent"] == "bug-agent"


def test_merge_keeps_nearby_quotes_when_titles_are_not_similar():
    first = {
        "line": 10,
        "severity": "high",
        "title": "SQL injection",
        "detail": "The query is injectable.",
        "code": "cursor.execute(query)",
        "evidence": "reasoning",
        "agent": "security-agent",
    }
    second = {
        **first,
        "line": 12,
        "title": "Missing return statement",
        "agent": "bug-agent",
    }

    merged, _score, _passed = agents.merge_and_score([first], [second])

    assert len(merged) == 2