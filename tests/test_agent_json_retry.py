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

    def fake_call_llm(system, user, force_local):
        calls.append((system, user, force_local))
        return responses.pop(0)

    monkeypatch.setattr(agents, "call_llm", fake_call_llm)

    result = agents.bug_agent(_state())

    assert result["bug_findings"] == []
    assert len(calls) == 2
    assert "STRICT FORMAT REMINDER" in calls[1][0]
    assert calls[1][2] is True
    assert "retried invalid JSON once" in result["log"][0]


def test_second_invalid_ollama_response_fails_after_one_retry(monkeypatch):
    calls = []

    def fake_call_llm(system, user, force_local):
        calls.append(system)
        return "not json", "ollama"

    monkeypatch.setattr(agents, "call_llm", fake_call_llm)

    with pytest.raises(ValueError, match="valid JSON array"):
        agents.bug_agent(_state())

    assert len(calls) == 2


def _finding(code, line=1):
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
    assert reminder in agents.SECURITY_SYSTEM
    assert reminder in agents.BUG_SYSTEM


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