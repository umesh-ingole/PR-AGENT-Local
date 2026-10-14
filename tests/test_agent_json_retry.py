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