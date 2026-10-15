import agents
import graph
import llm_gateway


def _initial_state():
    return {
        "filename": "sample.py",
        "language": "python",
        "original_code": "print('ok')\n",
        "current_code": "print('ok')\n",
        "force_local": False,
        "max_iterations": 3,
        "providers": [],
        "log": [],
        "score_history": [],
        "penalty_history": [],
        "response_errors": [],
    }


def _mock_graph_setup(monkeypatch):
    monkeypatch.setattr(
        graph,
        "static_node",
        lambda _state: {
            "static_tool": "mock",
            "static_report": "",
            "iteration": 0,
            "secret_findings": [],
            "ast_findings": [],
            "log": [],
        },
    )
    monkeypatch.setattr(graph, "rag_node", lambda _state: {"context": "", "log": []})
    monkeypatch.setattr(
        agents,
        "write_summary_with_provider",
        lambda _state: ("done", "mock"),
    )


def test_provider_exception_falls_back_once_then_finishes_with_unscored_warning(
    monkeypatch, caplog,
):
    _mock_graph_setup(monkeypatch)
    monkeypatch.setenv("LLM_PROVIDERS", "gemini,ollama")
    monkeypatch.setenv("GOOGLE_API_KEY", "offline-test-key")
    monkeypatch.setattr(llm_gateway, "_gemini_unavailable_until", 0.0)
    calls = {"gemini": 0, "ollama": 0}

    class FailingClient:
        def __init__(self, provider):
            self.provider = provider

        def invoke(self, _messages):
            calls[self.provider] += 1
            if self.provider == "gemini":
                raise TimeoutError("mock timeout")
            raise ConnectionError("mock offline")

    monkeypatch.setattr(
        llm_gateway, "_make", lambda provider: FailingClient(provider)
    )
    coder_calls = []
    monkeypatch.setattr(
        agents, "coder_agent", lambda state: coder_calls.append(state) or {}
    )

    result = graph.build_graph().invoke(_initial_state())
    warnings = [
        finding for finding in result["merged"]
        if finding["title"] == "LLM unavailable"
    ]

    assert calls == {"gemini": 2, "ollama": 2}
    assert result["score"] == 100
    assert result["raw_penalty"] == 0
    assert result["passed"] is False
    assert result["review_incomplete"] is True
    assert len(warnings) == 1
    assert warnings[0]["evidence"] == "tool"
    assert warnings[0]["agent"] == "llm-unavailable"
    assert "provider=gemini error=TimeoutError" in warnings[0]["detail"]
    assert "provider=ollama error=ConnectionError" in warnings[0]["detail"]
    assert agents.review_incomplete_banner(result) == agents.REVIEW_INCOMPLETE_BANNER
    assert agents.REVIEW_INCOMPLETE_BANNER in result["log"]
    assert coder_calls == []
    assert "provider=gemini error=TimeoutError" in caplog.text
    assert "provider=ollama error=ConnectionError" in caplog.text


def test_invalid_reply_twice_finishes_without_scoring_or_fixing_warning(monkeypatch):
    _mock_graph_setup(monkeypatch)
    calls = []

    def invalid_reply(system, _user, _force_local):
        calls.append(system)
        return "not JSON", "ollama"

    monkeypatch.setattr(agents, "call_llm", invalid_reply)
    coder_calls = []
    monkeypatch.setattr(
        agents, "coder_agent", lambda state: coder_calls.append(state) or {}
    )

    result = graph.build_graph().invoke(_initial_state())
    warning = next(
        finding for finding in result["merged"]
        if finding["title"] == "LLM unavailable"
    )

    assert len(calls) == 4
    assert result["score"] == 100
    assert result["raw_penalty"] == 0
    assert result["passed"] is False
    assert result["review_incomplete"] is True
    assert warning["evidence"] == "tool"
    assert warning["agent"] == "llm-unavailable"
    assert "InvalidJSON" in warning["detail"]
    assert coder_calls == []


def test_successful_agents_leave_review_complete_without_banner(monkeypatch):
    _mock_graph_setup(monkeypatch)
    monkeypatch.setattr(agents, "call_llm", lambda *_args: ("[]", "mock"))
    coder_calls = []
    monkeypatch.setattr(
        agents, "coder_agent", lambda state: coder_calls.append(state) or {}
    )

    result = graph.build_graph().invoke(_initial_state())

    assert result["passed"] is True
    assert result["review_incomplete"] is False
    assert agents.review_incomplete_banner(result) == ""
    assert agents.REVIEW_INCOMPLETE_BANNER not in result["log"]
    assert coder_calls == []


def test_coder_failure_leaves_source_unchanged_and_records_warning(monkeypatch):
    source = "dangerous_call()\n"
    finding = {
        "line": 1,
        "severity": "high",
        "title": "Unsafe call",
        "detail": "Replace this call safely.",
        "code": "dangerous_call()",
        "evidence": "reasoning",
        "agent": "bug-agent",
    }
    monkeypatch.setattr(
        agents,
        "call_llm",
        lambda *_args: (_ for _ in ()).throw(TimeoutError("mock timeout")),
    )

    result = agents.coder_agent({
        **_initial_state(),
        "current_code": source,
        "force_local": True,
        "merged": [finding],
    })

    assert result["current_code"] == source
    assert result["response_errors"][0]["title"] == "LLM unavailable"
    assert result["response_errors"][0]["evidence"] == "tool"
    assert "provider=ollama error=TimeoutError" in result["response_errors"][0]["detail"]


def test_coder_prompt_excludes_response_error_and_llm_warning(monkeypatch):
    warning = agents._llm_unavailable_finding(
        "Bug Agent", "provider=ollama error=TimeoutError"
    )
    response_error = {
        **warning,
        "agent": "agent-response-error",
        "title": "Bug Agent response error",
    }
    actual = {
        "line": 1,
        "severity": "low",
        "title": "Unsafe call",
        "detail": "Replace this call safely.",
        "code": "dangerous_call()",
        "evidence": "reasoning",
        "agent": "bug-agent",
    }
    captured = {}

    def fake_call_llm(_system, user, _force_local):
        captured["user"] = user
        return "```python\nprint('ok')\n```", "ollama"

    monkeypatch.setattr(agents, "call_llm", fake_call_llm)
    result = agents.coder_agent({
        **_initial_state(),
        "current_code": "dangerous_call()\n",
        "merged": [warning, response_error, actual],
    })

    assert "Unsafe call" in captured["user"]
    assert "LLM unavailable" not in captured["user"]
    assert "Bug Agent response error" not in captured["user"]
    assert result["current_code"] == "print('ok')\n"


def test_coder_skips_model_when_only_warnings_are_present(monkeypatch):
    warning = agents._llm_unavailable_finding(
        "Security Agent", "provider=ollama error=ConnectionError"
    )
    monkeypatch.setattr(
        agents,
        "call_llm",
        lambda *_args: (_ for _ in ()).throw(AssertionError("Coder was called")),
    )

    result = agents.coder_agent({
        **_initial_state(),
        "merged": [warning],
    })

    assert result["current_code"] == "print('ok')\n"
    assert result["iteration"] == 0
    assert "no actionable findings" in result["log"][0]


def test_summary_provider_exception_uses_fallback_and_adds_report_warning(monkeypatch):
    monkeypatch.setattr(
        agents,
        "write_summary_with_provider",
        lambda _state: (_ for _ in ()).throw(TimeoutError("mock timeout")),
    )
    state = {
        **_initial_state(),
        "force_local": True,
        "score": 80,
        "merged": [{
            "line": 1,
            "severity": "high",
            "title": "Unsafe call",
            "detail": "Fix it.",
            "code": "dangerous_call()",
            "evidence": "reasoning",
            "agent": "security-agent",
        }],
    }

    result = graph.finalize_node(state)

    assert result["summary"].startswith("Final score 80/100")
    assert result["review_incomplete"] is True
    assert result["passed"] is False
    assert result["merged"][0] == state["merged"][0]
    warning = result["merged"][1]
    assert warning["title"] == "LLM unavailable"
    assert warning["evidence"] == "tool"
    assert "provider=ollama error=TimeoutError" in warning["detail"]
    assert result["response_errors"] == [warning]
    assert warning["stage"] == "Summary Agent"
    assert agents.review_incomplete_banner(result) == agents.REVIEW_INCOMPLETE_BANNER
    assert any("LLM unavailable" in line for line in result["log"])
