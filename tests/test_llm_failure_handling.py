import ast
from pathlib import Path

import agents
import graph
import llm_gateway
import run_pipeline


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


def _load_app_report_builder():
    app_path = Path(agents.__file__).with_name("app.py")
    app_tree = ast.parse(app_path.read_text(encoding="utf-8"))
    report_function = next(
        node for node in app_tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "build_report_md"
    )
    isolated_module = ast.Module(body=[report_function], type_ignores=[])
    namespace = {"agents": agents}
    from config import LANG_LABEL
    namespace["LANG_LABEL"] = LANG_LABEL
    exec(compile(isolated_module, str(app_path), "exec"), namespace)
    return namespace["build_report_md"]


def test_both_review_agents_fail_marks_review_incomplete_and_shows_banner(
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
        llm_gateway, "_make",
        lambda provider, **_kwargs: FailingClient(provider),
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
    banner = (
        "Review incomplete: the model was unavailable. "
        "The score reflects tool checks only."
    )
    assert agents.review_incomplete_banner(result) == banner
    assert banner in result["log"]
    assert any("Security Agent found 0 issue(s)" in line for line in result["log"])
    assert any("Bug-Detection Agent found 0 issue(s)" in line for line in result["log"])
    assert coder_calls == []
    assert "provider=gemini error=TimeoutError" in caplog.text
    assert "provider=ollama error=ConnectionError" in caplog.text


def test_invalid_reply_twice_finishes_without_scoring_or_fixing_warning(monkeypatch):
    _mock_graph_setup(monkeypatch)
    calls = []

    def invalid_reply(system, _user, _force_local, **_kwargs):
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


def test_both_review_agents_succeed_and_no_banner_is_shown(monkeypatch):
    _mock_graph_setup(monkeypatch)
    monkeypatch.setattr(
        agents, "call_llm", lambda *_args, **_kwargs: ("[]", "mock")
    )
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


def test_single_review_agent_failure_continues_through_another_round(monkeypatch):
    source = "def parse(value):\n    return eval(value)\n"
    safe_code = "import ast\n\ndef parse(value):\n    return ast.literal_eval(value)\n"
    security_calls = []
    coder_calls = []

    monkeypatch.setattr(
        graph, "run_static_analysis", lambda *_args: ("pylint", "[no issues]")
    )
    monkeypatch.setattr(graph, "retrieve_context", lambda *_args: "")
    monkeypatch.setattr(graph, "run_pylint_errors_only", lambda _code: [])

    def security_agent(_state):
        security_calls.append(True)
        if len(security_calls) == 1:
            warning = agents._llm_unavailable_finding(
                "Security Agent", "provider=ollama error=TimeoutError"
            )
            return {
                "security_findings": [],
                "security_agent_failed": True,
                "response_errors": [warning],
                "providers": [],
                "log": ["Security Agent unavailable"],
            }
        return {
            "security_findings": [],
            "security_agent_failed": False,
            "response_errors": [],
            "providers": ["mock"],
            "log": ["Security Agent recovered"],
        }

    def bug_agent(_state):
        return {
            "bug_findings": [{
                "line": 2,
                "severity": "medium",
                "title": "Input parsing issue",
                "detail": "Handle this input parsing issue.",
                "code": "    return eval(value)",
                "evidence": "reasoning",
                "agent": "bug-agent",
            }],
            "bug_agent_failed": False,
            "response_errors": [],
            "providers": ["mock"],
            "log": ["Bug Agent succeeded"],
        }

    def coder_agent(state):
        coder_calls.append(state)
        return {
            "current_code": safe_code,
            "iteration": state.get("iteration", 0) + 1,
            "providers": ["mock"],
            "interface_findings": [],
            "log": ["Coder Agent applied fixes (round 1) via mock; re-checking"],
        }

    monkeypatch.setattr(graph.agents, "security_agent", security_agent)
    monkeypatch.setattr(graph.agents, "bug_agent", bug_agent)
    monkeypatch.setattr(graph.agents, "coder_agent", coder_agent)
    monkeypatch.setattr(
        agents, "write_summary_with_provider", lambda _state: ("done", "mock")
    )

    result = graph.build_graph().invoke({
        **_initial_state(),
        "current_code": source,
        "original_code": source,
        "force_local": True,
        "max_iterations": 2,
    })

    assert len(security_calls) == 2
    assert len(coder_calls) == 1
    assert any(
        finding["title"] == "Use of eval()"
        for finding in coder_calls[0]["merged"]
    )
    assert any(
        finding["title"] == "Input parsing issue"
        for finding in coder_calls[0]["merged"]
    )
    assert result["iteration"] == 1
    assert result["review_incomplete"] is False
    assert any(
        "Review round incomplete: Security Agent failed; continuing" in line
        for line in result["log"]
    )


def test_ollama_response_error_details_reach_unavailable_finding(
    monkeypatch, capsys,
):
    secret = "mock-review-secret"
    monkeypatch.setenv("LLM_PROVIDERS", "ollama")
    monkeypatch.setenv("MOCK_SECRET_TOKEN", secret)
    monkeypatch.setattr(llm_gateway.time, "sleep", lambda _seconds: None)

    class ResponseError(Exception):
        status_code = 500

    class FailingOllama:
        def invoke(self, _messages):
            raise ResponseError("response failed " + secret + " " + "y" * 230)

    monkeypatch.setattr(
        llm_gateway, "_make",
        lambda _provider, **_kwargs: FailingOllama(),
    )
    result = agents.security_agent({
        **_initial_state(),
        "static_tool": "pylint",
        "static_report": "",
        "context": "",
        "force_local": True,
    })

    detail = result["response_errors"][0]["detail"]
    assert "status_code=500" in detail
    assert "response failed" in detail
    assert secret not in detail
    assert secret not in capsys.readouterr().out


def test_app_report_places_exact_incomplete_banner_near_the_top():
    warning = agents._llm_unavailable_finding(
        "Security Agent", "provider=ollama error=TimeoutError"
    )
    report = _load_app_report_builder()({
        "filename": "sample.py",
        "language": "python",
        "score": 100,
        "iteration": 0,
        "summary": "Tool-only review.",
        "merged": [warning],
        "review_incomplete": True,
        "static_tool": "mock",
        "static_report": "",
    })

    banner_line = f"> **{agents.REVIEW_INCOMPLETE_BANNER}**"
    assert banner_line in report.splitlines()[:4]


def test_coder_failure_alone_does_not_mark_review_incomplete(monkeypatch):
    warning = agents._llm_unavailable_finding(
        "Coder Agent", "provider=ollama error=TimeoutError"
    )
    monkeypatch.setattr(
        agents,
        "summary_agent",
        lambda _state: {
            "merged": [warning],
            "score": 100,
            "passed": True,
            "score_history": [100],
            "log": [],
        },
    )

    update = graph.summary_node({
        "response_errors": [warning],
        "penalty_history": [],
        "dropped_finding_counts": [],
        "dropped_finding_count_index": 0,
    })

    assert update["review_incomplete"] is False
    assert update["passed"] is True
    assert agents.review_incomplete_banner(update) == ""
    assert agents.REVIEW_INCOMPLETE_BANNER not in update["log"]


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
    assert result.get("review_incomplete", False) is False
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

    def fake_call_llm(_system, user, _force_local, **_kwargs):
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


def test_run_pipeline_prints_exact_incomplete_banner_at_final_report(
    tmp_path, monkeypatch, capsys,
):
    project_dir = tmp_path / "project"
    sample_dir = project_dir / "sample"
    sample_dir.mkdir(parents=True)
    (sample_dir / "buggy_sample.py").write_text("print('offline')\n", encoding="utf-8")
    run_dir = project_dir / "runs" / "offline-run"
    run_dir.parent.mkdir()
    run_dir.mkdir()
    warning = agents._llm_unavailable_finding(
        "Bug Agent", "provider=ollama error=TimeoutError"
    )

    class FakeGraph:
        def stream(self, _initial, stream_mode):
            assert stream_mode == "updates"
            yield {
                "summary": {
                    "score": 100,
                    "raw_penalty": 0,
                    "passed": False,
                    "review_incomplete": True,
                    "merged": [warning],
                    "iteration": 0,
                    "log": [],
                }
            }
            yield {"finalize": {"summary": "Tool-only review.", "log": []}}

    monkeypatch.setattr(run_pipeline, "__file__", str(project_dir / "run_pipeline.py"))
    monkeypatch.setattr(run_pipeline, "create_run_directory", lambda _path: run_dir)
    monkeypatch.setattr(run_pipeline, "build_graph", FakeGraph)
    monkeypatch.setattr(
        run_pipeline, "static_analysis_mode", lambda: "direct calls (USE_MCP=0)"
    )

    run_pipeline.main([])

    lines = capsys.readouterr().out.splitlines()
    assert agents.REVIEW_INCOMPLETE_BANNER in lines
