from pathlib import Path

import graph
from run_pipeline import pipeline_settings


def _make_state(force_local: bool, max_iterations: int = 2):
    return {
        "filename": "sample.py",
        "language": "python",
        "original_code": "print('hi')\n",
        "current_code": "print('hi')\n",
        "force_local": force_local,
        "max_iterations": max_iterations,
        "providers": [],
        "log": [],
        "score_history": [],
        "penalty_history": [],
        "iteration": 0,
    }


def test_force_local_true_exact_serial_order_and_no_overlap(monkeypatch):
    calls = []
    summary_round = {"n": 0}

    def fake_static(state):
        calls.append("static")
        return {"static_tool": "pylint", "static_report": "ok", "iteration": 0, "log": ["static"]}

    def fake_rag(state):
        calls.append("rag")
        return {"context": "ctx", "log": ["rag"]}

    def fake_security(state):
        calls.append("security")
        return {
            "security_findings": [{"line": 1, "severity": "high", "title": "bad", "detail": "bad", "evidence": "reasoning", "agent": "security"}] if state.get("iteration", 0) == 0 else [],
            "providers": ["mock"],
            "log": ["security"],
        }

    def fake_bug(state):
        calls.append("bug")
        return {
            "bug_findings": [{"line": 1, "severity": "medium", "title": "bad", "detail": "bad", "evidence": "reasoning", "agent": "bug"}] if state.get("iteration", 0) == 0 else [],
            "providers": ["mock"],
            "log": ["bug"],
        }

    def fake_summary(state):
        calls.append("summary")
        summary_round["n"] += 1
        passed = summary_round["n"] >= 2
        findings = [] if passed else [{"line": 1, "severity": "high", "title": "bad", "detail": "bad", "evidence": "reasoning", "agent": "security"}]
        return {
            "merged": findings,
            "score": 95 if passed else 60,
            "passed": passed,
            "score_history": [95 if passed else 60],
            "log": ["summary"],
        }

    def fake_coder(state):
        calls.append("coder")
        return {
            "current_code": "print('fixed')\n",
            "iteration": state.get("iteration", 0) + 1,
            "providers": ["mock"],
            "log": ["coder"],
        }

    def fake_finalize(state):
        calls.append("finalize")
        return {"summary": "summary", "log": ["finalize"]}

    monkeypatch.setattr(graph, "static_node", fake_static)
    monkeypatch.setattr(graph, "rag_node", fake_rag)
    monkeypatch.setattr(graph, "finalize_node", fake_finalize)
    monkeypatch.setattr(graph, "run_static_analysis", lambda *_args, **_kwargs: ("pylint", "ok"))
    monkeypatch.setattr(graph.agents, "security_agent", fake_security)
    monkeypatch.setattr(graph.agents, "bug_agent", fake_bug)
    monkeypatch.setattr(graph.agents, "summary_agent", fake_summary)
    monkeypatch.setattr(graph.agents, "coder_agent", fake_coder)
    monkeypatch.setattr(graph.agents, "write_summary_with_provider", lambda state: ("summary", "mock"))

    compiled = graph.build_graph()
    compiled.invoke(_make_state(force_local=True, max_iterations=2))

    assert calls == [
        "static", "rag", "security", "bug", "summary",
        "coder", "security", "bug", "summary", "finalize",
    ]
    assert calls.index("security") < calls.index("bug")
    assert calls[2:4] == ["security", "bug"]
    assert calls[6:8] == ["security", "bug"]


def test_force_local_false_runs_security_and_bug_each_round(monkeypatch):
    calls = []
    summary_round = {"n": 0}

    def fake_static(state):
        calls.append("static")
        return {"static_tool": "pylint", "static_report": "ok", "iteration": 0, "log": ["static"]}

    def fake_rag(state):
        calls.append("rag")
        return {"context": "ctx", "log": ["rag"]}

    def fake_parallel(state):
        calls.append("parallel")
        return state

    def fake_security(state):
        calls.append("security")
        return {"security_findings": [], "providers": ["mock"], "log": ["security"]}

    def fake_bug(state):
        calls.append("bug")
        return {"bug_findings": [], "providers": ["mock"], "log": ["bug"]}

    def fake_summary(state):
        calls.append("summary")
        summary_round["n"] += 1
        passed = summary_round["n"] >= 2
        return {
            "merged": [],
            "score": 95 if passed else 60,
            "passed": passed,
            "score_history": [95 if passed else 60],
            "log": ["summary"],
        }

    def fake_coder(state):
        calls.append("coder")
        return {
            "current_code": "print('fixed')\n",
            "iteration": state.get("iteration", 0) + 1,
            "providers": ["mock"],
            "log": ["coder"],
        }

    def fake_finalize(state):
        calls.append("finalize")
        return {"summary": "summary", "log": ["finalize"]}

    monkeypatch.setattr(graph, "static_node", fake_static)
    monkeypatch.setattr(graph, "rag_node", fake_rag)
    monkeypatch.setattr(graph, "parallel_node", fake_parallel)
    monkeypatch.setattr(graph, "finalize_node", fake_finalize)
    monkeypatch.setattr(graph, "run_static_analysis", lambda *_args, **_kwargs: ("pylint", "ok"))
    monkeypatch.setattr(graph.agents, "security_agent", fake_security)
    monkeypatch.setattr(graph.agents, "bug_agent", fake_bug)
    monkeypatch.setattr(graph.agents, "summary_agent", fake_summary)
    monkeypatch.setattr(graph.agents, "coder_agent", fake_coder)
    monkeypatch.setattr(graph.agents, "write_summary_with_provider", lambda state: ("summary", "mock"))

    compiled = graph.build_graph()
    compiled.invoke(_make_state(force_local=False, max_iterations=2))

    assert calls[:2] == ["static", "rag"]
    assert calls.count("parallel") == 2
    assert calls.count("security") == 2
    assert calls.count("bug") == 2
    assert calls.count("summary") == 2
    assert calls.count("coder") == 1
    assert calls[-1] == "finalize"
    assert calls.index("parallel") < calls.index("security")
    assert calls.index("parallel") < calls.index("bug")


def test_loop_stops_at_max_iterations_even_if_score_never_passes(monkeypatch):
    calls = []

    def fake_static(state):
        calls.append("static")
        return {"static_tool": "pylint", "static_report": "ok", "iteration": 0, "log": ["static"]}

    def fake_rag(state):
        calls.append("rag")
        return {"context": "ctx", "log": ["rag"]}

    def fake_security(state):
        calls.append("security")
        return {"security_findings": [], "providers": ["mock"], "log": ["security"]}

    def fake_bug(state):
        calls.append("bug")
        return {"bug_findings": [], "providers": ["mock"], "log": ["bug"]}

    def fake_summary(state):
        calls.append("summary")
        return {
            "merged": [],
            "score": 60,
            "passed": False,
            "score_history": [60],
            "log": ["summary"],
        }

    def fake_coder(state):
        calls.append("coder")
        return {
            "current_code": "print('fixed')\n",
            "iteration": state.get("iteration", 0) + 1,
            "providers": ["mock"],
            "log": ["coder"],
        }

    def fake_finalize(state):
        calls.append("finalize")
        return {"summary": "summary", "log": ["finalize"]}

    monkeypatch.setattr(graph, "static_node", fake_static)
    monkeypatch.setattr(graph, "rag_node", fake_rag)
    monkeypatch.setattr(graph, "finalize_node", fake_finalize)
    monkeypatch.setattr(graph, "run_static_analysis", lambda *_args, **_kwargs: ("pylint", "ok"))
    monkeypatch.setattr(graph.agents, "security_agent", fake_security)
    monkeypatch.setattr(graph.agents, "bug_agent", fake_bug)
    monkeypatch.setattr(graph.agents, "summary_agent", fake_summary)
    monkeypatch.setattr(graph.agents, "coder_agent", fake_coder)
    monkeypatch.setattr(graph.agents, "write_summary_with_provider", lambda state: ("summary", "mock"))

    compiled = graph.build_graph()
    compiled.invoke(_make_state(force_local=True, max_iterations=1))

    assert calls == [
        "static", "rag", "security", "bug", "summary",
        "coder", "security", "bug", "summary", "finalize",
    ]
    assert calls.count("coder") == 1
    assert calls[-1] == "finalize"


def test_force_local_true_never_passes_coder_runs_exactly_twice(monkeypatch):
    calls = []
    summary_round = {"n": 0}

    def fake_static(state):
        calls.append("static")
        return {"static_tool": "pylint", "static_report": "ok", "iteration": 0, "log": ["static"]}

    def fake_rag(state):
        calls.append("rag")
        return {"context": "ctx", "log": ["rag"]}

    def fake_security(state):
        calls.append("security")
        return {"security_findings": [], "providers": ["mock"], "log": ["security"]}

    def fake_bug(state):
        calls.append("bug")
        return {"bug_findings": [], "providers": ["mock"], "log": ["bug"]}

    def fake_summary(state):
        calls.append("summary")
        summary_round["n"] += 1
        score = 40 + 10 * summary_round["n"]
        return {
            "merged": [{"severity": "critical"}] * (4 - summary_round["n"]),
            "score": score,
            "passed": False,
            "score_history": [score],
            "log": ["summary"],
        }

    def fake_coder(state):
        calls.append("coder")
        return {
            "current_code": "print('fixed')\n",
            "iteration": state.get("iteration", 0) + 1,
            "providers": ["mock"],
            "log": ["coder"],
        }

    def fake_finalize(state):
        calls.append("finalize")
        return {"summary": "summary", "log": ["finalize"]}

    monkeypatch.setattr(graph, "static_node", fake_static)
    monkeypatch.setattr(graph, "rag_node", fake_rag)
    monkeypatch.setattr(graph, "finalize_node", fake_finalize)
    monkeypatch.setattr(graph, "run_static_analysis", lambda *_args, **_kwargs: ("pylint", "ok"))
    monkeypatch.setattr(graph.agents, "security_agent", fake_security)
    monkeypatch.setattr(graph.agents, "bug_agent", fake_bug)
    monkeypatch.setattr(graph.agents, "summary_agent", fake_summary)
    monkeypatch.setattr(graph.agents, "coder_agent", fake_coder)
    monkeypatch.setattr(graph.agents, "write_summary_with_provider", lambda state: ("summary", "mock"))

    compiled = graph.build_graph()
    compiled.invoke(_make_state(force_local=True, max_iterations=2))

    assert calls.count("coder") == 2
    assert calls[-1] == "finalize"


def test_summary_records_new_unverified_drop_count_per_round(monkeypatch):
    monkeypatch.setattr(
        graph.agents,
        "summary_agent",
        lambda _state: {
            "score": 100,
            "passed": True,
            "merged": [],
            "score_history": [100],
            "log": [],
        },
    )
    state = {
        "penalty_history": [],
        "dropped_finding_counts": [2, 1],
        "dropped_finding_count_index": 0,
    }

    first = graph.summary_node(state)
    assert first["dropped_unverified_count"] == 3
    assert first["dropped_unverified_history"] == [3]
    assert first["dropped_finding_count_index"] == 2

    second = graph.summary_node({
        **state,
        "penalty_history": [0],
        "dropped_finding_counts": [2, 1, 4],
        "dropped_finding_count_index": 2,
    })
    assert second["dropped_unverified_count"] == 4
    assert second["dropped_unverified_history"] == [4]


def test_first_round_does_not_stop_without_previous_penalty(monkeypatch):
    monkeypatch.setattr(
        graph.agents,
        "summary_agent",
        lambda _state: {
            "score": 55,
            "passed": False,
            "merged": [],
            "score_history": [55],
            "log": ["summary"],
        },
    )
    state = {
        "score_history": [],
        "penalty_history": [],
        "force_local": True,
        "max_iterations": 3,
        "iteration": 0,
        "passed": False,
    }

    update = graph.summary_node(state)
    next_state = {**state, **update}

    assert update["stop_no_improvement"] is False
    assert not any("stopping early" in line for line in update["log"])
    assert graph.route_after_summary(next_state) == "coder"


def test_force_local_false_defaults_to_three_iterations_in_all_entry_points():
    cloud_state = {"passed": False, "iteration": 2, "force_local": False}
    assert graph.route_after_summary(cloud_state) == "coder"
    cloud_state["iteration"] = 3
    assert graph.route_after_summary(cloud_state) == "finalize"

    project_root = Path(__file__).resolve().parents[1]
    app_source = (project_root / "app.py").read_text(encoding="utf-8")
    assert "default_rounds = 2 if force_local else MAX_ITERATIONS" in app_source
    assert pipeline_settings(cloud=True)["max_iterations"] == 3


def test_early_stop_uses_raw_penalty_and_respects_round_cap(monkeypatch):
    def findings_for_penalty(penalty):
        severities = []
        while penalty >= 40:
            severities.append({"severity": "critical"})
            penalty -= 40
        while penalty >= 20:
            severities.append({"severity": "high"})
            penalty -= 20
        while penalty >= 8:
            severities.append({"severity": "medium"})
            penalty -= 8
        while penalty >= 2:
            severities.append({"severity": "low"})
            penalty -= 2
        assert penalty == 0
        return severities

    for previous, current, expected_route in (
        (120, 100, "coder"),
        (100, 120, "finalize"),
        (100, 100, "finalize"),
    ):
        monkeypatch.setattr(
            graph.agents,
            "summary_agent",
            lambda _state, penalty=current: {
                "merged": findings_for_penalty(penalty),
                "score": 0,
                "passed": False,
                "score_history": [0],
                "log": ["summary"],
            },
        )
        state = {
            "score_history": [0],
            "penalty_history": [previous],
            "force_local": True,
            "max_iterations": 3,
            "iteration": 0,
            "passed": False,
        }

        update = graph.summary_node(state)
        next_state = {**state, **update}

        assert update["score"] == 0
        assert update["raw_penalty"] == current
        assert update["penalty_history"] == [current]
        assert graph.route_after_summary(next_state) == expected_route

    monkeypatch.setattr(
        graph.agents,
        "summary_agent",
        lambda _state: {
            "merged": findings_for_penalty(100),
            "score": 0,
            "passed": False,
            "score_history": [0],
            "log": ["summary"],
        },
    )
    capped_state = {
        "score_history": [0],
        "penalty_history": [120],
        "force_local": True,
        "max_iterations": 2,
        "iteration": 2,
        "passed": False,
    }
    capped_update = graph.summary_node(capped_state)
    capped_next_state = {**capped_state, **capped_update}
    assert capped_update["stop_no_improvement"] is False
    assert graph.route_after_summary(capped_next_state) == "finalize"


def test_stops_after_raw_penalty_fails_to_improve(monkeypatch):
    calls = []
    score_round = {"n": 0}

    def fake_static(state):
        calls.append("static")
        return {"static_tool": "pylint", "static_report": "ok", "iteration": 0, "log": ["static"]}

    def fake_rag(state):
        calls.append("rag")
        return {"context": "ctx", "log": ["rag"]}

    def fake_security(state):
        calls.append("security")
        return {"security_findings": [], "providers": ["mock"], "log": ["security"]}

    def fake_bug(state):
        calls.append("bug")
        return {"bug_findings": [], "providers": ["mock"], "log": ["bug"]}

    def fake_summary(state):
        calls.append("summary")
        score_round["n"] += 1
        score = 70
        return {
            "merged": [],
            "score": score,
            "passed": False,
            "score_history": [score],
            "log": ["summary"],
        }

    def fake_coder(state):
        calls.append("coder")
        return {
            "current_code": "print('fixed')\n",
            "iteration": state.get("iteration", 0) + 1,
            "providers": ["mock"],
            "log": ["coder"],
        }

    def fake_finalize(state):
        calls.append("finalize")
        return {"summary": "summary", "log": ["finalize"]}

    monkeypatch.setattr(graph, "static_node", fake_static)
    monkeypatch.setattr(graph, "rag_node", fake_rag)
    monkeypatch.setattr(graph, "finalize_node", fake_finalize)
    monkeypatch.setattr(graph, "run_static_analysis", lambda *_args, **_kwargs: ("pylint", "ok"))
    monkeypatch.setattr(graph.agents, "security_agent", fake_security)
    monkeypatch.setattr(graph.agents, "bug_agent", fake_bug)
    monkeypatch.setattr(graph.agents, "summary_agent", fake_summary)
    monkeypatch.setattr(graph.agents, "coder_agent", fake_coder)
    monkeypatch.setattr(graph.agents, "write_summary_with_provider", lambda state: ("summary", "mock"))

    result = graph.build_graph().invoke(_make_state(force_local=True, max_iterations=3))

    assert calls.count("summary") == 2
    assert calls.count("coder") == 1
    assert calls[-1] == "finalize"
    assert any("Raw penalty did not improve (0 -> 0)" in line for line in result["log"])


def test_finalize_logs_summary_llm_provider(monkeypatch):
    monkeypatch.setattr(
        graph.agents,
        "write_summary_with_provider",
        lambda _state: ("summary", "gemini"),
    )

    result = graph.finalize_node({})

    assert result["summary"] == "summary"
    assert "Summary report generated via gemini" in result["log"]


def test_cloud_three_rounds_record_only_summary_scores_and_findings(monkeypatch):
    calls = {"security": 0, "bug": 0, "summary": 0}
    security_finding = {
        "line": 1, "severity": "medium", "title": "Security finding",
        "detail": "example", "evidence": "tool", "agent": "security-agent",
    }
    bug_finding = {
        "line": 2, "severity": "low", "title": "Bug finding",
        "detail": "example", "evidence": "tool", "agent": "bug-agent",
    }

    monkeypatch.setattr(
        graph, "static_node",
        lambda _state: {"static_tool": "none", "static_report": "ok", "iteration": 0, "log": []},
    )
    monkeypatch.setattr(graph, "rag_node", lambda _state: {"context": "ctx", "log": []})
    monkeypatch.setattr(graph, "finalize_node", lambda _state: {"summary": "done", "log": []})

    def fake_security(_state):
        calls["security"] += 1
        return {
            "security_findings": [security_finding],
            "providers": ["mock"],
            "log": ["security emitted"],
        }

    def fake_bug(_state):
        calls["bug"] += 1
        return {
            "bug_findings": [bug_finding],
            "providers": ["mock"],
            "log": ["bug emitted"],
        }

    def fake_summary(state):
        calls["summary"] += 1
        score = 40 + (calls["summary"] * 10)
        findings = [
            dict(security_finding),
            dict(bug_finding),
        ]
        severities = {
            1: ("critical", "high"),
            2: ("high", "medium"),
            3: ("medium", "low"),
        }[calls["summary"]]
        for finding, severity in zip(findings, severities):
            finding["severity"] = severity
        return {
            "merged": findings,
            "score": score,
            "passed": False,
            "score_history": [score],
            "log": [f"summary {calls['summary']}"],
        }

    def fake_coder(state):
        return {
            "current_code": state["current_code"],
            "iteration": state.get("iteration", 0) + 1,
            "providers": ["mock"],
            "log": ["coder emitted"],
        }

    monkeypatch.setattr(graph.agents, "security_agent", fake_security)
    monkeypatch.setattr(graph.agents, "bug_agent", fake_bug)
    monkeypatch.setattr(graph.agents, "summary_agent", fake_summary)
    monkeypatch.setattr(graph.agents, "coder_agent", fake_coder)
    monkeypatch.setattr(graph, "run_static_analysis", lambda *_args: ("none", "ok"))

    initial = _make_state(force_local=False, max_iterations=2)
    updates = list(graph.build_graph().stream(initial, stream_mode="updates"))
    summary_updates = [update["summary"] for update in updates if "summary" in update]
    parallel_updates = [update["parallel"] for update in updates if "parallel" in update]

    assert len(summary_updates) == 3
    assert [update["score"] for update in summary_updates] == [50, 60, 70]
    assert all(not update or "score" not in update for update in parallel_updates)
    assert all(
        not update or "security_findings" not in update and "bug_findings" not in update
        for update in parallel_updates
    )

    calls.update(security=0, bug=0, summary=0)
    result = graph.build_graph().invoke(initial)

    assert result["score_history"] == [50, 60, 70]
    assert calls == {"security": 3, "bug": 3, "summary": 3}
    assert result["merged"] == [security_finding, bug_finding]
    assert result["log"].count("security emitted") == 3
    assert result["log"].count("bug emitted") == 3
