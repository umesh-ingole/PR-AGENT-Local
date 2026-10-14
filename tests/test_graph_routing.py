from pathlib import Path

import graph


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
    monkeypatch.setattr(graph.agents, "write_summary", lambda state: "summary")

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
    monkeypatch.setattr(graph.agents, "write_summary", lambda state: "summary")

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
    monkeypatch.setattr(graph.agents, "write_summary", lambda state: "summary")

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
    monkeypatch.setattr(graph.agents, "write_summary", lambda state: "summary")

    compiled = graph.build_graph()
    compiled.invoke(_make_state(force_local=True, max_iterations=2))

    assert calls.count("coder") == 2
    assert calls[-1] == "finalize"


def test_first_round_does_not_stop_without_previous_score(monkeypatch):
    monkeypatch.setattr(
        graph.agents,
        "summary_agent",
        lambda _state: {
            "score": 55,
            "passed": False,
            "score_history": [55],
            "log": ["summary"],
        },
    )
    state = {
        "score_history": [],
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
    runner_source = (project_root / "run_pipeline.py").read_text(encoding="utf-8")
    assert "default_rounds = 2 if force_local else MAX_ITERATIONS" in app_source
    assert '"max_iterations": 2 if force_local else MAX_ITERATIONS' in runner_source


def test_stops_after_score_fails_to_improve(monkeypatch):
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
    monkeypatch.setattr(graph.agents, "write_summary", lambda state: "summary")

    result = graph.build_graph().invoke(_make_state(force_local=True, max_iterations=3))

    assert calls.count("summary") == 2
    assert calls.count("coder") == 1
    assert calls[-1] == "finalize"
    assert any("Score did not improve (70 -> 70)" in line for line in result["log"])
