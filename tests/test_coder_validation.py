import graph
import agents


ORIGINAL_CODE = 'def get_key():\n    return "old"\n'
INVALID_FIX = 'def get_key():\n    return os.environ["API_KEY"]\n'
VALID_FIX = 'import os\n\ndef get_key():\n    return os.environ["API_KEY"]\n'


def _mock_validation_environment(monkeypatch, candidates):
    calls = []

    def fake_coder(state):
        calls.append(dict(state))
        candidate = candidates.pop(0)
        return {
            "current_code": candidate,
            "iteration": state.get("iteration", 0) + 1,
            "providers": ["mock"],
            "interface_findings": [],
            "log": [
                "Coder Agent applied fixes "
                f"(round {state.get('iteration', 0) + 1}) via mock; re-checking"
            ],
        }

    def fake_pylint(code):
        if "os.environ" in code and "import os" not in code:
            return ["E0602:Undefined variable 'os'"]
        return []

    monkeypatch.setattr(graph.agents, "coder_agent", fake_coder)
    monkeypatch.setattr(graph, "run_pylint_errors_only", fake_pylint)
    monkeypatch.setattr(graph, "run_static_analysis", lambda *_args: ("pylint", "ok"))
    monkeypatch.setattr(graph, "scan_secrets", lambda _code: [])
    return calls


def _state():
    return {
        "language": "python",
        "current_code": ORIGINAL_CODE,
        "original_code": ORIGINAL_CODE,
        "merged": [],
        "iteration": 0,
    }


def test_missing_os_import_fix_is_retried_and_valid_retry_is_accepted(monkeypatch):
    calls = _mock_validation_environment(monkeypatch, [INVALID_FIX, VALID_FIX])

    update = graph.coder_node(_state())

    assert len(calls) == 2
    assert calls[1]["current_code"] == INVALID_FIX
    assert calls[1]["validation_errors"] == ["E0602:Undefined variable 'os'"]
    assert update["current_code"] == VALID_FIX
    assert update["interface_findings"] == []
    assert sum(
        line.startswith("Coder Agent applied fixes (round ")
        for line in update["log"]
    ) == 1


def test_valid_fix_is_accepted_without_retry(monkeypatch):
    calls = _mock_validation_environment(monkeypatch, [VALID_FIX])

    update = graph.coder_node(_state())

    assert len(calls) == 1
    assert update["current_code"] == VALID_FIX
    assert update["interface_findings"] == []


def test_python_parse_failure_is_retried(monkeypatch):
    invalid_syntax = "def get_key(:\n    return 'broken'\n"
    calls = _mock_validation_environment(monkeypatch, [invalid_syntax, VALID_FIX])

    update = graph.coder_node(_state())

    assert len(calls) == 2
    assert calls[1]["validation_errors"] == [
        "SyntaxError: invalid syntax (line 1)",
    ]
    assert update["current_code"] == VALID_FIX


def test_two_invalid_attempts_keep_previous_code_and_report_rejection(monkeypatch):
    _mock_validation_environment(monkeypatch, [INVALID_FIX, INVALID_FIX])

    update = graph.coder_node(_state())

    assert update["current_code"] == ORIGINAL_CODE
    rejected = [
        finding for finding in update["interface_findings"]
        if finding["title"] == "Coder Agent fix rejected"
    ]
    assert len(rejected) == 1
    assert rejected[0]["severity"] == "high"
    assert "E0602:Undefined variable 'os'" in rejected[0]["detail"]
    assert any("fix rejected after retry" in line for line in update["log"])
    assert not any(
        line.startswith("Coder Agent applied fixes (round ")
        for line in update["log"]
    )


def test_validation_errors_are_appended_to_coder_prompt(monkeypatch):
    captured = {}

    def fake_call_llm(_system, user, _force_local):
        captured["user"] = user
        return f"```python\n{ORIGINAL_CODE}```", "mock"

    monkeypatch.setattr(agents, "call_llm", fake_call_llm)
    state = {
        **_state(),
        "validation_errors": ["E0602:Undefined variable 'os'"],
    }

    agents.coder_agent(state)

    assert "VALIDATION ERRORS (exact messages):" in captured["user"]
    assert "E0602:Undefined variable 'os'" in captured["user"]
    assert "Fix these errors" in captured["user"]
