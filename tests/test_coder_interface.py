import agents


def _state(original_code):
    return {
        "original_code": original_code,
        "current_code": original_code,
        "language": "python",
        "merged": [{"line": 1, "severity": "high", "title": "Example", "detail": "Fix"}],
        "force_local": True,
        "iteration": 0,
    }


def _mock_coder(monkeypatch, fixed_code):
    monkeypatch.setattr(
        agents,
        "call_llm",
        lambda *_args, **_kwargs: (f"```python\n{fixed_code}\n```", "ollama"),
    )


def test_coder_reports_removed_function_and_changed_parameters(monkeypatch):
    original = "def keep(left, right):\n    return left + right\n\ndef removed(value):\n    return value\n"
    fixed = "def keep(left):\n    return left\n"
    _mock_coder(monkeypatch, fixed)

    result = agents.coder_agent(_state(original))

    assert len(result["interface_findings"]) == 1
    finding = result["interface_findings"][0]
    assert finding["severity"] == "medium"
    assert finding["evidence"] == "tool"
    assert finding["agent"] == "interface-check"
    assert "keep" in finding["detail"]
    assert "removed" in finding["detail"]
    assert any("interface changed" in line.lower() for line in result["log"])


def test_interface_finding_is_merged_without_code_verification(monkeypatch):
    original = "def compute(value):\n    return value\n"
    _mock_coder(monkeypatch, "def compute():\n    return 1\n")
    coder_result = agents.coder_agent(_state(original))

    summary = agents.summary_agent({
        "security_findings": [],
        "bug_findings": [],
        "interface_findings": coder_result["interface_findings"],
    })

    assert summary["merged"] == coder_result["interface_findings"]
    assert summary["merged"][0]["agent"] == "interface-check"


def test_interface_check_skips_when_original_or_fixed_python_does_not_parse(monkeypatch):
    cases = [
        ("def broken(:\n    pass\n", "def fixed():\n    pass\n", "original"),
        ("def original():\n    pass\n", "def broken(:\n    pass\n", "fixed"),
    ]

    for original, fixed, which in cases:
        _mock_coder(monkeypatch, fixed)
        result = agents.coder_agent(_state(original))

        assert result["interface_findings"] == []
        assert any(f"{which} Python code does not parse" in line for line in result["log"])


def test_coder_prompt_preserves_interfaces_behavior_and_secret_policy():
    assert "do not change function signatures" in agents.CODER_SYSTEM.lower()
    assert "return types" in agents.CODER_SYSTEM.lower()
    assert "exception types" in agents.CODER_SYSTEM.lower()
    assert "# review:" in agents.CODER_SYSTEM.lower()
    assert 'os.environ["name"]' in agents.CODER_SYSTEM.lower()
    assert "shlex.split" in agents.CODER_SYSTEM
    assert "same object type" in agents.CODER_SYSTEM