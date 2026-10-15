from pathlib import Path

import agents
import graph
from tools import scan_python_patterns


def test_sample_detects_requested_python_patterns():
    source_path = Path(__file__).resolve().parents[1] / "sample" / "buggy_sample.py"
    source = source_path.read_text(encoding="utf-8")

    findings = scan_python_patterns(source)

    assert {
        "Mutable default argument",
        "Exception silently ignored",
        "Use of eval()",
        "Shell enabled in subprocess call",
        "File opened outside a with statement",
    } <= {finding["title"] for finding in findings}
    assert all(finding["code"] == source.splitlines()[finding["line"] - 1] for finding in findings)
    assert all(finding["evidence"] == "tool" for finding in findings)
    assert all(finding["agent"] == "ast-scan" for finding in findings)


def test_clean_python_code_has_no_ast_pattern_findings():
    source = (
        "def read(path):\n"
        "    with open(path) as stream:\n"
        "        return stream.read()\n"
        "\n"
        "def parse(text, options=None):\n"
        "    if options is None:\n"
        "        options = {}\n"
        "    return text\n"
    )

    assert scan_python_patterns(source) == []


def test_mutable_default_literals_and_constructors_are_detected():
    source = (
        "def one(value=[]):\n    return value\n"
        "def two(value={}):\n    return value\n"
        "def three(value=set()):\n    return value\n"
        "def four(value=dict()):\n    return value\n"
    )

    findings = scan_python_patterns(source)

    assert [finding["line"] for finding in findings] == [1, 3, 5, 7]
    assert all(finding["title"] == "Mutable default argument" for finding in findings)


def test_open_inside_with_is_allowed_but_nested_function_open_is_reported():
    source = (
        "with open('file.txt') as stream:\n"
        "    data = stream.read()\n"
        "with open('other.txt') as stream:\n"
        "    def later():\n"
        "        return open('unmanaged.txt')\n"
    )

    findings = scan_python_patterns(source)

    assert len(findings) == 1
    assert findings[0]["line"] == 5


def test_scanner_detects_all_supported_call_patterns():
    source = (
        "exec(source)\n"
        "os.system(command)\n"
        "pickle.load(stream)\n"
        "pickle.loads(payload)\n"
        "yaml.load(payload)\n"
        "yaml.load(payload, Loader=SafeLoader)\n"
        "yaml.load(payload, SafeLoader)\n"
        "subprocess.Popen(command, shell=True)\n"
        "try:\n"
        "    do_work()\n"
        "except ValueError:\n"
        "    handle_error()\n"
    )

    findings = scan_python_patterns(source)

    assert [finding["line"] for finding in findings] == [1, 2, 3, 4, 5, 8]
    assert [finding["severity"] for finding in findings] == [
        "critical", "high", "high", "high", "high", "high",
    ]


def test_static_node_adds_ast_findings_to_report_and_summary_without_model(
    monkeypatch,
):
    source = "def work(items=[]):\n    return items\n"
    monkeypatch.setattr(graph, "run_static_analysis", lambda *_args: ("pylint", ""))

    initial = graph.static_node({"language": "python", "current_code": source})

    assert any(item["agent"] == "ast-scan" for item in initial["ast_findings"])
    assert "Mutable default argument" in initial["static_report"]
    monkeypatch.setattr(
        agents,
        "call_llm",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("summary must not call the model")
        ),
    )
    summary = agents.summary_agent({
        "security_findings": [],
        "bug_findings": [],
        "secret_findings": [],
        "ast_findings": initial["ast_findings"],
        "interface_findings": [],
    })
    assert summary["merged"] == initial["ast_findings"]


def test_coder_node_refreshes_python_ast_findings(monkeypatch):
    source = "def work(items=[]):\n    return items\n"
    monkeypatch.setattr(graph, "run_static_analysis", lambda *_args: ("pylint", ""))
    monkeypatch.setattr(graph, "run_pylint_errors_only", lambda _code: [])
    monkeypatch.setattr(
        graph.agents,
        "coder_agent",
        lambda _state: {
            "current_code": source,
            "iteration": 1,
            "providers": [],
            "log": [],
        },
    )

    update = graph.coder_node({
        "language": "python",
        "current_code": source,
        "original_code": source,
    })

    assert any(
        finding["title"] == "Mutable default argument"
        for finding in update["ast_findings"]
    )


def test_ast_finding_wins_dedupe_tie_with_tool_evidence():
    ast_finding = scan_python_patterns("eval(value)\n")[0]
    model_finding = {
        **ast_finding,
        "agent": "security-agent",
        "evidence": "reasoning",
    }

    merged, _score, _passed = agents.merge_and_score(
        [model_finding], [], [ast_finding]
    )

    assert len(merged) == 1
    assert merged[0]["agent"] == "ast-scan"
    assert merged[0]["evidence"] == "tool"


def test_coder_prompt_includes_low_severity_ast_and_secret_findings(monkeypatch):
    captured = {}
    findings = [
        {
            "line": 1,
            "severity": "low",
            "title": "AST finding",
            "detail": "Fix AST issue",
            "agent": "ast-scan",
        },
        {
            "line": 2,
            "severity": "low",
            "title": "Secret finding",
            "detail": "Fix secret issue",
            "agent": "secrets-scan",
        },
        {
            "line": 3,
            "severity": "low",
            "title": "Other low finding",
            "detail": "Not selected",
            "agent": "security-agent",
        },
    ]

    def fake_call_llm(_system, user, _force_local):
        captured["user"] = user
        return "```python\npass\n```", "mock"

    monkeypatch.setattr(agents, "call_llm", fake_call_llm)
    monkeypatch.setattr(agents, "_check_function_interfaces", lambda *_args: ([], []))

    agents.coder_agent({
        "language": "python",
        "current_code": "pass\n",
        "merged": findings,
        "force_local": True,
    })

    assert "AST finding" in captured["user"]
    assert "Secret finding" in captured["user"]
    assert "Other low finding" not in captured["user"]
    assert "mutable default, use None" in agents.CODER_SYSTEM
    assert "catch the specific exceptions" in agents.CODER_SYSTEM
