import subprocess

import pytest

import static_analysis
import tools


def test_windows_defaults_to_direct_static_analysis(monkeypatch):
    monkeypatch.delenv("USE_MCP", raising=False)
    monkeypatch.setattr(static_analysis.sys, "platform", "win32")
    monkeypatch.setattr(
        static_analysis.tools,
        "analyze",
        lambda language, code: ("pylint", f"direct:{language}:{code}"),
    )
    monkeypatch.setattr(
        static_analysis.asyncio,
        "run",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("MCP should be disabled by default on Windows")
        ),
    )

    assert static_analysis.run_static_analysis("python", "pass") == (
        "pylint", "direct:python:pass",
    )


def test_timeout_kills_windows_process_tree_and_reaps_process(monkeypatch):
    calls = []

    class FakeProcess:
        def __init__(self, command, **_kwargs):
            self.command = command
            self.pid = 4321 if command[0] == "checker" else 8765
            self.returncode = 0 if command[0] == "taskkill" else -9
            self.communicate_calls = 0
            calls.append(command)

        def communicate(self, timeout=None):
            self.communicate_calls += 1
            if self.command[0] == "checker" and self.communicate_calls == 1:
                raise subprocess.TimeoutExpired(self.command, timeout, output="partial")
            return "", ""

        def kill(self):
            self.returncode = -9

    monkeypatch.setattr(tools.sys, "platform", "win32")
    monkeypatch.setattr(tools, "TIMEOUT", 7)
    monkeypatch.setattr(tools.subprocess, "Popen", FakeProcess)

    with pytest.raises(subprocess.TimeoutExpired):
        tools._run_process(["checker", "input.py"])

    assert calls == [
        ["checker", "input.py"],
        ["taskkill", "/PID", "4321", "/T", "/F"],
    ]


def test_static_tool_timeout_is_reported(monkeypatch):
    monkeypatch.setattr(
        tools,
        "_run_process",
        lambda _command: (_ for _ in ()).throw(
            subprocess.TimeoutExpired(["checker"], 60)
        ),
    )
    monkeypatch.setattr(tools.shutil, "which", lambda _command: "checker.exe")

    assert tools._run(["checker"]) == "[checker timed out after 60s]"


def test_cppcheck_uses_standard_windows_install_path(monkeypatch):
    monkeypatch.setattr(tools.shutil, "which", lambda _cmd: None)
    monkeypatch.setattr(
        tools.os.path,
        "exists",
        lambda path: path == r"C:\Program Files\Cppcheck\cppcheck.exe",
    )

    assert tools._resolve_command(["cppcheck", "--version"]) == [
        r"C:\Program Files\Cppcheck\cppcheck.exe",
        "--version",
    ]
