"""Deterministic static-analysis tools (green 'Static Analysis' box).

Plain functions. mcp_server.py exposes them over MCP; static_analysis.py calls
them either through MCP or directly.
"""
import os
import shutil
import subprocess
import sys
import tempfile

TIMEOUT = 60


def _run(cmd: list[str]) -> str:
    if shutil.which(cmd[0]) is None and cmd[0] != sys.executable:
        return f"[{cmd[0]} is not installed on this machine - skipped]"
    try:
        p = subprocess.run(cmd, capture_output=True, text=True, timeout=TIMEOUT)
    except subprocess.TimeoutExpired:
        return f"[{cmd[0]} timed out after {TIMEOUT}s]"
    return ((p.stdout or "") + (p.stderr or "")).strip() or "[no issues reported]"


def _write_temp(code: str, suffix: str) -> str:
    fd, path = tempfile.mkstemp(suffix=suffix)
    with os.fdopen(fd, "w", encoding="utf-8") as f:
        f.write(code)
    return path


def _with_file(code: str, suffix: str, build_cmd) -> str:
    path = _write_temp(code, suffix)
    try:
        return _run(build_cmd(path)).replace(path, "<file>")
    finally:
        os.remove(path)


def run_cppcheck(code: str, cpp: bool = False) -> str:
    suffix = ".cpp" if cpp else ".c"
    return _with_file(
        code, suffix,
        lambda p: ["cppcheck", "--enable=warning,style,performance,portability",
                   "--template={line}:{severity}:{id}:{message}", "--quiet", p],
    )


def run_pylint(code: str) -> str:
    return _with_file(
        code, ".py",
        lambda p: [sys.executable, "-m", "pylint", "--score=n",
                   "--msg-template={line}:{category}:{symbol}:{msg}", p],
    )


def run_checkstyle(code: str) -> str:
    # Checkstyle needs the public class name to match the file name; a temp
    # name will raise one extra warning about that. It is safe to ignore.
    return _with_file(
        code, ".java",
        lambda p: ["checkstyle", "-c", "/google_checks.xml", p],
    )


def run_rubocop(code: str) -> str:
    return _with_file(
        code, ".rb",
        lambda p: ["rubocop", "--format", "emacs", p],
    )


def analyze(language: str, code: str) -> tuple[str, str]:
    """Return (tool_name, raw_output) for the language."""
    if language == "c":
        return "cppcheck", run_cppcheck(code, cpp=False)
    if language == "cpp":
        return "cppcheck", run_cppcheck(code, cpp=True)
    if language == "python":
        return "pylint", run_pylint(code)
    if language == "java":
        return "checkstyle", run_checkstyle(code)
    if language == "ruby":
        return "rubocop", run_rubocop(code)
    return "none", "[no static analyzer configured for this language]"
