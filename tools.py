"""Deterministic static-analysis tools (green 'Static Analysis' box).

Plain functions. mcp_server.py exposes them over MCP; static_analysis.py calls
them either through MCP or directly.
"""
import os
import re
import shutil
import subprocess
import sys
import tempfile

TIMEOUT = 60

_SENSITIVE_NAME_RE = re.compile(r"key|secret|token|password|passwd|credential", re.I)
_KEY_TOKEN_NAME_RE = re.compile(r"key|token", re.I)
_IGNORED_NAME_SUFFIXES = ("_type", "_field", "_name", "_id", "_url", "_path")
_SENSITIVE_ASSIGNMENT_RE = re.compile(
    r'''^\s*(?P<name>[A-Za-z_]\w*)\s*(?::[^=\n]+)?\s*=\s*'''
    r'''(?:[rRuUbBfF]{0,2})?(?P<quote>''' + "'''|\\\"\\\"\\\"|'|\\\"" + r''')'''
    r'''(?P<value>.*?)(?P=quote)'''
)
_ENV_FALLBACK_RE = re.compile(
    r'''\bos\.(?:getenv|environ\.get)\s*\(\s*'''
    r'''(?P<key>['"])(?P<name>[^'"]+)(?P=key)\s*,\s*'''
    r'''(?P<quote>['"])(?P<value>.*?)(?P=quote)\s*\)'''
)
_SECRET_VALUE_PATTERNS = (
    ("OpenAI-style API key", re.compile(r"\bsk-[A-Za-z0-9_-]{8,}\b")),
    ("AWS access key", re.compile(r"\bAKIA[0-9A-Z]{16}\b", re.I)),
    ("private key", re.compile(r"-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY(?:-----)?")),
)
_PLACEHOLDERS = {
    "changeme", "change_me", "change-me", "placeholder", "your_api_key",
    "your-api-key", "example", "dummy", "test", "none", "null", "redacted", "default",
}


def _is_placeholder(value: str, name: str = "") -> bool:
    normalized = value.strip().strip("'\"").strip().lower()
    return (
        not normalized
        or normalized in _PLACEHOLDERS
        or normalized == name.lower()
        or bool(re.fullmatch(r"(?:x{3,}|\*{3,}|<[^>]+>)", normalized))
    )


def _matches_known_secret(value: str) -> bool:
    return any(pattern.search(value) for _, pattern in _SECRET_VALUE_PATTERNS)


def _is_plain_word(value: str) -> bool:
    return (
        bool(re.fullmatch(r"[A-Za-z]+", value))
        and (value.islower() or value.isupper())
    )


def _is_sensitive_literal(value: str, name: str) -> bool:
    if _is_placeholder(value, name):
        return False
    if _matches_known_secret(value):
        return True
    if _KEY_TOKEN_NAME_RE.search(name):
        return len(value) >= 8 and not _is_plain_word(value)
    return True


def scan_secrets(code: str) -> list[dict]:
    """Find hard-coded credentials deterministically without returning their values."""
    findings_by_line: dict[int, dict] = {}

    def add_finding(line: int, title: str, detail: str) -> None:
        findings_by_line.setdefault(line, {
            "line": line,
            "severity": "high",
            "title": title,
            "detail": detail,
            "evidence": "tool",
            "agent": "secrets-scan",
        })

    for line_number, line in enumerate(code.splitlines(), 1):
        assignment = _SENSITIVE_ASSIGNMENT_RE.match(line)
        assigned_name = assignment.group("name") if assignment else ""
        for match in _ENV_FALLBACK_RE.finditer(line):
            name = match.group("name")
            value = match.group("value")
            sensitive_name = assigned_name or name
            if (
                (_SENSITIVE_NAME_RE.search(assigned_name) or _SENSITIVE_NAME_RE.search(name))
                and not _is_placeholder(value, sensitive_name)
            ):
                add_finding(
                    line_number,
                    "Hard-coded fallback for a secret",
                    "A non-placeholder string fallback is supplied to an environment lookup. "
                    "Remove the fallback and configure the secret outside source code.",
                )

        if not assignment:
            continue
        name = assignment.group("name")
        value = assignment.group("value")
        if (
            _SENSITIVE_NAME_RE.search(name)
            and not name.lower().endswith(_IGNORED_NAME_SUFFIXES)
            and _is_sensitive_literal(value, name)
        ):
            add_finding(
                line_number,
                f"Hard-coded secret in {name}",
                f"A string literal is assigned to sensitive name '{name}'. "
                "Load credentials from an environment variable or secret manager.",
            )

    for pattern_name, pattern in _SECRET_VALUE_PATTERNS:
        for match in pattern.finditer(code):
            line_number = code.count("\n", 0, match.start()) + 1
            if _is_placeholder(match.group(0)):
                continue
            add_finding(
                line_number,
                f"Hard-coded {pattern_name}",
                f"A string literal matches a known {pattern_name} format. "
                "Remove it from source and rotate it if it is active.",
            )

    return [findings_by_line[line] for line in sorted(findings_by_line)]


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
