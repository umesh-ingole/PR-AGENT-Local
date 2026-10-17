"""Deterministic static-analysis tools (green 'Static Analysis' box).

Plain functions. mcp_server.py exposes them over MCP; static_analysis.py calls
them either through MCP or directly.
"""
import ast
import os
import re
import shutil
import subprocess
import sys
import tempfile

TIMEOUT = 60
_COMMON_WINDOWS_TOOL_PATHS = {
    "cppcheck": (
        r"C:\Program Files\Cppcheck\cppcheck.exe",
        r"C:\Program Files (x86)\Cppcheck\cppcheck.exe",
    )
}

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


def scan_python_patterns(code: str) -> list[dict]:
    """Find selected Python code patterns using syntax only, without inference."""
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return []

    source_lines = code.splitlines()
    findings: dict[tuple[int, str], dict] = {}

    def add(node: ast.AST, severity: str, title: str, detail: str) -> None:
        line = getattr(node, "lineno", None)
        if not isinstance(line, int) or not 1 <= line <= len(source_lines):
            return
        findings.setdefault((line, title), {
            "line": line,
            "severity": severity,
            "title": title,
            "detail": detail,
            "code": source_lines[line - 1],
            "evidence": "tool",
            "agent": "ast-scan",
        })

    def is_mutable_default(node: ast.expr) -> bool:
        return (
            isinstance(node, (ast.List, ast.Dict, ast.Set, ast.ListComp,
                              ast.DictComp, ast.SetComp))
            or (
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Name)
                and node.func.id in {"list", "dict", "set"}
            )
        )

    def is_pass_only(handler: ast.ExceptHandler) -> bool:
        return bool(handler.body) and all(
            isinstance(statement, ast.Pass) for statement in handler.body
        )

    def is_open_call(node: ast.Call) -> bool:
        if isinstance(node.func, ast.Name) and node.func.id == "open":
            return True
        if not isinstance(node.func, ast.Attribute) or node.func.attr != "open":
            return False
        path_factory = node.func.value
        factory = path_factory.func if isinstance(path_factory, ast.Call) else None
        return (
            isinstance(factory, ast.Name)
            and factory.id == "Path"
            or isinstance(factory, ast.Attribute)
            and factory.attr == "Path"
        )

    def is_inside_with(node: ast.AST) -> bool:
        parent = parents.get(node)
        while parent is not None:
            if isinstance(parent, (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda)):
                return False
            if isinstance(parent, (ast.With, ast.AsyncWith)):
                return True
            parent = parents.get(parent)
        return False

    parents: dict[ast.AST, ast.AST] = {}
    for parent in ast.walk(tree):
        for child in ast.iter_child_nodes(parent):
            parents[child] = parent

    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            defaults = node.args.defaults + [
                default for default in node.args.kw_defaults if default is not None
            ]
            if any(is_mutable_default(default) for default in defaults):
                add(
                    node,
                    "high",
                    "Mutable default argument",
                    "A mutable object is used as a function default and may be shared "
                    "between calls. Use None and create the object inside the function.",
                )
        elif isinstance(node, ast.ExceptHandler):
            if (node.type is None or (
                isinstance(node.type, ast.Name) and node.type.id == "Exception"
            )) and is_pass_only(node):
                add(
                    node,
                    "medium",
                    "Exception silently ignored",
                    "A bare except or except Exception handler only passes. Catch the "
                    "specific exceptions this code can raise.",
                )
        elif isinstance(node, ast.Call):
            if isinstance(node.func, ast.Name) and node.func.id in {"eval", "exec"}:
                add(
                    node,
                    "critical",
                    f"Use of {node.func.id}()",
                    f"{node.func.id}() executes dynamically supplied code. Use a "
                    "constrained parser or an explicitly reviewed alternative.",
                )
            elif (
                isinstance(node.func, ast.Attribute)
                and isinstance(node.func.value, ast.Name)
                and node.func.value.id == "os"
                and node.func.attr == "system"
            ):
                add(node, "high", "Use of os.system()", "os.system() invokes a shell command.")
            elif (
                isinstance(node.func, ast.Attribute)
                and isinstance(node.func.value, ast.Name)
                and node.func.value.id == "pickle"
                and node.func.attr in {"load", "loads"}
            ):
                add(
                    node, "high", "Unsafe pickle deserialization",
                    "pickle.load/loads can execute code when deserializing untrusted data.",
                )
            elif (
                isinstance(node.func, ast.Attribute)
                and isinstance(node.func.value, ast.Name)
                and node.func.value.id == "yaml"
                and node.func.attr == "load"
                and len(node.args) < 2
                and not any(keyword.arg == "Loader" for keyword in node.keywords)
            ):
                add(
                    node, "high", "Unsafe yaml.load()",
                    "yaml.load() is called without an explicit Loader.",
                )
            elif (
                isinstance(node.func, ast.Attribute)
                and isinstance(node.func.value, ast.Name)
                and node.func.value.id == "subprocess"
                and node.func.attr in {
                    "run", "Popen", "call", "check_call", "check_output"
                }
                and any(
                    keyword.arg == "shell"
                    and isinstance(keyword.value, ast.Constant)
                    and keyword.value.value is True
                    for keyword in node.keywords
                )
            ):
                add(
                    node, "high", "Shell enabled in subprocess call",
                    "The subprocess call sets shell=True, enabling shell interpretation.",
                )
            elif is_open_call(node) and not is_inside_with(node):
                add(
                    node, "medium", "File opened outside a with statement",
                    "The file handle is opened without an enclosing with statement.",
                )

    return sorted(findings.values(), key=lambda item: (item["line"], item["title"]))


def _resolve_command(cmd: list[str]) -> list[str]:
    if not cmd:
        return cmd
    tool_name = cmd[0]
    if tool_name == sys.executable:
        return cmd

    resolved = shutil.which(tool_name)
    if resolved:
        return [resolved, *cmd[1:]]

    for alt in _COMMON_WINDOWS_TOOL_PATHS.get(tool_name, ()):
        if os.path.exists(alt):
            return [alt, *cmd[1:]]
    return cmd


def _run(cmd: list[str]) -> str:
    original_name = cmd[0]
    cmd = _resolve_command(cmd)
    if shutil.which(cmd[0]) is None and cmd[0] != sys.executable:
        return f"[{original_name} is not installed on this machine - skipped]"
    try:
        p = _run_process(cmd)
    except subprocess.TimeoutExpired:
        return f"[{original_name} timed out after {TIMEOUT}s]"
    return ((p.stdout or "") + (p.stderr or "")).strip() or "[no issues reported]"


def _run_process(cmd: list[str]) -> subprocess.CompletedProcess:
    process = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    try:
        stdout, stderr = process.communicate(timeout=TIMEOUT)
    except subprocess.TimeoutExpired as exc:
        if sys.platform == "win32":
            try:
                killer = subprocess.Popen(
                    ["taskkill", "/PID", str(process.pid), "/T", "/F"],
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                )
                killer.communicate(timeout=10)
                if killer.returncode:
                    process.kill()
            except (OSError, subprocess.TimeoutExpired):
                process.kill()
        else:
            process.kill()
        stdout, stderr = process.communicate()
        raise subprocess.TimeoutExpired(
            cmd, TIMEOUT, output=stdout or exc.output, stderr=stderr or exc.stderr
        ) from exc
    return subprocess.CompletedProcess(cmd, process.returncode, stdout, stderr)


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


def run_pylint_errors_only(code: str) -> list[str]:
    path = _write_temp(code, ".py")
    try:
        result = _run_process(
            [
                sys.executable, "-m", "pylint", "--score=n", "--reports=n",
                "--disable=all", "--enable=E", "--msg-template={msg_id}:{msg}",
                path,
            ],
        )
        output = (result.stdout or "") + (result.stderr or "")
        errors = [
            line.strip()
            for line in output.splitlines()
            if re.match(r"^E\d{4}:", line.strip())
        ]
        if result.returncode and not errors:
            raise RuntimeError(
                f"Pylint errors-only check failed: {output.strip() or result.returncode}"
            )
        return errors
    except subprocess.TimeoutExpired as exc:
        raise RuntimeError(f"Pylint errors-only check timed out after {TIMEOUT}s") from exc
    finally:
        os.remove(path)


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
