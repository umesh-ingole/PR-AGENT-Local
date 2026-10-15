"""Blue boxes: Security, Bug-Detection, Summary, Coder agents."""
import ast
import json
import re

from config import PASS_SCORE, SEVERITIES, SEVERITY_PENALTY
from llm_gateway import call_llm

_LINE_NUMBER_PREFIX_RE = re.compile(r"^\s*(?:L\s*)?\d+(?:\s*[:|.]|\s+)\s*")

FINDING_SCHEMA = (
    'Return ONLY a JSON array (no prose, no markdown fences). Each item: '
    '{"line": <int or null>, "severity": "critical|high|medium|low", '
    '"title": "<short>", "detail": "<what is wrong and how to fix it>", '
    '"code": "<exact offending line copied from the file>", '
    '"evidence": "tool|reasoning"}. Use "tool" only when a static-analysis line '
    'supports the finding. Return [] if there are no issues.'
)

SECURITY_SYSTEM = (
    "You are the Security Agent in a code-review pipeline. Find security "
    "vulnerabilities only: injection, hard-coded secrets, unsafe memory use, "
    "weak crypto, unsafe deserialization, missing validation. Do not report "
    "style or non-security bugs. Ground findings in the static analysis output "
    "and retrieved guidance when relevant. This code may already have been "
    "corrected. Report only problems present in the code shown, and quote the line. "
    + FINDING_SCHEMA
)

BUG_SYSTEM = (
    "You are the Bug-Detection Agent in a code-review pipeline. Find logic "
    "errors, crashes, resource leaks, off-by-one errors, unhandled edge cases, "
    "and style violations that hurt correctness. Do not report security issues "
    "(another agent does). Ground findings in the static analysis output and "
    "retrieved guidance when relevant. This code may already have been "
    "corrected. Report only problems present in the code shown, and quote the line. "
    + FINDING_SCHEMA
)

CODER_SYSTEM = (
    "You are the Coder Agent. Fix the listed findings in the user's file with "
    "minimal changes. Do not change function signatures, return types, or "
    "exception types unless a finding strictly requires it. If a fix would "
    "change behavior, keep the old behavior where possible and add a '# REVIEW:' "
    "comment explaining the change. Never leave a hard-coded fallback for a "
    "secret; read it with os.environ[\"NAME\"] and no default. When removing "
    "shell=True from subprocess, split string commands with shlex.split and "
    "keep returning the same object type as before. Do not add check=True or "
    "other new exception-raising arguments to existing calls. Do not add new "
    "input restrictions or validation. Do not change the input format a function "
    "accepts, for example replacing eval with json.loads changes accepted syntax; "
    "if the finding requires it, use ast.literal_eval, which accepts the same "
    "Python-literal syntax. Make the smallest change that removes the problem. "
    "Return the COMPLETE corrected file in one fenced code block and nothing else."
)

SUMMARY_SYSTEM = (
    "You are the Summary Agent. Write a 3-5 sentence plain-English summary of "
    "this code review for a developer: what was found, what was fixed, and what "
    "still needs human attention. No headings, no bullet points."
)


def _numbered(code: str) -> str:
    return "\n".join(f"{i:>4}| {line}" for i, line in enumerate(code.splitlines(), 1))


def _parse_json_array(text: str) -> list[dict]:
    text = text.strip()
    if text.startswith("```"):
        match = re.fullmatch(r"```(?:json)?\s*(.*?)\s*```", text, re.S | re.I)
        if not match:
            raise ValueError("Response contains an incomplete JSON code fence")
        text = match.group(1).strip()
    try:
        data = json.loads(text)
    except json.JSONDecodeError as exc:
        raise ValueError("Response is not a valid JSON array") from exc
    if not isinstance(data, list) or any(not isinstance(item, dict) for item in data):
        raise ValueError("Response must be a JSON array of finding objects")
    return data


def _normalize(items: list[dict], agent: str) -> list[dict]:
    out = []
    for d in items:
        sev = str(d.get("severity", "low")).lower()
        if sev not in SEVERITIES:
            sev = "low"
        line = d.get("line")
        out.append({
            "line": line if isinstance(line, int) else None,
            "severity": sev,
            "title": str(d.get("title", "Issue"))[:120],
            "detail": str(d.get("detail", ""))[:600],
            "code": d["code"],
            "evidence": "tool" if d.get("evidence") == "tool" else "reasoning",
            "agent": agent,
        })
    return out


def _verify_finding_code(finding: dict, source_lines: list[str]) -> bool:
    quoted = finding.get("code")
    if not isinstance(quoted, str):
        return False
    quote_variants = []
    for strip_prefix in (False, True):
        quoted_lines = []
        for line in quoted.splitlines():
            candidate = _LINE_NUMBER_PREFIX_RE.sub("", line, count=1) if strip_prefix else line
            quoted_lines.append(re.sub(r"\s+", "", candidate))
        while quoted_lines and not quoted_lines[0]:
            quoted_lines.pop(0)
        while quoted_lines and not quoted_lines[-1]:
            quoted_lines.pop()
        if quoted_lines and quoted_lines not in quote_variants:
            quote_variants.append(quoted_lines)
    if not quote_variants:
        return False

    line = finding.get("line")
    if line is None:
        first_line_indices = range(len(source_lines))
    elif isinstance(line, int) and not isinstance(line, bool) and line > 0:
        start = max(0, line - 3)
        end = min(len(source_lines), line + 2)
        first_line_indices = range(start, end)
    else:
        return False

    normalized_source = [re.sub(r"\s+", "", source_line) for source_line in source_lines]
    for quoted_lines in quote_variants:
        for first_line in first_line_indices:
            candidate_lines = normalized_source[first_line:first_line + len(quoted_lines)]
            if len(candidate_lines) != len(quoted_lines):
                continue
            if all(
                quote_line in source_line if quote_line else not source_line
                for quote_line, source_line in zip(quoted_lines, candidate_lines)
            ):
                return True
    return False


def _verified_findings(items: list[dict], code: str) -> tuple[list[dict], int]:
    source_lines = code.splitlines()
    verified = [item for item in items if _verify_finding_code(item, source_lines)]
    return verified, len(items) - len(verified)


def _review(system: str, agent: str, state: dict) -> tuple[list[dict], list[str], bool, int]:
    user = (
        f"Language: {state['language']}\n\n"
        f"STATIC ANALYSIS ({state['static_tool']}):\n{state['static_report'][:3000]}\n\n"
        f"RETRIEVED GUIDANCE:\n{state['context']}\n\n"
        f"CODE (line-numbered):\n{_numbered(state['current_code'])}"
    )
    text, provider = call_llm(system, user, state.get("force_local", False))
    providers = [provider]
    retried = False
    try:
        items = _parse_json_array(text)
    except ValueError:
        if provider != "ollama":
            raise
        strict_system = (
            system
            + "\n\nSTRICT FORMAT REMINDER: Your previous response was invalid. "
            "Return only one valid JSON array of finding objects. Do not include "
            "prose, markdown fences, or explanations. Return [] if there are no issues."
        )
        text, provider = call_llm(strict_system, user, state.get("force_local", False))
        providers.append(provider)
        items = _parse_json_array(text)
        retried = True
    items, dropped = _verified_findings(items, state["current_code"])
    return _normalize(items, agent), providers, retried, dropped


def security_agent(state: dict) -> dict:
    findings, providers, retried, dropped = _review(SECURITY_SYSTEM, "security-agent", state)
    retry_note = "; retried invalid JSON once" if retried else ""
    return {"security_findings": findings, "providers": providers,
        "log": [f"Security Agent found {len(findings)} issue(s) via {', '.join(providers)}{retry_note}",
            f"Security Agent dropped {dropped} finding(s) as unverified"]}


def bug_agent(state: dict) -> dict:
    findings, providers, retried, dropped = _review(BUG_SYSTEM, "bug-agent", state)
    retry_note = "; retried invalid JSON once" if retried else ""
    return {"bug_findings": findings, "providers": providers,
        "log": [f"Bug-Detection Agent found {len(findings)} issue(s) via {', '.join(providers)}{retry_note}",
            f"Bug-Detection Agent dropped {dropped} finding(s) as unverified"]}


def _function_interfaces(source: str) -> dict[str, list[tuple[tuple[str, str, bool], ...]]]:
    tree = ast.parse(source)
    interfaces: dict[str, list[tuple[tuple[str, str, bool], ...]]] = {}

    class InterfaceVisitor(ast.NodeVisitor):
        def __init__(self):
            self.scope: list[str] = []

        def visit_ClassDef(self, node: ast.ClassDef) -> None:
            self.scope.append(node.name)
            for child in node.body:
                self.visit(child)
            self.scope.pop()

        def _visit_function(self, node: ast.FunctionDef | ast.AsyncFunctionDef) -> None:
            args = node.args
            positional = args.posonlyargs + args.args
            default_start = len(positional) - len(args.defaults)
            parameters = []
            for index, arg in enumerate(positional):
                kind = "positional-only" if index < len(args.posonlyargs) else "positional"
                parameters.append((arg.arg, kind, index >= default_start))
            if args.vararg:
                parameters.append((args.vararg.arg, "vararg", False))
            parameters.extend(
                (arg.arg, "keyword-only", default is not None)
                for arg, default in zip(args.kwonlyargs, args.kw_defaults)
            )
            if args.kwarg:
                parameters.append((args.kwarg.arg, "kwarg", False))

            qualified_name = ".".join(self.scope + [node.name])
            interfaces.setdefault(qualified_name, []).append(tuple(parameters))
            self.scope.append(node.name)
            for child in node.body:
                self.visit(child)
            self.scope.pop()

        def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
            self._visit_function(node)

        def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
            self._visit_function(node)

    InterfaceVisitor().visit(tree)
    return interfaces


def _check_function_interfaces(original: str, fixed: str) -> tuple[list[dict], list[str]]:
    try:
        original_interfaces = _function_interfaces(original)
    except (SyntaxError, ValueError, TypeError) as exc:
        return [], [
            "Coder Agent interface check skipped: original Python code does not parse "
            f"(line {getattr(exc, 'lineno', None) or 'unknown'})"
        ]

    try:
        fixed_interfaces = _function_interfaces(fixed)
    except (SyntaxError, ValueError, TypeError) as exc:
        return [], [
            "Coder Agent interface check skipped: fixed Python code does not parse "
            f"(line {getattr(exc, 'lineno', None) or 'unknown'})"
        ]

    removed = sorted(set(original_interfaces) - set(fixed_interfaces))
    changed = sorted(
        name for name in original_interfaces.keys() & fixed_interfaces.keys()
        if original_interfaces[name] != fixed_interfaces[name]
    )
    if not removed and not changed:
        return [], []

    changes = []
    if removed:
        changes.append("removed function(s): " + ", ".join(removed))
    if changed:
        changes.append("changed parameters/defaults/variadic arguments: " + ", ".join(changed))
    detail = "The corrected file changed its function interface: " + "; ".join(changes) + "."
    finding = {
        "line": None,
        "severity": "medium",
        "title": "Function interface changed",
        "detail": detail,
        "evidence": "tool",
        "agent": "interface-check",
    }
    return [finding], ["Coder Agent interface changed: " + "; ".join(changes)]


def merge_and_score(
    security: list[dict], bugs: list[dict], secrets: list[dict] | None = None,
    interface: list[dict] | None = None,
) -> tuple[list[dict], int, bool]:
    """Deterministic merge + severity score, so the pass/fail decision is traceable."""
    best: dict[tuple, dict] = {}
    for f in security + bugs + (secrets or []) + (interface or []):
        key = (f["line"], re.sub(r"\W+", "", f["title"].lower())[:30])
        if key not in best or SEVERITIES.index(f["severity"]) < SEVERITIES.index(best[key]["severity"]):
            best[key] = f
    merged = sorted(best.values(), key=lambda f: (SEVERITIES.index(f["severity"]), f["line"] or 0))
    score = max(0, 100 - sum(SEVERITY_PENALTY[f["severity"]] for f in merged))
    has_blocker = any(f["severity"] in ("critical", "high") for f in merged)
    return merged, score, (score >= PASS_SCORE and not has_blocker)


def summary_agent(state: dict) -> dict:
    merged, score, passed = merge_and_score(
        state.get("security_findings", []), state.get("bug_findings", []),
        state.get("secret_findings", []), state.get("interface_findings", []))
    return {
        "merged": merged, "score": score, "passed": passed,
        "score_history": [score],
        "log": [f"Summary Agent: {len(merged)} merged finding(s), score {score}/100 "
                f"({'good enough' if passed else 'needs fixing'})"],
    }


def coder_agent(state: dict) -> dict:
    to_fix = [f for f in state["merged"] if f["severity"] in ("critical", "high", "medium")] or state["merged"]
    listing = "\n".join(
        f"- line {f['line']}: [{f['severity']}] {f['title']} - {f['detail']}" for f in to_fix)
    user = f"Language: {state['language']}\n\nFINDINGS TO FIX:\n{listing}\n\nFILE:\n{state['current_code']}"
    validation_errors = state.get("validation_errors", [])
    if validation_errors:
        user += (
            "\n\nVALIDATION ERRORS (exact messages):\n"
            + "\n".join(validation_errors)
            + "\nFix these errors while preserving the existing behavior and interfaces."
        )
    text, provider = call_llm(CODER_SYSTEM, user, state.get("force_local", False))
    m = re.search(r"```[a-zA-Z+#]*\n(.*?)```", text, re.S)
    fixed = (m.group(1) if m else text).rstrip() + "\n"
    iteration = state.get("iteration", 0) + 1
    log = [f"Coder Agent applied fixes (round {iteration}) via {provider}; re-checking"]
    interface_findings = []
    if state.get("language") == "python":
        interface_findings, interface_log = _check_function_interfaces(
            state.get("original_code", state["current_code"]), fixed)
        log.extend(interface_log)
    else:
        log.append(
            f"Coder Agent interface check skipped for non-Python language: {state.get('language')}"
        )
    return {"current_code": fixed, "iteration": iteration, "providers": [provider],
            "interface_findings": interface_findings, "log": log}


def write_summary_with_provider(state: dict) -> tuple[str, str | None]:
    counts = {s: sum(1 for f in state["merged"] if f["severity"] == s) for s in SEVERITIES}
    user = (f"Language: {state['language']}\nFinal score: {state['score']}/100\n"
            f"Remaining findings by severity: {counts}\nFix rounds: {state.get('iteration', 0)}\n"
            f"Remaining findings: {json.dumps(state['merged'][:15])}")
    try:
        text, provider = call_llm(SUMMARY_SYSTEM, user, state.get("force_local", False))
        return text.strip(), provider
    except Exception:
        return (
            f"Final score {state['score']}/100 after {state.get('iteration', 0)} fix round(s). "
            f"Remaining: {counts}.",
            None,
        )


def write_summary(state: dict) -> str:
    return write_summary_with_provider(state)[0]
