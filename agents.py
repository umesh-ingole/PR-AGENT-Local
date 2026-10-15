"""Blue boxes: Security, Bug-Detection, Summary, Coder agents."""
import ast
import difflib
import json
import os
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
    "input restrictions or validation. For a mutable default, use None and "
    "create the object inside the function, preserving behavior for callers. "
    "For a bare except or except Exception handler that only passes, catch the "
    "specific exceptions the code can raise. Do not change the input format a function "
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
        raise ValueError(
            f"Response is not valid JSON: {exc.msg} "
            f"(line {exc.lineno}, column {exc.colno})"
        ) from exc
    if not isinstance(data, list) or any(not isinstance(item, dict) for item in data):
        raise ValueError("Response must be a JSON array of finding objects")
    return data


def _report_line_numbers(report: str) -> set[int]:
    lines = set()
    for entry in report.splitlines():
        match = re.search(r"(?:^|:)(\d+):", entry)
        if match:
            lines.add(int(match.group(1)))
    return lines


def _normalize(
    items: list[dict], agent: str, tool_lines: set[int] | None = None,
) -> list[dict]:
    out = []
    tool_lines = tool_lines or set()
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
            "evidence": "tool" if line in tool_lines else "reasoning",
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


def _provider_failure_detail(
    exc: Exception, force_local: bool, attempted_providers: list[str] | None = None,
) -> str:
    failures = getattr(exc, "failures", ())
    if failures:
        return "; ".join(
            f"provider={provider} error={error_type}"
            for provider, error_type in failures
        )
    providers = attempted_providers or (
        ["ollama"] if force_local else [
            provider.strip()
            for provider in os.getenv("LLM_PROVIDERS", "gemini,ollama").split(",")
            if provider.strip()
        ][:2]
    )
    if not providers:
        providers = ["unknown"]
    return "; ".join(
        f"provider={provider} error={type(exc).__name__}"
        for provider in providers
    )


def _llm_unavailable_finding(agent: str, detail: str) -> dict:
    return {
        "line": None,
        "severity": "low",
        "title": "LLM unavailable",
        "detail": detail,
        "code": "",
        "evidence": "tool",
        "agent": "llm-unavailable",
    }


def _is_llm_warning(finding: dict) -> bool:
    return finding.get("agent") in {"llm-unavailable", "agent-response-error"}


def _review(
    system: str, agent: str, state: dict,
) -> tuple[list[dict], list[str], bool, int, dict | None]:
    user = (
        f"Language: {state['language']}\n\n"
        f"STATIC ANALYSIS ({state['static_tool']}):\n{state['static_report'][:3000]}\n\n"
        f"RETRIEVED GUIDANCE:\n{state['context']}\n\n"
        f"CODE (line-numbered):\n{_numbered(state['current_code'])}"
    )
    try:
        text, provider = call_llm(system, user, state.get("force_local", False))
    except Exception as exc:
        provider_details = _provider_failure_detail(
            exc, state.get("force_local", False)
        )
        response_error = _llm_unavailable_finding(
            agent,
            f"{agent} could not complete the review; {provider_details}.",
        )
        return [], [], False, 0, response_error

    providers = [provider]
    retried = False
    try:
        items = _parse_json_array(text)
    except ValueError as first_error:
        retry_error = ""
        retry_failure_details = ""
        strict_system = (
            system
            + "\n\nSTRICT FORMAT REMINDER: Your previous response was invalid. "
            "Return only one valid JSON array of finding objects. Do not include "
            "prose, markdown fences, or explanations. Return [] if there are no issues."
        )
        retried = True
        retry_user = (
            user
            + "\n\nPREVIOUS RESPONSE JSON PARSE ERROR (exact):\n"
            + str(first_error)
            + "\nReturn a corrected JSON array only."
        )
        try:
            text, provider = call_llm(
                strict_system, retry_user, state.get("force_local", False)
            )
            providers.append(provider)
        except Exception as exc:
            retry_error = f"Retry provider call failed: {type(exc).__name__}"
            retry_failure_details = _provider_failure_detail(
                exc, state.get("force_local", False)
            )
            items = []
        else:
            try:
                items = _parse_json_array(text)
            except ValueError as exc:
                retry_error = str(exc)
                retry_failure_details = (
                    f"provider={provider} error=InvalidJSON"
                )
                items = []
            else:
                retry_error = ""

        if retry_error:
            if not retry_failure_details:
                retry_failure_details = (
                    f"provider={providers[0]} error=InvalidJSON; "
                    f"provider={provider} error=InvalidJSON"
                )
            response_error = _llm_unavailable_finding(
                agent,
                f"{agent} could not produce valid JSON after one retry; "
                f"{retry_failure_details}.",
            )
            items = []
        else:
            response_error = None
    else:
        response_error = None
    items, dropped = _verified_findings(items, state["current_code"])
    tool_lines = _report_line_numbers(state.get("static_report", ""))
    tool_lines.update(
        finding["line"]
        for finding in state.get("secret_findings", [])
        if isinstance(finding.get("line"), int)
    )
    return (
        _normalize(items, agent, tool_lines),
        providers,
        retried,
        dropped,
        response_error,
    )
def security_agent(state: dict) -> dict:
    findings, providers, retried, dropped, response_error = _review(
        SECURITY_SYSTEM, "security-agent", state
    )
    retry_note = "; retried invalid JSON once" if retried else ""
    return {"security_findings": findings, "providers": providers,
        "dropped_finding_counts": [dropped],
        "response_errors": [response_error] if response_error else [],
        "log": [f"Security Agent found {len(findings)} issue(s) via {', '.join(providers)}{retry_note}",
            f"Security Agent dropped {dropped} finding(s) as unverified"]
            + ([f"LLM unavailable: {response_error['detail']}"]
               if response_error else [])}


def bug_agent(state: dict) -> dict:
    findings, providers, retried, dropped, response_error = _review(
        BUG_SYSTEM, "bug-agent", state
    )
    retry_note = "; retried invalid JSON once" if retried else ""
    return {"bug_findings": findings, "providers": providers,
        "dropped_finding_counts": [dropped],
        "response_errors": [response_error] if response_error else [],
        "log": [f"Bug-Detection Agent found {len(findings)} issue(s) via {', '.join(providers)}{retry_note}",
            f"Bug-Detection Agent dropped {dropped} finding(s) as unverified"]
            + ([f"LLM unavailable: {response_error['detail']}"]
               if response_error else [])}


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
    best: list[dict] = []
    for f in security + bugs + (secrets or []) + (interface or []):
        exact_title = re.sub(r"\W+", "", f["title"].lower())[:30]
        quoted_code = re.sub(r"\s+", "", f.get("code", ""))
        duplicates = []
        for index, existing in enumerate(best):
            same_existing_key = (
                f["line"] == existing["line"]
                and exact_title == re.sub(r"\W+", "", existing["title"].lower())[:30]
            )
            close_cross_agent_quote = (
                bool(quoted_code)
                and f.get("agent") != existing.get("agent")
                and re.sub(r"\s+", "", existing.get("code", "")) == quoted_code
                and isinstance(f.get("line"), int)
                and isinstance(existing.get("line"), int)
                and abs(f["line"] - existing["line"]) <= 2
                and difflib.SequenceMatcher(
                    None, f["title"].lower(), existing["title"].lower()
                ).ratio() >= 0.6
            )
            if same_existing_key or close_cross_agent_quote:
                duplicates.append(index)
        if not duplicates:
            best.append(f)
            continue
        group = [best[index] for index in duplicates] + [f]
        strongest = min(
            group,
            key=lambda item: (
                SEVERITIES.index(item["severity"]),
                0 if item.get("agent") in {"ast-scan", "secrets-scan"} else 1,
            ),
        )
        first_duplicate = duplicates[0]
        best[first_duplicate] = strongest
        for index in reversed(duplicates[1:]):
            best.pop(index)

    merged = sorted(best, key=lambda f: (SEVERITIES.index(f["severity"]), f["line"] or 0))
    scored_findings = [finding for finding in merged if not _is_llm_warning(finding)]
    score = max(
        0,
        100 - sum(SEVERITY_PENALTY[f["severity"]] for f in scored_findings),
    )
    has_blocker = any(
        f["severity"] in ("critical", "high") for f in scored_findings
    )
    return merged, score, (score >= PASS_SCORE and not has_blocker)


def summary_agent(state: dict) -> dict:
    merged, score, passed = merge_and_score(
        state.get("security_findings", []), state.get("bug_findings", []),
        state.get("secret_findings", []) + state.get("ast_findings", []),
        state.get("interface_findings", []) + state.get("response_errors", []))
    return {
        "merged": merged, "score": score, "passed": passed,
        "score_history": [score],
        "log": [f"Summary Agent: {len(merged)} merged finding(s), score {score}/100 "
                f"({'good enough' if passed else 'needs fixing'})"],
    }


def coder_agent(state: dict) -> dict:
    actionable_findings = [
        finding for finding in state["merged"] if not _is_llm_warning(finding)
    ]
    to_fix = [
        finding for finding in actionable_findings
        if (
            finding["severity"] in ("critical", "high", "medium")
            or finding.get("agent") in {"ast-scan", "secrets-scan"}
        )
    ] or actionable_findings
    if not to_fix and not state.get("validation_errors"):
        return {
            "current_code": state["current_code"],
            "iteration": state.get("iteration", 0),
            "providers": [],
            "interface_findings": [],
            "log": ["Coder Agent skipped: no actionable findings"],
        }
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
    try:
        text, provider = call_llm(
            CODER_SYSTEM, user, state.get("force_local", False)
        )
    except Exception as exc:
        provider_details = _provider_failure_detail(
            exc, state.get("force_local", False)
        )
        warning = _llm_unavailable_finding(
            "Coder Agent",
            f"Coder Agent could not apply fixes; {provider_details}. "
            "The source file was left unchanged.",
        )
        return {
            "current_code": state["current_code"],
            "iteration": state.get("iteration", 0),
            "providers": [],
            "response_errors": [warning],
            "interface_findings": [],
            "log": [f"LLM unavailable: {warning['detail']}"],
        }
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
    counts = {
        severity: sum(
            1 for finding in state["merged"]
            if finding["severity"] == severity and not _is_llm_warning(finding)
        )
        for severity in SEVERITIES
    }
    user = (f"Language: {state['language']}\nFinal score: {state['score']}/100\n"
            f"Remaining findings by severity: {counts}\nFix rounds: {state.get('iteration', 0)}\n"
            f"Remaining findings: {json.dumps(state['merged'][:15])}")
    text, provider = call_llm(SUMMARY_SYSTEM, user, state.get("force_local", False))
    return text.strip(), provider


def _fallback_summary(state: dict) -> str:
    counts = {
        severity: sum(
            1 for finding in state["merged"]
            if finding["severity"] == severity and not _is_llm_warning(finding)
        )
        for severity in SEVERITIES
    }
    return (
        f"Final score {state['score']}/100 after "
        f"{state.get('iteration', 0)} fix round(s). Remaining: {counts}."
    )


def write_summary(state: dict) -> str:
    try:
        return write_summary_with_provider(state)[0]
    except Exception:
        return _fallback_summary(state)
