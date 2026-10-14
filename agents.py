"""Blue boxes: Security, Bug-Detection, Summary, Coder agents."""
import json
import re

from config import PASS_SCORE, SEVERITIES, SEVERITY_PENALTY
from llm_gateway import call_llm

FINDING_SCHEMA = (
    'Return ONLY a JSON array (no prose, no markdown fences). Each item: '
    '{"line": <int or null>, "severity": "critical|high|medium|low", '
    '"title": "<short>", "detail": "<what is wrong and how to fix it>", '
    '"evidence": "tool|reasoning"}. Use "tool" only when a static-analysis line '
    'supports the finding. Return [] if there are no issues.'
)

SECURITY_SYSTEM = (
    "You are the Security Agent in a code-review pipeline. Find security "
    "vulnerabilities only: injection, hard-coded secrets, unsafe memory use, "
    "weak crypto, unsafe deserialization, missing validation. Do not report "
    "style or non-security bugs. Ground findings in the static analysis output "
    "and retrieved guidance when relevant. " + FINDING_SCHEMA
)

BUG_SYSTEM = (
    "You are the Bug-Detection Agent in a code-review pipeline. Find logic "
    "errors, crashes, resource leaks, off-by-one errors, unhandled edge cases, "
    "and style violations that hurt correctness. Do not report security issues "
    "(another agent does). Ground findings in the static analysis output and "
    "retrieved guidance when relevant. " + FINDING_SCHEMA
)

CODER_SYSTEM = (
    "You are the Coder Agent. Fix the listed findings in the user's file with "
    "minimal changes. Keep behavior, structure, names, and comments unless a "
    "finding requires changing them. Return the COMPLETE corrected file in one "
    "fenced code block and nothing else."
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
            "evidence": "tool" if d.get("evidence") == "tool" else "reasoning",
            "agent": agent,
        })
    return out


def _review(system: str, agent: str, state: dict) -> tuple[list[dict], str, bool]:
    user = (
        f"Language: {state['language']}\n\n"
        f"STATIC ANALYSIS ({state['static_tool']}):\n{state['static_report'][:3000]}\n\n"
        f"RETRIEVED GUIDANCE:\n{state['context']}\n\n"
        f"CODE (line-numbered):\n{_numbered(state['current_code'])}"
    )
    text, provider = call_llm(system, user, state.get("force_local", False))
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
        items = _parse_json_array(text)
        retried = True
    return _normalize(items, agent), provider, retried


def security_agent(state: dict) -> dict:
    findings, provider, retried = _review(SECURITY_SYSTEM, "security-agent", state)
    retry_note = "; retried invalid JSON once" if retried else ""
    return {"security_findings": findings, "providers": [provider],
            "log": [f"Security Agent found {len(findings)} issue(s) via {provider}{retry_note}"]}


def bug_agent(state: dict) -> dict:
    findings, provider, retried = _review(BUG_SYSTEM, "bug-agent", state)
    retry_note = "; retried invalid JSON once" if retried else ""
    return {"bug_findings": findings, "providers": [provider],
            "log": [f"Bug-Detection Agent found {len(findings)} issue(s) via {provider}{retry_note}"]}


def merge_and_score(
    security: list[dict], bugs: list[dict], secrets: list[dict] | None = None
) -> tuple[list[dict], int, bool]:
    """Deterministic merge + severity score, so the pass/fail decision is traceable."""
    best: dict[tuple, dict] = {}
    for f in security + bugs + (secrets or []):
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
        state.get("secret_findings", []))
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
    text, provider = call_llm(CODER_SYSTEM, user, state.get("force_local", False))
    m = re.search(r"```[a-zA-Z+#]*\n(.*?)```", text, re.S)
    fixed = (m.group(1) if m else text).rstrip() + "\n"
    iteration = state.get("iteration", 0) + 1
    return {"current_code": fixed, "iteration": iteration, "providers": [provider],
            "log": [f"Coder Agent applied fixes (round {iteration}) via {provider}; re-checking"]}


def write_summary(state: dict) -> str:
    counts = {s: sum(1 for f in state["merged"] if f["severity"] == s) for s in SEVERITIES}
    user = (f"Language: {state['language']}\nFinal score: {state['score']}/100\n"
            f"Remaining findings by severity: {counts}\nFix rounds: {state.get('iteration', 0)}\n"
            f"Remaining findings: {json.dumps(state['merged'][:15])}")
    try:
        text, _ = call_llm(SUMMARY_SYSTEM, user, state.get("force_local", False))
        return text.strip()
    except Exception:
        return (f"Final score {state['score']}/100 after {state.get('iteration', 0)} fix round(s). "
                f"Remaining: {counts}.")
