"""LangGraph state machine that mirrors the architecture diagram.

START -> static -> rag -> [security || bug] -> summary -> good enough?
                                         ^                 | no -> coder -> [security || bug]
                                         |                 | yes -> finalize -> END
"""
import ast
import operator
from typing import Annotated, TypedDict

from langgraph.graph import END, START, StateGraph

import agents
from config import MAX_ITERATIONS, SEVERITY_PENALTY
from rag import retrieve_context
from static_analysis import run_static_analysis
from tools import run_pylint_errors_only, scan_python_patterns, scan_secrets


class ReviewState(TypedDict, total=False):
    filename: str
    language: str
    original_code: str
    current_code: str
    force_local: bool
    max_iterations: int
    static_tool: str
    static_report: str
    secret_findings: list
    ast_findings: list
    context: str
    security_findings: list
    bug_findings: list
    response_errors: Annotated[list, operator.add]
    interface_findings: list
    merged: list
    score: int
    raw_penalty: int
    passed: bool
    stop_no_improvement: bool
    iteration: int
    summary: str
    # parallel nodes write these, so they need reducers
    providers: Annotated[list, operator.add]
    log: Annotated[list, operator.add]
    score_history: Annotated[list, operator.add]
    penalty_history: Annotated[list, operator.add]
    rejected_fix_count: Annotated[int, operator.add]
    dropped_finding_counts: Annotated[list, operator.add]
    dropped_unverified_history: Annotated[list, operator.add]
    dropped_finding_count_index: int


def static_node(state: ReviewState) -> dict:
    code = state["current_code"]
    tool, report = run_static_analysis(state["language"], code)
    secret_findings = scan_secrets(code)
    ast_findings = scan_python_patterns(code) if state["language"] == "python" else []
    report = _append_secret_findings(report, secret_findings + ast_findings)
    log = [f"Static analysis ran {tool}"]
    if secret_findings:
        log.append(f"Secrets scan found {len(secret_findings)} hard-coded credential(s)")
    if ast_findings:
        log.append(f"Python AST scan found {len(ast_findings)} pattern(s)")
    return {"static_tool": tool, "static_report": report, "secret_findings": secret_findings,
            "ast_findings": ast_findings,
            "iteration": 0, "log": log}


def _append_secret_findings(report: str, findings: list[dict]) -> str:
    if not findings:
        return report
    entries = [f"{item['line']}:{item['severity']}:{item['title']} - {item['detail']}"
               for item in findings]
    return f"{report}\n\nDeterministic secrets scan:\n" + "\n".join(entries)


def rag_node(state: ReviewState) -> dict:
    ctx = retrieve_context(state["language"], state["current_code"], state["static_report"])
    return {"context": ctx, "log": ["RAG retrieved style-guide context"]}


def coder_node(state: ReviewState) -> dict:
    update = agents.coder_agent(state)
    previous_code = state["current_code"]
    rejection_errors = []
    if state["language"] == "python":
        baseline_errors = run_pylint_errors_only(previous_code)
        candidate_code = update["current_code"]
        rejection_errors = _python_validation_errors(candidate_code, baseline_errors)
        if rejection_errors:
            retry_state = {
                **state,
                "current_code": candidate_code,
                "validation_errors": rejection_errors,
            }
            retry_update = agents.coder_agent(retry_state)
            retry_errors = _python_validation_errors(
                retry_update["current_code"], baseline_errors)
            retry_fix_logs = [
                line for line in retry_update.get("log", [])
                if line.startswith("Coder Agent applied fixes (round ")
            ]
            update["log"] = [
                line for line in update.get("log", [])
                if not line.startswith("Coder Agent applied fixes (round ")
            ]
            update["providers"] = (
                update.get("providers", []) + retry_update.get("providers", [])
            )
            update["log"] = update.get("log", []) + [
                line for line in retry_update.get("log", [])
                if not line.startswith("Coder Agent applied fixes (round ")
            ]
            update["interface_findings"] = retry_update.get("interface_findings", [])
            update["iteration"] = update.get("iteration", state.get("iteration", 0) + 1)
            if retry_errors:
                error_detail = "; ".join(retry_errors)
                update["current_code"] = previous_code
                update.setdefault("interface_findings", []).append({
                    "line": None,
                    "severity": "high",
                    "title": "Coder Agent fix rejected",
                    "detail": (
                        "Both Coder Agent attempts failed Python validation. "
                        f"Rejected validation errors: {error_detail}"
                    ),
                    "evidence": "tool",
                    "agent": "python-validation",
                })
                update.setdefault("log", []).append(
                    "Coder Agent fix rejected after retry; keeping the previous code. "
                    f"Validation errors: {error_detail}"
                )
                update["rejected_fix_count"] = 1
            else:
                update["current_code"] = retry_update["current_code"]
                if retry_fix_logs:
                    update.setdefault("log", []).append(retry_fix_logs[-1])
    if state["language"] == "python" and not rejection_errors:
        update["current_code"] = candidate_code
    # Refresh the linter output on the fixed code so the re-check is grounded.
    code = update["current_code"]
    tool, report = run_static_analysis(state["language"], code)
    secret_findings = scan_secrets(code)
    ast_findings = scan_python_patterns(code) if state["language"] == "python" else []
    report = _append_secret_findings(report, secret_findings + ast_findings)
    update.update({"static_tool": tool, "static_report": report,
                   "secret_findings": secret_findings, "ast_findings": ast_findings})
    if secret_findings:
        update.setdefault("log", []).append(
            f"Secrets scan found {len(secret_findings)} hard-coded credential(s)")
    if ast_findings:
        update.setdefault("log", []).append(
            f"Python AST scan found {len(ast_findings)} pattern(s)")
    return update


def _python_validation_errors(candidate_code: str, baseline_errors: list[str]) -> list[str]:
    errors = []
    try:
        ast.parse(candidate_code)
    except SyntaxError as exc:
        errors.append(f"SyntaxError: {exc.msg} (line {exc.lineno})")
    candidate_errors = run_pylint_errors_only(candidate_code)
    errors.extend(message for message in candidate_errors if message not in baseline_errors)
    return errors


def route_after_summary(state: ReviewState) -> str:
    limit = state.get("max_iterations", 2 if state.get("force_local", False) else MAX_ITERATIONS)
    if state["passed"] or state.get("stop_no_improvement", False) or state.get("iteration", 0) >= limit:
        return "finalize"
    return "coder"


def summary_node(state: ReviewState) -> dict:
    update = agents.summary_agent(state)
    raw_penalty = sum(SEVERITY_PENALTY[f["severity"]] for f in update["merged"])
    update["raw_penalty"] = raw_penalty
    update["penalty_history"] = [raw_penalty]
    previous_penalties = state.get("penalty_history", [])
    if previous_penalties and raw_penalty >= previous_penalties[-1]:
        update["stop_no_improvement"] = True
        update.setdefault("log", []).append(
            f"Raw penalty did not improve ({previous_penalties[-1]} -> {raw_penalty}); "
            "stopping early"
        )
    else:
        update["stop_no_improvement"] = False
    dropped_counts = state.get("dropped_finding_counts", [])
    previous_index = state.get("dropped_finding_count_index", 0)
    round_dropped = sum(dropped_counts[previous_index:])
    update["dropped_finding_count_index"] = len(dropped_counts)
    update["dropped_unverified_count"] = round_dropped
    update["dropped_unverified_history"] = [round_dropped]
    return update


def finalize_node(state: ReviewState) -> dict:
    summary, provider = agents.write_summary_with_provider(state)
    if provider:
        summary_log = f"Summary report generated via {provider}"
    else:
        summary_log = "Summary report generated with deterministic fallback; no LLM provider succeeded"
    return {"summary": summary, "log": [summary_log, "Report and fixed file ready"]}


def route_rag_node(state: ReviewState) -> str:
    return "security" if state.get("force_local", False) else "parallel"


def route_security_node(state: ReviewState) -> str:
    return "bug" if state.get("force_local", False) else "summary"


def parallel_node(state: ReviewState) -> dict:
    return {}


def build_graph():
    g = StateGraph(ReviewState)
    g.add_node("static", static_node)
    g.add_node("rag", rag_node)
    g.add_node("parallel", parallel_node)
    g.add_node("security", agents.security_agent)
    g.add_node("bug", agents.bug_agent)
    g.add_node("summary", summary_node)
    g.add_node("coder", coder_node)
    g.add_node("finalize", finalize_node)

    g.add_edge(START, "static")
    g.add_edge("static", "rag")
    g.add_conditional_edges("rag", route_rag_node, {
        "security": "security",
        "parallel": "parallel",
    })
    g.add_edge("parallel", "security")
    g.add_edge("parallel", "bug")
    g.add_conditional_edges("security", route_security_node, {
        "bug": "bug",
        "summary": "summary",
    })
    g.add_edge("bug", "summary")
    g.add_conditional_edges("summary", route_after_summary,
                            {"coder": "coder", "finalize": "finalize"})
    g.add_conditional_edges("coder", route_rag_node, {
        "security": "security",
        "parallel": "parallel",
    })
    g.add_edge("finalize", END)
    return g.compile()
