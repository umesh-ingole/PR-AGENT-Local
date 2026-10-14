"""LangGraph state machine that mirrors the architecture diagram.

START -> static -> rag -> [security || bug] -> summary -> good enough?
                                         ^                 | no -> coder -> [security || bug]
                                         |                 | yes -> finalize -> END
"""
import operator
from typing import Annotated, TypedDict

from langgraph.graph import END, START, StateGraph

import agents
from config import MAX_ITERATIONS
from rag import retrieve_context
from static_analysis import run_static_analysis
from tools import scan_secrets


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
    context: str
    security_findings: list
    bug_findings: list
    interface_findings: list
    merged: list
    score: int
    passed: bool
    iteration: int
    summary: str
    # parallel nodes write these, so they need reducers
    providers: Annotated[list, operator.add]
    log: Annotated[list, operator.add]
    score_history: Annotated[list, operator.add]


def static_node(state: ReviewState) -> dict:
    code = state["current_code"]
    tool, report = run_static_analysis(state["language"], code)
    secret_findings = scan_secrets(code)
    report = _append_secret_findings(report, secret_findings)
    log = [f"Static analysis ran {tool}"]
    if secret_findings:
        log.append(f"Secrets scan found {len(secret_findings)} hard-coded credential(s)")
    return {"static_tool": tool, "static_report": report, "secret_findings": secret_findings,
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
    # Refresh the linter output on the fixed code so the re-check is grounded.
    code = update["current_code"]
    tool, report = run_static_analysis(state["language"], code)
    secret_findings = scan_secrets(code)
    report = _append_secret_findings(report, secret_findings)
    update.update({"static_tool": tool, "static_report": report,
                   "secret_findings": secret_findings})
    if secret_findings:
        update.setdefault("log", []).append(
            f"Secrets scan found {len(secret_findings)} hard-coded credential(s)")
    return update


def route_after_summary(state: ReviewState) -> str:
    limit = state.get("max_iterations", MAX_ITERATIONS)
    if state["passed"] or state.get("iteration", 0) >= limit:
        return "finalize"
    return "coder"


def finalize_node(state: ReviewState) -> dict:
    return {"summary": agents.write_summary(state), "log": ["Report and fixed file ready"]}


def route_rag_node(state: ReviewState) -> str:
    return "security" if state.get("force_local", False) else "parallel"


def route_security_node(state: ReviewState) -> str:
    return "bug" if state.get("force_local", False) else "summary"


def parallel_node(state: ReviewState) -> ReviewState:
    return state


def build_graph():
    g = StateGraph(ReviewState)
    g.add_node("static", static_node)
    g.add_node("rag", rag_node)
    g.add_node("parallel", parallel_node)
    g.add_node("security", agents.security_agent)
    g.add_node("bug", agents.bug_agent)
    g.add_node("summary", agents.summary_agent)
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
