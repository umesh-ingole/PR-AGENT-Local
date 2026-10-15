"""Streamlit UI: 'User Uploads Code (Website)' -> ... -> 'User Downloads'."""
import difflib
import os

import streamlit as st

from config import (EXT_TO_LANG, LANG_LABEL, MAX_ITERATIONS, PASS_SCORE,
                    SEVERITIES, load_streamlit_secrets)

load_streamlit_secrets()
from graph import build_graph  # noqa: E402  (after secrets so env is populated)

st.set_page_config(page_title="AI Code Reviewer", page_icon="🔍", layout="wide")

SEV_ICON = {"critical": "🔴", "high": "🟠", "medium": "🟡", "low": "🔵"}
STEP_LABEL = {
    "static": "Static analysis (MCP)",
    "rag": "RAG retrieval",
    "security": "Security Agent",
    "bug": "Bug-Detection Agent",
    "summary": "Summary Agent",
    "coder": "Coder Agent",
    "finalize": "Report + fixed file",
}


@st.cache_resource
def get_graph():
    return build_graph()


def build_report_md(state: dict) -> str:
    lines = [f"# Code review: {state['filename']}", "",
             f"Language: {LANG_LABEL[state['language']]}  ",
             f"Final score: {state['score']}/100  ",
             f"Fix rounds: {state.get('iteration', 0)}", "",
             "## Summary", state.get("summary", ""), "", "## Findings"]
    if not state["merged"]:
        lines.append("No findings.")
    for f in state["merged"]:
        loc = f"line {f['line']}" if f["line"] else "general"
        lines.append(f"- **{f['severity'].upper()}** ({loc}, {f['agent']}, evidence: {f['evidence']}) "
                     f"{f['title']}: {f['detail']}")
    lines += ["", f"## Static analysis ({state['static_tool']})", "```", state["static_report"], "```"]
    return "\n".join(lines)


# ---------------------------------------------------------------- sidebar
with st.sidebar:
    st.header("Settings")
    force_local = st.toggle("Use local model only (Ollama)", value=False,
                            help="Skips Gemini and uses Qwen2.5-Coder through Ollama.")
    default_rounds = 2 if force_local else MAX_ITERATIONS
    max_rounds = st.slider("Max fix rounds", 1, 5, default_rounds)
    st.caption(f"A file passes at score {PASS_SCORE}+ with no critical or high findings.")
    st.divider()
    st.caption("Provider order: Gemini, then Ollama fallback.")
    st.caption("Keys found: " + ", ".join(
        k for k in ("GOOGLE_API_KEY",) if os.getenv(k)) or "none (local model only)")

# ------------------------------------------------------------------- main
st.title("AI Code Reviewer")
st.write("Upload a C, C++, Java, Python, or Ruby file. You get a severity-ranked report and a corrected file.")

uploaded = st.file_uploader("Source file", type=[e.lstrip(".") for e in EXT_TO_LANG])
use_sample = st.button("Try the sample Python file")

code, filename = None, None
if uploaded:
    filename = uploaded.name
    code = uploaded.getvalue().decode("utf-8", errors="replace")
elif use_sample or st.session_state.get("sample"):
    st.session_state["sample"] = True
    filename = "buggy_sample.py"
    with open(os.path.join(os.path.dirname(__file__), "sample", filename), encoding="utf-8") as f:
        code = f.read()

if code is None:
    st.info("Choose a file to begin.")
    st.stop()

ext = os.path.splitext(filename)[1].lower()
language = EXT_TO_LANG.get(ext)
if language is None:
    st.error(f"Unsupported file type {ext}. Supported: {', '.join(EXT_TO_LANG)}")
    st.stop()

st.caption(f"{filename} · detected {LANG_LABEL[language]} · {len(code.splitlines())} lines")
with st.expander("Preview uploaded code"):
    st.code(code, language=language if language != "cpp" else "cpp")

if st.button("Review and fix", type="primary"):
    init = {"filename": filename, "language": language, "original_code": code,
            "current_code": code, "force_local": force_local, "max_iterations": max_rounds,
            "providers": [], "log": [], "score_history": [], "penalty_history": [],
            "rejected_fix_count": 0}
    final = dict(init)
    try:
        with st.status("Running review pipeline...", expanded=True) as status:
            for chunk in get_graph().stream(init, stream_mode="updates"):
                for node, update in chunk.items():
                    if not update:
                        continue
                    for line in update.get("log", []):
                        st.write(f"**{STEP_LABEL.get(node, node)}**: {line}")
                    for key, value in update.items():
                        if key in (
                            "providers", "log", "score_history", "penalty_history",
                            "rejected_fix_count",
                        ):
                            default = 0 if key == "rejected_fix_count" else []
                            final[key] = final.get(key, default) + value
                        else:
                            final[key] = value
            status.update(label="Review complete", state="complete", expanded=False)
        st.session_state["result"] = final
    except Exception as exc:
        st.error(f"Pipeline stopped: {exc}")

result = st.session_state.get("result")
if result and result["filename"] == filename:
    st.divider()
    counts = {s: sum(1 for f in result["merged"] if f["severity"] == s) for s in SEVERITIES}
    cols = st.columns(5)
    cols[0].metric("Score", f"{result['score']}/100")
    for col, s in zip(cols[1:], SEVERITIES):
        col.metric(f"{SEV_ICON[s]} {s}", counts[s])

    if len(result["score_history"]) > 1:
        st.caption("Score per round: " + " → ".join(str(s) for s in result["score_history"]))

    tab_report, tab_diff, tab_static, tab_log = st.tabs(
        ["Report", "Original vs fixed", "Static analysis", "Pipeline log"])

    with tab_report:
        st.write(result.get("summary", ""))
        for f in result["merged"]:
            loc = f"line {f['line']}" if f["line"] else "general"
            with st.expander(f"{SEV_ICON[f['severity']]} {f['title']} ({loc})"):
                st.write(f["detail"])
                st.caption(f"Found by {f['agent']} · evidence: {f['evidence']}")
        if not result["merged"]:
            st.success("No findings remain.")

    with tab_diff:
        diff = "".join(difflib.unified_diff(
            result["original_code"].splitlines(True), result["current_code"].splitlines(True),
            "original", "fixed"))
        st.code(diff or "No changes were made.", language="diff")
        st.caption("Findings above are from the last re-check of the fixed file.")

    with tab_static:
        st.caption(f"Tool: {result['static_tool']}")
        st.code(result["static_report"])
        st.caption("Retrieved context sent to the agents:")
        st.code(result["context"], language="markdown")

    with tab_log:
        st.write("Providers used: " + ", ".join(sorted(set(result["providers"]))) if result["providers"] else "none")
        st.code("\n".join(result["log"]))

    stem, suffix = os.path.splitext(filename)
    d1, d2 = st.columns(2)
    d1.download_button("Download fixed file", result["current_code"],
                       file_name=f"{stem}_fixed{suffix}", type="primary")
    d2.download_button("Download report (.md)", build_report_md(result),
                       file_name=f"{stem}_report.md")
