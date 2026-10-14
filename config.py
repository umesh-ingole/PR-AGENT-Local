"""Central settings. Reads .env locally and st.secrets on Streamlit Cloud."""
import os
from dotenv import load_dotenv

load_dotenv()


def load_streamlit_secrets() -> None:
    """Copy st.secrets into os.environ so the rest of the code uses one path."""
    try:
        import streamlit as st
        for key, value in st.secrets.items():
            if isinstance(value, str):
                os.environ.setdefault(key, value)
    except Exception:
        pass


EXT_TO_LANG = {
    ".c": "c", ".h": "c",
    ".cpp": "cpp", ".cc": "cpp", ".cxx": "cpp", ".hpp": "cpp",
    ".java": "java",
    ".py": "python",
    ".rb": "ruby",
}
LANG_LABEL = {"c": "C", "cpp": "C++", "java": "Java", "python": "Python", "ruby": "Ruby"}

SEVERITIES = ["critical", "high", "medium", "low"]
SEVERITY_PENALTY = {"critical": 40, "high": 20, "medium": 8, "low": 2}

PASS_SCORE = 80      # "Score good enough?" threshold
MAX_ITERATIONS = 3   # hard stop so the fix loop can never run forever
RAG_TOP_K = 2        # RARe (Meng et al.): more retrieved context can hurt
